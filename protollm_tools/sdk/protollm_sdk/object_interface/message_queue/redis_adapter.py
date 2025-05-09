"""Redis implementation of :pyclass:`protollm_sdk.object_interface.message_queue.base.BaseMessageQueue`.

The adapter supports two back‑ends depending on whether *max_priority* was
specified at queue declaration time:

* **List mode** (no priority) uses ``LPUSH/RPOP``.  It provides FIFO order and
  can block indefinitely via ``BRPOP``.
* **Priority mode** stores messages inside a **sorted set** and fetches them via
  ``ZPOPMAX`` so higher scores (priorities) are delivered first.

This implementation is intentionally pragmatic – it trades strict delivery
semantics for simplicity and zero external dependencies beyond ``redis-py``.
It is suitable for local workloads and integration tests; production clusters
should consider Redis Streams or an additional Lua script for atomicity.
"""
from __future__ import annotations

import json
import logging
import time
import uuid
from typing import Any, Callable, Optional

import redis

from .base import BaseMessageQueue, ReceivedMessage

log = logging.getLogger(__name__)


class RedisQueue(BaseMessageQueue):  # noqa: WPS230
    """Synchronous Redis adapter with optional priority support."""

    backend_name = "redis"

    # ------------------------------------------------------------------
    # Construction / connection
    # ------------------------------------------------------------------
    def __init__(
        self,
        url: str = "redis://localhost:6379/0",
        *,
        decode_responses: bool = False,
    ) -> None:
        self._url = url
        self._redis: redis.Redis | None = None
        self._decode = decode_responses
        # Track per‐queue max_priority to decide list vs ZSET
        self._queue_conf: dict[str, int | None] = {}

    # ------------------------------------------------------------------
    # Base overrides
    # ------------------------------------------------------------------
    def connect(self) -> None:  # noqa: D401
        if self._redis:
            return
        self._redis = redis.from_url(self._url, decode_responses=self._decode)
        try:
            self._redis.ping()
        except redis.RedisError as exc:  # noqa: WPS329
            raise ConnectionError("Cannot connect to Redis") from exc

    def close(self) -> None:  # noqa: D401
        if self._redis is None:
            return
        try:
            self._redis.close()  # type: ignore[attr-defined]
        finally:
            self._redis = None

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _list_key(queue: str) -> str:  # noqa: D401
        return f"{queue}:list"

    @staticmethod
    def _zset_key(queue: str) -> str:  # noqa: D401
        return f"{queue}:zset"

    @staticmethod
    def _msg_key(queue: str) -> str:  # noqa: D401
        return f"{queue}:msg"

    @staticmethod
    def _pending_key(queue: str) -> str:  # noqa: D401
        return f"{queue}:pending"

    # ------------------------------------------------------------------
    # Queue declaration
    # ------------------------------------------------------------------
    def declare_queue(  # noqa: D401
        self,
        name: str,
        *,
        durable: bool = True,  # ignored; Redis is durable by default
        auto_delete: bool = False,  # ignored
        max_priority: int | None = None,
        **kwargs: Any,
    ) -> None:
        self._queue_conf[name] = max_priority
        # Nothing to do in Redis – keys are created lazily
        if max_priority is not None:
            log.info("Declaring Redis priority queue '%s' (max %s)", name, max_priority)
        else:
            log.info("Declaring Redis list queue '%s'", name)

    # ------------------------------------------------------------------
    # Publishing
    # ------------------------------------------------------------------
    def publish(  # noqa: WPS211
        self,
        queue: str,
        message: bytes | str,
        *,
        priority: int | None = None,
        routing_key: str | None = None,  # unused
        headers: dict[str, Any] | None = None,
        persistent: bool = True,  # ignored; persistence handled by Redis config
        **kwargs: Any,
    ) -> None:
        assert self._redis, "connect() must be called first"
        body: bytes = message.encode() if isinstance(message, str) else message
        msg_id = str(uuid.uuid4())
        meta = json.dumps({"headers": headers or {}, "priority": priority})
        # Store payload and metadata
        self._redis.hset(self._msg_key(queue), mapping={msg_id: body})
        self._redis.hset(self._msg_key(f"{queue}:meta"), mapping={msg_id: meta})

        if self._queue_conf.get(queue):  # priority mode
            score = float(priority or 0)
            self._redis.zadd(self._zset_key(queue), {msg_id: score})
        else:  # list mode – LPUSH for FIFO
            self._redis.lpush(self._list_key(queue), msg_id)

    # ------------------------------------------------------------------
    # Consumption helpers
    # ------------------------------------------------------------------
    def _pop_message_id(self, queue: str, timeout: float | None) -> str | None:  # noqa: D401, WPS211
        """Return next message‑id or ``None`` if timeout."""
        assert self._redis
        start_time = time.monotonic()
        while True:
            if self._queue_conf.get(queue):  # priority ZSET
                result = self._redis.zpopmax(self._zset_key(queue), 1)
                if result:
                    return result[0][0]
            else:  # list
                if timeout is None:
                    result = self._redis.brpop(self._list_key(queue), timeout=0)
                    return result[1] if result else None
                else:  # poll; Redis BRPOP doesn't accept float timeout
                    result = self._redis.rpop(self._list_key(queue))
                    if result:
                        return result
            if timeout is not None and (time.monotonic() - start_time) >= timeout:
                return None
            time.sleep(0.1)

    def get(  # noqa: D401
        self,
        queue: str,
        *,
        timeout: float | None = None,
        auto_ack: bool = False,
        **kwargs: Any,
    ) -> Optional[ReceivedMessage]:
        assert self._redis, "connect() must be called first"
        msg_id = self._pop_message_id(queue, timeout)
        if msg_id is None:
            return None

        body = self._redis.hget(self._msg_key(queue), msg_id)
        meta_raw = self._redis.hget(self._msg_key(f"{queue}:meta"), msg_id) or b"{}"
        meta = json.loads(meta_raw)
        if auto_ack:
            # Remove message data immediately
            pipe = self._redis.pipeline()
            pipe.hdel(self._msg_key(queue), msg_id)
            pipe.hdel(self._msg_key(f"{queue}:meta"), msg_id)
            pipe.execute()
        else:
            # Put into pending for potential re‑queueing
            self._redis.hset(self._pending_key(queue), msg_id, meta_raw)
        return ReceivedMessage(
            body=body,
            delivery_tag=msg_id,
            headers=meta.get("headers", {}),
            routing_key=None,
            priority=meta.get("priority"),
        )

    def consume(  # noqa: D401, WPS211
        self,
        queue: str,
        callback: Callable[[ReceivedMessage], None],
        *,
        auto_ack: bool = False,
        prefetch: int = 1,  # ignored; Redis delivers one message at a time
        **kwargs: Any,
    ) -> None:
        assert self._redis, "connect() must be called first"
        while True:
            msg = self.get(queue, timeout=None, auto_ack=auto_ack)
            if msg is not None:
                callback(msg)

    # ------------------------------------------------------------------
    # Ack / nack
    # ------------------------------------------------------------------
    def ack(self, delivery_tag: Any) -> None:  # noqa: D401
        assert self._redis
        msg_id = str(delivery_tag)
        # Remove from pending & full storage
        pipe = self._redis.pipeline()
        pipe.hdel(self._pending_key(""), msg_id)  # noqa: WPS221 – key built in get
        pipe.hdel(self._msg_key(""), msg_id)
        pipe.execute()

    def nack(self, delivery_tag: Any, *, requeue: bool = True) -> None:  # noqa: D401, WPS211
        assert self._redis
        msg_id = str(delivery_tag)
        meta_raw = self._redis.hget(self._pending_key(""), msg_id)
        if meta_raw is None:
            return  # already processed or unknown
        meta = json.loads(meta_raw)
        priority = meta.get("priority", 0) or 0
        queue = meta.get("queue", "default")
        # Remove from pending
        self._redis.hdel(self._pending_key(queue), msg_id)
        if requeue:
            if self._queue_conf.get(queue):
                self._redis.zadd(self._zset_key(queue), {msg_id: float(priority)})
            else:
                self._redis.rpush(self._list_key(queue), msg_id)
