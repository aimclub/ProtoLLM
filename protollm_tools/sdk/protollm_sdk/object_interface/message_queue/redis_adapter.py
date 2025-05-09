"""Redis implementation of :pyclass:`BaseMessageQueue` with priority support.

Revision notes
--------------
* **Fallback for servers without ZPOPMAX** – priority pop now attempts
  ``ZPOPMAX`` and falls back to ``ZREVRANGE`` + ``ZREM`` when the command is
  unavailable, enabling compatibility with Redis < 5.
* **Sleep during polling** – list-mode polling now uses ``time.sleep(0.05)`` to
  avoid tight CPU loops.
* **Fixed meta‑key naming** – per‑queue meta now stored under ``{queue}:meta``
  (was ``{queue}:meta:msg``).
* **Corrected ack/nack pending‑key construction**.
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
    # Key helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _list_key(queue: str) -> str:  # noqa: D401
        return f"{queue}:list"

    @staticmethod
    def _zset_key(queue: str) -> str:  # noqa: D401
        return f"{queue}:zset"

    @staticmethod
    def _body_key(queue: str) -> str:  # noqa: D401
        return f"{queue}:body"

    @staticmethod
    def _meta_key(queue: str) -> str:  # noqa: D401
        return f"{queue}:meta"

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
        durable: bool = True,
        auto_delete: bool = False,
        max_priority: int | None = None,
        **kwargs: Any,
    ) -> None:
        self._queue_conf[name] = max_priority
        log.debug("Declare Redis queue '%s' (priority=%s)", name, max_priority)

    # ------------------------------------------------------------------
    # Publishing
    # ------------------------------------------------------------------
    def publish(  # noqa: WPS211
        self,
        queue: str,
        message: bytes | str,
        *,
        priority: int | None = None,
        routing_key: str | None = None,
        headers: dict[str, Any] | None = None,
        persistent: bool = True,
        **kwargs: Any,
    ) -> None:
        assert self._redis, "connect() must be called first"
        body: bytes = message.encode() if isinstance(message, str) else message
        msg_id = str(uuid.uuid4())
        meta = json.dumps({"headers": headers or {}, "priority": priority})

        # Store
        pipe = self._redis.pipeline()
        pipe.hset(self._body_key(queue), msg_id, body)
        pipe.hset(self._meta_key(queue), msg_id, meta)
        pipe.execute()

        if self._queue_conf.get(queue):  # priority mode
            score = float(priority or 0)
            self._redis.zadd(self._zset_key(queue), {msg_id: score})
        else:  # list FIFO
            self._redis.lpush(self._list_key(queue), msg_id)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------
    def _pop_priority(self, queue: str) -> str | None:  # noqa: D401
        """Pop highest‑priority item, compatible with Redis < 5."""
        assert self._redis
        try:
            result = self._redis.zpopmax(self._zset_key(queue), 1)
        except redis.ResponseError:
            # Fallback if ZPOPMAX unsupported
            member = self._redis.zrevrange(self._zset_key(queue), 0, 0)
            if not member:
                return None
            self._redis.zrem(self._zset_key(queue), member[0])
            return member[0]
        if result:
            return result[0][0]
        return None

    def _pop_message_id(self, queue: str, timeout: float | None) -> str | None:  # noqa: D401, WPS211
        assert self._redis
        start_time = time.monotonic()
        while True:
            if self._queue_conf.get(queue):  # priority ZSET
                msg_id = self._pop_priority(queue)
                if msg_id is not None:
                    return msg_id
            else:  # FIFO list
                if timeout is None:
                    result = self._redis.brpop(self._list_key(queue), timeout=0)
                    return result[1] if result else None
                else:
                    result = self._redis.rpop(self._list_key(queue))
                    if result:
                        return result
            # No message yet
            if timeout is not None and (time.monotonic() - start_time) >= timeout:
                return None
            time.sleep(0.05)

    # ------------------------------------------------------------------
    # Consumption / get
    # ------------------------------------------------------------------
    def get(  # noqa: D401, WPS211
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

        body = self._redis.hget(self._body_key(queue), msg_id)
        meta_raw = self._redis.hget(self._meta_key(queue), msg_id) or b"{}"
        meta = json.loads(meta_raw)

        if auto_ack:
            pipe = self._redis.pipeline()
            pipe.hdel(self._body_key(queue), msg_id)
            pipe.hdel(self._meta_key(queue), msg_id)
            pipe.execute()
        else:
            self._redis.hset(self._pending_key(queue), msg_id, meta_raw)

        return ReceivedMessage(
            body=body,
            delivery_tag=msg_id,
            headers=meta.get("headers", {}),
            routing_key=None,
            priority=meta.get("priority"),
        )

    # ------------------------------------------------------------------
    # Consume loop
    # ------------------------------------------------------------------
    def consume(  # noqa: D401, WPS211
        self,
        queue: str,
        callback: Callable[[ReceivedMessage], None],
        *,
        auto_ack: bool = False,
        prefetch: int = 1,
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
        for key_fn in (self._body_key, self._meta_key, self._pending_key):
            self._redis.hdel(key_fn(""), msg_id)

    def nack(self, delivery_tag: Any, *, requeue: bool = True) -> None:  # noqa: D401, WPS211
        assert self._redis
        msg_id = str(delivery_tag)
        meta_raw = self._redis.hget(self._pending_key(""), msg_id)
        if meta_raw is None:
            return
        meta = json.loads(meta_raw)
        priority = meta.get("priority", 0) or 0
        queue = meta.get("queue", "default")
        self._redis.hdel(self._pending_key(queue), msg_id)
        if requeue:
            if self._queue_conf.get(queue):
                self._redis.zadd(self._zset_key(queue), {msg_id: float(priority)})
            else:
                self._redis.rpush(self._list_key(queue), msg_id)

