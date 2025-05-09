"""RabbitMQ implementation of :pyclass:`protollm_sdk.object_interface.message_queue.base.BaseMessageQueue`.

Changes in this revision
------------------------
* **Robust `get()` implementation** – handles cases where
  ``channel.basic_get`` returns ``None`` or ``(None, None, None)`` and supports
  the *timeout* parameter (simple polling with ``time.sleep``).

The rest of the adapter remains unchanged.
"""

import logging
import threading
import time
from typing import Any, Callable, Optional

import pika
from pika import BasicProperties, BlockingConnection, ConnectionParameters, PlainCredentials

from .base import BaseMessageQueue, ReceivedMessage

log = logging.getLogger(__name__)


class RabbitMQQueue(BaseMessageQueue):  # noqa: WPS230
    """Synchronous RabbitMQ adapter."""

    backend_name = "rabbitmq"

    # ------------------------------------------------------------------
    # Construction / connection
    # ------------------------------------------------------------------
    def __init__(  # noqa: WPS211
        self,
        *,
        host: str = "localhost",
        port: int = 5672,
        virtual_host: str = "/",
        username: str = "guest",
        password: str = "guest",
        heartbeat: int | None = 60,
        blocked_connection_timeout: int | None = 30,
    ) -> None:
        self._params = ConnectionParameters(
            host=host,
            port=port,
            virtual_host=virtual_host,
            credentials=PlainCredentials(username, password),
            heartbeat=heartbeat,
            blocked_connection_timeout=blocked_connection_timeout,
        )
        self._connection: BlockingConnection | None = None
        self._channel: pika.channel.Channel | None = None
        self._consumer_tags: list[str] = []
        self._consume_thread: threading.Thread | None = None

    # ------------------------------------------------------------------
    # Base overrides
    # ------------------------------------------------------------------
    def connect(self) -> None:  # noqa: D401
        if self._connection and self._connection.is_open:
            return  # already connected
        log.debug("Connecting to RabbitMQ → %s", self._params.host)
        self._connection = BlockingConnection(self._params)
        self._channel = self._connection.channel()

    def close(self) -> None:  # noqa: D401
        if not self._connection:
            return
        for tag in self._consumer_tags:
            try:
                self._channel.basic_cancel(tag)
            except Exception:  # noqa: BLE001
                log.exception("Failed to cancel consumer %s", tag)
        if self._channel and self._channel.is_open:
            self._channel.close()
        if self._connection.is_open:
            self._connection.close()
        self._connection = None
        self._channel = None

    # ------------------------------------------------------------------
    # Queue declaration
    # ------------------------------------------------------------------
    def declare_queue(  # noqa: D401, WPS211
        self,
        name: str,
        *,
        durable: bool = True,
        auto_delete: bool = False,
        max_priority: int | None = None,
        **kwargs: Any,
    ) -> None:
        assert self._channel, "connect() must be called first"
        arguments: dict[str, Any] | None = None
        if max_priority is not None:
            arguments = {"x-max-priority": int(max_priority)}
        self._channel.queue_declare(
            queue=name,
            durable=durable,
            auto_delete=auto_delete,
            arguments=arguments,
            **kwargs,
        )

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
        assert self._channel, "connect() must be called first"
        body: bytes = message.encode() if isinstance(message, str) else message
        properties = BasicProperties(
            priority=priority,
            headers=headers,
            delivery_mode=2 if persistent else 1,
        )
        self._channel.basic_publish(
            exchange="",
            routing_key=routing_key or queue,
            body=body,
            properties=properties,
            **kwargs,
        )

    # ------------------------------------------------------------------
    # Consumption helpers
    # ------------------------------------------------------------------
    def _translate_message(self, method, props, body) -> ReceivedMessage:  # noqa: D401, N802, WPS110
        return ReceivedMessage(
            body=body,
            delivery_tag=method.delivery_tag,
            headers=getattr(props, "headers", {}) or {},
            routing_key=method.routing_key,
            priority=getattr(props, "priority", None),
        )

    def get(  # noqa: D401, WPS211
        self,
        queue: str,
        *,
        timeout: float | None = None,
        auto_ack: bool = False,
        **kwargs: Any,
    ) -> Optional[ReceivedMessage]:
        """Fetch one message with optional *timeout* (seconds)."""
        assert self._channel, "connect() must be called first"
        start = time.monotonic()
        while True:
            result = self._channel.basic_get(queue=queue, auto_ack=auto_ack)
            if result:
                # pika 1.x: (method, header, body) – method=None when queue empty
                if isinstance(result, tuple):
                    method = result[0]
                    if method is not None:
                        return self._translate_message(*result)
                else:  # some custom adapter could return object
                    method = result.method_frame  # type: ignore[attr-defined]
                    if method is not None:
                        return self._translate_message(
                            method,
                            result.properties,  # type: ignore[attr-defined]
                            result.body,  # type: ignore[attr-defined]
                        )
            # No message yet
            if timeout is not None and (time.monotonic() - start) >= timeout:
                return None
            time.sleep(0.1)

    def consume(  # noqa: D401, WPS211
        self,
        queue: str,
        callback: Callable[[ReceivedMessage], None],
        *,
        auto_ack: bool = False,
        prefetch: int = 1,
        **kwargs: Any,
    ) -> None:
        assert self._channel, "connect() must be called first"
        self._channel.basic_qos(prefetch_count=prefetch)

        def _on_message(ch, method, props, body):  # noqa: D401, N802
            msg = self._translate_message(method, props, body)
            callback(msg)
            if auto_ack is False:
                # Application must ack/nack explicitly
                pass

        tag = self._channel.basic_consume(queue=queue, on_message_callback=_on_message, auto_ack=auto_ack)
        self._consumer_tags.append(tag)

        def _start_consuming():  # noqa: D401
            try:
                self._channel.start_consuming()
            except KeyboardInterrupt:
                pass

        self._consume_thread = threading.Thread(target=_start_consuming, daemon=True)
        self._consume_thread.start()
        self._consume_thread.join()

    # ------------------------------------------------------------------
    # Ack / nack
    # ------------------------------------------------------------------
    def ack(self, delivery_tag: Any) -> None:  # noqa: D401
        assert self._channel, "connect() must be called first"
        self._channel.basic_ack(delivery_tag=delivery_tag)

    def nack(self, delivery_tag: Any, *, requeue: bool = True) -> None:  # noqa: D401
        assert self._channel, "connect() must be called first"
        self._channel.basic_nack(delivery_tag=delivery_tag, requeue=requeue)

