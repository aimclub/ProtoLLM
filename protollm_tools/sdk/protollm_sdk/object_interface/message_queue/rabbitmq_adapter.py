"""RabbitMQ implementation of :pyclass:`protollm_sdk.object_interface.message_queue.base.BaseMessageQueue`.

The adapter is intentionally lightweight and relies on ``pika.BlockingConnection``
so it can live in the same synchronous world as the rest of the ProtoLLM SDK.

Example
-------
>>> from protollm_sdk.object_interface.message_queue.rabbitmq_adapter import RabbitMQQueue
>>> with RabbitMQQueue(host="localhost") as mq:
...     mq.declare_queue("tasks", max_priority=10)
...     mq.publish("tasks", "run-me", priority=5)
"""

import logging
import threading
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
        # Thread for long‐running consume loop
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
    def get(  # noqa: D401
        self,
        queue: str,
        *,
        timeout: float | None = None,
        auto_ack: bool = False,
        **kwargs: Any,
    ) -> Optional[ReceivedMessage]:
        assert self._channel, "connect() must be called first"
        method, props, body = self._channel.basic_get(queue=queue, auto_ack=auto_ack)
        if method is None:
            return None  # empty queue
        return ReceivedMessage(
            body=body,
            delivery_tag=method.delivery_tag,
            headers=props.headers or {},
            routing_key=method.routing_key,
            priority=getattr(props, "priority", None),
        )

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
            msg = ReceivedMessage(
                body=body,
                delivery_tag=method.delivery_tag,
                headers=props.headers or {},
                routing_key=method.routing_key,
                priority=getattr(props, "priority", None),
            )
            callback(msg)
            if auto_ack is False:
                # Application must ack/nack explicitly
                pass

        tag = self._channel.basic_consume(queue=queue, on_message_callback=_on_message, auto_ack=auto_ack)
        self._consumer_tags.append(tag)

        # Run the consumer loop in a separate thread so ``consume`` blocks but
        # can be interrupted via KeyboardInterrupt.
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
