"""Base abstractions for message-queue back-ends with priority support.

This module defines a minimal synchronous interface that concrete adapters such
as RabbitMQ or Redis Streams must implement to interact with ProtoLLM.  The
interface is intentionally small and generic so the rest of the SDK can remain
broker-agnostic while still exposing the common primitives every backend
provides.

Additions compared to the previous revision
-------------------------------------------
* **Priority queues** – the API now understands message priorities, allowing
  high-priority tasks to overtake older low-priority ones when the underlying
  broker supports it.
"""

import abc
from dataclasses import dataclass
from typing import Any, Callable, Optional

__all__ = [
    "ReceivedMessage",
    "BaseMessageQueue",
]


# ---------------------------------------------------------------------------
# Public data structures
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class ReceivedMessage:  # noqa: D101, WPS600
    """Container for a message fetched from a queue.

    Attributes:
        body: Raw message payload.  *str* payloads **must** be UTF-8 encoded by
            implementations before creating the dataclass instance.
        delivery_tag: Backend-specific identifier used to acknowledge or reject
            the message (e.g. RabbitMQ delivery-tag, Redis Streams entry ID).
        headers: Optional key-value metadata delivered with the message.
        routing_key: Optional routing key / stream ID with which the message
            was published.
        priority: Optional integer priority.  Higher numbers mean higher
            priority.  ``None`` = not set / backend default.
    """

    body: bytes
    delivery_tag: Any | None = None
    headers: dict[str, Any] | None = None
    routing_key: str | None = None
    priority: int | None = None


# ---------------------------------------------------------------------------
# Abstract base class
# ---------------------------------------------------------------------------


class BaseMessageQueue(abc.ABC):  # noqa: D101, WPS230
    """Abstract interface for queue back-ends.

    Concrete subclasses **must** provide a one-for-one implementation of each
    abstract method.  This synchronous interface mirrors the style of
    ``ResultStorage`` – the rest of the SDK depends only on these signatures
    and remains oblivious to the actual broker in use.  Async implementations
    may wrap a synchronous adapter or provide an alternate adapter that fulfils
    the same contract.
    """

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    backend_name: str = "abstract"

    @abc.abstractmethod
    def connect(self) -> None:  # noqa: D401
        """Establish a connection with the message broker.

        Multiple calls **should** be idempotent – they must not open duplicate
        network connections when already connected.

        Raises:
            ConnectionError: If the broker cannot be reached or the credentials
                are invalid.
        """

    @abc.abstractmethod
    def close(self) -> None:  # noqa: D401
        """Gracefully close the connection and release resources."""

    # ------------------------------------------------------------------
    # Queue / stream declaration
    # ------------------------------------------------------------------
    @abc.abstractmethod
    def declare_queue(  # noqa: D401, WPS211
            self,
            name: str,
            *,
            durable: bool = True,
            auto_delete: bool = False,
            max_priority: int | None = None,
            **kwargs: Any,
    ) -> None:
        """Ensure that *name* exists in the backend.

        Args:
            name: Logical name of the queue / stream / topic.
            durable: Persist queue data to disk across restarts when supported.
            auto_delete: Remove the queue when the last consumer disconnects,
                if supported by the backend.
            max_priority: Maximum priority level supported by the queue.  Only
                meaningful for brokers that implement priority natively.  If
                *None*, priority support is disabled or determined by backend
                defaults.
            **kwargs: Additional adapter-specific parameters (e.g. ``max_length``
                for Redis Streams).
        """

    # ------------------------------------------------------------------
    # Publishing
    # ------------------------------------------------------------------
    @abc.abstractmethod
    def publish(  # noqa: D401, WPS211
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
        """Push *message* to *queue*.

        Args:
            queue: Target queue / stream name.
            message: Message payload.  *str* payloads **must** be UTF-8 encoded
                by the adapter before transmission.
            priority: Integer from 0 upward – higher numbers mean higher
                priority.  Ignored if the queue was declared without
                *max_priority*.
            routing_key: Optional routing key / stream ID.
            headers: Arbitrary message metadata (converted to headers when the
                backend supports them).
            persistent: Ask the broker to persist the message to disk (when
                supported).
        """

    # ------------------------------------------------------------------
    # Consumption
    # ------------------------------------------------------------------
    @abc.abstractmethod
    def get(  # noqa: D401
            self,
            queue: str,
            *,
            timeout: float | None = None,
            auto_ack: bool = False,
            **kwargs: Any,
    ) -> Optional[ReceivedMessage]:
        """Fetch a single message from *queue*.

        Args:
            queue: Queue / stream name.
            timeout: Seconds to wait for a message; ``None`` blocks forever.
            auto_ack: If *True*, the implementation **must** acknowledge the
                message before returning it.

        Returns:
            The received message or ``None`` if the timeout elapsed.
        """

    @abc.abstractmethod
    def consume(  # noqa: D401, WPS211
            self,
            queue: str,
            callback: Callable[[ReceivedMessage], None],
            *,
            auto_ack: bool = False,
            prefetch: int = 1,
            **kwargs: Any,
    ) -> None:
        """Start a blocking consume loop on *queue*.

        The adapter **must** call *callback* for every received message.  If
        *auto_ack* is *False*, the *callback* is responsible for calling
        :py:meth:`ack` or :py:meth:`nack` using the *delivery_tag* from the
        :pyclass:`ReceivedMessage`.

        Args:
            queue: Queue / stream name.
            callback: User-supplied function that handles each message.
            auto_ack: Whether to acknowledge messages automatically.
            prefetch: Maximum number of unacknowledged messages the broker is
                allowed to deliver at once (``0`` = backend default).
        """

    # ------------------------------------------------------------------
    # Acknowledgement
    # ------------------------------------------------------------------
    @abc.abstractmethod
    def ack(self, delivery_tag: Any) -> None:  # noqa: D401
        """Acknowledge successful processing of *delivery_tag*."""

    @abc.abstractmethod
    def nack(self, delivery_tag: Any, *, requeue: bool = True) -> None:  # noqa: D401
        """Negatively acknowledge *delivery_tag*.

        Args:
            delivery_tag: Identifier returned by the backend on message receipt.
            requeue: If *True*, the message will be re-queued; otherwise it will
                be discarded or dead-lettered depending on broker policy.
        """

    # ------------------------------------------------------------------
    # Context-manager helpers
    # ------------------------------------------------------------------
    def __enter__(self) -> "BaseMessageQueue":  # noqa: D401
        self.connect()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:  # noqa: D401
        self.close()
