"""Unit- and integration tests for :pyclass:`RedisQueue`.

New in this revision
--------------------
* Added **redis_url** fixture usage for integration tests instead of hard-coded
  connection strings.
* Added two extra integration tests that initialise the adapter from an
  existing ``redis.Redis`` client object rather than a URL.
"""
import uuid

import pytest
import redis as _redis_module  # noqa: WPS433 – import inside tests

from protollm_sdk.object_interface.message_queue.redis_adapter import RedisQueue

# ---------------------------------------------------------------------------
#                        Shared fixtures from test suite
# ---------------------------------------------------------------------------
# Provided by project-level conftest.py:
#   * redis_client – real Redis connection for integration tests
#   * redis_url    – simple fixture returning "redis://localhost:6379/0"
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
#                               In-memory stub
# ---------------------------------------------------------------------------


class _FakePipeline:  # noqa: WPS110 – inner helper
    def __init__(self, parent):
        self.parent = parent
        self.ops: list = []

    def hset(self, *a, **kw):  # noqa: D401, N802
        self.ops.append(("hset", a, kw))
        return self

    def hdel(self, *a, **kw):  # noqa: D401, N802
        self.ops.append(("hdel", a, kw))
        return self

    def execute(self):  # noqa: D401
        return []


class _FakeRedis:  # noqa: WPS110 – unit-test stub
    def __init__(self):
        self.zadd_calls: list = []
        self.lpush_calls: list = []

    def ping(self):
        return True

    def close(self):
        pass

    # --- pipeline ---------------------------------------------------------
    def pipeline(self):
        return _FakePipeline(self)

    # --- list / zset ------------------------------------------------------
    def zadd(self, key, mapping):
        self.zadd_calls.append((key, mapping))
        return 1

    def zpopmax(self, *_, **__):
        return []

    def lpush(self, key, value):
        self.lpush_calls.append((key, value))
        return 1

    def rpop(self, key):
        return None

    def brpop(self, key, timeout):
        return None

    # --- hash -------------------------------------------------------------
    def hset(self, *_, **__):
        pass

    def hget(self, *_, **__):
        return None

    def hdel(self, *_, **__):
        pass


_ORIG_FROM_URL = _redis_module.from_url


@pytest.fixture(autouse=True)
def _patch_redis(monkeypatch, request):
    """Patch ``redis.from_url`` for **unit** tests only."""

    if request.node.get_closest_marker("integration") is not None:
        # Ensure real Redis for integration tests
        monkeypatch.setattr(
            "protollm_sdk.object_interface.message_queue.redis_adapter.redis.from_url",
            _ORIG_FROM_URL,
            raising=True,
        )
        return None

    fake = _FakeRedis()

    monkeypatch.setattr(
        "protollm_sdk.object_interface.message_queue.redis_adapter.redis.from_url",
        lambda *a, **kw: fake,
        raising=True,
    )
    return fake


# ---------------------------------------------------------------------------
#                            Unit-level tests
# ---------------------------------------------------------------------------

def test_publish_priority_zadd(_patch_redis):
    mq = RedisQueue(url="redis://fake")
    mq.connect()
    mq.declare_queue("q", max_priority=5)
    mq.publish("q", "msg", priority=3)
    assert _patch_redis.zadd_calls


def test_publish_list_lpush(_patch_redis):
    mq = RedisQueue(url="redis://fake")
    mq.connect()
    mq.declare_queue("q")
    mq.publish("q", "msg")
    assert _patch_redis.lpush_calls


# ---------------------------------------------------------------------------
#                 Integration tests – URL-based initialisation
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_priority_ordering_redis_url(redis_url):
    mq = RedisQueue(url=redis_url, decode_responses=False)
    queue_name = f"pytest_{uuid.uuid4().hex}"

    with mq:
        mq.declare_queue(queue_name, max_priority=10)
        mq.publish(queue_name, "low", priority=1)
        mq.publish(queue_name, "high", priority=9)

        first = mq.get(queue_name, timeout=5, auto_ack=True)
        second = mq.get(queue_name, timeout=5, auto_ack=True)

        assert first is not None and second is not None
        assert first.body == b"high"
        assert second.body == b"low"


@pytest.mark.integration
def test_fifo_ordering_redis_url(redis_url):
    mq = RedisQueue(url=redis_url, decode_responses=False)
    queue_name = f"pytest_{uuid.uuid4().hex}"

    with mq:
        mq.declare_queue(queue_name)
        mq.publish(queue_name, "first")
        mq.publish(queue_name, "second")

        first = mq.get(queue_name, timeout=5, auto_ack=True)
        second = mq.get(queue_name, timeout=5, auto_ack=True)

        assert first is not None and second is not None
        assert first.body == b"first"
        assert second.body == b"second"


# ---------------------------------------------------------------------------
#              Integration tests – client-object initialisation
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_priority_ordering_redis_client(redis_client):
    # Build adapter using an existing client instance
    mq = RedisQueue(url="redis://unused", decode_responses=False)
    mq._redis = redis_client  # inject connection directly before connect()
    queue_name = f"pytest_{uuid.uuid4().hex}"

    with mq:
        mq.declare_queue(queue_name, max_priority=10)
        mq.publish(queue_name, "low", priority=1)
        mq.publish(queue_name, "high", priority=9)

        first = mq.get(queue_name, timeout=5, auto_ack=True)
        second = mq.get(queue_name, timeout=5, auto_ack=True)

        assert first and second and first.body == b"high" and second.body == b"low"


@pytest.mark.integration
def test_fifo_ordering_redis_client(redis_client):
    mq = RedisQueue(url="redis://unused", decode_responses=False)
    mq._redis = redis_client
    queue_name = f"pytest_{uuid.uuid4().hex}"

    with mq:
        mq.declare_queue(queue_name)
        mq.publish(queue_name, "first")
        mq.publish(queue_name, "second")

        first = mq.get(queue_name, timeout=5, auto_ack=True)
        second = mq.get(queue_name, timeout=5, auto_ack=True)

        assert first and second and first.body == b"first" and second.body == b"second"

