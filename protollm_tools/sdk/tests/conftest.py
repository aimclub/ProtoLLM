import redis
import pytest

@pytest.fixture(scope="session")
def redis_client():
    """Shared Redis connection for the whole test session."""
    pool = redis.ConnectionPool(host="localhost", port=6379, db=0, decode_responses=False)
    client = redis.Redis(connection_pool=pool)
    try:
        client.ping()
    except redis.ConnectionError:
        pytest.skip("Redis is not available on localhost:6379")
    client.flushdb()              # clean start
    yield client
    client.flushdb()