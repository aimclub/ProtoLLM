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


@pytest.fixture
def redis_url():
    """Redis URL for the tests."""
    return "redis://localhost:6379/0"

@pytest.fixture
def rabbitmq_connection_params():
    """RabbitMQ connection parameters."""
    return {
        "host": "localhost",
        "port": 5672,
        "login": "admin",
        "password": "admin",
    }