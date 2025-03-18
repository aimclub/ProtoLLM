import asyncio
import json
import time
from unittest.mock import MagicMock, patch, AsyncMock, Mock
import pytest

from protollm_sdk.object_interface import RedisWrapper

@pytest.fixture
def mock_redis():
    with patch("redis.Redis.from_url", return_value=MagicMock()) as mock_redis:
        yield mock_redis

@pytest.fixture
def mock_async_redis():
    with patch("aioredis.from_url", return_value=AsyncMock()) as mock_redis:
        yield mock_redis

@pytest.fixture
def redis_wrapper():
    return RedisWrapper(redis_host="localhost", redis_port=6379)


@pytest.fixture(autouse=True)
def cleanup_redis(redis_wrapper):
    with patch.object(redis_wrapper, '_get_redis', return_value=MagicMock()) as mock_redis:
        mock_redis().flushall()

@pytest.mark.ci
def test_save_item_publishes_message(redis_wrapper, mock_redis):
    key = "test_key"
    item = {"field": "value"}

    mock_redis_instance = mock_redis.return_value
    pubsub = MagicMock()
    mock_redis_instance.pubsub.return_value = pubsub
    pubsub.subscribe.return_value = None

    mock_message = {
        "channel": key.encode("utf-8"),
        "data": b"set"
    }
    pubsub.get_message.return_value = mock_message

    redis_wrapper.save_item(key, item)

    message = None
    timeout = 5
    start_time = time.time()

    while time.time() - start_time < timeout:
        message = pubsub.get_message(ignore_subscribe_messages=True, timeout=0.1)
        if message:
            break

    assert message is not None, "Message was not received from the channel within the timeout"
    assert message["channel"] == key.encode("utf-8")
    assert message["data"] == b"set"

@pytest.mark.ci
def test_get_item_raises_exception(redis_wrapper, mock_redis):
    """Tests that get_item raises an exception on error."""
    key = "test_key"

    mock_redis_instance = mock_redis.return_value
    mock_redis_instance.get.side_effect = Exception(f"The receipt of the element with the {key} prefix was interrupted")

    with pytest.raises(Exception, match=f"The receipt of the element with the {key} prefix was interrupted"):
        redis_wrapper.get_item(key)

@pytest.mark.ci
def test_save_and_get_item(redis_wrapper, mock_redis):
    key = "test_key"
    item = {"field": "value"}

    mock_redis_instance = mock_redis.return_value
    mock_redis_instance.set.return_value = True
    redis_wrapper.save_item(key, item)

    mock_redis_instance.get.return_value = json.dumps(item).encode('utf-8')
    retrieved_item = redis_wrapper.get_item(key)
    assert retrieved_item == b'{"field": "value"}'
    assert json.loads(retrieved_item) == item

@pytest.mark.ci
def test_save_item_raises_exception(redis_wrapper, mock_redis):
    key = "test_key"
    item = {"field": "value"}

    mock_redis_instance = mock_redis.return_value
    mock_redis_instance.set.side_effect = Exception(f"Saving the result with the {key} prefix has been interrupted")

    with pytest.raises(Exception, match=f"Saving the result with the {key} prefix has been interrupted"):
        redis_wrapper.save_item(key, item)

@pytest.mark.ci
def test_check_key(redis_wrapper, mock_redis):
    key = "test_key"
    item = {"field": "value"}

    mock_redis_instance = mock_redis.return_value
    mock_redis_instance.get.return_value = None

    assert redis_wrapper.check_key(key) is False
    redis_wrapper.save_item(key, item)
    mock_redis_instance.get.return_value = json.dumps(item).encode('utf-8')
    assert redis_wrapper.check_key(key) is True

@pytest.mark.ci
def test_check_key_raises_exception(redis_wrapper, mock_redis):
    key = "test_key"

    mock_redis_instance = mock_redis.return_value
    mock_redis_instance.get.side_effect = Exception(f"An error occurred while processing the {key} key")

    with pytest.raises(Exception, match=f"An error occurred while processing the {key} key"):
        redis_wrapper.check_key(key)

@pytest.mark.ci
@pytest.mark.asyncio
async def test_wait_item(redis_wrapper, mock_async_redis):
    key = "test_key"
    item = {"field": "value"}

    mock_pubsub = Mock()
    mock_message = {
        "channel": key.encode("utf-8"),
        "data": b"set",
    }
    mock_pubsub.subscribe = AsyncMock(return_value=None)
    mock_pubsub.get_message = AsyncMock(
        side_effect=[None, mock_message]
    )

    mock_redis_instance = mock_async_redis.return_value
    mock_redis_instance.pubsub = mock_pubsub
    mock_redis_instance.get = AsyncMock(return_value=json.dumps(item).encode("utf-8"))
    mock_redis_instance.pubsub.side_effect = lambda: mock_pubsub

    async def mock_aenter(_):
        return mock_redis_instance

    async def mock_aexit(obj, exc_type, exc, tb):
        return None

    mock_async_redis.__aenter__ = mock_aenter
    mock_async_redis.__aexit__ = mock_aexit

    result = await redis_wrapper.wait_item(key, timeout=5)

    assert result == b'{"field": "value"}'

    mock_pubsub.subscribe.assert_called_once_with(f"{key}")
    mock_pubsub.get_message.assert_called()



@pytest.mark.ci
@pytest.mark.asyncio
async def test_wait_item_raises_exception(redis_wrapper, mock_async_redis):
    key = "test_key"

    mock_pubsub = Mock()
    mock_message = {
        "channel": key.encode("utf-8"),
        "data": b"set",
    }
    mock_pubsub.subscribe = AsyncMock(return_value=None)
    mock_pubsub.get_message = AsyncMock(
        side_effect=[None, mock_message]
    )

    mock_redis_instance = mock_async_redis.return_value
    mock_redis_instance.pubsub = mock_pubsub
    mock_redis_instance.get = AsyncMock(
        side_effect=Exception(f"The receipt of the test element with the {key} prefix was interrupted"))
    mock_redis_instance.pubsub.side_effect = lambda: mock_pubsub

    async def mock_aenter(_):
        return mock_redis_instance

    async def mock_aexit(obj, exc_type, exc, tb):
        return None

    mock_async_redis.__aenter__ = mock_aenter
    mock_async_redis.__aexit__ = mock_aexit

    with pytest.raises(Exception, match=f"The receipt of the test element with the {key} prefix was interrupted"):
        await redis_wrapper.wait_item(key, timeout=1)
