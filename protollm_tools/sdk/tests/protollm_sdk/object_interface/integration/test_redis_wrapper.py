import asyncio
import json
import time

import pytest

from protollm_sdk.object_interface import RedisWrapper


@pytest.fixture
def redis_wrapper():
    return RedisWrapper(redis_host="localhost", redis_port=6379)


@pytest.fixture(autouse=True)
def cleanup_redis(redis_wrapper):
    with redis_wrapper._get_redis() as redis:
        redis.flushall()

@pytest.mark.local
def test_save_item_publishes_message(redis_wrapper):
    key = "test_key"
    item = {"field": "value"}

    with redis_wrapper._get_redis() as redis:
        pubsub = redis.pubsub()
        pubsub.subscribe(key)

        redis_wrapper.save_item(key, item)

        message = None
        timeout = 5
        start_time = time.time()

        while time.time() - start_time < timeout:
            message = pubsub.get_message(ignore_subscribe_messages=True, timeout=0.1)
            if message:
                break

        assert message is not None, "Сообщение не получено из канала в течение таймаута"
        assert message["channel"] == key.encode("utf-8")
        assert message["data"] == b"set"

@pytest.mark.local
def test_get_item_raises_exception(redis_wrapper):
    """Тестирует, что get_item выбрасывает исключение при ошибке."""
    key = "test_key"

    redis_wrapper.url = "redis://invalid_host:6379"
    with pytest.raises(Exception, match=f"The receipt of the element with the {key} prefix was interrupted"):
        redis_wrapper.get_item(key)

@pytest.mark.local
def test_save_and_get_item(redis_wrapper):
    key = "test_key"
    item = {"field": "value"}

    redis_wrapper.save_item(key, item)
    retrieved_item = redis_wrapper.get_item(key)
    assert retrieved_item == b'{"field": "value"}'
    assert json.loads(retrieved_item) == item

@pytest.mark.local
def test_save_item_raises_exception(redis_wrapper):
    key = "test_key"
    item = {"field": "value"}

    redis_wrapper.url = "redis://invalid_host:6379"
    with pytest.raises(Exception, match=f"Saving the result with the {key} prefix has been interrupted"):
        redis_wrapper.save_item(key, item)

@pytest.mark.local
def test_check_key(redis_wrapper):
    key = "test_key"
    item = {"field": "value"}

    assert redis_wrapper.check_key(key) is False
    redis_wrapper.save_item(key, item)
    assert redis_wrapper.check_key(key) is True

@pytest.mark.local
def test_check_key_raises_exception(redis_wrapper):
    key = "test_key"
    redis_wrapper.url = "redis://invalid_host:6379"
    with pytest.raises(Exception, match=f"An error occurred while processing the {key} key"):
        redis_wrapper.check_key(key)

@pytest.mark.local
@pytest.mark.asyncio
async def test_wait_item(redis_wrapper):
    key = "test_key"
    item = {"field": "value"}
    wait_task = asyncio.create_task(redis_wrapper.wait_item(key, timeout=5))
    await asyncio.sleep(1)

    with redis_wrapper._get_redis() as redis:
        redis.set(key, json.dumps(item))
        redis.publish(key, 'set')

    result = await wait_task
    assert result == b'{"field": "value"}'

@pytest.mark.local
@pytest.mark.asyncio
async def test_wait_item_raises_exception(redis_wrapper):
    key = "test_key"
    redis_wrapper.url = "redis://invalid_host:6379"
    with pytest.raises(Exception, match=f"The receipt of the element with the {key} prefix was interrupted"):
        await redis_wrapper.wait_item(key, timeout=1)
