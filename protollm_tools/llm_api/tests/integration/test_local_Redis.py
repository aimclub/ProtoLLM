import asyncio
import uuid

import pytest
from protollm_sdk.models.job_context_models import ResponseModel
from protollm_sdk.object_interface.redis_wrapper import RedisWrapper

from protollm_api.backend.broker import get_result


@pytest.fixture(scope="module")
def redis_client(test_local_config):
    assert test_local_config.redis_host == "localhost"
    client = RedisWrapper(test_local_config.redis_host, test_local_config.redis_port)
    return client


@pytest.mark.asyncio
async def test_get_result_from_local_redis(test_local_config, redis_client):
    task_id = str(uuid.uuid4())
    redis_key = f"{test_local_config.redis_prefix}:{task_id}"
    expected_content = {"content": "success"}

    result_task = asyncio.create_task(
        get_result(test_local_config, task_id, redis_client)
    )

    await asyncio.sleep(1)

    redis_client.save_item(redis_key, expected_content)

    response = await result_task

    assert isinstance(response, ResponseModel)
    assert response.content == "success"
