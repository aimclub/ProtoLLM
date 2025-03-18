import uuid

import pytest
from protollm_sdk.models.job_context_models import (
    ChatCompletionModel,
    PromptMeta,
    ChatCompletionUnit,
    ChatCompletionTransactionModel,
    PromptTypes,
)
from protollm_sdk.models.job_context_models import ResponseModel
from protollm_sdk.object_interface.redis_wrapper import RedisWrapper

from protollm_api.backend.broker import get_result
from protollm_api.backend.broker import send_task


@pytest.fixture(scope="module")
def redis_client(test_real_config):
    assert test_real_config.redis_host != "localhost"
    client = RedisWrapper(test_real_config.redis_host, test_real_config.redis_port)
    return client


@pytest.mark.asyncio
@pytest.mark.skip(reason="Test waits infinitely in GitHub Action")
async def test_task_in_queue(test_real_config, redis_client):
    task_id = str(uuid.uuid4())
    prompt = ChatCompletionModel(
        job_id=task_id,
        meta=PromptMeta(),
        messages=[ChatCompletionUnit(role="user", content="Сколько будет 2+2*2?")],
    )
    transaction = ChatCompletionTransactionModel(
        prompt=prompt, prompt_type=PromptTypes.CHAT_COMPLETION.value
    )

    await send_task(test_real_config, test_real_config.queue_name, transaction)

    result = await get_result(test_real_config, task_id, redis_client)

    assert isinstance(result, ResponseModel)
    assert result.content != ""
