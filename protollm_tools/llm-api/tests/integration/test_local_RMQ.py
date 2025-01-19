import json
import uuid

import pika
import pytest
from protollm_sdk.models.job_context_models import (ChatCompletionModel, PromptMeta, ChatCompletionUnit,
                                                      ChatCompletionTransactionModel, PromptTypes)
from protollm_sdk.object_interface import RabbitMQWrapper

from protollm_api.backend.broker import send_task

@pytest.fixture(scope="module")
def rabbit_client(test_local_config):
    assert test_local_config.rabbit_host == "localhost"
    client = RabbitMQWrapper(test_local_config.rabbit_host,
                             test_local_config.rabbit_port,
                             test_local_config.rabbit_login,
                             test_local_config.rabbit_password)
    return client


@pytest.fixture(scope="module")
def rabbitmq_connection(test_local_config):
    assert test_local_config.rabbit_host == "localhost"
    connection = pika.BlockingConnection(
        pika.ConnectionParameters(
            host=test_local_config.rabbit_host,
            port=test_local_config.rabbit_port,
            virtual_host='/',
            credentials=pika.PlainCredentials(
                username=test_local_config.rabbit_login,
                password=test_local_config.rabbit_password
            )
        )
    )
    channel = connection.channel()

    yield channel

    connection.close()


@pytest.mark.asyncio
async def test_task_in_queue(test_local_config, rabbitmq_connection, rabbit_client):
    queue_name = "test_queue"
    prompt = ChatCompletionModel(
        job_id=str(uuid.uuid4()),
        meta=PromptMeta(),
        messages=[ChatCompletionUnit(role="user", content="test request")]
    )
    transaction = ChatCompletionTransactionModel(prompt=prompt, prompt_type=PromptTypes.CHAT_COMPLETION.value)

    await send_task(test_local_config, queue_name, transaction, rabbit_client)

    method_frame, header_frame, body = rabbitmq_connection.basic_get(queue=queue_name, auto_ack=True)

    assert method_frame is not None
    task = json.loads(body)

    assert task["id"] == prompt.job_id
    assert task["task"] == "generate"

    kwargs = task["kwargs"]
    assert kwargs["prompt_type"] == transaction.prompt_type

    resp_prompt = kwargs["prompt"]
    assert resp_prompt["job_id"] == prompt.job_id

    meta = resp_prompt["meta"]
    assert meta["temperature"] == prompt.meta.temperature
    assert meta["tokens_limit"] == prompt.meta.tokens_limit
    assert meta["stop_words"] == prompt.meta.stop_words
    assert meta["model"] == prompt.meta.model

    message = resp_prompt["messages"]
    assert len(message) == 1
    assert message[0]["content"] == prompt.messages[0].content
    assert message[0]["role"] == prompt.messages[0].role

    method_frame, header_frame, body = rabbitmq_connection.basic_get(queue=queue_name, auto_ack=True)
    assert method_frame is None, "В очереди больше одной задачи"
