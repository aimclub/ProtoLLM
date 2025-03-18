import json
from unittest.mock import AsyncMock, patch, ANY

import pytest
from fastapi import FastAPI
from httpx import AsyncClient, ASGITransport
from protollm_sdk.models.job_context_models import ResponseModel, PromptTransactionModel, PromptModel, \
    PromptTypes, ChatCompletionModel, ChatCompletionTransactionModel
from protollm_sdk.object_interface import RabbitMQWrapper

from protollm_api.backend.endpoints import get_router


@pytest.fixture
def test_app(test_local_config):
    app = FastAPI()
    app.include_router(get_router(test_local_config))
    return app


@pytest.mark.asyncio
async def test_generate_endpoint(test_app, test_local_config):
    async with AsyncClient(transport=ASGITransport(app=test_app), base_url="http://testserver") as client:
        with patch("protollm_api.backend.endpoints.send_task", new_callable=AsyncMock) as send_task_mock, \
                patch("protollm_api.backend.endpoints.get_result", new_callable=AsyncMock) as get_result_mock:
            get_result_mock.return_value = ResponseModel(content="Test Response")

            prompt = {
                "job_id": "test-job-id",
                "priority": 3,
                "meta": {
                    "temperature": 0.2,
                    "tokens_limit": 0,
                    "stop_words": [
                        "string"
                    ],
                    "model": "string"
                },
                "content": "string"
            }

            response = await client.post(
                "/generate",
                json=prompt
            )

            prompt_data = PromptModel.model_validate_json(json.dumps(prompt))
            transaction_model = ChatCompletionTransactionModel(
                prompt=ChatCompletionModel.from_prompt_model(prompt_data),
                prompt_type=PromptTypes.CHAT_COMPLETION.value
            )
            send_task_mock.assert_called_once_with(test_local_config, "llm-api-queue", transaction_model, ANY)

            get_result_mock.assert_called_once_with(test_local_config, "test-job-id", ANY)

            assert response.status_code == 200
            assert response.json() == {"content": "Test Response"}


@pytest.mark.asyncio
async def test_chat_completion_endpoint(test_app, test_local_config):
    async with AsyncClient(transport=ASGITransport(app=test_app), base_url="http://testserver") as client:
        with patch("protollm_api.backend.endpoints.send_task", new_callable=AsyncMock) as send_task_mock, \
                patch("protollm_api.backend.endpoints.get_result", new_callable=AsyncMock) as get_result_mock:
            get_result_mock.return_value = ResponseModel(content="Test Response")

            prompt = {
                "job_id": "test-job-id",
                "meta": {
                    "temperature": 0.2,
                    "tokens_limit": 0,
                    "stop_words": [
                        "string"
                    ],
                    "model": "string"
                },
                "messages": [{"role": "user", "content": "string"}]
            }

            response = await client.post(
                "/chat_completion",
                json=prompt
            )

            prompt_data = ChatCompletionModel.model_validate_json(json.dumps(prompt))
            transaction_model = ChatCompletionTransactionModel(
                prompt=prompt_data,
                prompt_type=PromptTypes.CHAT_COMPLETION.value
            )
            send_task_mock.assert_called_once_with(test_local_config, "llm-api-queue", transaction_model, ANY)

            get_result_mock.assert_called_once_with(test_local_config, "test-job-id", ANY)

            assert response.status_code == 200
            assert response.json() == {"content": "Test Response"}
