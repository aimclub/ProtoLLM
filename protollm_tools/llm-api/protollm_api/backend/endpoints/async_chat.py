from fastapi import APIRouter
from protollm_sdk.models.job_context_models import ResponseModel, PromptModel, ChatCompletionTransactionModel, \
    ChatCompletionModel, PromptTypes
from protollm_sdk.object_interface import RedisWrapper, RabbitMQWrapper

from protollm_api.backend.broker import send_task, logger, get_result
from protollm_api.config import Config


def get_sync_chat_router(config: Config, redis_db: RedisWrapper, rabbitmq: RabbitMQWrapper) -> APIRouter:
    router = APIRouter(
        prefix="",
        tags=["root"],
        responses={404: {"description": "Not found"}},
    )

    @router.post('/generate', response_model=ResponseModel)
    async def generate(prompt_data: PromptModel, queue_name: str = config.queue_name):
        transaction_model = ChatCompletionTransactionModel(
            prompt=ChatCompletionModel.from_prompt_model(prompt_data),
            prompt_type=PromptTypes.CHAT_COMPLETION.value
        )
        await send_task(config, queue_name, transaction_model, rabbitmq)
        logger.info(f"Task {prompt_data.job_id} was sent to LLM.")
        return await get_result(config, prompt_data.job_id, redis_db)

    @router.post('/chat_completion', response_model=ResponseModel)
    async def chat_completion(prompt_data: ChatCompletionModel, queue_name: str = config.queue_name):
        transaction_model = ChatCompletionTransactionModel(
            prompt=prompt_data,
            prompt_type=PromptTypes.CHAT_COMPLETION.value
        )
        await send_task(config, queue_name, transaction_model, rabbitmq)
        logger.info(f"Task {prompt_data.job_id} was sent to LLM.")
        return await get_result(config, prompt_data.job_id, redis_db)

    return router