import logging
from fastapi import APIRouter
from protollm_api.backend.broker import send_task, get_result
from protollm_api.config import Config
from protollm_sdk.models.job_context_models import (
    PromptModel, ResponseModel, ChatCompletionModel,
    PromptTransactionModel, ChatCompletionTransactionModel,
    PromptTypes
)
from protollm_sdk.object_interface.redis_wrapper import RedisWrapper
from protollm_sdk.object_interface.rabbit_mq_wrapper import RabbitMQWrapper

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_router(config: Config) -> APIRouter:
    router = APIRouter(
        prefix="",
        tags=["root"],
        responses={404: {"description": "Not found"}},
    )

    redis_db = RedisWrapper(config.redis_host, config.redis_port)
    rabbitmq = RabbitMQWrapper(config.redis_host, config.redis_port, config.rabbit_login, config.rabbit_password)

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
