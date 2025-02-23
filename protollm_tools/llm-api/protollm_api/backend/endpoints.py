import logging
from typing import Any, List
from fastapi import APIRouter

from protollm_api.backend.broker import (
    send_task,
    get_result,
    add_queue,
    delete_queue,
    fetch_queues_meta,
    purge_queue,
    get_active_workers,
    update_queue,
    get_all_messages,
    delete_message
)
from protollm_api.backend.models.queue_management import (
    QueueDeclarationModel,
    QueueManagementModel,
    QueuesFetchModel,
    QueueUpdateModel,
    UpdateContentModel,
    ActiveWorkersFetchModel
)
from protollm_api.config import Config
from protollm_sdk.models.job_context_models import (
    PromptModel,
    ResponseModel,
    ChatCompletionModel,
    ChatCompletionTransactionModel,
    PromptTypes
)
from protollm_sdk.object_interface.redis_wrapper import RedisWrapper
from protollm_sdk.object_interface.mongo_db_wrapper import MongoDBWrapper
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
    rabbitmq = RabbitMQWrapper(
        rabbit_host=config.rabbit_host,
        rabbit_port=config.rabbit_port,
        rabbit_user=config.rabbit_login,
        rabbit_password=config.rabbit_password
    )
    mongo_db = MongoDBWrapper(
        mongodb_host=config.mongodb_host,
        mongodb_port=config.mongodb_port,
        mongodb_user=config.mongodb_user,
        mongodb_password=config.mongodb_password,
        database=config.mongodb_database_name,
        collection=config.mongodb_collection_name,
    )

    @router.post('/generate', response_model=ResponseModel)
    async def generate(prompt_data: PromptModel, queue_name: str = config.queue_name):
        transaction_model = ChatCompletionTransactionModel(
            prompt=ChatCompletionModel.from_prompt_model(prompt_data),
            prompt_type=PromptTypes.CHAT_COMPLETION.value
        )
        await send_task(config, queue_name, transaction_model, rabbitmq)
        logger.info(f"Task {prompt_data.job_id} was sent to LLM.")
        return await get_result(config, prompt_data.job_id, mongo_db)

    @router.post('/chat_completion', response_model=ResponseModel)
    async def chat_completion(prompt_data: ChatCompletionModel, queue_name: str = config.queue_name):
        transaction_model = ChatCompletionTransactionModel(
            prompt=prompt_data,
            prompt_type=PromptTypes.CHAT_COMPLETION.value
        )
        await send_task(config, queue_name, transaction_model, rabbitmq)
        logger.info(f"Task {prompt_data.job_id} was sent to LLM.")
        return await get_result(config, prompt_data.job_id, mongo_db)

    @router.post("/queues/declare/{queue_name}")
    async def declare_rabbit_queue(queue_name: str, model: str, durable: bool, description: str, arguments: dict[str, Any]):
        declaration_model = QueueDeclarationModel(
            id=f"{config.mongodb_database_name}:{queue_name}",
            queue_name=queue_name,
            model=model,
            durable=durable,
            arguments=arguments,
            description=description
        )
        logger.info(f"Queue {queue_name} declaration started")
        return await add_queue(declaration_model, rabbitmq=rabbitmq, mongodb=mongo_db)

    @router.post("/queues/delete/{queue_name}")
    async def delete_rabbit_queue(queue_name: str):
        deletion_model = QueueManagementModel(
            id=f"{config.mongodb_database_name}:{queue_name}",
            queue_name=queue_name
        )
        logger.info(f"Queue {queue_name} deletion started")
        return await delete_queue(deletion_model, rabbitmq=rabbitmq, mongodb=mongo_db)

    @router.post("/queues/update_metadata/{queue_name}")
    async def update_rabbit_queue_metadata(queue_name: str, new_model: str, new_description: str):
        modification_model = QueueUpdateModel(
            id=f"{config.mongodb_database_name}:{queue_name}",
            queue_name=queue_name,
            update=UpdateContentModel(
                model=new_model,
                description=new_description
            )
        )
        logger.info(f"Attempting to update queue {queue_name} metadata in MongoDB")
        return await update_queue(modification_model, mongodb=mongo_db)

    @router.get("/queues/all", response_model=List[QueuesFetchModel])
    async def fetch_rabbit_queues():
        logger.info(f"Attempting to fetch RabbitMQ queues meta")
        return await fetch_queues_meta(config, rabbitmq=rabbitmq, mongodb=mongo_db)

    @router.post("/queues/purge/{queue_name}")
    async def purge_rabbit_queue(queue_name: str):
        purge_model = QueueManagementModel(
            id=f"{config.mongodb_database_name}:{queue_name}",
            queue_name=queue_name
        )
        logger.info(f"Attempting to purge queue {queue_name}")
        return await purge_queue(purge_model, rabbitmq=rabbitmq)

    @router.post("/queues/{queue_name}/remove_message/{message_id}")
    async def remove_message_rabbit_queue(queue_name: str, message_id: str):
        logger.info(f"Attempting to delete message {message_id} from queue {queue_name}")
        return await delete_message(queue_name, message_id, rabbitmq=rabbitmq)

    @router.post("/queue/{queue_name}/messages")
    async def get_messages_from_rabbit_queue(queue_name: str):
        model = QueueManagementModel(
            queue_name=queue_name
        )
        logger.info(f"Attempting to get messages from queue {queue_name}")
        return await get_all_messages(model, rabbitmq=rabbitmq)

    @router.get("/queues/active_workers", response_model=ActiveWorkersFetchModel)
    async def fetch_rabbit_active_workers():
        logger.info(f"Attempting to get all active workers")
        return await get_active_workers(config)

    return router
