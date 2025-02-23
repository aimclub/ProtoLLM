import json
import logging
import httpx

from typing import List
from fastapi import HTTPException

from protollm_api.config import Config
from protollm_api.backend.models.queue_management import (
    QueueDeclarationModel, QueueManagementModel, QueuesFetchModel, ActiveWorkersFetchModel, QueueUpdateModel
)
from protollm_sdk.object_interface.mongo_db_wrapper import MongoDBWrapper
from protollm_sdk.object_interface.rabbit_mq_wrapper import RabbitMQWrapper
from protollm_sdk.models.job_context_models import (
    ResponseModel, ChatCompletionTransactionModel, PromptTransactionModel
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def send_task(config: Config,
                    queue_name: str,
                    transaction: PromptTransactionModel | ChatCompletionTransactionModel,
                    rabbitmq: RabbitMQWrapper,
                    task_type='generate'):
    """
    Sends a task to the RabbitMQ queue.

    Args:
        config (Config): Configuration object containing RabbitMQ connection details.
        queue_name (str): Name of the RabbitMQ queue where the task will be published.
        transaction (PromptTransactionModel | ChatCompletionTransactionModel): Transaction data to be sent.
        rabbitmq (RabbitMQWrapper): Rabbit wrapper object to interact with the Rabbit queue.
        task_type (str, optional): The type of task to be executed (default is 'generate').

    Raises:
        Exception: If there is an error during the connection or message publishing process.
    """
    task = {
        "type": "task",
        "task": task_type,
        "args": [],
        "kwargs": transaction.model_dump(),
        "id": transaction.prompt.job_id,
        "retries": 0,
        "eta": None
    }

    rabbitmq.publish_message(queue_name, task)

async def get_result(config: Config, task_id: str, mongo_db: MongoDBWrapper) -> ResponseModel:
    """
    Retrieves the result of a task from MongoDB.

    Args:
        config (Config): Configuration object containing Redis connection details.
        task_id (str): ID of the task whose result is to be retrieved.
        mongo_db (MongoDBWrapper):  MongoDB wrapper object to interact with MongoDB database.

    Returns:
        ResponseModel: Parsed response model containing the result.

    Raises:
        Exception: If the result is not found within the timeout period or other errors occur.
    """
    logger.info(f"Trying to get data from MongoDB")
    logger.info(f"MongoDB key: {config.mongodb_database_name}:{config.mongodb_collection_name}:{task_id}")
    while True:
        try:

            # Wait for the result to be available in MongoDB
            result = await mongo_db.get_single_document(
                pattern={"_id": f"{config.mongodb_database_name}:{config.mongodb_collection_name}:{task_id}"}
            )
            result["_id"] = str(result["_id"])

            break
        except Exception as ex:
            logger.info(f"Retrying to get data from MongoDB: {ex}")

    return ResponseModel(content=result["content"])

async def add_queue(
        model: QueueDeclarationModel,
        rabbitmq: RabbitMQWrapper,
        mongodb: MongoDBWrapper
) -> str:
    """
    Declares a new specified queue at RabbitMQ and saves metadata to the Redis.

    Args:
        model (QueueDeclarationModel): Pydantic model specified for a queue declaration.
        rabbitmq (RabbitMQWrapper): Rabbit wrapper object to interact with the Rabbit queue.
        mongodb (MongoDBWrapper): MongoDB wrapper object to interact with MongoDB database.
    Returns:
        None
    """
    with rabbitmq.get_channel() as channel:
        channel.queue_declare(
            queue=model.queue_name,
            durable=model.durable,
            arguments=model.arguments
        )
        logger.info(f"Queue {model.queue_name} successfully declared at RabbitMQ")

    data = model.model_dump()
    data["_id"] = data.pop("id")
    logger.info(f"{data}")
    result = await mongodb.insert_single_document(document=data)

    logger.info(f"Queue metadata {data} was successfully saved into MongoDB")
    return str(result.inserted_id)

async def delete_queue(
        model: QueueManagementModel,
        rabbitmq: RabbitMQWrapper,
        mongodb: MongoDBWrapper,
) -> str:
    """
    Removes specified queue from RabbitMQ and Redis using the api.

    Args:
        model (QueueManagementModel): Pydantic model specified for a queue management.
        rabbitmq (RabbitMQWrapper): Rabbit wrapper object to interact with the Rabbit queue.
        mongodb (MongoDBWrapper): MongoDB wrapper object to interact with MongoDB database.

    Returns:
        None
    """
    with rabbitmq.get_channel() as channel:
        channel.queue_delete(queue=model.queue_name)
        logger.info(f"Queue {model.queue_name} was successfully deleted from RabbitMQ")

    data = model.model_dump(include={"id": True, "queue_name": True})
    data["_id"] = data.pop("id")

    result = await mongodb.delete_single_document(pattern=data)
    logger.info(f"Queue {model.queue_name} was successfully deleted from MongoDB")

    return str(result.raw_result)

async def update_queue(model: QueueUpdateModel, mongodb: MongoDBWrapper) -> str:
        """
        Modifies a specific queue record in MongoDB database.

        Args:
            model (QueueUpdateModel): Pydantic model for a record modification that belongs to a specific queue.
            mongodb (MongoDBWrapper): MongoDB wrapper object to interact with MongoDB database.

        Returns:
            A count of modified documents
        """
        pattern = {"_id": model.id, "queue_name": model.queue_name}
        update = model.update.model_dump()

        result = await mongodb.update_single_document(
            pattern=pattern,
            update=update
        )
        logger.info(f"Queue {model.queue_name} was successfully updated in MongoDB")

        return str(result.modified_count)

async def fetch_queues_meta(
        config: Config,
        mongodb: MongoDBWrapper,
        rabbitmq: RabbitMQWrapper
) -> List[QueuesFetchModel]:
    """
    Retrieves all queues and their metadata from RabbitMQ and Redis database.

    Args:
        config (Config): Configuration object containing connection details.
        mongodb (MongoDBWrapper): MongoDB wrapper object to interact with MongoDB database.
        rabbitmq (RabbitMQWrapper): Rabbit wrapper object to interact with the Rabbit queue.
    Returns:
        List[QueuesFetchModel] - all queues with their metadata
    """

    url = f"http://{config.rabbit_host}:{config.rabbit_web_port}/api/queues/"

    retrieved_data = []

    async with httpx.AsyncClient() as client:

        response = await client.get(url, auth=(config.rabbit_login, config.rabbit_password))
        logger.info(f"Queues were requested from RabbitMQ management API")

        if response.status_code != 200:
            raise HTTPException(status_code=response.status_code)

        # iterating over requested queues
        for q in response.json():

            meta = await mongodb.get_single_document(pattern={"queue_name": q["name"]})

            if meta:

                meta["_id"] = str(meta["_id"])

                with rabbitmq.get_channel() as channel:

                    # getting the count of consumers and messages
                    queue = channel.queue_declare(queue=meta["queue_name"], passive=True)
                    consumers, messages = queue.method.consumer_count, queue.method.message_count

                    retrieved_data.append(
                        QueuesFetchModel(
                            id=meta["_id"],
                            queue_name=meta["queue_name"],
                            model=meta["model"],
                            description=meta["description"],
                            consumers_count=int(consumers),
                            messages_count=int(messages)
                        )
                    )
        return retrieved_data

async def get_active_workers(config: Config):
    """
    Requests all active workers (consumers) for RabbitMQ.

    Args:
        config (Config): Configuration object containing connection details.

    Returns:
        ActiveWorkersFetchModel - a model with all workers which active at current RabbiMQ instance.
    """
    url = f"http://{config.rabbit_host}:{config.rabbit_web_port}/api/consumers/"

    async with httpx.AsyncClient() as client:

        response = await client.get(url, auth=(config.rabbit_login, config.rabbit_password))

        if response.status_code != 200:
            raise HTTPException(status_code=response.status_code)

        return ActiveWorkersFetchModel(content=[w for w in response.json() if w["active"]])

async def purge_queue(model: QueueManagementModel, rabbitmq: RabbitMQWrapper) -> None:
    """
    Purges all messages from a specified RabbitMQ queue

    Args:
        model (QueueManagementModel): Pydantic model specified for a queue management
        rabbitmq (RabbitMQWrapper): Rabbit wrapper object to interact with the Rabbit queue.

    Returns:
        None
    """
    queue = model.queue_name

    with rabbitmq.get_channel() as channel:
        channel.queue_purge(queue)
        logger.info(f"Queue {queue} was successfully purged.")

async def get_all_messages(model: QueueManagementModel, rabbitmq: RabbitMQWrapper) -> list:
    """
    Collects and returns all messages for specified RabbitMQ queue.

    Args:
        model (QueueManagementModel): Pydantic model specified for a queue management
        rabbitmq (RabbitMQWrapper): Rabbit wrapper object to interact with the Rabbit queue.

    Returns:
        A list of messages from transmitted RabbitMQ queue.
    """
    messages_list = []

    with rabbitmq.get_channel() as channel:

        # iterating over consumption generator without acknowledgement
        for method, properties, body in channel.consume(queue=model.queue_name, auto_ack=False, inactivity_timeout=1):

            # canceling the consumption
            if not method:
                channel.cancel()
                break

            # collecting messages
            messages_list.append(json.loads(bytes.decode(body)))

    return messages_list


async def delete_message(queue_name: str, message_id: str, rabbitmq: RabbitMQWrapper) -> None:
    """
    Deletes a message from a transmitted RabbitMQ queue and requeue remaining messages.

    Args:
        queue_name: The name of RabbitMQ queue where needs to delete a message.
        message_id: id of the message that need to be deleted from transmitted queue.
        rabbitmq (RabbitMQWrapper): Rabbit wrapper object to interact with the Rabbit queue.

    Returns:
        None
    """
    to_requeue = []

    def _callback(ch, method, properties, body, delete_id) -> None:
        msg = json.loads(bytes.decode(body))

        if msg.get("id") != delete_id:
            to_requeue.append((properties, msg))

        # message acknowledgement
        ch.basic_ack(delivery_tag=method.delivery_tag)

    try:
        with rabbitmq.get_channel() as channel:

            # iterating over consumption generator without acknowledgement
            for method, properties, body in channel.consume(queue=queue_name, auto_ack=False, inactivity_timeout=1):

                if method:
                    _callback(channel, method, properties, body, delete_id=message_id)
                else:
                    channel.cancel()
                    break

            # requeue the remaining messages
            for properties, message in to_requeue:

                channel.basic_publish(
                    exchange='',
                    routing_key=queue_name,
                    body=json.dumps(message),
                    properties=properties
                )
    except Exception as ex:
        logger.info(f"Error occurs during the message deletion: Exception -> {ex}")
        raise ex
