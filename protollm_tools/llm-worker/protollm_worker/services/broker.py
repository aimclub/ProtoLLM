import json
import logging

import pika
from protollm_sdk.models.job_context_models import PromptModel, ChatCompletionModel, PromptTransactionModel, \
    PromptWrapper, ChatCompletionTransactionModel
from protollm_sdk.object_interface import RabbitMQWrapper
from protollm_sdk.object_interface.redis_wrapper import RedisWrapper

from protollm_worker.config import Config
from protollm_worker.models.base import BaseLLM

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LLMWrap:
    """
    A wrapper for handling interactions with an LLM model, Redis database, and RabbitMQ message broker.

    This class provides a mechanism for consuming messages from RabbitMQ, processing them with a language model,
    and storing the results in Redis.
    """

    def __init__(self,
                 llm_model: BaseLLM,
                 config: Config):
        """
        Initialize the LLMWrap class with the necessary configurations.

        :param llm_model: The language model to use for processing prompts.
        :type llm_model: BaseLLM
        :param config: Set for setting Redis and RabbitMQ.
        :type config: Config
        """
        self.llm = llm_model
        logger.info('Loaded model')

        self.redis_bd = RedisWrapper(config.redis_host, config.redis_port)
        self.rabbitMQ = RabbitMQWrapper(config.rabbit_host, config.rabbit_port, config.rabbit_login, config.rabbit_password)
        self.redis_prefix = config.redis_prefix
        logger.info('Connected to Redis')

        self.models = {
            'single_generate': PromptModel,
            'chat_completion': ChatCompletionModel,
        }

        self.queue_name = config.queue_name

    def start_connection(self):
        """
        Establish a connection to the RabbitMQ broker and start consuming messages from the specified queue.
        """
        self.rabbitMQ.consume_messages(self.queue_name, self._callback)
        logger.info('Started consuming messages')

    def _dump_from_body(self, message_body) -> PromptModel | ChatCompletionModel:
        """
        Deserialize the message body into a PromptModel or ChatCompletionModel.

        :param message_body: The body of the message to deserialize.
        :type message_body: dict
        :return: A deserialized PromptModel or ChatCompletionModel.
        :rtype: PromptModel | ChatCompletionModel
        """
        return PromptModel(**message_body['kwargs'])

    def _callback(self, ch, method, properties, body):
        """
        Callback function to handle messages consumed from RabbitMQ.

        This function processes the message using the language model and saves the result in Redis.

        :param ch: The channel object.
        :type ch: pika.adapters.blocking_connection.BlockingChannel
        :param method: Delivery method object.
        :type method: pika.spec.Basic.Deliver
        :param properties: Message properties.
        :type properties: pika.spec.BasicProperties
        :param body: The message body.
        :type body: bytes
        """
        logger.info(json.loads(body))
        prompt_wrapper = PromptWrapper(prompt=json.loads(body)['kwargs'])
        transaction: PromptTransactionModel | ChatCompletionTransactionModel = prompt_wrapper.prompt
        func_result = self.llm(transaction)

        logger.info(f'The LLM response for task {transaction.prompt.job_id} has been generated')
        logger.info(f'{self.redis_prefix}:{transaction.prompt.job_id}\n{func_result}')
        self.redis_bd.save_item(f'{self.redis_prefix}:{transaction.prompt.job_id}', {"content": func_result})
        logger.info(f'The response for task {transaction.prompt.job_id} was written to Redis')
