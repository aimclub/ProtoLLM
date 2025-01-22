import json
import logging
from contextlib import contextmanager
import pika

logger = logging.getLogger(__name__)


class RabbitMQWrapper:
    def __init__(self, rabbit_host: str, rabbit_port: int, rabbit_user: str, rabbit_password: str, virtual_host: str = '/'):
        """
        Initialize RabbitMQ wrapper.

        :param rabbit_host: RabbitMQ host
        :param rabbit_port: RabbitMQ port
        :param rabbit_user: RabbitMQ username
        :param rabbit_password: RabbitMQ password
        :param virtual_host: RabbitMQ virtual host
        """
        self.connection_params = pika.ConnectionParameters(
            host=rabbit_host,
            port=rabbit_port,
            virtual_host=virtual_host,
            credentials=pika.PlainCredentials(rabbit_user, rabbit_password)
        )

    @contextmanager
    def get_channel(self):
        """
        Provide a channel for RabbitMQ operations.

        :yield: pika channel
        """
        connection = pika.BlockingConnection(self.connection_params)
        channel = connection.channel()
        try:
            yield channel
        finally:
            channel.close()
            connection.close()

    def publish_message(self, queue_name: str, message: dict, priority: int = None):
        """
        Publish a message to a specified queue with an optional priority.

        :param queue_name: Name of the queue to publish to
        :param message: Message to publish (dictionary will be serialized to JSON)
        :param priority: Optional priority of the message (0-255)
        """
        try:
            with self.get_channel() as channel:
                arguments = {}
                if priority is not None:
                    arguments['x-max-priority'] = 10

                channel.queue_declare(queue=queue_name, durable=True, arguments=arguments)

                properties = pika.BasicProperties(
                    delivery_mode=2,
                    priority=priority if priority is not None else 0
                )
                channel.basic_publish(
                    exchange='',
                    routing_key=queue_name,
                    body=json.dumps(message),
                    properties=properties
                )
                logger.info(
                    f"Message published to queue '{queue_name}' with priority {priority if priority is not None else 'None'}")
        except Exception as ex:
            logger.error(f"Failed to publish message to queue '{queue_name}'. Error: {ex}")
            raise Exception(f"Failed to publish message to queue '{queue_name}'. Error: {ex}") from ex

    def consume_messages(self, queue_name: str, callback):
        """
        Start consuming messages from a specified queue.

        :param queue_name: Name of the queue to consume from
        :param callback: Callback function to process messages
        """
        try:
            connection = pika.BlockingConnection(self.connection_params)
            channel = connection.channel()

            channel.queue_declare(queue=queue_name, durable=True)

            channel.basic_consume(
                queue=queue_name,
                on_message_callback=callback,
                auto_ack=True
            )
            logger.info(f"Started consuming messages from queue '{queue_name}'")
            channel.start_consuming()
        except Exception as ex:
            logger.error(f"Failed to consume messages from queue '{queue_name}'. Error: {ex}")
            raise Exception(f"Failed to consume messages from queue '{queue_name}'. Error: {ex}")