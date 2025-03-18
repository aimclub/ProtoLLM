import pika
import pytest
import json
from unittest.mock import MagicMock, patch
from protollm_sdk.object_interface import RabbitMQWrapper

@pytest.fixture
def mock_pika():
    with patch("pika.BlockingConnection") as mock_connection:
        mock_channel = MagicMock()
        mock_connection.return_value.channel.return_value = mock_channel
        yield mock_channel

@pytest.fixture
def rabbit_wrapper(mock_pika):
    return RabbitMQWrapper(
        rabbit_host="localhost",
        rabbit_port=5672,
        rabbit_user="admin",
        rabbit_password="admin",
    )

@pytest.mark.ci
def test_publish_message_with_priority(rabbit_wrapper, mock_pika):
    """
    Tests successful message publishing to a queue with priority.
    """
    queue_name = "test_queue"
    message = {"kwargs": {"prompt": {"priority": 3}}}


    rabbit_wrapper.publish_message(queue_name, message)

    mock_pika.queue_declare.assert_called_once_with(
        queue=queue_name,
        durable=True,
        arguments={"x-max-priority": 5}
    )

    mock_pika.basic_publish.assert_called_once_with(
        exchange="",
        routing_key=queue_name,
        body=json.dumps(message),
        properties=pika.BasicProperties(
            delivery_mode=2,
            priority=3
        ),
    )

@pytest.mark.ci
def test_consume_message(rabbit_wrapper, mock_pika):
    queue_name = "test_queue"
    callback = MagicMock()

    rabbit_wrapper.consume_messages(queue_name, callback)

    mock_pika.queue_declare.assert_called_once_with(queue=queue_name, durable=True, arguments={'x-max-priority': 5})
    mock_pika.basic_consume.assert_called_once_with(
        queue=queue_name,
        on_message_callback=callback,
        auto_ack=True,
    )
    mock_pika.start_consuming.assert_called_once()

@pytest.mark.ci
def test_publish_message_exception(mock_pika):
    mock_pika.basic_publish.side_effect = Exception("Mocked exception")
    rabbit_wrapper = RabbitMQWrapper(
        rabbit_host="localhost",
        rabbit_port=5672,
        rabbit_user="admin",
        rabbit_password="admin",
    )

    queue_name = "test_queue"
    message = {"key": "value"}

    with pytest.raises(Exception, match="Failed to publish message"):
        rabbit_wrapper.publish_message(queue_name, message)

@pytest.mark.ci
def test_consume_message_exception(mock_pika):
    mock_pika.start_consuming.side_effect = Exception("Mocked exception")
    rabbit_wrapper = RabbitMQWrapper(
        rabbit_host="localhost",
        rabbit_port=5672,
        rabbit_user="admin",
        rabbit_password="admin",
    )

    queue_name = "test_queue"
    callback = MagicMock()

    with pytest.raises(Exception, match="Failed to consume messages"):
        rabbit_wrapper.consume_messages(queue_name, callback)
