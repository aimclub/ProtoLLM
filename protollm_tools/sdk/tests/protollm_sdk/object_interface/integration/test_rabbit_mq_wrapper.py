import pika
import pytest
import json
from time import sleep
import threading
from protollm_sdk.object_interface import RabbitMQWrapper

@pytest.fixture(scope="module")
def rabbit_wrapper():
    """
    Fixture to initialize RabbitMQWrapper with local RabbitMQ.
    """
    rabbit_host = "localhost"
    rabbit_port = 5672
    rabbit_user = "admin"
    rabbit_password = "admin"
    wrapper = RabbitMQWrapper(
        rabbit_host=rabbit_host,
        rabbit_port=rabbit_port,
        rabbit_user=rabbit_user,
        rabbit_password=rabbit_password,
    )
    return wrapper

@pytest.fixture(autouse=True)
def cleanup_queues(rabbit_wrapper):
    """
    Fixture to clean up all queues before each test.
    """
    with rabbit_wrapper.get_channel() as channel:
        channel.queue_declare(queue="test_queue", durable=True)
        channel.queue_purge("test_queue")

        channel.queue_declare(queue="test_priority_queue", durable=True, arguments={"x-max-priority": 10})
        channel.queue_purge("test_priority_queue")

@pytest.mark.local
def test_publish_message(rabbit_wrapper):
    """
    Tests successful message publishing to a queue.
    """
    queue_name = "test_queue"
    message = {"key": "value"}

    rabbit_wrapper.publish_message(queue_name, message)

    with rabbit_wrapper.get_channel() as channel:
        method_frame, header_frame, body = channel.basic_get(queue_name, auto_ack=True)
        assert method_frame is not None, "Message not found in the queue"
        assert json.loads(body) == message, "Message in the queue does not match the sent message"

@pytest.mark.local
def test_publish_message_with_priority(rabbit_wrapper):
    """
    Tests successful message publishing to a queue with priority.
    """
    queue_name = "test_priority_queue"
    message = {"key": "value"}
    priority = 5

    rabbit_wrapper.publish_message(queue_name, message, priority=priority)

    with rabbit_wrapper.get_channel() as channel:
        method_frame, header_frame, body = channel.basic_get(queue_name, auto_ack=True)
        assert method_frame is not None, "Message not found in the queue"
        assert json.loads(body) == message, "Message in the queue does not match the sent message"
        assert header_frame.priority == priority, f"Expected priority {priority}, but got {header_frame.priority}"


@pytest.mark.local
def test_consume_message(rabbit_wrapper):
    """
    Tests successful message consumption from a queue and stopping consuming.
    """
    queue_name = "test_queue"
    message = {"key": "value"}

    rabbit_wrapper.publish_message(queue_name, message)

    consumed_messages = []

    def callback(ch, method, properties, body):
        consumed_messages.append(json.loads(body))

    consuming_thread = threading.Thread(
        target=rabbit_wrapper.consume_messages, args=(queue_name, callback)
    )
    consuming_thread.daemon = True
    consuming_thread.start()

    sleep(2)

    assert len(consumed_messages) == 1, "Message was not consumed"
    assert consumed_messages[0] == message, "Consumed message does not match the sent message"

@pytest.mark.local
def test_publish_message_exception(rabbit_wrapper):
    """
    Tests exception handling during message publishing.
    """
    queue_name = "test_queue"
    message = {"key": "value"}

    invalid_wrapper = RabbitMQWrapper(
        rabbit_host="invalid_host",
        rabbit_port=5672,
        rabbit_user="guest",
        rabbit_password="guest",
    )
    with pytest.raises(Exception, match="Failed to publish message"):
        invalid_wrapper.publish_message(queue_name, message)

@pytest.mark.local
def test_consume_message_exception(rabbit_wrapper):
    """
    Tests exception handling during message consumption.
    """
    queue_name = "test_queue"

    invalid_wrapper = RabbitMQWrapper(
        rabbit_host="invalid_host",
        rabbit_port=5672,
        rabbit_user="guest",
        rabbit_password="guest",
    )

    def callback(ch, method, properties, body):
        pass

    with pytest.raises(Exception, match="Failed to consume messages"):
        invalid_wrapper.consume_messages(queue_name, callback)
