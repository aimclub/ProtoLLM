import json
import logging
from contextlib import asynccontextmanager, contextmanager

import aioredis
import redis

logger = logging.getLogger(__name__)


class RedisWrapper:
    def __init__(self, redis_host: str, redis_port: str | int | None = None):
        self.url = f"redis://{redis_host}:{redis_port}" if redis_port else redis_host

    @asynccontextmanager
    async def get_redis_async(self):
        rd = aioredis.from_url(self.url)
        try:
            yield rd
        finally:
            await rd.close()

    @contextmanager
    def _get_redis(self):
        rd = redis.Redis.from_url(self.url)
        try:
            yield rd
        finally:
            rd.close()

    def save_item(self, key, item: dict) -> None:
        try:
            with self._get_redis() as redis:
                redis.set(key, json.dumps(item))
                redis.publish(key, 'set')
                logger.info(f"The result with the prefix {key} has been successfully "
                            f"written to the redis")
        except Exception as ex:
            msg = f"Saving the result with the {key} prefix has been interrupted. Error: {ex}."
            logger.info(msg)
            raise Exception(msg)

    def get_item(self, key) -> bytes:
        try:
            with self._get_redis() as redis:
                response = redis.get(key)
                logger.info(f"The item with the {key} prefix was successfully retrieved from the redis")
                return response
        except Exception as ex:
            msg = f"The receipt of the element with the {key} prefix was interrupted. Error: {ex}"
            logger.info(msg)
            raise Exception(msg)

    def check_key(self, key) -> bool:
        try:
            with self._get_redis() as redis:
                response = redis.get(key)
                logger.info(f"The request for the {key} key has been successfully processed.")
                return True if response is not None else False
        except Exception as ex:
            msg = f"An error occurred while processing the {key} key. Error: {ex}"
            logger.info(msg)
            raise Exception(msg)

    async def wait_item(self, key, timeout: float = 60) -> bytes:
        """
            Waits for an item to appear in the specified Redis channel.

            Note:
                There is a very small chance that the message might be published
                before the intended channel is subscribed to. This could result
                in the `wait_item` method getting stuck in a loop.
            """
        try:
            async with self.get_redis_async() as redis:
                pubsub = redis.pubsub()
                await pubsub.subscribe(f"{key}")
                while True:
                    message = await pubsub.get_message(ignore_subscribe_messages=True, timeout=timeout)
                    if message is not None:
                        event = message['data']
                        if event == b'set':
                            value = await redis.get(key)
                            if value is not None:
                                logger.info(f"The item with the {key} prefix was successfully retrieved from the redis")
                                return value
        except Exception as ex:
            msg = f"The receipt of the element with the {key} prefix was interrupted. Error: {ex}"
            logger.info(msg)
            raise Exception(msg)
