from fastapi import FastAPI
from protollm_sdk.object_interface import RedisWrapper, RabbitMQWrapper

from protollm_api.config import Config

from protollm_api.backend.endpoints.sync_chat import get_sync_chat_router

app = FastAPI()
config = Config.read_from_env()
redis_db = RedisWrapper(config.redis_host, config.redis_port)
rabbitmq = RabbitMQWrapper(config.rabbit_host, config.rabbit_port, config.rabbit_login, config.rabbit_password)
app.include_router(get_sync_chat_router(config, redis_db, rabbitmq))

