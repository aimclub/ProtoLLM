import logging

from fastapi import WebSocket
from pydantic import BaseModel


logger = logging.getLogger(__name__)


class SocketConnector:
    def __init__(self):
        logger.info(f"Socket connector initialized")
        self.websocket = None

    async def connect(self, websocket: WebSocket):
        logger.info("Connecting to socket")
        self.websocket = websocket
        await self.websocket.accept()

    async def disconnect(self):
        self.websocket = None

    async def send_encoded_model(self, message: BaseModel):
        await self.websocket.send_json(message.model_dump())


async def get_socket_connector() -> SocketConnector:
    return SocketConnector()
