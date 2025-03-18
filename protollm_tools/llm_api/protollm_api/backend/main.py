from fastapi import FastAPI
from protollm_api.config import Config
from protollm_api.backend.endpoints import get_router

app = FastAPI()

config = Config.read_from_env()

app.include_router(get_router(config))
