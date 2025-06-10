import uvicorn
from fastapi import FastAPI

from protollm_api.backend.endpoints import get_router
from protollm_api.config import Config

app = FastAPI()

config = Config.read_from_env()

app.include_router(get_router(config))
print("config", config)

uvicorn.run(app, host="127.0.0.1", port=6672)