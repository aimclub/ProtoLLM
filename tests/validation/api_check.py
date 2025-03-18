from fastapi import FastAPI
from protollm_tools.llm_api.protollm_api.config import Config
from protollm_tools.llm_api.protollm_api.backend.endpoints import get_router

app = FastAPI()

config = Config.read_from_env()

app.include_router(get_router(config))

"""
curl -X POST "http://localhost:8000/generate" -H "Content-Type: application/json" -d '{
  "job_id": "12345",
  "meta": {
    "temperature": 0.5,
    "tokens_limit": 1000,
    "stop_words": ["stop"],
    "model": "gpt-model"
  },
  "content": "What is AI?"
}'

curl -X POST "http://localhost:8000/chat_completion" -H "Content-Type: application/json" -d '{
  "job_id": "12345",
  "meta": {
    "temperature": 0.5,
    "tokens_limit": 1000,
    "stop_words": ["stop"],
    "model": "gpt-model"
  },
  "messages": [
    {"role": "user", "content": "What is AI?"},
    {"role": "assistant", "content": "Artificial Intelligence is..."}
  ]}'
"""
