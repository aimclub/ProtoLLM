from typing import Any
from protollm_agents.sdk.models import (
    CompletionModel,
    ChatModel,
    MultimodalModel,
    EmbeddingAPIModel,
    TokenizerModel,
)
from protollm_agents.sdk.vector_stores import BaseVectorStore
from pydantic import BaseModel, Field


class Storage(BaseModel):
    llm_models: dict[str, CompletionModel | ChatModel] = Field(...)
    multimodal_models: dict[str, MultimodalModel] = Field(...)
    embeddings: dict[str, EmbeddingAPIModel] = Field(...)
    tokenizers: dict[str, TokenizerModel] = Field(...)
    vector_store_clients: dict[str, BaseVectorStore] = Field(...)


storage: Storage | None = None


def get_storage() -> Storage:
    if storage is None:
        raise ValueError("Storage is not initialized")
    return storage
