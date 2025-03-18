from dataclasses import dataclass, field

from langchain_core.tools import Tool
from protollm_agents.sdk.base import ModelType, VectorStoreType, AgentType
from protollm_agents.sdk.models import (
    TokenizerModel,
    CompletionModel,
    ChatModel,
    MultimodalModel,
    EmbeddingAPIModel,
)


@dataclass
class Context:
    embeddings: dict[str, EmbeddingAPIModel] = field(default_factory=dict)
    llms: dict[str, CompletionModel | ChatModel] = field(default_factory=dict)
    multimodals: dict[str, MultimodalModel] = field(default_factory=dict)
    tokenizers: dict[str, TokenizerModel] = field(default_factory=dict)
    vector_stores: dict[str, VectorStoreType] = field(default_factory=dict)
    agents: dict[str, AgentType] = field(default_factory=dict)
    tools: dict[str, Tool] = field(default_factory=dict)
