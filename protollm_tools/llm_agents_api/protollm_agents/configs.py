from typing import Literal
import logging
from protollm_agents.sdk.models import (
    EmbeddingAPIModel,
    CompletionModel,
    MultimodalModel,
    TokenizerModel,
    ChatModel,
)
from pydantic import BaseModel, Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
from protollm_agents.sdk.vector_stores import ChromaVectorStore

logger = logging.getLogger(__name__)


class AgentConfig(BaseModel):
    agent_id: str | None = Field(default=None, description="ID of the agent")
    name: str = Field(..., description="Name of the agent")
    description: str = Field(..., description="Description of the agent")
    class_path: str = Field(..., description="Path to the agent class")
    default_params: dict = Field(..., description="Default parameters for the agent")


class ModelConfig(BaseModel):
    type: Literal["embedding", "completion", "multimodal", "tokenizer", "chat"] = Field(
        ..., description="Type of the model"
    )
    params: dict = Field(..., description="Parameters of the model")

    @model_validator(mode="after")
    def validate_params(self):
        if self.type == "embedding":
            self.params = EmbeddingAPIModel.model_validate(self.params)
        elif self.type == "completion":
            self.params = CompletionModel.model_validate(self.params)
        elif self.type == "chat":
            self.params = ChatModel.model_validate(self.params)
        elif self.type == "multimodal":
            self.params = MultimodalModel.model_validate(self.params)
        elif self.type == "tokenizer":
            self.params = TokenizerModel.model_validate(self.params)
        else:
            raise ValueError(f"Invalid model type: {self.type}")
        return self


class VectorStoreConnectionConfig(BaseModel):
    type: Literal["chroma", "elasticsearch"] = Field(
        ..., description="Type of the vector store"
    )
    params: dict = Field(..., description="Parameters of the vector store")

    @model_validator(mode="after")
    def validate_params(self):
        if self.type == "chroma":
            self.params = ChromaVectorStore.model_validate(self.params)
        else:
            raise ValueError(f"Invalid vector store type: {self.type}")
        return self


class EntrypointConfig(BaseSettings):
    app_port: int = Field(default=8080)
    app_host: str = Field(default="0.0.0.0")
    redis_host: str | None = Field(default=None)
    redis_port: int | None = Field(default=None)
    redis_db: int = Field(default=0)
    postgres_host: str | None = Field(default=None)
    postgres_port: int | None = Field(default=None)
    postgres_user: str | None = Field(default=None)
    postgres_password: str | None = Field(default=None)
    postgres_db: str | None = Field(default=None)
    agents: list[AgentConfig] = Field(default_factory=list)
    models: list[ModelConfig] = Field(default_factory=list)
    vector_stores: list[VectorStoreConnectionConfig] = Field(default_factory=list)

    is_admin: bool = Field(default=False)

    @model_validator(mode="after")
    def validate_prerequisites(self):
        if not (
            (
                is_admin := (
                    self.redis_host is not None
                    and self.redis_port is not None
                    and self.postgres_host is not None
                    and self.postgres_port is not None
                    and self.postgres_user is not None
                    and self.postgres_password is not None
                    and self.postgres_db is not None
                )
            )
            or (
                self.redis_host is None
                and self.redis_port is None
                and self.postgres_host is None
                and self.postgres_port is None
                and self.postgres_user is None
                and self.postgres_password is None
                and self.postgres_db is None
            )
        ):
            raise ValueError(
                "Either all or none of the Redis and Postgres parameters must be provided"
            )
        self.is_admin = is_admin
        return self

    model_config = SettingsConfigDict(env_file="../.env")
