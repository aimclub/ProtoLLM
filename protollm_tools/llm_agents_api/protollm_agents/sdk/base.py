from abc import ABC, abstractmethod
from typing import Any, Type, TypeVar
import uuid
from pydantic import BaseModel, Field, SerializationInfo, field_serializer

from langchain_core.runnables import Runnable
from langchain_core.tools import Tool

from langchain_community.vectorstores import VectorStore


class BaseRunnableModel(BaseModel, ABC):
    name: str = Field(..., description="Name of the model")

    @abstractmethod
    def to_runnable(self) -> Runnable: ...


ModelType = TypeVar("ModelType", bound=BaseRunnableModel)


class BaseVectorStore(BaseModel, ABC):
    model_config = {"arbitrary_types_allowed": True}

    name: str = Field(..., description="Name of the vector store")
    description: str = Field(..., description="Description of the vector store")
    embeddings_model_name: str = Field(
        ..., description="Name of the embeddings model to use"
    )

    embeddings_model: Any = None

    def initialize_embeddings_model(self, models: dict[str, ModelType]):
        if self.embeddings_model is not None:
            return
        try:
            self.embeddings_model = models[self.embeddings_model_name].to_runnable()
        except KeyError:
            raise ValueError(f"Embeddings model {self.embeddings_model_name} not found")

    @abstractmethod
    def to_vector_store(self) -> VectorStore: ...


VectorStoreType = TypeVar("VectorStoreType", bound=BaseVectorStore)


class BaseAgent(ABC):
    agent_id: uuid.UUID = Field(default_factory=uuid.uuid4)
    name: str = Field(..., description="Name of the agent")
    description: str = Field(..., description="Description of the agent")
    arguments: Type["Arguments"] = Field(..., description="Arguments of the agent")
    module_name: str = Field(default=None, description="Module name of the agent")
    class_name: str = Field(default=None, description="Class name of the agent")

    class Arguments(BaseModel, ABC):
        pass

    def __init__(
        self,
        name: str,
        description: str,
        arguments: Arguments,
        agent_id: uuid.UUID | None = None,
        module_name: str | None = None,
        class_name: str | None = None,
    ):
        self.agent_id = agent_id or uuid.uuid4()
        self.name = name
        self.description = description
        self.arguments = arguments
        self.module_name = module_name or self.__module__
        self.class_name = class_name or self.__class__.__name__

    def to_dict(self) -> dict:
        return {
            "agent_id": self.agent_id,
            "name": self.name,
            "description": self.description,
            "arguments": self.arguments.model_dump(),
            "module_name": self.module_name,
            "class_name": self.class_name,
        }

    @classmethod
    def get_arguments_class(cls) -> Type[Arguments]:
        return cls.Arguments

    @abstractmethod
    async def to_tool(self) -> Tool: ...

    @abstractmethod
    async def to_runnable(self) -> Runnable: ...

    @classmethod
    def get_arguments_class(cls) -> Type[BaseModel]:
        return cls.Arguments


AgentType = TypeVar("AgentType", bound=BaseAgent)


class Event(BaseModel, ABC):
    event_id: uuid.UUID = Field(default_factory=uuid.uuid4)
    agent_id: uuid.UUID = Field(
        ..., description="ID of the agent that generated the event"
    )
    description: str | None = Field(None, description="Description of the event")
    is_eos: bool = Field(
        False, description="Whether the event is the end of the stream"
    )

    @field_serializer("event_id")
    def serialize_event_id(self, event_id: uuid.UUID, _info: SerializationInfo) -> str:
        return str(event_id)

    @field_serializer("agent_id")
    def serialize_agent_id(self, agent_id: uuid.UUID, _info: SerializationInfo) -> str:
        return str(agent_id)


class AgentAnswer(BaseModel, ABC):
    result: Any

    @field_serializer("result")
    @abstractmethod
    def serialize_result(self, result: Any, _info: SerializationInfo) -> str: ...
