from pydantic import Field
from langchain_core.runnables import Runnable
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_community.llms.vllm import VLLMOpenAI
from langchain_openai import ChatOpenAI
from abc import ABC, abstractmethod
from transformers import AutoTokenizer

from protollm_agents.sdk.base import BaseRunnableModel


class BaseOpenAIModel(BaseRunnableModel, ABC):
    temperature: float = Field(default=0.01, description="Temperature of the model")
    top_p: float = Field(default=0.95, description="Top-p of the model")
    streaming: bool = Field(
        default=False, description="Whether to stream from the model"
    )
    model: str = Field(..., description="Model path / name on server")
    url: str = Field(..., description="URL of the model")
    api_key: str = Field(..., description="API key to access the model via LLM API")


class CompletionModel(BaseOpenAIModel):
    def to_runnable(self) -> Runnable:
        llm = VLLMOpenAI(
            name=self.name,
            model=self.model,
            temperature=self.temperature,
            top_p=self.top_p,
            streaming=self.streaming,
            openai_api_key=self.api_key,
            openai_api_base=self.url,
        )
        llm = llm.bind(stop=["<|eot_id|>"])
        return llm


class ChatModel(BaseOpenAIModel):
    def to_runnable(self) -> Runnable:
        llm = ChatOpenAI(
            name=self.name,
            model=self.model,
            temperature=self.temperature,
            top_p=self.top_p,
            streaming=self.streaming,
            openai_api_key=self.api_key,
            openai_api_base=self.url,
        )
        llm = llm.bind(stop=["<|eot_id|>"])
        return llm


class EmbeddingAPIModel(BaseRunnableModel):
    model: str = Field(..., description="Model of the embedding")
    check_embedding_ctx_length: bool = Field(
        default=False, description="Whether to check embedding context length"
    )
    tiktoken_enabled: bool = Field(default=False, description="Whether to use tiktoken")
    tiktoken_model_name: str | None = Field(
        default=None, description="Name of the tiktoken model"
    )
    url: str = Field(..., description="URL of the model")
    api_key: str = Field(..., description="API key to access the model via LLM API")

    def to_runnable(self) -> OpenAIEmbeddings:
        return OpenAIEmbeddings(
            model=self.model,
            openai_api_key=self.api_key,
            openai_api_base=self.url,
            check_embedding_ctx_length=self.check_embedding_ctx_length,
            tiktoken_enabled=self.tiktoken_enabled,
            tiktoken_model_name=self.tiktoken_model_name,
        )


class MultimodalModel(BaseRunnableModel):

    @abstractmethod
    def to_runnable(self) -> Runnable:
        raise NotImplementedError("Multimodal model is not implemented yet")


class TokenizerModel(BaseRunnableModel):
    path_or_repo_id: str = Field(..., description="Tokenizer")

    def to_runnable(self) -> Runnable:
        return AutoTokenizer.from_pretrained(self.path_or_repo_id)
