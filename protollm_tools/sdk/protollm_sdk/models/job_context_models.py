from enum import Enum
from typing import Literal, Union

from pydantic import BaseModel, Field


class PromptTypes(Enum):
    SINGLE_GENERATION: str = "single_generation"
    CHAT_COMPLETION: str = "chat_completion"


class PromptMeta(BaseModel):
    temperature: float | None = 0.2
    tokens_limit: int | None = 8096
    stop_words: list[str] | None = None
    model: str | None = Field(default=None, examples=[None])


class PromptModel(BaseModel):
    job_id: str
    priority: int | None = Field(default=None, examples=[None])
    meta: PromptMeta
    content: str


class ChatCompletionUnit(BaseModel):
    """A model for element of chat completion"""
    role: str
    content: str


class ChatCompletionModel(BaseModel):
    """A model for chat completion order"""
    job_id: str
    priority: int | None = Field(default=None, examples=[None])
    source: str = "local"
    meta: PromptMeta
    messages: list[ChatCompletionUnit]

    @classmethod
    def from_prompt_model(cls, prompt_model: PromptModel) -> 'ChatCompletionModel':
        # Создаем первое сообщение из содержимого PromptModel
        initial_message = ChatCompletionUnit(
            role="user",  # Или другой подходящий role
            content=prompt_model.content
        )
        # Возвращаем новый экземпляр ChatCompletionModel
        return cls(
            job_id=prompt_model.job_id,
            priority=prompt_model.priority,
            meta=prompt_model.meta,
            messages=[initial_message]
        )


class PromptTransactionModel(BaseModel):
    prompt: PromptModel
    prompt_type: Literal[PromptTypes.SINGLE_GENERATION.value]


class ChatCompletionTransactionModel(BaseModel):
    prompt: ChatCompletionModel
    prompt_type: Literal[PromptTypes.CHAT_COMPLETION.value]


class PromptWrapper(BaseModel):
    prompt: Union[PromptTransactionModel, ChatCompletionTransactionModel] = Field(..., discriminator='prompt_type')


class ResponseModel(BaseModel):
    content: str


class LLMResponse(BaseModel):
    job_id: str
    text: str


class TextEmbedderRequest(BaseModel):
    job_id: str
    inputs: str
    truncate: bool


class ToEmbed(BaseModel):
    inputs: str
    truncate: bool


class TextEmbedderResponse(BaseModel):
    embeddings: list[float]
