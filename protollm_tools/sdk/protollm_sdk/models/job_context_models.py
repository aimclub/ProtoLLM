from enum import Enum
from typing import Literal, Union, Optional

from pydantic import BaseModel, Field

from protollm_sdk.models.utils import validate_image_base64, generate_job_id


class PromptTypes(Enum):
    SINGLE_GENERATION: str = "single_generation"
    CHAT_COMPLETION: str = "chat_completion"


class Role(Enum):
    """
    Roles for chat completion
    USER - "user" for messages from the user
    ASSISTANT - "assistant" for messages from the assistant (LLM)
    SYSTEM - "system" for setting up the assistant's behavior`
    """
    USER: str = "user"
    ASSISTANT: str = "assistant"
    SYSTEM: str = "system"


class ContentType(Enum):
    TEXT: str = "text"
    IMAGE: str = "image"


class PromptMeta(BaseModel):
    temperature: float | None = 0.2
    tokens_limit: int | None = 8096
    stop_words: list[str] | None = None
    model: str | None = Field(default=None, examples=[None])


class MessageContentUnit(BaseModel):
    type: ContentType = Field(default=ContentType.TEXT, examples=[ContentType.TEXT, ContentType.IMAGE])
    text: str | None = Field(default=None, examples=["Hello, world!"])
    image: str | None = Field(default=None, description="f\"data:image/jpeg;base64,{base64_image}\"")

    def model_post_init(self, __context):
        if self.type == ContentType.TEXT:
            assert self.text is not None, "Text content must be provided for type 'text'."
        elif self.type == ContentType.IMAGE:
            assert self.image is not None, "Image content must be provided for type 'image'."
            parsing_message = validate_image_base64(self.image)
            assert len(parsing_message) == 0, f"Image content is not valid base64: {parsing_message}"
        else:
            raise ValueError(f"Unsupported content type: {self.type}")


class PromptModel(BaseModel):
    job_id: Optional[str] = Field(default_factory=generate_job_id)
    priority: int | None = Field(default=None, examples=[None])
    meta: PromptMeta
    content: str | list[MessageContentUnit]


class ChatCompletionUnit(BaseModel):
    """A model for element of chat completion"""

    role: Role = Field(default="user", examples=[Role.USER, Role.SYSTEM, Role.ASSISTANT])
    content: str | list[MessageContentUnit]


class ChatCompletionModel(BaseModel):
    """A model for chat completion order"""
    job_id: Optional[str] = Field(default_factory=generate_job_id)
    priority: int | None = Field(default=None, examples=[None])
    source: str = "local"
    meta: PromptMeta
    messages: list[ChatCompletionUnit]

    @classmethod
    def from_prompt_model(cls, prompt_model: PromptModel) -> 'ChatCompletionModel':
        initial_message = ChatCompletionUnit(
            role=Role.USER,
            content=prompt_model.content
        )
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
