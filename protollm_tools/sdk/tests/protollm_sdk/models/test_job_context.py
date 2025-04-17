import uuid

import pytest
from pydantic import ValidationError

from protollm_sdk.models.job_context_models import (
    PromptTypes,
    Role,
    ContentType,
    PromptMeta,
    MessageContentUnit,
    PromptModel,
    ChatCompletionModel,
    PromptTransactionModel,
    ChatCompletionTransactionModel,
    PromptWrapper,
    LLMResponse,
    TextEmbedderRequest,
    TextEmbedderResponse,
    generate_job_id,
)


@pytest.mark.ci
def test_from_prompt_model():
    prompt_model = PromptModel(
        job_id="test_job_123",
        meta=PromptMeta(
            temperature=0.5,
            tokens_limit=100,
            stop_words=["stop", "words"],
            model="gpt-3"
        ),
        content="This is a test prompt"
    )

    chat_completion = ChatCompletionModel.from_prompt_model(prompt_model)

    assert chat_completion.job_id == prompt_model.job_id
    assert chat_completion.meta == prompt_model.meta

    assert len(chat_completion.messages) == 1

    assert chat_completion.messages[0].role == Role.USER
    assert chat_completion.messages[0].content == prompt_model.content

    assert chat_completion.meta.temperature == 0.5
    assert chat_completion.meta.tokens_limit == 100
    assert chat_completion.meta.stop_words == ["stop", "words"]
    assert chat_completion.meta.model == "gpt-3"


# Helper function to provide valid image data URL for testing purposes.
def valid_image_data_url():
    # 1x1 transparent PNG image in base64 format.
    return (
        "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8"
        "/wIAAgcBBH5C60cAAAAASUVORK5CYII="
    )


# Test for creating a valid PromptModel with different content types.
@pytest.mark.parametrize("content_input", [
    "Simple text prompt.",
    [MessageContentUnit(type=ContentType.TEXT.value, text="Hello world!")],
])
def test_prompt_model_creation(content_input):
    # Create meta data for prompt model
    meta = PromptMeta(temperature=0.5, tokens_limit=1000, stop_words=["STOP"], model="gpt-3")
    prompt_model = PromptModel(meta=meta, content=content_input)

    # Assert that a valid UUID is generated for job_id
    try:
        uuid_obj = uuid.UUID(prompt_model.job_id)
    except ValueError:
        pytest.fail("Job ID is not a valid UUID.")

    # Verify meta properties and content assignment
    assert prompt_model.meta.temperature == 0.5, "Meta temperature should be 0.5."
    assert prompt_model.content == content_input, "Prompt content does not match input."


# Test for creating a ChatCompletionModel using the from_prompt_model class method.
@pytest.mark.parametrize("chat_content", [
    "Chat message text.",
    [MessageContentUnit(type=ContentType.TEXT.value, text="Chat message provided as a list.")],
])
def test_chat_completion_model_from_prompt(chat_content):
    meta = PromptMeta(temperature=0.7, tokens_limit=1500, stop_words=[], model="gpt-3")
    prompt_model = PromptModel(meta=meta, content=chat_content)
    chat_model = ChatCompletionModel.from_prompt_model(prompt_model)

    # Assert that the job_id is transferred from PromptModel
    assert chat_model.job_id == prompt_model.job_id, "Job IDs should be identical between prompt and chat completion models."

    # Verify that the initial message role is 'user'
    initial_message = chat_model.messages[0]
    assert initial_message.role == Role.USER, "The initial chat message role should be USER."
    assert initial_message.content == chat_content, "Chat message content should match the prompt content."


# Test to ensure correct transaction model is selected based on the prompt_type discriminator.
@pytest.mark.parametrize("transaction_input, expected_class", [
    (
            {
                "prompt": {
                    "job_id": generate_job_id(),
                    "priority": None,
                    "meta": {"temperature": 0.3, "tokens_limit": 2048, "stop_words": None, "model": None},
                    "content": "Test prompt content."
                },
                "prompt_type": PromptTypes.SINGLE_GENERATION.value
            },
            PromptTransactionModel
    ),
    (
            {
                "prompt": {
                    "job_id": generate_job_id(),
                    "priority": None,
                    "source": "local",
                    "meta": {"temperature": 0.2, "tokens_limit": 4096, "stop_words": None, "model": "gpt-4"},
                    "messages": [{"role": "user", "content": "Test chat message."}]
                },
                "prompt_type": PromptTypes.CHAT_COMPLETION.value
            },
            ChatCompletionTransactionModel
    ),
])
def test_transaction_models(transaction_input, expected_class):
    # Creating PromptWrapper which should correctly identify the transaction type based on the discriminator 'prompt_type'
    wrapper = PromptWrapper(prompt=transaction_input)
    # Assert that the parsed prompt is an instance of the expected transaction model class
    assert isinstance(wrapper.prompt,
                      expected_class), "Parsed prompt should be an instance of the expected transaction model."


# Test for LLMResponse creation with valid job_id and text.
@pytest.mark.parametrize("response_input", [
    {"job_id": generate_job_id(), "text": "Response text from LLM."},
])
def test_llm_response_model(response_input):
    response = LLMResponse(**response_input)
    # Assert job_id is a string and text matches the input
    assert isinstance(response.job_id, str), "LLMResponse job_id should be a string."
    assert response.text == response_input["text"], "LLMResponse text should match the input text."


# Test for creating a valid TextEmbedderRequest.
@pytest.mark.parametrize("embedder_request_input", [
    {"job_id": generate_job_id(), "inputs": "Text to embed.", "truncate": False},
])
def test_text_embedder_request_model(embedder_request_input):
    request = TextEmbedderRequest(**embedder_request_input)
    # Assert the fields are correctly set in the model
    assert request.job_id == embedder_request_input["job_id"], "TextEmbedderRequest job_id does not match input."
    assert request.inputs == embedder_request_input["inputs"], "TextEmbedderRequest inputs should match provided value."
    assert request.truncate is False, "TextEmbedderRequest truncate value should be False."


# Test for creating a valid TextEmbedderResponse with embeddings.
@pytest.mark.parametrize("embedder_response_input", [
    {"embeddings": [0.1, 0.2, 0.3, 0.4]},
])
def test_text_embedder_response_model(embedder_response_input):
    response = TextEmbedderResponse(**embedder_response_input)
    # Assert that embeddings are provided as a list of floats
    assert isinstance(response.embeddings, list), "Embeddings should be a list."
    assert all(isinstance(val, float) for val in response.embeddings), "Each embedding should be a float."


# Test for MessageContentUnit creation, verifying behavior based on content type.
@pytest.mark.parametrize("unit_input, expected_error", [
    (
            {"type": ContentType.TEXT.value, "text": "Test text content."},
            None  # No error expected for valid text content
    ),
    (
            {"type": ContentType.IMAGE.value, "image": valid_image_data_url()},
            None  # No error expected for valid image data URL
    ),
    (
            {"type": ContentType.IMAGE.value, "image": "data:image/png;base64,NotAValidBase64"},
            pytest.raises(ValidationError)  # Expect an assertion error due to invalid image base64 data
    ),
])
def test_message_content_unit(unit_input, expected_error):
    if expected_error is None:
        unit = MessageContentUnit(**unit_input)
        assert unit is not None, "MessageContentUnit should be created successfully with valid input."
    else:
        with expected_error:
            MessageContentUnit(**unit_input)
