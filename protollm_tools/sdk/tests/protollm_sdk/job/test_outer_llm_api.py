from unittest.mock import patch, MagicMock

import pytest
from httpx import URL

from protollm_sdk.jobs.outer_llm_api import OuterLLMAPI
from protollm_sdk.models.job_context_models import PromptModel, ResponseModel, ChatCompletionModel, \
    ChatCompletionUnit, PromptMeta


@pytest.fixture
def outer_llm_api():
    return OuterLLMAPI("key")


@pytest.fixture
def mock_openai():
    with patch('protollm_sdk.jobs.outer_llm_api.OpenAI') as mock_openai:
        mock_openai().chat.completions.create = MagicMock()
        yield mock_openai().chat.completions.create

@pytest.mark.ci
def test_outer_llmapi_initialization():
    outer_llm_api = OuterLLMAPI("key")
    assert outer_llm_api.model == "openai/gpt-4o-2024-08-06"
    assert outer_llm_api.timeout_sec == 10 * 60
    assert isinstance(outer_llm_api.client.api_key, str)
    assert len(outer_llm_api.client.api_key) == len(
        "key")
    assert outer_llm_api.client.base_url == URL("https://api.vsegpt.ru/v1/")

@pytest.mark.ci
def test_outer_llmapi_inference_success(mock_openai, outer_llm_api):
    """
    Test inference method of OuterLLMAPI for a successful response.
    """
    mock_response = MagicMock()
    mock_response.choices = [MagicMock(message=MagicMock(content="Test success"))]
    mock_openai.return_value = mock_response

    mock_request = MagicMock(spec=PromptModel)
    mock_request.content = "test prompt"
    mock_meta = MagicMock()
    mock_meta.temperature = 0.7
    mock_meta.tokens_limit = 100
    mock_meta.model = None
    mock_request.meta = mock_meta  # Добавляем meta к запросу

    response = outer_llm_api.inference(mock_request)

    mock_openai.assert_called_once_with(
        model="openai/gpt-4o-2024-08-06",
        messages=[{"role": "user", "content": "test prompt"}],
        temperature=0.7,
        n=1,
        max_tokens=100,
        timeout=10 * 60
    )

    assert isinstance(response, ResponseModel)
    assert response.content == "Test success"

@pytest.mark.ci
def test_outer_llmapi_inference_server_error(mock_openai, outer_llm_api):
    """
    Test inference method of OuterLLMAPI for a 500 error.
    """
    mock_openai.side_effect = Exception("The LLM server is not available.")

    mock_request = MagicMock(spec=PromptModel)
    mock_request.content = "test prompt"
    mock_meta = MagicMock()
    mock_meta.temperature = 0.7
    mock_meta.tokens_limit = 100
    mock_meta.model = None
    mock_request.meta = mock_meta

    with pytest.raises(Exception,
                       match="The response generation has been interrupted. Error: The LLM server is not available."):
        outer_llm_api.inference(mock_request)

@pytest.mark.ci
def test_outer_llmapi_chat_completion_success(mock_openai, outer_llm_api):
    """
    Test chat_completion method of OuterLLMAPI for a successful response.
    """
    mock_response = MagicMock()
    mock_response.choices = [MagicMock(message=MagicMock(content="Test chat success"))]
    mock_openai.return_value = mock_response

    mock_request = MagicMock(spec=ChatCompletionModel)

    mock_request.messages = [ChatCompletionUnit(role="user", content="test chat")]

    mock_meta = MagicMock(spec=PromptMeta)
    mock_meta.temperature = 0.7
    mock_meta.tokens_limit = 100
    mock_meta.model = None
    mock_request.meta = mock_meta

    response = outer_llm_api.chat_completion(mock_request)

    mock_openai.assert_called_once_with(
        model="openai/gpt-4o-2024-08-06",
        messages=[{"role": "user", "content": "test chat"}],
        temperature=0.7,
        n=1,
        max_tokens=100,
        timeout=10 * 60
    )

    assert isinstance(response, ResponseModel)
    assert response.content == "Test chat success"

@pytest.mark.ci
def test_outer_llmapi_chat_completion_server_error(mock_openai, outer_llm_api):
    """
    Test chat_completion method of OuterLLMAPI for a 500 error.
    """
    mock_openai.side_effect = Exception("The LLM server is not available.")

    mock_request = MagicMock(spec=ChatCompletionModel)

    mock_request.messages = [ChatCompletionUnit(role="user", content="test chat")]

    mock_meta = MagicMock(spec=PromptMeta)
    mock_meta.temperature = 0.7
    mock_meta.tokens_limit = 100
    mock_meta.model = None
    mock_request.meta = mock_meta

    with pytest.raises(Exception,
                       match="The response generation has been interrupted. Error: The LLM server is not available."):
        outer_llm_api.chat_completion(mock_request)
