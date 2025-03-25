from unittest.mock import patch, MagicMock
from urllib.parse import urljoin

import pytest

from protollm_sdk.jobs.llm_api import LLMAPI
from protollm_sdk.models.job_context_models import PromptModel, ResponseModel, ChatCompletionModel


@pytest.fixture
def llm_api():
    """
    Fixture for creating an LLMAPI object.
    """
    return LLMAPI("localhost", 8080)


@pytest.fixture
def mock_post():
    """
    Fixture for mocking the httpx.Client.post method.
    """
    with patch('httpx.Client.post') as mock_post:
        yield mock_post

@pytest.mark.ci
def test_llmapi_initialization():
    """
    Test that LLMAPI is initialized correctly with and without a port.
    """
    llm_api_no_port = LLMAPI("localhost")
    assert llm_api_no_port.path == "http://localhost"

    llm_api_with_port = LLMAPI("localhost", 8080)
    assert llm_api_with_port.path == "http://localhost:8080"

@pytest.mark.ci
def test_llmapi_inference_success(mock_post, llm_api):
    """
    Test inference method of LLMAPI for a successful response.
    """
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"content": "Test success"}
    mock_post.return_value = mock_response

    mock_request = MagicMock(spec=PromptModel)
    mock_request.model_dump_json.return_value = '{"prompt": "test prompt"}'

    response = llm_api.inference(mock_request)

    mock_post.assert_called_once_with(
        urljoin("http://localhost:8080", "/generate"),
        headers={"Content-type": "application/json"},
        params={},
        data='{"prompt": "test prompt"}',
        timeout=10 * 60
    )

    assert isinstance(response, ResponseModel)

@pytest.mark.ci
def test_llmapi_inference_server_error(mock_post, llm_api):
    """
    Test inference method of LLMAPI for a 500 error.
    """
    mock_response = MagicMock()
    mock_response.status_code = 500
    mock_post.return_value = mock_response

    mock_request = MagicMock(spec=PromptModel)

    with pytest.raises(Exception,
                       match='The response generation has been interrupted. Error: The LLM server is not available..'):
        llm_api.inference(mock_request)

@pytest.mark.ci
def test_llmapi_inference_validation_error(mock_post, llm_api):
    """
    Test inference method of LLMAPI for a 422 validation error.
    """
    mock_response = MagicMock()
    mock_response.status_code = 422
    mock_response.json.return_value = {"detail": "Validation failed"}
    mock_post.return_value = mock_response

    mock_request = MagicMock(spec=PromptModel)

    with pytest.raises(Exception,
                       match="The response generation has been interrupted. Error: Data model validation error.."):
        llm_api.inference(mock_request)

@pytest.mark.ci
def test_llmapi_chat_completion_success(mock_post, llm_api):
    """
    Test chat_completion method of LLMAPI for a successful response.
    """
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"content": "Test chat success"}
    mock_post.return_value = mock_response

    mock_request = MagicMock(spec=ChatCompletionModel)
    mock_request.model_dump_json.return_value = '{"chat": "test chat"}'

    response = llm_api.chat_completion(mock_request)

    mock_post.assert_called_once_with(
        urljoin("http://localhost:8080", "/chat_completion"),
        headers={"Content-type": "application/json"},
        params={},
        data='{"chat": "test chat"}',
        timeout=10 * 60
    )

    assert isinstance(response, ResponseModel)

@pytest.mark.ci
def test_llmapi_chat_completion_server_error(mock_post, llm_api):
    """
    Test inference method of LLMAPI for a 500 error.
    """
    mock_response = MagicMock()
    mock_response.status_code = 500
    mock_post.return_value = mock_response

    mock_request = MagicMock(spec=PromptModel)

    with pytest.raises(Exception,
                       match='The response generation has been interrupted. Error: The LLM server is not available..'):
        llm_api.chat_completion(mock_request)

@pytest.mark.ci
def test_llmapi_chat_completion_validation_error(mock_post, llm_api):
    """
    Test inference method of LLMAPI for a 422 validation error.
    """
    mock_response = MagicMock()
    mock_response.status_code = 422
    mock_response.json.return_value = {"detail": "Validation failed"}
    mock_post.return_value = mock_response

    mock_request = MagicMock(spec=PromptModel)

    with pytest.raises(Exception,
                       match="The response generation has been interrupted. Error: Data model validation error.."):
        llm_api.chat_completion(mock_request)
