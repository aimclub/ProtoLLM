import uuid
from unittest.mock import patch

import httpx
import pytest

from protollm_sdk.config import Config
from protollm_sdk.jobs.text_embedder import TextEmbedder
from protollm_sdk.models.job_context_models import TextEmbedderRequest, TextEmbedderResponse


# ---------------------------- Initialisations tests ----------------------------
@pytest.mark.ci
@pytest.mark.parametrize(
    "host, port, expected_path",
    [
        ("localhost", 8080, "http://localhost:8080"),
        ("example.com", None, "http://example.com"),
        ("127.0.0.1", 9942, "http://127.0.0.1:9942"),
        ("text-embedder.com", "6672", "http://text-embedder.com:6672"),
    ]
)
def test_text_embedder_initialization(host, port, expected_path):
    """
    Tests that the URL is correctly initialized with different values of text_emb_host and text_emb_port.
    """
    text_embedder = TextEmbedder(text_emb_host=host, text_emb_port=port)
    assert text_embedder.path == expected_path
    assert isinstance(text_embedder.client, httpx.Client)

@pytest.mark.ci
def test_text_embedder_timeout_default():
    """
    Tests that the default timeout value is correctly set.
    """
    text_embedder = TextEmbedder(text_emb_host="localhost", text_emb_port=8080)
    assert text_embedder.timeout_sec == 10 * 60  # Default timeout is 10 minutes

@pytest.mark.ci
@pytest.mark.parametrize(
    "custom_timeout",
    [30, 60, 120]
)
def test_text_embedder_custom_timeout(custom_timeout):
    """
    Tests that the custom timeout value is correctly set.
    """
    text_embedder = TextEmbedder(text_emb_host="localhost", text_emb_port=8080, timeout_sec=custom_timeout)
    assert text_embedder.timeout_sec == custom_timeout

@pytest.mark.ci
def test_text_embedder_client_initialization():
    """
    Tests that httpx.Client is initialized when the object is created.
    """
    text_embedder = TextEmbedder(text_emb_host="localhost", text_emb_port=8080)
    assert isinstance(text_embedder.client, httpx.Client)


# ---------------------------- Fixtures ----------------------------

@pytest.fixture
def text_embedder():
    """
    Fixture to create a TextEmbedder instance using environment variables.
    """
    return TextEmbedder(text_emb_host=Config.text_embedder_host, text_emb_port=Config.text_embedder_port)


@pytest.fixture
def text_embedder_request():
    """
    Fixture to create a sample TextEmbedderRequest for testing.
    """
    data = {
        "job_id": str(uuid.uuid4()),
        "inputs": "A Greek was crossing the river when he saw a crab in the water.",
        "truncate": False
    }
    return TextEmbedderRequest(**data)


# ---------------------------- Function Tests ----------------------------
@pytest.mark.local
def test_text_embedder_inference(text_embedder, text_embedder_request):
    """
    Tests that the inference method returns a valid TextEmbedderResponse.
    """
    res = text_embedder.inference(text_embedder_request)
    assert isinstance(res, TextEmbedderResponse)

@pytest.mark.ci
def test_text_embedder_connection_error(text_embedder, text_embedder_request):
    """
    Tests that a ConnectionError is raised when the server returns a 500 status code.
    """
    with patch.object(text_embedder.client, 'post') as mock_post:
        mock_post.return_value.status_code = 500
        with pytest.raises(Exception, match='The embedding was interrupted. Error: The LLM server is not available.'):
            text_embedder.inference(text_embedder_request)

@pytest.mark.ci
def test_text_embedder_validation_error(text_embedder, text_embedder_request):
    """
    Tests that a ValueError is raised when the server returns a 422 status code.
    """
    with patch.object(text_embedder.client, 'post') as mock_post:
        mock_post.return_value.status_code = 422
        mock_post.return_value.json.return_value = {"detail": "Validation failed"}
        with pytest.raises(Exception,
                           match="The embedding was interrupted. Error: Data model validation error. {'detail': 'Validation failed'}"):
            text_embedder.inference(text_embedder_request)
