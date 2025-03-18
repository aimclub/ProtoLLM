from unittest.mock import patch, MagicMock
from urllib.parse import urljoin

import httpx
import pytest

from protollm_sdk.jobs.vector_db import VectorDB

@pytest.mark.ci
def test_vector_db_initialization_without_port():
    """
    Test that VectorDB is initialized correctly without a port.
    """
    vector_db = VectorDB("localhost")
    assert vector_db.url == "http://localhost"
    assert isinstance(vector_db.client, httpx.Client)

@pytest.mark.ci
def test_vector_db_initialization_with_port():
    """
    Test that VectorDB is initialized correctly with a port.
    """
    vector_db_with_port = VectorDB("localhost", 8080)
    assert vector_db_with_port.url == "http://localhost:8080"
    assert isinstance(vector_db_with_port.client, httpx.Client)

@pytest.mark.ci
@patch('httpx.Client.get')
def test_vector_db_api_v1(mock_get):
    """
    Test the api_v1 method of VectorDB class.
    """
    mock_response = MagicMock()
    mock_response.json.return_value = {"status": "success"}
    mock_get.return_value = mock_response

    vector_db = VectorDB("localhost", 8080)

    response = vector_db.api_v1()

    mock_get.assert_called_once_with(
        urljoin("http://localhost:8080", "/api/v1"),
        headers={"Content-type": "application/json"},
        timeout=10 * 60
    )

    assert response == {"status": "success"}
