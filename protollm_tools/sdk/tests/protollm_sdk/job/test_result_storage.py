import json
from unittest.mock import MagicMock, patch

import pytest

from protollm_sdk.jobs.result_storage import ResultStorage


@pytest.fixture
def mock_redis():
    with patch("redis.Redis.from_url", return_value=MagicMock()) as mock_redis:
        yield mock_redis


@pytest.fixture
def result_storage():
    return ResultStorage(redis_host="localhost", redis_port=6379, prefix="test_job")

@pytest.mark.ci
def test_for_job(result_storage):
    """
    Test that for_job correctly creates a new ResultStorage instance with a new job prefix.
    """
    with patch('protollm_sdk.jobs.result_storage.ResultStorage') as mock_result_storage:
        new_storage = result_storage.for_job("new_job")

        mock_result_storage.assert_called_once_with(redis_host="redis://localhost:6379", redis_port=None,
                                                    prefix="new_job")

        assert new_storage is mock_result_storage.return_value

@pytest.mark.ci
def test_build_key(result_storage):
    """
    Test the build_key method for formatting keys.
    """
    key = result_storage.build_key(job_id="12345", prefix="test_prefix")
    assert key == "test_prefix:12345"

    key_without_prefix = result_storage.build_key(job_id="12345", prefix=None)
    assert key_without_prefix == "12345"

@pytest.mark.ci
def test_save_dict_success(mock_redis, result_storage):
    """
    Test saving a dictionary to Redis.
    """
    mock_redis_instance = mock_redis.return_value

    result_storage.save_dict("12345", {"status": "success"})

    mock_redis_instance.set.assert_called_once_with("test_job:12345", json.dumps({"status": "success"}))

@pytest.mark.ci
def test_save_dict_failure(mock_redis, result_storage):
    """
    Test failure when saving a dictionary to Redis.
    """
    mock_redis_instance = mock_redis.return_value

    mock_redis_instance.set.side_effect = Exception("Redis Error")

    with pytest.raises(Exception, match="Saving the result with the key test_job:12345 has been interrupted"):
        result_storage.save_dict("12345", {"status": "success"})

    mock_redis_instance.set.assert_called_once_with("test_job:12345", json.dumps({"status": "success"}))

@pytest.mark.ci
def test_load_bytes_or_str_dict_success(mock_redis, result_storage):
    """
    Test loading a dictionary from Redis.
    """
    mock_redis_instance = mock_redis.return_value

    mock_redis_instance.get.return_value = json.dumps({"status": "success"})

    result = result_storage.load_dict("12345")

    mock_redis_instance.get.assert_called_once_with("test_job:12345")

    assert result == {"status": "success"}

@pytest.mark.ci
def test_load_not_bytes_or_str_dict_success(mock_redis, result_storage):
    """
    Test loading a dictionary from Redis.
    """
    mock_redis_instance = mock_redis.return_value

    mock_redis_instance.get.return_value = 123

    result = result_storage.load_dict("12345")

    mock_redis_instance.get.assert_called_once_with("test_job:12345")

    assert result == 123

@pytest.mark.ci
def test_load_dict_failure(mock_redis, result_storage):
    """
    Test failure when loading a dictionary from Redis.
    """
    mock_redis_instance = mock_redis.return_value

    mock_redis_instance.get.side_effect = Exception("Redis Error")

    with pytest.raises(Exception, match="Failed to load the result with the key test_job:12345 from redis."):
        result_storage.load_dict("12345")

    mock_redis_instance.get.assert_called_once_with("test_job:12345")
