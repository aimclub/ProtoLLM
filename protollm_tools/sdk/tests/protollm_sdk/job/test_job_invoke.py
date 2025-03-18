from unittest import mock
from unittest.mock import MagicMock, patch

import pytest

from protollm_sdk.jobs.job_invoke import BlockingJobResult, WorkerJobResult, JobInvoker, InvokeType


@pytest.fixture
def blocking_job_result():
    """
    Fixture to create a BlockingJobResult with mocked storage and result functionality.
    """
    mock_storage = MagicMock()

    mock_storage.for_job.return_value.load_dict.return_value = {"status": "success", "job_id": "12345"}

    job_result = BlockingJobResult(storage=mock_storage, job_name="test_job", job_id="12345")

    return job_result, mock_storage


@pytest.fixture
def worker_job_result():
    """
    Fixture to create a BlockingJobResult with mocked storage and result functionality.
    """
    mock_storage = MagicMock()

    mock_storage.for_job.return_value.load_dict.return_value = {"status": "success", "job_id": "12345"}

    job_result = WorkerJobResult(storage=mock_storage, job_name="test_job", job_id="12345")

    return job_result, mock_storage


@pytest.fixture
def job_invoker():
    mock_task = MagicMock()
    mock_storage = MagicMock()

    invoker = JobInvoker(
        abstract_celery_task=mock_task,
        result_storage=mock_storage,
        invoke_type=InvokeType.Worker
    )

    return invoker, mock_task, mock_storage


# ---------------------------- Functional test of BlockingJobResult ----------------------------

@pytest.mark.ci
def test_blocking_job_result_initialization(blocking_job_result):
    """
    Test that BlockingJobResult initializes correctly and calls _get_result.
    """
    blocking_job_result, mock_storage = blocking_job_result

    assert blocking_job_result.job_id == "12345"
    assert isinstance(blocking_job_result.result, dict)
    mock_storage.for_job.assert_called_once_with("test_job")

@pytest.mark.ci
def test_blocking_job_result_get_result(blocking_job_result):
    """
    Test that get_result returns the correct result.
    """
    blocking_job_result, mock_storage = blocking_job_result

    result = blocking_job_result.get_result()

    assert result == {"status": "success", "job_id": "12345"}
    mock_storage.for_job().load_dict.assert_called_once_with("12345")

@pytest.mark.ci
@patch('time.sleep', return_value=None)
def test_blocking_job_result_timeout(mock_sleep, blocking_job_result):
    """
    Test that BlockingJobResult raises TimeoutError after the timeout is reached.
    """
    blocking_job_result, mock_storage = blocking_job_result

    mock_storage.for_job().load_dict.return_value = None

    with patch('time.monotonic', side_effect=[0, 1, 2, 300]):  # Simulate waiting 0/1/2/300 seconds
        with pytest.raises(TimeoutError, match="Couldn't retrieve result in 5min 0sec"):
            blocking_job_result._get_result()

    assert mock_sleep.call_count > 0

@pytest.mark.ci
@patch('time.sleep', return_value=None)
def test_blocking_job_result_retries_and_succeeds(mock_sleep, blocking_job_result):
    """
    Test that BlockingJobResult retries the result retrieval and eventually succeeds.
    """
    blocking_job_result, mock_storage = blocking_job_result

    mock_storage.for_job().load_dict.side_effect = [None, None, None, {"status": "success", "job_id": "12345"}]

    result = blocking_job_result._get_result()

    assert result == {"status": "success", "job_id": "12345"}
    assert mock_storage.for_job().load_dict.call_count == 5

@pytest.mark.ci
def test_blocking_job_ping_result(blocking_job_result):
    """
    Test that _ping_result calls the storage and returns the correct result.
    """
    blocking_job_result, mock_storage = blocking_job_result

    mock_storage.load_dict.return_value = {"status": "success", "job_id": "12345"}

    result = blocking_job_result._ping_result()

    assert result == {"status": "success", "job_id": "12345"}
    mock_storage.for_job().load_dict.assert_called_with("12345")
    assert mock_storage.for_job().load_dict.call_count == 2


# ---------------------------- Functional test of WorkerJobResult ----------------------------

@pytest.mark.ci
def test_worker_job_result_initialization(worker_job_result):
    """
    Test that WorkerJobResult initializes correctly and calls _get_result.
    """
    worker_job_result, mock_storage = worker_job_result

    assert worker_job_result.job_id == "12345"
    mock_storage.for_job.assert_called_once_with("test_job")

@pytest.mark.ci
def test_worker_job_result_get_result(worker_job_result):
    """
    Test that get_result returns the correct result.
    """
    worker_job_result, mock_storage = worker_job_result

    result = worker_job_result.get_result()

    assert result == {"status": "success", "job_id": "12345"}
    mock_storage.for_job().load_dict.assert_called_once_with("12345")

@pytest.mark.ci
@patch('time.sleep', return_value=None)
def test_worker_job_result_timeout(mock_sleep, worker_job_result):
    """
    Test that BlockingJobResult raises TimeoutError after the timeout is reached.
    """
    worker_job_result, mock_storage = worker_job_result

    mock_storage.for_job().load_dict.return_value = None

    with patch('time.monotonic', side_effect=[0, 1, 2, 300, 600]):  # Simulate waiting 0/1/2/300 seconds
        with pytest.raises(TimeoutError, match="Couldn't retrieve result in 10min 0sec"):
            worker_job_result.get_result()

    assert mock_sleep.call_count > 0

@pytest.mark.ci
@patch('time.sleep', return_value=None)
def test_worker_job_result_retries_and_succeeds(mock_sleep, worker_job_result):
    """
    Test that BlockingJobResult retries the result retrieval and eventually succeeds.
    """
    worker_job_result, mock_storage = worker_job_result

    mock_storage.for_job().load_dict.side_effect = [None, None, None, {"status": "success", "job_id": "12345"}]

    result = worker_job_result.get_result()

    assert result == {"status": "success", "job_id": "12345"}
    assert mock_storage.for_job().load_dict.call_count == 4

@pytest.mark.ci
def test_worker_job_ping_result(worker_job_result):
    """
    Test that _ping_result calls the storage and returns the correct result.
    """
    worker_job_result, mock_storage = worker_job_result

    mock_storage.load_dict.return_value = {"status": "success", "job_id": "12345"}

    result = worker_job_result._ping_result()

    assert result == {"status": "success", "job_id": "12345"}
    mock_storage.for_job().load_dict.assert_called_with("12345")
    assert mock_storage.for_job().load_dict.call_count == 1


# ---------------------------- Functional test of JobInvoke ----------------------------

@pytest.mark.ci
def test_job_invoker_worker_success(job_invoker):
    """
    Test JobInvoker with InvokeType.Worker.
    """
    invoker, mock_task, mock_storage = job_invoker

    mock_job_class = MagicMock()

    result = invoker.invoke(mock_job_class)

    mock_task.apply_async.assert_called_once_with(args=(mock_job_class, mock.ANY), kwargs={})
    assert isinstance(result, WorkerJobResult)

@pytest.mark.ci
def test_job_invoker_blocking_success(job_invoker):
    """
    Test JobInvoker with InvokeType.Blocking.
    """
    invoker, mock_task, mock_storage = job_invoker

    invoker._invoke_type = InvokeType.Blocking

    mock_job_class = MagicMock()

    result = invoker.invoke(mock_job_class)

    mock_task.assert_called_once_with(mock_job_class, mock.ANY)
    assert isinstance(result, BlockingJobResult)

@pytest.mark.ci
def test_job_invoker_no_task(job_invoker):
    """
    Test JobInvoker when no task is provided.
    """
    invoker, mock_task, mock_storage = job_invoker

    invoker._task = None

    mock_job_class = MagicMock()

    with pytest.raises(Exception, match="Calling JobInvoker without abstract task."):
        invoker.invoke(mock_job_class)

@pytest.mark.ci
def test_job_invoker_unknown_invoke_type(job_invoker):
    """
    Test JobInvoker with an unknown InvokeType.
    """
    invoker, mock_task, mock_storage = job_invoker

    invoker._invoke_type = "UnknownType"

    mock_job_class = MagicMock()

    with pytest.raises(NotImplementedError, match="Unknown invoke type"):
        invoker.invoke(mock_job_class)
