import threading
import time
import uuid

import pytest

from protollm_sdk.object_interface.result_storage.models import (
    JobStatusError,
    JobStatusErrorType,
    JobStatusType,
)
from protollm_sdk.object_interface.result_storage import RedisResultStorage

# --------------------------------------------------------------------------- #
#  Fixtures                                                                    #
# --------------------------------------------------------------------------- #

@pytest.fixture
def storage(redis_client):
    """Fresh RedisResultStorage for each test."""
    return RedisResultStorage(redis_client=redis_client)


# --------------------------------------------------------------------------- #
#  Basic operations                                                            #
# --------------------------------------------------------------------------- #

def test_create_and_get(storage):
    job_id = f"job-{uuid.uuid4()}"
    storage.create_job_status(job_id)

    status = storage.get_job_status(job_id)
    assert status.status is JobStatusType.PENDING
    assert status.status_message == "Job is created"
    assert status.is_completed is False


@pytest.mark.parametrize(
    "new_status, msg",
    [
        (JobStatusType.IN_PROGRESS, "working"),
    ],
)
def test_update_job_status(storage, new_status, msg):
    job_id = f"job-{uuid.uuid4()}"
    storage.create_job_status(job_id)

    storage.update_job_status(job_id, new_status, msg)
    status = storage.get_job_status(job_id)

    assert status.status is new_status
    assert status.status_message == msg
    assert status.is_completed is False


def test_complete_job_success(storage):
    job_id = f"job-{uuid.uuid4()}"
    storage.create_job_status(job_id)

    storage.complete_job(job_id, result='{"answer":42}', status_message="done")

    status = storage.get_job_status(job_id)
    assert status.status is JobStatusType.COMPLETED
    assert status.result == '{"answer":42}'
    assert status.is_completed is True
    assert status.error is None


def test_complete_job_error(storage):
    job_id = f"job-{uuid.uuid4()}"
    storage.create_job_status(job_id)

    err = JobStatusError(type=JobStatusErrorType.ValidationError, msg="boom")
    storage.complete_job(job_id, error=err, status_message="failed")

    status = storage.get_job_status(job_id)
    assert status.status is JobStatusType.ERROR
    assert status.error == err
    assert status.is_completed is True


def test_delete_job_status(storage, redis_client):
    job_id = f"job-{uuid.uuid4()}"
    storage.create_job_status(job_id)

    storage.delete_job_status(job_id)
    assert redis_client.get(job_id) is None


# --------------------------------------------------------------------------- #
#  Pub/Sub behaviour                                                           #
# --------------------------------------------------------------------------- #

def test_subscribe_receives_updates(storage):
    job_id = f"job-{uuid.uuid4()}"
    storage.create_job_status(job_id)

    generator = storage.subscribe(job_id, timeout=5)

    def updater():
        time.sleep(0.2)
        storage.update_job_status(job_id, JobStatusType.IN_PROGRESS, "running")
        time.sleep(0.2)
        storage.complete_job(job_id, result="{}")

    thread = threading.Thread(target=updater)
    thread.start()

    updates = list(generator)      # stops automatically on completion
    thread.join()

    assert len(updates) == 2
    assert updates[0].status is JobStatusType.IN_PROGRESS
    assert updates[1].status is JobStatusType.COMPLETED
    assert updates[1].is_completed is True


def test_wait_completeness_success(storage):
    job_id = f"job-{uuid.uuid4()}"
    storage.create_job_status(job_id)

    def completer():
        time.sleep(0.3)
        storage.complete_job(job_id, result='{"ok":1}')

    thread = threading.Thread(target=completer)
    thread.start()

    final_status = storage.wait_completeness(job_id, timeout=5)
    thread.join()

    assert final_status.status is JobStatusType.COMPLETED
    assert final_status.is_completed is True


def test_wait_completeness_timeout(storage):
    job_id = f"job-{uuid.uuid4()}"
    storage.create_job_status(job_id)

    with pytest.raises(TimeoutError):
        storage.wait_completeness(job_id, timeout=1)