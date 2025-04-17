import json
from typing import Any

import pytest
from pydantic import ValidationError

from protollm_sdk.object_interface.result_storage.models import (
    JobStatus,
    JobStatusError,
    JobStatusErrorType,
    JobStatusType,
)


# Positive tests (test valid values)

@pytest.mark.parametrize(
    "status, status_message, result",
    [
        (JobStatusType.PENDING, "queued", None),
        (JobStatusType.IN_PROGRESS, "running", None),
        (JobStatusType.COMPLETED, "done", '{"answer": 42}'),
    ],
)
def test_job_status_valid(status: JobStatusType, status_message: str, result: str | None) -> None:
    model = JobStatus(status=status, status_message=status_message, result=result)
    assert model.status is status
    assert model.status_message == status_message
    assert model.result == result
    assert isinstance(model.last_update, str)
    assert model.is_completed is False


@pytest.mark.parametrize(
    "err_type, message",
    [
        (JobStatusErrorType.ValidationError, "invalid field"),
        (JobStatusErrorType.ConnectionError, "redis unavailable"),
        (JobStatusErrorType.TimeoutError, "took too long"),
    ],
)
def test_job_status_error_valid(err_type: JobStatusErrorType, message: str) -> None:
    err = JobStatusError(type=err_type, msg=message)
    assert err.type is err_type
    assert err.msg == message


def test_job_status_json_roundtrip() -> None:
    original = JobStatus(
        status=JobStatusType.COMPLETED,
        status_message="finished",
        result='{"ok": true}',
        is_completed=True,
    )
    dumped = original.model_dump_json()
    loaded = JobStatus.model_validate_json(dumped)
    assert original == loaded
    json.loads(dumped)


# Negative tests (test errors)

@pytest.mark.parametrize(
    "field, bad_value",
    [
        ("status", "unknown_status"),
        ("status", 123),
        ("error", {"type": "NonExistentError", "msg": "bad"}),
        ("is_completed", "not_bool"),
    ],
)
def test_job_status_validation_errors(field: str, bad_value: Any) -> None:
    kwargs = {
        "status": JobStatusType.PENDING,
        "status_message": "queued",
        "result": None,
        "error": None,
        "is_completed": False,
        field: bad_value
    }
    with pytest.raises(ValidationError):
        JobStatus(**kwargs)


@pytest.mark.parametrize(
    "bad_type",
    ["NotAnErrorType", 42, None],
)
def test_job_status_error_invalid_type(bad_type: Any) -> None:
    with pytest.raises(ValidationError):
        JobStatusError(type=bad_type, msg="oops")
