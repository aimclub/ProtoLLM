import logging
import uuid

import pytest

from protollm_sdk.celery.app import task_test, abstract_task
from protollm_sdk.celery.job import ResultStorageJob
from protollm_sdk.jobs.job import Job
from protollm_sdk.jobs.utility import construct_job_context


@pytest.fixture
def result_storage():
    return {"question": "What is the ultimate question answer?",
            "answers": "42"}

@pytest.mark.ci
def test_task_test_unknown_job_class(caplog):
    task_id = str(uuid.uuid4())
    task_class = "unknown_class"

    with pytest.raises(Exception, match=f"Unknown job class: '{task_class}'"):
        task_test(task_class=task_class, task_id=task_id)

    assert f"Error in task '{task_id}'. Unknown job class: '{task_class}'." in caplog.text

@pytest.mark.local
def test_task_test_known_job_class(caplog, result_storage):
    caplog.set_level(logging.INFO)
    task_id = str(uuid.uuid4())
    task_class = ResultStorageJob.__name__

    result = task_test(task_class=task_class, task_id=task_id, kwargs=result_storage)

    assert f"Starting task '{task_id}'. Job '{task_class}'." in caplog.text
    assert result is None


class DummyJob(Job):
    def __init__(self):
        self.ran = False

    def run(self, task_id, ctx, **kwargs):
        self.ran = True
        self.task_id = task_id
        self.ctx = ctx
        self.kwargs = kwargs


@pytest.fixture
def dummy_job():
    return DummyJob()

@pytest.mark.ci
def test_abstract_task_class_input(caplog, dummy_job):
    caplog.set_level("INFO")

    task_id = "test_task_id"
    task_class = DummyJob

    abstract_task(task_class=task_class, task_id=task_id, test_arg="value")

    assert f"Starting task '{task_id}'. Job 'DummyJob'." in caplog.text

    assert construct_job_context("DummyJob", abstract_task) is not None

    job_instance = task_class()
    job_instance.run(task_id=task_id, ctx=construct_job_context("DummyJob", abstract_task), test_arg="value")
    assert job_instance.ran is True
    assert job_instance.task_id == task_id
    assert job_instance.kwargs == {"test_arg": "value"}

@pytest.mark.ci
def test_abstract_task_instance_input(caplog, dummy_job):
    caplog.set_level("INFO")

    task_id = "test_task_id"

    abstract_task(task_class=dummy_job, task_id=task_id, test_arg="value")

    assert f"Starting task '{task_id}'. Job 'DummyJob'." in caplog.text

    assert construct_job_context("DummyJob", abstract_task) is not None

    assert dummy_job.ran is True
    assert dummy_job.task_id == task_id
    assert dummy_job.kwargs == {"test_arg": "value"}
