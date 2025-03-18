import os
from unittest.mock import patch

import pytest

from protollm_sdk.config import Config
from protollm_sdk.jobs.job_context import JobContext
from protollm_sdk.jobs.job_invoke import InvokeType
from protollm_sdk.jobs.utility import construct_job_context

@pytest.mark.ci
def test_construct_job_context_real():
    """
    Test construct_job_context function with real environment variables and connections.
    This test assumes that all services (Redis, LLM API, Text Embedder, etc.) are properly running.
    """
    job_name = "test_job"
    job_context = construct_job_context(job_name)

    assert isinstance(job_context, JobContext)

    assert job_context.llm_api is not None
    assert job_context.outer_llm_api is not None
    assert job_context.text_embedder is not None
    assert job_context.result_storage is not None
    assert job_context.vector_db is not None
    assert job_context.job_invoker is not None

@pytest.mark.ci
def test_construct_job_context_with_invoke_type_worker():
    """
    Test construct_job_context with a missing environment variable.
    """
    with patch.dict(os.environ, {"JOB_INVOCATION_TYPE": "worker"}):
        assert os.getenv("JOB_INVOCATION_TYPE") == "worker"

        job_name = "test_job"
        job_context = construct_job_context(job_name)
        assert job_context.job_invoker._invoke_type == InvokeType.Worker

@pytest.mark.ci
def test_construct_job_context_with_invoke_type_blocking():
    """
    Test construct_job_context with a missing environment variable.
    """
    with patch.dict(os.environ, {"JOB_INVOCATION_TYPE": "blocking"}):
        assert os.getenv("JOB_INVOCATION_TYPE") == "blocking"
        Config.reload_invocation_type()
        job_name = "test_job"
        job_context = construct_job_context(job_name)
        assert job_context.job_invoker._invoke_type == InvokeType.Blocking

    Config.reload_invocation_type()

@pytest.mark.ci
def test_construct_job_context_with_wrong_invoke_type():
    """
    Test construct_job_context with a missing environment variable.
    """
    with patch.dict(os.environ, {"JOB_INVOCATION_TYPE": "cringe"}):
        assert os.getenv("JOB_INVOCATION_TYPE") == "cringe"
        Config.reload_invocation_type()

        with pytest.raises(ValueError, match="Found unknown invocation type 'cringe'."):
            job_name = "test_job"
            construct_job_context(job_name)

    Config.reload_invocation_type()
