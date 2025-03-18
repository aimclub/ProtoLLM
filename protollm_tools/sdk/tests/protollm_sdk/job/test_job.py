import uuid

import pytest

from protollm_sdk.celery.app import task_test
from protollm_sdk.celery.job import (
    LLMAPIJob, TextEmbedderJob, ResultStorageJob, VectorDBJob
)
from protollm_sdk.models.job_context_models import LLMResponse, TextEmbedderResponse


@pytest.fixture
def llm_request():
    random_id = uuid.uuid4()
    prompt_msg = "What has a head like cat, feet like a kat, tail like a cat, but isn't a cat?"
    meta = {"temperature": 0.5,
            "tokens_limit": 10,
            "stop_words": ["Stop"]}
    llm_request = {"job_id": str(random_id),
                   "meta": meta,
                   "content": prompt_msg}
    return llm_request


@pytest.fixture
def text_embedder_request():
    return {"job_id": "0",
            "inputs": "Everybody steals and throws, they cut each other and hang each other... "
                      "In general, normal civilized life is going on. McDonald's everywhere. "
                      "I don't see them here, by the way. That can't be good.",
            "truncate": False}


@pytest.fixture
def result_storage():
    return {"question": "What is the ultimate question answer?",
            "answers": "42"}


@pytest.mark.skip(reason="LLM die sometimes because stupid questions")
def test_llm_request(llm_request):
    result = task_test.apply_async(args=(LLMAPIJob.__name__, llm_request["job_id"]), kwargs=llm_request)
    r = result.get()
    res = LLMResponse(job_id=llm_request["job_id"], text=r.content)
    assert isinstance(res, LLMResponse)

@pytest.mark.local
def test_text_embedder_request(text_embedder_request):
    random_id = uuid.uuid4()
    result = task_test.apply_async(args=(TextEmbedderJob.__name__, random_id), kwargs=text_embedder_request)
    assert isinstance(result.get(), TextEmbedderResponse)

@pytest.mark.local
def test_result_storage(result_storage):
    random_id = uuid.uuid4()
    task_test.apply_async(args=(ResultStorageJob.__name__, random_id), kwargs=result_storage)

@pytest.mark.skip(reason="We don't have local vector DB")
def test_ping_vector_db():
    random_id = uuid.uuid4()
    result = task_test.apply_async(args=(VectorDBJob.__name__, random_id))
    r = result.get()
    assert isinstance(r, dict)
