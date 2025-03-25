import uuid

import pytest

from protollm_sdk.jobs.utility import construct_job_context
from protollm_sdk.models.job_context_models import PromptModel, ChatCompletionModel

@pytest.fixture
def llm_request():
    random_id = str(uuid.uuid4())
    prompt_msg = "What has a head like cat, feet like a kat, tail like a cat, but isn't a cat?"
    meta = {"temperature": 0.5,
            "tokens_limit": 1000,
            "stop_words": ["Stop"]}
    llm_request = {"job_id": random_id,
                   "meta": meta,
                   "content": prompt_msg}
    request = PromptModel(**llm_request)
    return request

@pytest.fixture
def llm_job_context(llm_request):
    jc = construct_job_context(llm_request.job_id)
    return jc

@pytest.mark.local
def test_api_interface(llm_request, llm_job_context):
    response = llm_job_context.llm_api.inference(llm_request)
    print(response)

@pytest.mark.local
def test_api_interface_with_queue_name(llm_request, llm_job_context):
    response = llm_job_context.llm_api.inference(llm_request, "wq_outer_vsegpt")
    print(response)

@pytest.mark.local
def test_api_chat_completion(llm_request, llm_job_context):
    response = llm_job_context.llm_api.chat_completion(ChatCompletionModel.from_prompt_model(llm_request))
    print(response)

@pytest.mark.local
def test_api_chat_completion_with_queue_name(llm_request, llm_job_context):
    response = llm_job_context.llm_api.chat_completion(ChatCompletionModel.from_prompt_model(llm_request), "wq_outer_vsegpt")
    print(response)
