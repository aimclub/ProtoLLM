import uuid

from protollm_sdk.config import Config
from protollm_sdk.celery.app import task_test
from protollm_sdk.celery.job import TextEmbedderJob, ResultStorageJob, LLMAPIJob, \
    VectorDBJob, OuterLLMAPIJob  # , LangchainLLMAPIJob
from protollm_sdk.object_interface import RedisWrapper


def embed():
    """An example of using an embedder with celery"""
    text_embedder_request = {"job_id": "0",
                             "inputs": "Какой-то умный текст. Или не очень умный.",
                             "truncate": False}
    random_id = uuid.uuid4()
    result = task_test.apply_async(args=(TextEmbedderJob.__name__, random_id), kwargs=text_embedder_request)
    print(result.get())


def store_results():
    """An example of using a storage with celery"""
    random_id = uuid.uuid4()
    result_storage = {"job_id": random_id,
                      "result": {"question": "Очень умный вопрос.",
                                 "answers": "Не очень умный ответ"}}

    result = task_test.apply_async(args=(ResultStorageJob.__name__, random_id), kwargs=result_storage)
    print(result.get())


def llm_resp():
    """An example of using a llm with celery"""
    meta = {"temperature": 0.5,
            "tokens_limit": 10,
            "stop_words": None}
    llm_request = {"job_id": str(uuid.uuid4()),
                   "meta": meta,
                   "content": "Сколько попугаев Какаду в одном метре?"}
    result = task_test.apply_async(args=(LLMAPIJob.__name__, llm_request["job_id"]), kwargs=llm_request)

    print(result.get())


async def out_llm_resp(redis_client: RedisWrapper):
    """An example of using a outer llm with celery"""
    meta = {"temperature": 0.2,
            "tokens_limit": 4096,
            "stop_words": None}
    llm_request = {"job_id": str(uuid.uuid4()),
                   "meta": meta,
                   "content": "Монтаж оголовников, Сборка опор/порталов, Подвеска провода, Укладка активного соляного заземления"}

    task_test.apply_async(args=(OuterLLMAPIJob.__name__, llm_request["job_id"]), kwargs=llm_request)
    result = await redis_client.wait_item(f"{OuterLLMAPIJob.__name__}:{llm_request['job_id']}", timeout=60)

def get_dict(key):
    rd = RedisWrapper(redis_host=Config.redis_host,
                      redis_port=Config.redis_port)
    resp = rd.get_item(key)
    print(resp)


def vector_db():
    random_id = uuid.uuid4()
    result = task_test.apply_async(args=(VectorDBJob.__name__, random_id), task_id=random_id)
    print(result.get())
