from protollm_sdk.config import Config
from protollm_sdk.jobs.job_context import JobContext
from protollm_sdk.jobs.job_invoke import JobInvoker, InvokeType
from protollm_sdk.jobs.llm_api import LLMAPI
from protollm_sdk.jobs.outer_llm_api import OuterLLMAPI
from protollm_sdk.jobs.result_storage import ResultStorage
from protollm_sdk.jobs.text_embedder import TextEmbedder
from protollm_sdk.jobs.vector_db import VectorDB


def construct_job_context(job_name: str, abstract_task=None) -> JobContext:
    """
    Create JobContext object with object access to functions and services, based on environment variable values.

    :param job_name: job name
    :type job_name: str
    :param abstract_task: optional reference to celery.task for recursive Job calling from Job
    :return: JobContext object
    """

    llm_api = LLMAPI(Config.llm_api_host, Config.llm_api_port)

    out_llm_api = OuterLLMAPI(Config.outer_llm_key)

    text_embedder = TextEmbedder(Config.text_embedder_host, Config.text_embedder_port)

    result_storage = ResultStorage(
        redis_host=Config.redis_host, redis_port=Config.redis_port, prefix=job_name
    )

    invoke_type_str = Config.job_invocation_type.lower()
    match invoke_type_str:
        case "worker":
            invoke_type = InvokeType.Worker
        case "blocking":
            invoke_type = InvokeType.Blocking
        case _:
            raise ValueError(f"Found unknown invocation type '{invoke_type_str}'.")
    job_invoker = JobInvoker(abstract_task, result_storage, invoke_type)

    vector_db = VectorDB(
        vector_bd_host=Config.vector_bd_host, vector_db_port=Config.vector_db_port
    )

    return JobContext(
        llm_api=llm_api,
        outer_llm_api=out_llm_api,
        text_embedder=text_embedder,
        result_storage=result_storage,
        vector_db=vector_db,
        job_invoker=job_invoker,
    )
