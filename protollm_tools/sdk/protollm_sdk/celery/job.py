import logging

from protollm_sdk.jobs.job import Job
from protollm_sdk.jobs.job_context import JobContext
from protollm_sdk.jobs.job_invoke import InvokeType
from protollm_sdk.models.job_context_models import TextEmbedderRequest, PromptModel


class TextEmbedderJob(Job):
    """Defining the embedder class inherited from Job"""

    def run(self, task_id: str, ctx: JobContext, **kwargs):
        """Redefining the run function"""
        request = TextEmbedderRequest(**kwargs)
        resp = ctx.text_embedder.inference(request)
        return resp


class ResultStorageJob(Job):
    """Defining the storage class inherited from Job"""

    def run(self, task_id: str, ctx: JobContext, **kwargs):
        """Redefining the run function"""
        ctx.result_storage.save_dict(job_id=task_id, result=kwargs)


class LLMAPIJob(Job):
    """Defining the llm class inherited from Job"""

    def run(self, task_id: str, ctx: JobContext, **kwargs):
        """Redefining the run function"""
        request = PromptModel(**kwargs)
        resp = ctx.llm_api.inference(request)
        return resp


class OuterLLMAPIJob(Job):
    """Defining the llm class inherited from Job"""

    def run(self, task_id: str, ctx: JobContext, **kwargs):
        """Redefining the run function"""
        request = PromptModel(**kwargs)
        resp = ctx.outer_llm_api.inference(request)
        ctx.result_storage.save_dict(task_id, resp)
        # TODO add return??


class VectorDBJob(Job):
    """Defining the vector_db class inherited from Job"""

    def run(self, task_id: str, ctx: JobContext, **kwargs):
        """Redefining the run function"""
        resp = ctx.vector_db.api_v1()
        return resp


# class LangchainLLMAPIJob(Job):
#     """Defining the langchain class inherited from Job"""
#      def run(self, task_id: str, ctx: JobContext, **kwargs):
#         """Redefining the run function"""
#         request = LLMRequest(**kwargs)
#         request.job_id = task_id
#         resp = ctx.langchain_llm_api._call(request)
#         return resp


class TestInvocationJob(Job):
    def run(self, task_id: str, ctx: JobContext, **kwargs):
        logger = logging.getLogger(self.__class__.__name__)
        logger.warning(f"Running {self.__class__.__name__}")

        my_kwargs = dict(my_kwarg="simple_value")

        # This lets you manipulate the behaviour of the job invoker.
        # Normally, you shouldn't set this manually, but you can for the testing purposes.
        # - InvokeType.Worker (default on server) - the called job is executed on the remote worker.
        #   Call returns immediately, so you can do something before waiting for the result.
        # - InvokeType.Blocking - The called job is executed on the caller therad, and the result is accessible right afther the call returns.
        ctx.job_invoker._invoke_type = (
            InvokeType.Worker
            if kwargs.get("invoker", "blocking") == "worker"
            else InvokeType.Blocking
        )

        # Job calling is executed via passing the job class reference and the corresponding kwargs.
        job_result = ctx.job_invoker.invoke(ResultStorageJob, **my_kwargs)
        logger.warning(
            f"{self.__class__.__name__} sent {ResultStorageJob.__class__.__name__} task with kwargs {my_kwargs}"
        )

        # The get_result behaviour depends on the InvokeType (described above).
        result = job_result.get_result()
        logger.warning(f"{self.__class__.__name__} got result {result}")

        ctx.result_storage.save_dict(task_id, result)
