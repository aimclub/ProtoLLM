import logging
import time
from abc import ABC, abstractmethod
from enum import Enum, auto
from typing import Type, Callable
from uuid import uuid4

from protollm_sdk.jobs.result_storage import ResultStorage


class InvokeType(Enum):
    """
    Enum for job invocation type
    """

    Blocking = auto()
    Worker = auto()


class JobResult(ABC):
    """
    Abstract class for job result
    """

    def __init__(self, storage: ResultStorage, job_name: str, job_id: str):
        """
        Initialize JobResult

        :param storage: storage for arbitrary job
        :type storage: ResultStorage
        :param job_name: job name
        :type job_name: str
        :param job_id: job id
        :type job_id: str
        """
        self.job_id = job_id
        self.storage = storage.for_job(job_name)

    @abstractmethod
    def get_result(self) -> dict:
        """
        Return result of job

        :return: dict
        """
        ...


class BlockingJobResult(JobResult):
    """
    Class for blocking job result.
    Executed in the caller thread, and the result is accessible immediately.
    """

    def __init__(self, storage: ResultStorage, job_name: str, job_id: str):
        """
        Initialize BlockingJobResult

        :param storage: storage for job
        :type storage: ResultStorage
        :param job_name: job name
        :type job_name: str
        :param job_id: job id
        :type job_id: str
        """
        super().__init__(storage, job_name, job_id)
        self.result = self._get_result()

    def get_result(self) -> dict:
        """
        Return result of job

        :return: dict
        """
        return self.result

    def _get_result(self) -> dict:
        t = time.monotonic()
        timeout = 5 * 60
        wait = 0.25

        while time.monotonic() - t < timeout:
            result = self._ping_result()
            if result is not None:
                return result
            time.sleep(wait)

        raise TimeoutError(
            f"Couldn't retrieve result in {timeout // 60}min {timeout % 60}sec"
        )

    def _ping_result(self) -> dict | None:
        return self.storage.load_dict(self.job_id)


class WorkerJobResult(JobResult):
    """
    Class for worker job result
    Executed on celery worker. Result will be awaited.
    """

    def __init__(self, storage: ResultStorage, job_name: str, job_id: str):
        """
        Initialize WorkerJobResult

        :param storage: storage for job
        :type storage: ResultStorage
        :param job_name: job name
        :type job_name: str
        :param job_id: job id
        :type job_id: str
        """
        super().__init__(storage, job_name, job_id)

    def get_result(self) -> dict:
        """
        Return result of job

        :return: dict
        """
        t = time.monotonic()
        timeout = 10 * 60
        wait = 0.5

        while time.monotonic() - t < timeout:
            result = self._ping_result()
            if result is not None:
                return result
            time.sleep(wait)

        raise TimeoutError(
            f"Couldn't retrieve result in {timeout // 60}min {timeout % 60}sec"
        )

    def _ping_result(self) -> dict | None:
        """
        Ping result of job

        :return: dict | None
        """
        return self.storage.load_dict(self.job_id)


class JobInvoker:
    """
    Class for invoking job
    """

    def __init__(
        self,
        abstract_celery_task: Callable[[type, str, ...], None] | None,
        result_storage: ResultStorage,
        invoke_type: InvokeType = InvokeType.Worker,
    ):
        """
        Initialize JobInvoker

        :param abstract_celery_task: reference to celery.task
        :type abstract_celery_task: Callable[[type, str, ...], None]
        :param result_storage: storage for job
        :type result_storage: ResultStorage
        :param invoke_type: job invocation type
        :type invoke_type: InvokeType
        """

        self._logger = logging.getLogger(self.__class__.__name__)
        self._task = abstract_celery_task
        self._result_storage = result_storage
        self._invoke_type = invoke_type

    def invoke(self, job_class: Type, **kwargs) -> JobResult:
        """
        Invoke job based on specified invoke type

        :param job_class: job class
        :type job_class: Type
        :param kwargs: additional arguments
        :type kwargs: dict
        :return: JobResult
        """

        if self._task is None:
            msg = "Calling JobInvoker without abstract task."
            self._logger.error(msg)
            raise Exception(msg)

        task_id = str(uuid4())
        job_name = (
            job_class.__name__
            if isinstance(job_class, type)
            else job_class.__class__.__name__
        )

        self._logger.info(f"Invoking '{job_name}' with task id'{task_id}'.")

        match self._invoke_type:
            case InvokeType.Blocking:
                self._task(job_class, task_id, **kwargs)
                return BlockingJobResult(self._result_storage, job_name, task_id)
            case InvokeType.Worker:
                self._task.apply_async(args=(job_class, task_id), kwargs=kwargs)
                return WorkerJobResult(self._result_storage, job_name, task_id)
            case _:
                msg = f"Unknown invoke type: {self._invoke_type}."
                self._logger.error(msg)
                raise NotImplementedError(msg)
