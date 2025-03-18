from abc import ABC, abstractmethod
from typing import TypeVar

from protollm_sdk.jobs.job_context import JobContext


class Job(ABC):
    """
    Job interface for integration with outer modules to SDK.
    All the required data for job should be passed and parameters should be defined in advance,
    this also applies to the run method
    """

    @abstractmethod
    def run(self, job_id: str, ctx: JobContext, **kwargs):
        """
        Run the job. The job can use a number of functions defined in the module and service functions from ctx.
        After that, using the ctx, it saves the result to Redis.
        The method should not return any data. If an error occurs,
        it is thrown inside the run method via the `raise`, `raise ex` or `raise … from ex` statement.

        :param job_id: job id
        :type job_id: str
        :param ctx: contextual services
        :type ctx: JobContext
        :param kwargs: The data and parameters required to complete the task are set via key arguments
        :return: None
        :raises TypeError: Error

        """
        pass


TJob = TypeVar('TJob', bound=Job)
