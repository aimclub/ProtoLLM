import logging
from typing import Optional

import redis

from protollm_sdk.models.utils import current_time
from protollm_sdk.object_interface.result_storage.base import ResultStorage
from protollm_sdk.object_interface.result_storage.models import JobStatusType, JobStatusError, \
    JobStatus


class RedisResultStorage(ResultStorage):
    """Redis-based implementation of the ResultStorage interface.

    This class allows creating, updating, and completing jobs in Redis, and
    subscribing to job completion using Redis Pub/Sub mechanism.
    """

    def __init__(
            self,
            redis_client: Optional[redis.Redis] = None,
            redis_host: Optional[str] = 'localhost',
            redis_port: Optional[str | int] = 6379
    ):
        """Initialize RedisResultStorage.
        If redis_client is not provided, a new Redis client will be created,
        otherwise the provided client will be used and redis_host and redis_port will be ignored.

        Args:
            redis_client (Optional[redis.Redis]): Optional Redis client instance.
            redis_host (Optional[str]): The Redis host.
            redis_port (Optional[str | int]): The Redis port.
        """
        if redis_client is None:
            pool = redis.ConnectionPool(host=redis_host, port=redis_port, db=0)
            self._redis = redis.Redis(connection_pool=pool)
            self.url = f"redis://{redis_host}:{redis_port}/0"
        else:
            self._redis = redis_client
            self.url = self._redis.connection_pool.connection_kwargs.get('url', f"redis://{redis_host}:{redis_port}/0")
        self.logger = logging.getLogger(self.__class__.__name__)

    def create_job_status(self, job_id: str) -> None:
        """Create a new job entry in Redis with pending status.

        Args:
            job_id (str): Unique identifier for the job.
        """
        try:
            # Initialize job result with PENDING status
            job_status = JobStatus(status=JobStatusType.PENDING, status_message="Job is created")
            self.__save_job(job_id, job_status)
            self.logger.info(f"Job {job_id} created with pending status.")
        except Exception as ex:
            self.logger.error(f"Failed to create job {job_id}. Error: {ex}")
            raise ex

    def __load_job(self, job_id: str) -> JobStatus:
        """Load job result from Redis.

        Args:
            job_id (str): Unique identifier for the job.

        Returns:
            JobStatus: The job result object.
        """
        data = self._redis.get(job_id)
        if data is None:
            raise Exception(f"Job {job_id} not found in Redis.")
        return JobStatus.model_validate_json(data)

    def __save_job(self, job_id: str, job: JobStatus) -> None:
        """Save job result to Redis.

        Args:
            job_id (str): Unique identifier for the job.
            job (JobStatus): The job result object.
        """
        self._redis.set(job_id, job.model_dump_json())
        self._redis.publish(job_id, 'set')

    def __update_job_status(
            self,
            job_id: str,
            status: JobStatusType,
            status_message: Optional[str] = None,
            result: Optional[str] = None,
            error: Optional[JobStatusError] = None
    ) -> None:

        """Update the job status.

        Args:
            job_id (str): The unique identifier for the job.
            status (JobStatusType): New status of the job.
            status_message (Optional[str]): Optional status message.
            result (Optional[JobResultType]): The result of the job if completed successfully.
            error (Optional[JobStatusError]): The error if the job failed.
        """
        job = self.__load_job(job_id)
        job.status = status
        job.status_message = status_message
        job.last_update = current_time()
        job.result = result
        job.error = error
        self.__save_job(job_id, job)

    def update_job_status(
            self,
            job_id: str,
            status: JobStatusType,
            status_message: Optional[str] = None
    ) -> None:
        """Update the status of an existing job.

        Args:
            job_id (str): Unique identifier for the job.
            status (JobStatusType): New status of the job.
            status_message (Optional[str]): Optional status message.
        """
        try:
            self.__update_job_status(job_id, status, status_message)
            self.logger.info(f"Job {job_id} updated to status {status.value}.")
        except Exception as ex:
            self.logger.error(f"Failed to update job {job_id}. Error: {ex}")
            raise ex

    def complete_job(
            self,
            job_id: str,
            result: Optional[str] = None,
            error: Optional[JobStatusError] = None,
            status_message: Optional[str] = None
    ) -> None:
        """Complete the job by setting its result or error.

        Args:
            job_id (str): Unique identifier for the job.
            result (Optional[T]): The result of the job if completed successfully.
            error (Optional[JobStatusError]): The error if the job failed.
            status_message (Optional[str]): Optional status message.
        """
        try:
            status = JobStatusType.COMPLETED if error is None else JobStatusType.ERROR
            self.__update_job_status(job_id, status, status_message, result, error)
            self.logger.info(f"Job {job_id} completed with status {status.value}.")
        except Exception as ex:
            self.logger.error(f"Failed to complete job {job_id}. Error: {ex}")
            raise ex

    def get_job_status(self, job_id: str) -> JobStatus:
        """Get the status of a job.

        Args:
            job_id (str): Unique identifier for the job.

        Returns:
            JobStatus: The job status object.
        """
        try:
            job = self.__load_job(job_id)
            self.logger.info(f"Job {job_id} status retrieved: {job.status.value}.")
            return job
        except Exception as ex:
            self.logger.error(f"Failed to get job {job_id} status. Error: {ex}")
            raise ex

    def delete_job_status(self, job_id: str) -> None:
        """Delete a job status from Redis.

        Args:
            job_id (str): Unique identifier for the job.
        """
        try:
            self._redis.delete(job_id)
            self.logger.info(f"Job {job_id} deleted from Redis.")
        except Exception as ex:
            self.logger.error(f"Failed to delete job {job_id}. Error: {ex}")
            raise ex

    async def wait_for_completion(
            self,
            job_id: str,
            timeout: float = 30,
            update_cycles: int = 5
    ) -> JobStatus:
        """Wait for the job to reach a terminal state (COMPLETED or ERROR).
        Max waiting time is timeout * update_cycles seconds.

        Args:
            job_id (str): Unique identifier for the job.
            timeout (float): Timeout in seconds for waiting for job completion.
            update_cycles (int): Number of cycles to check for job completion.
        Returns:
            JobStatus[T]: The final job result.
        """
        job = self.get_job_status(job_id)
        if job.status in (JobStatusType.COMPLETED, JobStatusType.ERROR):
            self.logger.info(f"Job {job_id} is already in terminal state: {job.status.value}.")
            return job

        current_iteration = 0
        try:
            pubsub = self._redis.pubsub()
            pubsub.subscribe(job_id)
        except Exception as ex:
            self.logger.error(f"Failed to subscribe to job {job_id}. Error: {ex}")
            raise ex

        try:
            while True:
                message = pubsub.get_message(ignore_subscribe_messages=False, timeout=timeout)
                if message is not None and message['data'] == b'set':
                    job: JobStatus = self.__load_job(job_id)

                    if job.status in (JobStatusType.COMPLETED, JobStatusType.ERROR):
                        self.logger.info(f"Job {job_id} reached terminal state: {job.status.value}.")
                        return job
                if current_iteration >= update_cycles:
                    raise TimeoutError()
        except TimeoutError:
            self.logger.info(f"Job {job_id} is still in progress after {timeout * update_cycles} seconds.")
            raise TimeoutError(f"Job {job_id} did not complete within the timeout period.")
        except Exception as ex:
            self.logger.error(f"Failed waiting for job {job_id} completion. Error: {ex}")
            raise ex
        finally:
            pubsub.unsubscribe(job_id)
