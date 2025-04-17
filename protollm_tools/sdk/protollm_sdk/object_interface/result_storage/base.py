from abc import ABC, abstractmethod
from typing import Optional, Iterable

from protollm_sdk.object_interface.result_storage.models import JobStatusType, JobStatusError, \
    JobStatus


class ResultStorage(ABC):
    """Abstract base class for result storage.

    This class defines the interface for a storage system that can create jobs,
    update job status, complete jobs with a result or error, and subscribe to
    job completion.
    """

    @abstractmethod
    def create_job_status(self, job_id: str) -> None:
        """Create a new job entry in the storage with pending status.

        Args:
            job_id (str): Unique identifier for the job.
        """
        pass

    @abstractmethod
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
        pass

    @abstractmethod
    def complete_job(
            self,
            job_id: str,
            result: Optional[str] = None,
            error: Optional[JobStatusError] = None,
            status_message: Optional[str] = None
    ) -> None:
        """Complete the job by setting the final result or error.

        Args:
            job_id (str): Unique identifier for the job.
            result (Optional): The result of the job if completed successfully.
            error (Optional[JobStatusError]): The error if the job failed.
            status_message (Optional[str]): Optional status message.
        """
        pass

    @abstractmethod
    def get_job_status(
            self,
            job_id: str
    ) -> JobStatus:
        """Get the current status of the job.

        Args:
            job_id (str): Unique identifier for the job.

        Returns:
            JobStatus: The current status of the job.
        """
        pass

    @abstractmethod
    def delete_job_status(
            self,
            job_id: str
    ) -> None:
        """Delete the job status entry from the storage.

        Args:
            job_id (str): Unique identifier for the job.
        """
        pass

    @abstractmethod
    def subscribe(self, job_id: str, timeout: float = 60) -> Iterable[JobStatus]:
        """Subscribe to job status updates. Break the loop when job is completed.

        Args:
            job_id (str): Unique identifier for the job.
            timeout (float): Timeout in seconds. After that raise  TimeoutError.
        """
        pass

    @abstractmethod
    def wait_completeness(
            self,
            job_id: str,
            timeout: float = 60
    ) -> JobStatus:
        """Wait for the job to reach a terminal state (COMPLETED or ERROR).

        Args:
            job_id (str): Unique identifier for the job.
            timeout (float): Timeout in seconds.

        Returns:
            JobStatus: The final job result.
        """
        pass
