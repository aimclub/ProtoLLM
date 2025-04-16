from enum import Enum
from typing import TypeVar, Generic, Optional

from pydantic import BaseModel, Field

from protollm_sdk.models.utils import current_time


class JobStatusType(Enum):
    """
    Enum for job status
    Possible statuses:
    PENDING: Job is pending and waiting to be processed
    IN_PROGRESS: Job is currently being processed
    COMPLETED: Job has been completed successfully
    ERROR: Job has encountered an error during processing
    """
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    ERROR = "error"


class JobStatusError(BaseModel):
    """
    Class for error message in Redis job result
    Attributes:
        type: Type of error, e.g. "ValidationError", "ConnectionError"
        msg: Error message
    """
    type: str
    msg: str



class JobStatus(BaseModel):
    """
    Class for job result in Redis
    Attributes:
        status: Status of the job
        status_message: Status message, e.g. "Job written to RabbitMQ"; "Job running on worker"
        result: Result of the job, if completed successfully
        error: Error if any
    """
    status: JobStatusType = Field(default=JobStatusType.PENDING, examples=[
        JobStatusType.PENDING,
        JobStatusType.IN_PROGRESS,
        JobStatusType.COMPLETED,
        JobStatusType.ERROR
    ], description="Job status")
    last_update: Optional[str] = Field(default_factory=current_time, description="Last update timestamp")
    status_message: Optional[str] = Field(default=None, description="Status message")
    result: Optional[str] = Field(default=None, description="Result of the job, if completed successfully. Format: JSON string")
    error: Optional[JobStatusError] = Field(default=None, description="Error message, if any")
