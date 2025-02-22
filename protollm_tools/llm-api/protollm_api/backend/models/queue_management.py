from typing import Optional, Any
from pydantic import BaseModel


class QueueManagementModel(BaseModel):
    """A base model for queues management operations"""
    id: Optional[str] = None
    queue_name: Optional[str] = None
    model: Optional[str] = None
    description : Optional[str] = None

class QueueDeclarationModel(QueueManagementModel):
    """A model for adding a new RabbitMQ queue"""
    durable: Optional[bool] = None
    arguments: Optional[dict[str, Any]] = None

class UpdateContentModel(BaseModel):
    model: Optional[str] = None
    description: Optional[str] = None

class QueueUpdateModel(QueueManagementModel):
    """A model for queues metadata modification"""
    update: Optional[UpdateContentModel] = None

class QueuesFetchModel(QueueManagementModel):
    """A model for access to metadata of specified queue"""
    consumers_count: Optional[int] = None
    messages_count: Optional[int] = None

class ActiveWorkersFetchModel(BaseModel):
    """A model for getting an active workers list"""
    content: Optional[list[Any]] = None
