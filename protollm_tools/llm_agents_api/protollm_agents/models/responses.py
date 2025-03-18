import uuid
from typing import Any
from pydantic import BaseModel, Field, field_serializer


class AgentResponse(BaseModel):
    agent_id: uuid.UUID = Field(..., description="The agent id")
    name: str = Field(..., description="The agent name")
    description: str = Field(..., description="The agent description")
    arguments: dict = Field(..., description="The agent arguments")


class ErrorMessage(BaseModel):
    type: str = "error"
    detail: str = "Error"


class TaskIdResponse(BaseModel):
    task_id: uuid.UUID = Field(..., description="The task id")

    @field_serializer("task_id")
    def serialize_task_id(self, task_id: uuid.UUID, _info: Any) -> str:
        return str(task_id)


class TaskResponse(TaskIdResponse):
    events: list[dict] = Field(..., description="The events")
