from typing import List, Optional, Union

from pydantic import BaseModel, Field


class Response(BaseModel):
    """Final response to the user."""

    response: str


class Plan(BaseModel):
    """Plan consisting of steps. Each step is a list of tasks.

    - Single-task step means sequential execution.
    - Multi-task step means parallel execution.
    """

    steps: List[List[str]] = Field(
        ...,
        description="List of steps. Each step is a list of tasks. One task = sequential, multiple = parallel.",
        example=[
            ["Prepare data"],
            ["Train model"],
            ["Predict for molecule1", "Predict for molecule2"],
        ],
    )


class ReplanAction(BaseModel):
    """Action returned by replanner — either a final response or an updated plan."""

    action: str = Field(..., description="Either 'response' or 'steps'")
    response: Optional[str] = Field(
        None, description="Final user-facing response if action = 'response'"
    )
    steps: Optional[List[List[str]]] = Field(
        None,
        description="Updated plan steps if action = 'steps'",
        example=[["Train model"], ["Predict for molecule1", "Predict for molecule2"]],
    )


class Worker(BaseModel):
    """Workers to call in the next step"""

    next: List[str] = Field(description="List of next workers to call")


class WorkerСhat(BaseModel):
    """Workers to call in the next step"""

    next: str = Field(description="Next worker to call")


class Chat(BaseModel):
    """Action to perform"""

    action: Union[Response, WorkerСhat] = Field(
        description="Action to perform. If you want to respond to user, use Response. "
        "If you need to further use tools to get the answer, use Next."
    )

    last_memory: Optional[str] = Field(
        description="last memory of the user, if any", default=""
    )
