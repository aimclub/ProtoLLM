import importlib
from typing import Literal
from pydantic import BaseModel, Field, ConfigDict
import uuid

from protollm_agents.sdk.base import BaseAgent


class Agent(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    agent_id: uuid.UUID = Field(..., description="ID of the agent")
    name: str = Field(..., description="Name of the agent")
    description: str = Field(..., description="Description of the agent")
    arguments: dict = Field(..., description="Agent parameters")
    module_name: str = Field(..., description="Module name of the agent")
    class_name: str = Field(..., description="Class name of the agent")
    agent_type: list[Literal["background", "streaming", "router", "ensemble"]] = Field(
        default_factory=list, description="Type of the agent"
    )

    @property
    def agent_instance(self) -> BaseAgent:
        module = importlib.import_module(self.module_name)
        agent_cls = getattr(module, self.class_name)
        return agent_cls(
            agent_id=self.agent_id,
            name=self.name,
            description=self.description,
            arguments=agent_cls.get_arguments_class().model_validate(self.arguments),
            module_name=self.module_name,
            class_name=self.class_name,
        )
