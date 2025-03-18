from typing import Annotated
from annotated_types import Len
from pydantic import BaseModel, Field, field_serializer
import uuid


class HistoryRecord(BaseModel):
    query: str = Field(..., description="The query to send to the agent")
    response: str = Field(..., description="The response from the agent")


class RouterSocketQuery(BaseModel):
    dialogue_id: uuid.UUID = Field(..., description="Unique dialogue id")
    query: str = Field(..., description="The query to send to the agent")
    chat_history: Annotated[list[HistoryRecord], Len(max_length=50)] = Field(
        default_factory=list, description="Previous queries in the dialogue"
    )

    @field_serializer("dialogue_id")
    def serialize_id_dialogue(self, dialogue_id: uuid.UUID, _info):
        return str(dialogue_id)

    @property
    def history_as_tuple_list(self) -> list[tuple[str, str]]:
        chat_history = []
        for record in self.chat_history:
            chat_history.extend([("human", record.query), ("ai", record.response)])
        return chat_history


class AgentSocketQuery(RouterSocketQuery):
    agent_id: uuid.UUID = Field(..., description="The ID of the agent to query")
    run_params: dict = Field(
        default_factory=dict, description="The parameters to run the agent with"
    )

    @field_serializer("agent_id")
    def serialize_id_agent(self, agent_id: uuid.UUID, _info):
        return str(agent_id)


class AgentRESTQuery(BaseModel):
    agent_id: uuid.UUID = Field(..., description="The ID of the agent to query")
    run_params: dict = Field(
        default_factory=dict, description="The parameters to run the agent with"
    )
    documents: list[str] = Field(
        default_factory=list, description="The documents to process"
    )

    @field_serializer("agent_id")
    def serialize_id_agent(self, agent_id: uuid.UUID, _info):
        return str(agent_id)
