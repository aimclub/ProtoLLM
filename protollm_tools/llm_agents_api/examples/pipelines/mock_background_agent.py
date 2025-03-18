from typing import AsyncGenerator
from protollm_agents.sdk import BackgroundAgent

from protollm_agents.sdk.base import Event
from protollm_agents.sdk.events import TextEvent, MultiDictEvent, StatusEvent, DictEvent
from protollm_agents.sdk.context import Context


class SummaryAgent(BackgroundAgent):

    class Arguments(BackgroundAgent.Arguments):
        a: int

    async def invoke(self, *args, **kwargs):
        pass

    async def stream(
        self, ctx: Context, arguments: Arguments, documents: list[str]
    ) -> AsyncGenerator[Event, None]:
        mock_events = [
            StatusEvent(
                agent_id=self.agent_id, result="Planning", name="Planning event"
            ),
            MultiDictEvent(
                agent_id=self.agent_id,
                result=[
                    {"id": "1", "page_content": "Mock document 1"},
                    {"id": "2", "page_content": "Mock document 2"},
                ],
            ),
            TextEvent(agent_id=self.agent_id, result="Mock", is_eos=False),
            TextEvent(agent_id=self.agent_id, result="Mock answer", is_eos=False),
            TextEvent(
                agent_id=self.agent_id, result="Mock answer to question", is_eos=False
            ),
            DictEvent(agent_id=self.agent_id, result=dict(a=1), is_eos=True),
        ]
        for event in mock_events:
            yield event

    async def to_tool(self, *args, **kwargs):
        pass

    async def to_runnable(self):
        pass
