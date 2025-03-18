from abc import ABC, abstractmethod
from collections import defaultdict
import json
from typing import Any, AsyncGenerator, Literal
import uuid
import logging
from redis.asyncio import Redis
from protollm_agents.models.responses import ErrorMessage
from protollm_agents.models.schemas import Agent
from protollm_agents.sdk.agents import (
    EnsembleAgent,
    BackgroundAgent,
    RouterAgent,
    StreamingAgent,
)
from protollm_agents.services.db_client import DBClient
from protollm_agents.services.exceptions import AgentNotFound, TaskNotFound

from protollm_agents.sdk.base import Event
from protollm_agents.sdk.context import Context
from protollm_agents.sdk.events import (
    TextEvent,
    MultiTextEvent,
    DictEvent,
    MultiDictEvent,
    StatusEvent,
    ErrorEvent,
)
from protollm_agents.services.storage import Storage, get_storage


logger = logging.getLogger(__name__)


class AgentsManager(ABC):
    router_agent_id: uuid.UUID | None = None
    ensemble_agent_id: uuid.UUID | None = None

    @abstractmethod
    async def initialize(self, agents: list[Any]): ...

    @abstractmethod
    async def list_agents(
        self, agent_type: Literal["streaming", "background", "all"] = "all"
    ) -> list[StreamingAgent | BackgroundAgent]: ...

    @abstractmethod
    async def get_agent(
        self, agent_id: uuid.UUID
    ) -> StreamingAgent | BackgroundAgent: ...

    @abstractmethod
    async def get_streaming_agent(self, agent_id: uuid.UUID) -> StreamingAgent: ...

    @abstractmethod
    async def get_background_agent(self, agent_id: uuid.UUID) -> BackgroundAgent: ...

    def get_router_agent(self) -> RouterAgent:
        if self.router_agent_id is None:
            raise ValueError("Router agent is not set")
        return self.get_streaming_agent(self.router_agent_id)

    def get_ensemble_agent(self) -> EnsembleAgent:
        if self.ensemble_agent_id is None:
            raise ValueError("Ensemble agent is not set")
        return self.get_streaming_agent(self.ensemble_agent_id)

    async def get_context(self, storage: Storage) -> Context:
        return Context(
            llms=storage.llm_models,
            multimodals=storage.multimodal_models,
            embeddings=storage.embeddings,
            tokenizers=storage.tokenizers,
            agents={
                agent.name: agent for agent in await self.list_agents(agent_type="all")
            },
            vector_stores=storage.vector_store_clients,
            tools=dict(),
        )

    async def run_background_agent(
        self,
        task_id: uuid.UUID,
        agent_id: uuid.UUID,
        arguments: dict,
        documents: list[str],
        storage: Storage,
        cache_client: Redis | None = None,
    ) -> None:
        try:
            agent = await self.get_background_agent(agent_id)
            default_arguments = agent.arguments.model_dump().copy()
            default_arguments.update(arguments)
            run_params = agent.get_arguments_class().model_validate(default_arguments)
            event = StatusEvent(
                agent_id=agent_id, result=f"Starting background agent {agent_id}"
            ).model_dump()
            if cache_client is not None:
                await cache_client.rpush(str(task_id), json.dumps(event))
            else:
                self._background_tasks[task_id].append(event)
            async for event in agent.stream(
                ctx=await self.get_context(storage),
                arguments=run_params,
                documents=documents,
            ):
                logger.info(f"Background agent {agent_id} event: {event}")
                event = event.model_dump()
                if cache_client is not None:
                    await cache_client.rpush(str(task_id), json.dumps(event))
                else:
                    self._background_tasks[task_id].append(event)
        except AgentNotFound as e:
            event = ErrorEvent(agent_id=agent_id, result=str(e)).model_dump()
            if cache_client is not None:
                await cache_client.rpush(str(task_id), json.dumps(event))
            else:
                self._background_tasks[task_id].append(event)
        except Exception as e:
            event = ErrorEvent(agent_id=agent_id, result=str(e)).model_dump()
            if cache_client is not None:
                await cache_client.rpush(str(task_id), json.dumps(event))
            else:
                self._background_tasks[task_id].append(event)

    async def get_background_agent_result(
        self, task_id: uuid.UUID, cache_client: Redis | None = None
    ) -> list[Event]:
        if cache_client is not None:
            events = await cache_client.lrange(str(task_id), 0, -1)
            print(events)
            events = [json.loads(event) for event in events]
            if len(events) == 0:
                raise TaskNotFound(f"Task with id {task_id} not found")
            return events
        else:
            if not task_id in self._background_tasks:
                raise TaskNotFound(f"Task with id {task_id} not found")
            return self._background_tasks[task_id]

    async def run_streaming_agent(
        self,
        agent_id: uuid.UUID,
        arguments: dict,
        history: list[tuple],
        query: str,
        storage: Storage,
    ) -> AsyncGenerator[Event, None]:
        try:
            agent = await self.get_streaming_agent(agent_id)
            default_arguments = agent.arguments.model_dump().copy()
            default_arguments.update(arguments)
            run_params = agent.get_arguments_class().model_validate(default_arguments)
            async for event in agent.stream(
                ctx=await self.get_context(storage),
                arguments=run_params,
                history=history,
                query=query,
            ):
                logger.info(f"Streaming agent {agent_id} event: {event}")
                yield self.prepare_event(agent_id=event.agent_id, event=event)
        except Exception as e:
            yield self.prepare_event(
                agent_id=agent_id, event=ErrorEvent(agent_id=agent_id, result=str(e))
            )
            raise e

    async def run_router_agent(
        self, history: list[tuple], query: str, storage: Storage
    ) -> AsyncGenerator[Event, None]:
        try:
            agent = await self.get_router_agent()
            async for event in agent.stream(
                ctx=await self.get_context(storage),
                arguments=agent.arguments,
                history=history,
                query=query,
            ):
                logger.info(f"Streaming router agent event: {event}")
                yield self.prepare_event(agent_id=event.agent_id, event=event)
        except Exception as e:
            yield self.prepare_event(
                agent_id=self.router_agent_id,
                event=ErrorEvent(agent_id=self.router_agent_id, result=str(e)),
            )
            raise e

    async def run_ensemble_agent(
        self, history: list[tuple], query: str, storage: Storage
    ) -> AsyncGenerator[Event, None]:
        try:
            agent = await self.get_ensemble_agent()
            async for event in agent.stream(
                ctx=await self.get_context(storage),
                arguments=agent.arguments,
                history=history,
                query=query,
            ):
                logger.info(f"Streaming router agent event: {event}")
                yield self.prepare_event(agent_id=event.agent_id, event=event)
        except Exception as e:
            yield self.prepare_event(
                agent_id=self.ensemble_agent_id,
                event=ErrorEvent(agent_id=self.ensemble_agent_id, result=str(e)),
            )
            raise e

    def prepare_event(self, agent_id: uuid.UUID, event: Event) -> Event | ErrorMessage:
        match event:
            case (
                TextEvent()
                | MultiTextEvent()
                | DictEvent()
                | MultiDictEvent()
                | StatusEvent()
                | ErrorEvent()
            ):
                return event
            case _:
                return ErrorMessage(detail=f"{agent_id} Got unknown event: {event}")


class InMemoryAgentsManager(AgentsManager):
    streaming_agents: dict[uuid.UUID, StreamingAgent] = dict()
    background_agents: dict[uuid.UUID, BackgroundAgent] = dict()
    _background_tasks: dict[uuid.UUID, list[Event]] = defaultdict(list)

    async def initialize(self, agents: list[Any]):
        for agent in agents:
            if isinstance(agent, StreamingAgent):
                self.streaming_agents[agent.agent_id] = agent
            elif isinstance(agent, BackgroundAgent):
                self.background_agents[agent.agent_id] = agent
            else:
                raise ValueError(f"Unknown agent type: {type(agent)}")
            if isinstance(agent, RouterAgent):
                if self.router_agent_id is not None:
                    raise ValueError("Main agent is already set")
                self.router_agent_id = agent.agent_id
            elif isinstance(agent, EnsembleAgent):
                if self.ensemble_agent_id is not None:
                    raise ValueError("Ensemble agent is already set")
                self.ensemble_agent_id = agent.agent_id

    async def list_agents(
        self,
        agent_type: Literal[
            "streaming", "background", "ensemble", "router", "all"
        ] = "all",
    ) -> list[StreamingAgent | BackgroundAgent | EnsembleAgent | RouterAgent]:
        if agent_type == "streaming":
            return list(self.streaming_agents.values())
        elif agent_type == "background":
            return list(self.background_agents.values())
        else:
            return list(self.streaming_agents.values()) + list(
                self.background_agents.values()
            )

    async def get_agent(self, agent_id: uuid.UUID) -> StreamingAgent | BackgroundAgent:
        if agent_id in self.streaming_agents:
            return self.streaming_agents[agent_id]
        elif agent_id in self.background_agents:
            return self.background_agents[agent_id]
        else:
            raise AgentNotFound(f"Agent with id {agent_id} not found")

    async def get_streaming_agent(self, agent_id: uuid.UUID) -> StreamingAgent:
        agent = self.streaming_agents.get(agent_id)
        if not agent:
            raise AgentNotFound(f"Streaming agent with id {agent_id} not found")
        return agent

    async def get_background_agent(self, agent_id: uuid.UUID) -> BackgroundAgent:
        agent = self.background_agents.get(agent_id)
        if not agent:
            raise AgentNotFound(f"Background agent with id {agent_id} not found")
        return agent


class DatabaseAgentsManager(AgentsManager):
    def __init__(self, db_client: DBClient):
        self.db_client = db_client

    async def initialize(self, agents: list[Any]):
        for agent in agents:
            agent_types = []
            if isinstance(agent, StreamingAgent):
                agent_types.append("streaming")
            elif isinstance(agent, BackgroundAgent):
                agent_types.append("background")
            else:
                raise ValueError(f"Unknown agent type: {type(agent)}")
            if isinstance(agent, RouterAgent):
                agent_types.append("router")
                if self.router_agent_id is not None:
                    raise ValueError("Main agent is already set")
                self.router_agent_id = agent.agent_id
            elif isinstance(agent, EnsembleAgent):
                agent_types.append("ensemble")
                if self.ensemble_agent_id is not None:
                    raise ValueError("Ensemble agent is already set")
                self.ensemble_agent_id = agent.agent_id

            await self.db_client.create_agent(
                Agent.model_validate({**agent.to_dict(), "agent_type": agent_types})
            )

    async def list_agents(
        self,
        agent_type: Literal[
            "streaming", "background", "router", "ensemble", "all"
        ] = "all",
    ) -> list[StreamingAgent | BackgroundAgent | RouterAgent | EnsembleAgent]:
        agents = await self.db_client.get_agents(agent_type=agent_type)
        return [agent.agent_instance for agent in agents]

    async def get_agent(
        self, agent_id: uuid.UUID
    ) -> StreamingAgent | BackgroundAgent | RouterAgent | EnsembleAgent:
        try:
            agent = await self.db_client.get_agent(agent_id)
            return agent.agent_instance
        except AgentNotFound as e:
            raise AgentNotFound(f"Agent with id {agent_id} not found")

    async def get_background_agent(self, agent_id: uuid.UUID) -> BackgroundAgent:
        agent = await self.get_agent(agent_id)
        if not isinstance(agent, BackgroundAgent):
            raise AgentNotFound(f"Agent with id {agent_id} is not a background agent")
        return agent

    async def get_streaming_agent(self, agent_id: uuid.UUID) -> StreamingAgent:
        agent = await self.get_agent(agent_id)
        if not isinstance(agent, StreamingAgent):
            raise AgentNotFound(f"Agent with id {agent_id} is not a streaming agent")
        return agent


agent_manager: InMemoryAgentsManager | DatabaseAgentsManager | None = None


def get_agents_manager() -> AgentsManager:
    return agent_manager
