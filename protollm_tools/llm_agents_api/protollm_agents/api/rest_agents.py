import logging
from typing import Literal
import uuid

from fastapi import (
    APIRouter,
    BackgroundTasks,
    Body,
    Depends,
    Query,
    status,
    HTTPException,
)

from protollm_agents.models.requests import AgentRESTQuery
from protollm_agents.models.responses import AgentResponse, TaskResponse, TaskIdResponse
from protollm_agents.services.agents_manager import (
    InMemoryAgentsManager,
    get_agents_manager,
)
from protollm_agents.services.exceptions import AgentNotFound, TaskNotFound
from protollm_agents.services.storage import Storage, get_storage
from protollm_agents.services.cache_client import get_cache
from redis.asyncio import Redis

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get(
    "/",
    summary="Get agents added to the system.",
    status_code=status.HTTP_200_OK,
    response_model=list[AgentResponse],
)
async def get_agents(
    agent_manager: InMemoryAgentsManager = Depends(get_agents_manager),
    agent_type: Literal["streaming", "background", "router", "all"] = Query(
        default="all", description="The type of agents to get"
    ),
) -> list[AgentResponse]:
    return [
        agent.to_dict()
        for agent in await agent_manager.list_agents(agent_type=agent_type)
    ]


@router.get(
    "/{idx}",
    summary="Get agent by the agent id.",
    status_code=status.HTTP_200_OK,
    response_model=AgentResponse,
)
async def get_agent_by_idx(
    idx: uuid.UUID, agent_manager: InMemoryAgentsManager = Depends(get_agents_manager)
) -> AgentResponse:
    try:
        return (await agent_manager.get_agent(idx)).to_dict()
    except AgentNotFound as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))


@router.post(
    "/task",
    summary="Create a long-running task for the agent (Summarization and dates extraction).",
    status_code=status.HTTP_200_OK,
    response_model=TaskIdResponse,
)
async def schedule_task(
    background_tasks: BackgroundTasks,
    request: AgentRESTQuery = Body(..., description="The request to schedule the task"),
    agent_manager: InMemoryAgentsManager = Depends(get_agents_manager),
    storage: Storage = Depends(get_storage),
    cache_client: Redis | None = Depends(get_cache),
) -> TaskIdResponse:
    task_id = uuid.uuid4()
    background_tasks.add_task(
        agent_manager.run_background_agent,
        task_id=task_id,
        agent_id=request.agent_id,
        arguments=request.run_params,
        documents=request.documents,
        storage=storage,
        cache_client=cache_client,
    )
    return TaskIdResponse(task_id=task_id)


@router.get(
    "/task/{task_id}",
    summary="Get the long-running task status by the task id.",
    status_code=status.HTTP_200_OK,
    response_model=TaskResponse,
)
async def get_task_events(
    task_id: uuid.UUID,
    agent_manager: InMemoryAgentsManager = Depends(get_agents_manager),
    cache_client: Redis | None = Depends(get_cache),
) -> TaskResponse:
    try:
        events = await agent_manager.get_background_agent_result(
            task_id=task_id, cache_client=cache_client
        )
    except TaskNotFound as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    return TaskResponse(task_id=task_id, events=events)
