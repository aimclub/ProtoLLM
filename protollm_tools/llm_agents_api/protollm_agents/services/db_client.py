from contextlib import asynccontextmanager
from typing import Literal
import uuid
from sqlalchemy import select
from sqlalchemy.ext.asyncio import (
    AsyncSession,
    AsyncEngine,
)
from sqlalchemy.exc import NoResultFound

from protollm_agents.models.schemas import Agent
from protollm_agents.models import models as db_models
from protollm_agents.services.exceptions import AgentNotFound


class DBClient:
    def __init__(self, session: AsyncSession):
        self.session = session

    def _get_session(self, session: AsyncSession | None = None) -> AsyncSession:
        return session or self.session

    @asynccontextmanager
    async def execute(self, session: AsyncSession | None = None):
        session = self._get_session(session)
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise

    async def create_agent(
        self, agent: Agent, session: AsyncSession | None = None
    ) -> Agent:
        agent_db = db_models.Agent(**agent.model_dump())
        async with self.execute(session) as sess_:
            sess_.add(agent_db)
            await sess_.flush()
            await sess_.refresh(agent_db)
            return Agent.model_validate(agent_db)

    async def get_agent(
        self, agent_id: uuid.UUID, session: AsyncSession | None = None
    ) -> Agent:
        async with self.execute(session) as sess_:
            query = select(db_models.Agent).where(db_models.Agent.agent_id == agent_id)
            result = await sess_.execute(query)
            try:
                agent_db = result.scalar_one()
            except NoResultFound:
                raise AgentNotFound(f"Agent with id {agent_id} not found")
            return Agent.model_validate(agent_db)

    async def get_agents(
        self,
        agent_type: Literal[
            "background", "streaming", "all", "router", "ensemble"
        ] = "all",
        session: AsyncSession | None = None,
    ) -> list[Agent]:
        async with self.execute(session) as sess_:
            query = select(db_models.Agent)
            if agent_type != "all":
                query = query.where(db_models.Agent.agent_type.contains([agent_type]))
            result = await sess_.execute(query)
            return [Agent.model_validate(row) for row in result.scalars()]


engine: AsyncEngine | None = None
SessionLocal: AsyncSession | None = None


async def init_db():
    async with engine.begin() as conn:
        await conn.run_sync(db_models.Base.metadata.drop_all)
        await conn.run_sync(db_models.Base.metadata.create_all)


async def get_db_client() -> DBClient:
    async with SessionLocal() as session:
        yield DBClient(session)
