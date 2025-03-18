from contextlib import asynccontextmanager
import logging

from fastapi import FastAPI
from fastapi.responses import ORJSONResponse
import uvicorn
from redis.asyncio import Redis

from protollm_agents.api import rest_agents, socket_agents
from langchain_core.runnables import Runnable

from protollm_agents.sdk.vector_stores import BaseVectorStore
from protollm_agents.sdk.models import (
    ChatModel,
    EmbeddingAPIModel,
    CompletionModel,
    MultimodalModel,
    TokenizerModel,
)
from protollm_agents.sdk.base import ModelType
from protollm_agents.services import db_client
from protollm_agents.services import cache_client
from protollm_agents.sdk.agents import (
    EnsembleAgent,
    BackgroundAgent,
    RouterAgent,
    StreamingAgent,
)
from protollm_agents.services import agents_manager
from protollm_agents.services import storage
from protollm_agents.configs import EntrypointConfig
import yaml
import importlib
from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)


logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


class Entrypoint:
    app: FastAPI | None = None
    agents: list[StreamingAgent | BackgroundAgent] = list()
    models: list[Runnable] = list()
    vector_stores: list[BaseVectorStore] = list()
    _is_admin = False

    def __init__(
        self,
        config_path: str | None = None,
        models: list[ModelType] | None = None,
        agents: list[StreamingAgent | BackgroundAgent] | None = None,
        vector_stores: list[BaseVectorStore] | None = None,
    ):
        self.config = self._load_config(config_path)
        if models is not None:
            self.models = models
        if agents is not None:
            self.agents = agents
        if vector_stores is not None:
            self.vector_stores = vector_stores

        app = FastAPI(
            title="Agents API",
            lifespan=self.lifespan,
            default_response_class=ORJSONResponse,
        )
        app.include_router(rest_agents.router)
        app.include_router(socket_agents.router)
        self.app = app

    def _load_config(self, config_path: str | None = None) -> EntrypointConfig | None:
        if config_path:
            with open(config_path, "r") as f:
                config = yaml.safe_load(f)
            config = EntrypointConfig(**config)
            logger.info(
                f"Loaded config from {config_path}. Running in admin mode: {config.is_admin}"
            )
            self._is_admin = config.is_admin
            return config

    @asynccontextmanager
    async def lifespan(self, app: FastAPI):
        logger.info(f"Running in {'admin' if self._is_admin else 'debug'} mode")
        if self.config is None and (not self.models or not self.agents):
            raise ValueError("Both models and agents must be provided")
        models = self.models
        agents = self.agents
        vector_stores = self.vector_stores
        router_agent_id, ensemble_agent_id = None, None
        for agent in agents:
            if isinstance(agent, RouterAgent):
                router_agent_id = agent.agent_id
            if isinstance(agent, EnsembleAgent):
                ensemble_agent_id = agent.agent_id
        if self.config is not None:
            for model in self.config.models:
                models.append(model.params)
            for agent in self.config.agents:
                module_name, class_name = agent.class_path.rsplit(".", 1)
                agent_cls = getattr(importlib.import_module(module_name), class_name)
                agent_instance = agent_cls(
                    agent_id=agent.agent_id,
                    name=agent.name,
                    description=agent.description,
                    arguments=agent_cls.get_arguments_class().model_validate(
                        agent.default_params
                    ),
                )
                agents.append(agent_instance)
                if isinstance(agent_instance, RouterAgent):
                    router_agent_id = agent_instance.agent_id
                if isinstance(agent_instance, EnsembleAgent):
                    ensemble_agent_id = agent_instance.agent_id
            for vector_store in self.config.vector_stores:
                vector_stores.append(vector_store.params)

        if router_agent_id is None:
            router_agent = RouterAgent(
                name="Router",
                description="Router agent",
                arguments=RouterAgent.get_arguments_class().model_validate({}),
            )
            agents.append(router_agent)
        if ensemble_agent_id is None:
            ensemble_agent = EnsembleAgent(
                name="Ensemble",
                description="Ensemble agent",
                arguments=EnsembleAgent.get_arguments_class().model_validate({}),
            )
            agents.append(ensemble_agent)

        for vector_store in vector_stores:
            vector_store.initialize_embeddings_model(
                {model.name: model for model in models}
            )

        storage.storage = storage.Storage(
            llm_models={
                model.name: model
                for model in models
                if isinstance(model, (CompletionModel, ChatModel))
            },
            multimodal_models={
                model.name: model
                for model in models
                if isinstance(model, MultimodalModel)
            },
            embeddings={
                model.name: model
                for model in models
                if isinstance(model, EmbeddingAPIModel)
            },
            vector_store_clients={
                vector_store.name: vector_store for vector_store in vector_stores
            },
            tokenizers={
                model.name: model
                for model in models
                if isinstance(model, TokenizerModel)
            },
        )
        if self._is_admin:
            logger.info(
                f"Initializing connection to Redis at {self.config.redis_host}:{self.config.redis_port}"
            )
            cache_client.cache = Redis(
                host=self.config.redis_host,
                port=self.config.redis_port,
                db=self.config.redis_db,
                decode_responses=True,
            )
            logger.info(
                f"Initializing connection to Postgres at {self.config.postgres_host}:{self.config.postgres_port}"
            )
            engine = create_async_engine(
                f"postgresql+asyncpg://{self.config.postgres_user}:{self.config.postgres_password}@{self.config.postgres_host}"
                f":{self.config.postgres_port}/{self.config.postgres_db}"
            )
            db_client.engine = engine
            db_client.SessionLocal = async_sessionmaker(
                autocommit=False,
                autoflush=False,
                bind=db_client.engine,
                class_=AsyncSession,
            )
            await db_client.init_db()
            session = db_client.SessionLocal()
            agents_manager.agent_manager = agents_manager.DatabaseAgentsManager(
                db_client.DBClient(session)
            )

        else:
            agents_manager.agent_manager = agents_manager.InMemoryAgentsManager()

        await agents_manager.agent_manager.initialize(agents)
        yield

        if self._is_admin:
            await session.close()
            await db_client.engine.dispose()
            await cache_client.cache.close()

    def run(self, port: int = 8080, host: str = "0.0.0.0"):
        uvicorn.run(
            self.app,
            host=host if not self.config else self.config.app_host,
            port=port if not self.config else self.config.app_port,
        )
