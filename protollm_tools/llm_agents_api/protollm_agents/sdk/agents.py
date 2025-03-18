from abc import ABC, abstractmethod
from typing import AsyncGenerator

from langchain_openai import ChatOpenAI

from protollm_agents.sdk.base import AgentAnswer, BaseAgent, Event
from protollm_agents.sdk.context import Context
from protollm_agents.sdk.pipelines.router_pipeline import RouterPipeline
from protollm_agents.sdk.pipelines.ensemble_router_pipeline import (
    EnsembleRouterPipeline,
)


class StreamingAgent(BaseAgent, ABC):
    """
    A streaming agent is an agent that streams its output to the client in real-time.
    """

    @abstractmethod
    async def stream(
        self,
        ctx: Context,
        arguments: BaseAgent.Arguments,
        history: list[tuple],
        query: str,
    ) -> AsyncGenerator[Event, None]: ...

    @abstractmethod
    async def invoke(
        self,
        ctx: Context,
        arguments: BaseAgent.Arguments,
        history: list[tuple],
        query: str,
    ) -> AgentAnswer: ...


class BackgroundAgent(BaseAgent, ABC):
    """
    A background agent is an agent that runs in the background and returns a result
    after it has finished processing.
    """

    @abstractmethod
    async def stream(
        self, ctx: Context, arguments: BaseAgent.Arguments, documents: list[str]
    ) -> AsyncGenerator[Event, None]: ...

    @abstractmethod
    async def invoke(
        self, ctx: Context, arguments: BaseAgent.Arguments, documents: list[str]
    ) -> AgentAnswer: ...


class RouterAgent(StreamingAgent):
    """
    A router agent is an agent that routes the query to the appropriate agent.
    """

    class Arguments(StreamingAgent.Arguments):
        max_input_tokens: int = 6144
        max_chat_history_token_length: int = 24576

        router_model: str = "router_llm"
        tools_names: list[str] = ["rag_education", "rag_environment"]
        tokenizer_name: str = "qwen_2.5"

        class Config:
            arbitrary_types_allowed = True

    async def stream(
        self, ctx: Context, arguments: Arguments, history: list[tuple], query: str
    ) -> AsyncGenerator[Event, None]:
        model_name = arguments.router_model
        tools_names = arguments.tools_names
        tokenizer_name = arguments.tokenizer_name
        model = ctx.llms.get(model_name).to_runnable()
        tools = []
        for tool in tools_names:
            tool_obj = await ctx.agents.get(tool).to_tool(ctx)
            tools.append(tool_obj)
        tokenizer = ctx.tokenizers.get(tokenizer_name).to_runnable()

        assert isinstance(model.bound, ChatOpenAI)
        assert tools is not None
        assert tokenizer is not None

        pipeline = RouterPipeline(
            agent_id=self.agent_id,
            model=model,
            tools=tools,
            tokenizer=tokenizer,
            max_input_tokens=arguments.max_input_tokens,
            max_chat_history_token_length=arguments.max_chat_history_token_length,
        )

        async for e in pipeline.stream(
            question=query, chat_history=history, raise_if_error=True
        ):
            yield e

    async def invoke(
        self, ctx: Context, arguments: Arguments, history: list[tuple], query: str
    ) -> AgentAnswer:
        model_name = arguments.router_model
        tools_names = arguments.tools_names
        tokenizer_name = arguments.tokenizer_name

        model = ctx.llms.get(model_name).to_runnable()
        tools = []
        for tool in tools_names:
            tool_obj = await ctx.agents.get(tool).to_tool(ctx)
            tools.append(tool_obj)
        tokenizer = ctx.tokenizers.get(tokenizer_name).to_runnable()
        assert isinstance(model.bound, ChatOpenAI)
        assert tools is not None
        assert tokenizer is not None

        pipeline = RouterPipeline(
            agent_id=self.agent_id,
            model=model,
            tools=tools,
            tokenizer=tokenizer,
            max_input_tokens=arguments.max_input_tokens,
            max_chat_history_token_length=arguments.max_chat_history_token_length,
        )
        response = pipeline.invoke(
            question=query, chat_history=history, raise_if_error=True
        )

        return response.result

    async def to_tool(self, *args, **kwargs):
        raise NotImplementedError

    async def to_runnable(self):
        return None


class EnsembleAgent(StreamingAgent):
    """
    An ensemble agent is an agent that uses ensemble to run a task.
    """

    class Arguments(StreamingAgent.Arguments):
        max_input_tokens: int = 6144
        max_chat_history_token_length: int = 24576

        router_model: str = "router_llm"
        tools_names: list[str] = ["rag_education", "rag_environment"]
        tokenizer_name: str = "qwen_2.5"

        class Config:
            arbitrary_types_allowed = True

    async def stream(
        self, ctx: Context, arguments: Arguments, history: list[tuple], query: str
    ) -> AsyncGenerator[Event, None]:
        model_name = arguments.router_model
        tools_names = arguments.tools_names
        tokenizer_name = arguments.tokenizer_name
        model = ctx.llms.get(model_name).to_runnable()
        tools = []
        for tool in tools_names:
            tool_obj = await ctx.agents.get(tool).to_tool(ctx)
            tools.append(tool_obj)
        tokenizer = ctx.tokenizers.get(tokenizer_name).to_runnable()

        assert isinstance(model.bound, ChatOpenAI)
        assert tools is not None
        assert tokenizer is not None

        pipeline = EnsembleRouterPipeline(
            agent_id=self.agent_id,
            model=model,
            tools=tools,
            tokenizer=tokenizer,
            max_input_tokens=arguments.max_input_tokens,
            max_chat_history_token_length=arguments.max_chat_history_token_length,
        )

        async for e in pipeline.stream(
            question=query, chat_history=history, raise_if_error=True
        ):
            yield e

    async def invoke(
        self, ctx: Context, arguments: Arguments, history: list[tuple], query: str
    ) -> AgentAnswer:
        model_name = arguments.router_model
        tools_names = arguments.tools_names
        tokenizer_name = arguments.tokenizer_name

        model = ctx.llms.get(model_name).to_runnable()
        tools = []
        for tool in tools_names:
            tool_obj = await ctx.agents.get(tool).to_tool(ctx)
            tools.append(tool_obj)
        tokenizer = ctx.tokenizers.get(tokenizer_name).to_runnable()
        assert isinstance(model.bound, ChatOpenAI)
        assert tools is not None
        assert tokenizer is not None

        pipeline = EnsembleRouterPipeline(
            agent_id=self.agent_id,
            model=model,
            tools=tools,
            tokenizer=tokenizer,
            max_input_tokens=arguments.max_input_tokens,
            max_chat_history_token_length=arguments.max_chat_history_token_length,
        )
        response = pipeline.invoke(
            question=query, chat_history=history, raise_if_error=True
        )

        return response.result

    async def to_tool(self, *args, **kwargs):
        raise NotImplementedError

    async def to_runnable(self):
        return None
