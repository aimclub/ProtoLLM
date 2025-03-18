from langchain_core.tools import StructuredTool
from typing import Optional
from langchain_core.runnables.config import RunnableConfig

from protollm_agents.sdk.agents import StreamingAgent
from protollm_agents.sdk.context import Context


from examples.pipelines.rag_pipeline import RAGPipeline, RAGPrompts


class RAGAgent(StreamingAgent):

    class Arguments(StreamingAgent.Arguments):
        max_input_tokens: int = 6144
        max_chat_history_token_length: int = 24576
        prompts: Optional[RAGPrompts] = None
        retrieving_top_k: int = 3
        generator_context_top_k: Optional[int] = None
        include_original_question_in_queries: bool = True

        planner_model_name: str = "planner_llm"
        generator_model_name: str = "generator_llm"
        tokenizer_name: str = "qwen_2.5"
        store_name: str = "vector_store_name"

        class Config:
            arbitrary_types_allowed = True

    async def stream(
        self, ctx: Context, arguments: Arguments, history: list[tuple], query: str
    ):

        planner_model_name = arguments.planner_model_name
        generator_model_name = arguments.generator_model_name
        tokenizer_name = arguments.tokenizer_name
        store_name = arguments.store_name

        planner_model = ctx.llms.get(planner_model_name).to_runnable()
        generator_model = ctx.llms.get(generator_model_name).to_runnable()
        tokenizer = ctx.tokenizers.get(tokenizer_name).to_runnable()
        store = ctx.vector_stores.get(store_name).to_vector_store()

        pipeline = RAGPipeline(
            agent_id=self.agent_id,
            planner_model=planner_model,
            generator_model=generator_model,
            tokenizer=tokenizer,
            store=store,
            max_input_tokens=arguments.max_input_tokens,
            max_chat_history_token_length=arguments.max_chat_history_token_length,
            retrieving_top_k=arguments.retrieving_top_k,
            generator_context_top_k=arguments.generator_context_top_k,
            include_original_question_in_queries=arguments.include_original_question_in_queries,
        )
        async for e in pipeline.stream(
            question=query, chat_history=history, raise_if_error=True
        ):
            yield e

    async def invoke(
        self,
        ctx: Context,
        arguments: Arguments,
        history: Optional[list[tuple]],
        query: str,
    ):
        planner_model_name = arguments.planner_model_name
        generator_model_name = arguments.generator_model_name
        tokenizer_name = arguments.tokenizer_name
        store_name = arguments.store_name

        planner_model = ctx.llms.get(planner_model_name).to_runnable()
        generator_model = ctx.llms.get(generator_model_name).to_runnable()
        tokenizer = ctx.tokenizers.get(tokenizer_name).to_runnable()
        store = ctx.vector_stores.get(store_name).to_vector_store()

        pipeline = RAGPipeline(
            agent_id=self.agent_id,
            planner_model=planner_model,
            generator_model=generator_model,
            tokenizer=tokenizer,
            store=store,
            max_input_tokens=arguments.max_input_tokens,
            max_chat_history_token_length=arguments.max_chat_history_token_length,
            retrieving_top_k=arguments.retrieving_top_k,
            generator_context_top_k=arguments.generator_context_top_k,
            include_original_question_in_queries=arguments.include_original_question_in_queries,
            runnable_config=None,
        )
        response = pipeline.invoke(
            question=query, chat_history=history, raise_if_error=True
        )

        return response.result

    async def to_runnable():
        return None

    async def to_tool(self, ctx: Context):
        async def pipe_invoke(question: str, runnable_config: RunnableConfig):
            planner_model_name = self.arguments.planner_model_name
            generator_model_name = self.arguments.generator_model_name
            tokenizer_name = self.arguments.tokenizer_name
            store_name = self.arguments.store_name

            planner_model = ctx.llms.get(planner_model_name).to_runnable()
            generator_model = ctx.llms.get(generator_model_name).to_runnable()
            tokenizer = ctx.tokenizers.get(tokenizer_name).to_runnable()
            store = ctx.vector_stores.get(store_name).to_vector_store()

            pipeline = RAGPipeline(
                agent_id=self.agent_id,
                planner_model=planner_model,
                generator_model=generator_model,
                tokenizer=tokenizer,
                store=store,
                max_input_tokens=self.arguments.max_input_tokens,
                max_chat_history_token_length=self.arguments.max_chat_history_token_length,
                retrieving_top_k=self.arguments.retrieving_top_k,
                generator_context_top_k=self.arguments.generator_context_top_k,
                include_original_question_in_queries=self.arguments.include_original_question_in_queries,
                runnable_config=runnable_config,
            )
            result = await pipeline.invoke(question=question, raise_if_error=True)
            return result.answer

        tool = StructuredTool.from_function(
            coroutine=pipe_invoke, name=self.name, description=self.description
        )
        return tool
