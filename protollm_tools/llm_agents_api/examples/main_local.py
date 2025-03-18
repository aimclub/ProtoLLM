import logging

from protollm_agents.entrypoint import Entrypoint
from protollm_agents.sdk.agents import StreamingAgent
from protollm_agents.sdk.models import (
    CompletionModel,
    ChatModel,
    TokenizerModel,
    EmbeddingAPIModel,
)
from protollm_agents.sdk.vector_stores import ChromaVectorStore
from examples.pipelines.rag_agent import RAGAgent


logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

HOST = "d.dgx"

if __name__ == "__main__":
    epoint = Entrypoint(
        models=[
            CompletionModel(
                name="planner_llm",
                model="/model",
                temperature=0.01,
                top_p=0.95,
                streaming=False,
                url=f"http://{HOST}:8001/v1",
                api_key="token-abc123",
            ),
            CompletionModel(
                name="generator_llm",
                model="/model",
                temperature=0.01,
                top_p=0.95,
                streaming=True,
                url=f"http://{HOST}:8001/v1",
                api_key="token-abc123",
            ),
            ChatModel(
                name="router_llm",
                model="/model",
                temperature=0.01,
                top_p=0.95,
                streaming=True,
                url=f"http://{HOST}:8001/v1",
                api_key="token-abc123",
            ),
            TokenizerModel(
                name="qwen_2.5",
                path_or_repo_id="Qwen/Qwen2.5-7B-Instruct",
            ),
            EmbeddingAPIModel(
                name="e5-mistral-7b-instruct",
                model="/models/e5-mistral-7b-instruct",
                url=f"http://{HOST}:58891/v1",
                api_key="token-abc123",
                check_embedding_ctx_length=False,
                tiktoken_enabled=False,
            ),
        ],
        agents=[
            RAGAgent(
                name="rag_domain1",
                description="Поиск по базе докуметов домена 1",
                arguments=RAGAgent.Arguments(
                    max_input_tokens=6144,
                    max_chat_history_token_length=24576,
                    retrieving_top_k=2,
                    generator_context_top_k=2,
                    include_original_question_in_queries=True,
                    planner_model_name="planner_llm",
                    generator_model_name="generator_llm",
                    tokenizer_name="qwen_2.5",
                    store_name="chroma_domain1",
                ),
            ),
            RAGAgent(
                name="rag_domain2",
                description="Поиск по базе докуметов домена 2",
                arguments=RAGAgent.Arguments(
                    max_input_tokens=6144,
                    max_chat_history_token_length=24576,
                    retrieving_top_k=2,
                    generator_context_top_k=2,
                    include_original_question_in_queries=True,
                    planner_model_name="planner_llm",
                    generator_model_name="generator_llm",
                    tokenizer_name="qwen_2.5",
                    store_name="chroma_domain2",
                ),
            ),
            RAGAgent(
                name="rag_domain3",
                description="Поиск по базе докуметов домена 3",
                arguments=RAGAgent.Arguments(
                    max_input_tokens=6144,
                    max_chat_history_token_length=24576,
                    retrieving_top_k=2,
                    generator_context_top_k=2,
                    include_original_question_in_queries=True,
                    planner_model_name="planner_llm",
                    generator_model_name="generator_llm",
                    tokenizer_name="qwen_2.5",
                    store_name="chroma_domain3",
                ),
            ),
        ],
        vector_stores=[
            ChromaVectorStore(
                name="chroma_domain1",
                description="Chroma vector store",
                embeddings_model_name="e5-mistral-7b-instruct",
                host=HOST,
                port=57777,
                collection_name="education",
            ),
            ChromaVectorStore(
                name="chroma_domain2",
                description="Chroma vector store",
                embeddings_model_name="e5-mistral-7b-instruct",
                host=HOST,
                port=57777,
                collection_name="environment",
            ),
            ChromaVectorStore(
                name="chroma_domain3",
                description="Chroma vector store",
                embeddings_model_name="e5-mistral-7b-instruct",
                host=HOST,
                port=57777,
                collection_name="union",
            ),
        ],
    )
    epoint.run()
