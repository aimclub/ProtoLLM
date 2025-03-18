import logging
import uuid
from typing import Optional, List, Any, Callable

import chromadb
from langchain_community.embeddings.huggingface_hub import HuggingFaceHubEmbeddings
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models import LLM

from protollm_sdk.jobs.job import Job
from protollm_sdk.jobs.job_context import JobContext
from protollm_sdk.models.job_context_models import (
    PromptModel,
    PromptMeta,
    ResponseModel,
)

from protollm.templates.prompt_templates.rag_prompt_templates import EOS, EOT
from protollm.rags.rag_core.retriever import (
    DocsSearcherModels,
    DocRetriever,
    RetrievingPipeline,
)
from protollm.rags.rag_core.utils import run_multiple_rag
from protollm.rags.settings.chroma_settings import settings

from protollm.raw_data_processing.docs_transformers.key_words_splitter import (
    KeywordExtractor,
)

logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler())
logger.setLevel(logging.INFO)


def get_simple_retriever(
    docs_searcher_models: DocsSearcherModels,
    top_k: int,
    preprocess_query: Optional[Callable[[str], str]] = None,
) -> DocRetriever:
    return DocRetriever(top_k, docs_searcher_models, preprocess_query)


def get_advanced_retriever(
    docs_searcher_models: DocsSearcherModels, top_ks: list[int] = None
):
    if top_ks is None:
        top_ks = [100, 50, 10]
    file_name_retriever = get_simple_retriever(docs_searcher_models, top_ks[0])
    keywords_retriever = get_simple_retriever(
        docs_searcher_models,
        top_ks[1],
        lambda query: KeywordExtractor().get_object_action_pair(query)[0],
    )
    content_retriever = get_simple_retriever(docs_searcher_models, top_ks[2])
    return [file_name_retriever, keywords_retriever, content_retriever]


def get_pipelines_retriever(
    list_retrievers: list[list[DocRetriever]], list_collection_names: list[list[str]]
) -> list[RetrievingPipeline]:
    return [
        RetrievingPipeline()
        .set_retrievers(retrievers)
        .set_collection_names(collection_names)
        for retrievers, collection_names in zip(list_retrievers, list_collection_names)
    ]


class CustomLLM(LLM):
    ctx: JobContext = None

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        temperature = kwargs.get("temperature", 0)
        tokens_limit = kwargs.get("tokens_limit", 8000)
        stop_words = kwargs.get("stop_words", [EOS, EOT])

        return self.ctx.outer_llm_api.inference(
            PromptModel(
                job_id=str(uuid.uuid4()),
                content=prompt,
                meta=PromptMeta(
                    temperature=temperature,
                    tokens_limit=tokens_limit,
                    stop_words=stop_words,
                ),
            )
        ).content

    def _llm_type(self) -> str:
        return "custom"


class RAGJob(Job):
    def run(self, job_id: str, ctx: JobContext, **kwargs: Any):
        """
        Run RAG pipeline and save the LLM response to outer database

        :param user_prompt: input question
        :param use_advanced_rag:
        :return: None
        """
        user_prompt: str | list[str] = kwargs["user_prompt"]
        use_advanced_rag: bool = kwargs.get("use_advanced_rag", False)
        client = chromadb.HttpClient(
            host=settings.chroma_host, port=settings.chroma_port
        )
        llm = CustomLLM(ctx=ctx, n=10)
        encoder = HuggingFaceHubEmbeddings(
            huggingfacehub_api_token="any-token", model=settings.embedding_host
        )
        docs_searcher_models = DocsSearcherModels(
            embedding_model=encoder, chroma_client=client
        )

        logger.info("Necessary setting setup")

        simple_pipeline = get_pipelines_retriever(
            [[get_simple_retriever(docs_searcher_models, 10)]],
            [[settings.collection_name]],
        )
        advanced_pipeline = get_pipelines_retriever(
            [
                [get_simple_retriever(docs_searcher_models, 10)],
                get_advanced_retriever(docs_searcher_models),
            ],
            [[settings.collection_name], settings.collection_names_for_advance],
        )

        pipelines = simple_pipeline if not use_advanced_rag else advanced_pipeline

        # Run RAG process and obtain the final LLM response
        llm_response = run_multiple_rag(user_prompt, llm, pipelines)
        logger.info(f'LLM responses the follow text: "{llm_response}"')

        # Save result to DB
        ctx.result_storage.save_dict(
            job_id, ResponseModel(content=llm_response).model_dump()
        )
