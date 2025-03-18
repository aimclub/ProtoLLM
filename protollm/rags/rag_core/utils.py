import logging

from langchain_core.documents import Document
from langchain_core.language_models import LLM

from protollm.templates.prompt_templates.rag_prompt_templates import (
    PROMPT_RANK,
    PROMPT_LLM_RESPONSE,
)
from protollm.rags.rag_core.reranker import LLMReranker
from protollm.rags.rag_core.retriever import (
    DocRetriever,
    DocsSearcherModels,
    RetrievingPipeline,
)

logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler())
logger.setLevel(logging.INFO)


def get_retriever(
    docs_searcher_models: DocsSearcherModels, top_k: int = 5
) -> DocRetriever:
    """Documents retriever object."""
    return DocRetriever(
        top_k=top_k,
        docs_searcher_models=docs_searcher_models,
    )


def run_rag(
    user_prompt: str,
    llm: LLM,
    retrievers: list[DocRetriever],
    collection_names: list[str],
    do_reranking: bool = False,
) -> str:
    """
    :param user_prompt: only prompt that was received from user
    :param collection_names: retriever gets docs from the collection with name 'collection_name'
    :param do_reranking: if reranking is necessary
    :return: response from LLM
    """

    # Retrieve
    logger.info("Retrieving ----------- IN PROGRESS")
    context = (
        RetrievingPipeline()
        .set_retrievers(retrievers)
        .set_collection_names(collection_names)
        .get_retrieved_docs(user_prompt)
    )
    logger.info("Retrieving ----------- DONE")

    # Rerank
    if do_reranking:
        logger.info("Reranking ----------- IN PROGRESS")
        reranker = LLMReranker(llm, PROMPT_RANK)
        res = reranker.rerank_context(context, user_prompt)
        logger.info("Reranking ----------- DONE")
    else:
        logger.info("Reranking ----------- SKIPPED")
        res = context

    # Get response
    logger.info("Generation ----------- IN PROGRESS")
    paragraphs = "\n".join(
        [f"Параграф {i + 1}: {doc.page_content}" for i, doc in enumerate(res)]
    )
    llm_response = llm.invoke(
        PROMPT_LLM_RESPONSE.format(paragraphs=paragraphs, question=user_prompt)
    )
    logger.info("Generation ----------- DONE")

    return llm_response


def run_multiple_rag(
    user_prompt: str, llm: LLM, retriever_pipelines: list[RetrievingPipeline]
) -> str:
    """
    :param user_prompt: only prompt that was received from user
    :param retriever_pipelines: ready to use retriever pipelines (retrievers and collection_names should be specified)
    :param do_reranking: if reranking is necessary
    :return: response from LLM
    """
    reranker = LLMReranker(llm, PROMPT_RANK)

    # Retrieve
    logger.info("Retrieving ----------- IN PROGRESS")
    contexts = [
        pipeline.get_retrieved_docs(user_prompt) for pipeline in retriever_pipelines
    ]
    logger.info("Retrieving ----------- DONE")

    max_len_context = max([len(context) for context in contexts])
    for ctx in contexts:
        if len(ctx) < max_len_context:
            ctx.extend([Document(page_content="")] * (max_len_context - len(ctx)))

    # Merge the most relevant paragraphs
    logger.info("Merging ----------- IN PROGRESS")
    context = reranker.merge_docs(user_prompt, contexts)

    # Get response
    logger.info("Generation ----------- IN PROGRESS")
    paragraphs = "\n".join(
        [f"Параграф {i + 1}: {doc.page_content}" for i, doc in enumerate(context)]
    )
    llm_response = llm.invoke(
        PROMPT_LLM_RESPONSE.format(paragraphs=paragraphs, question=user_prompt)
    )
    logger.info("Generation ----------- DONE")

    return llm_response
