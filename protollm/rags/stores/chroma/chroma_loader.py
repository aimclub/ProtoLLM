from typing import Optional
import logging

import chromadb
from langchain_community.embeddings.huggingface_hub import HuggingFaceHubEmbeddings
from langchain_community.vectorstores.chroma import Chroma

from protollm.rags.pipeline.etl_pipeline import DocsExtractPipeline
from protollm.rags.settings.pipeline_settings import PipelineSettings
from protollm.rags.settings.chroma_settings import (
    ChromaSettings,
    settings as default_settings,
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def load_documents_to_chroma_db(
    settings: Optional[ChromaSettings] = None,
    processing_batch_size: int = 100,
    loading_batch_size: int = 32,
    **kwargs,
) -> None:

    if settings is None:
        settings = default_settings

    logger.info(
        f"Initializing batch generator with processing_batch_size: {processing_batch_size},"
        f" loading_batch_size: {loading_batch_size}"
    )

    pipeline_settings = PipelineSettings.config_from_file(
        settings.docs_processing_config
    )

    store = Chroma(
        collection_name=settings.collection_name,
        embedding_function=HuggingFaceHubEmbeddings(
            model=settings.embedding_host,
            huggingfacehub_api_token="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",
        ),
        client=chromadb.HttpClient(
            host=settings.chroma_host, port=settings.chroma_port
        ),
    )

    # Documents loading and processing
    DocsExtractPipeline(pipeline_settings).go_to_next_step(
        docs_collection_path=settings.docs_collection_path
    ).update_docs_transformers(**kwargs).go_to_next_step(
        batch_size=processing_batch_size
    ).load(
        store, loading_batch_size=loading_batch_size
    )
