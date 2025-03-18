from dataclasses import dataclass

import logging

from protollm_sdk.jobs.outer_llm_api import OuterLLMAPI
from protollm_sdk.jobs.job_invoke import JobInvoker
from protollm_sdk.jobs.vector_db import VectorDB
from protollm_sdk.jobs.result_storage import ResultStorage
from protollm_sdk.jobs.text_embedder import TextEmbedder
from protollm_sdk.jobs.llm_api import LLMAPI

logger = logging.getLogger(__name__)


@dataclass
class JobContext:
    """
    The class contains contextual services for executing Job
    """

    llm_api: LLMAPI
    outer_llm_api: OuterLLMAPI
    text_embedder: TextEmbedder
    result_storage: ResultStorage
    vector_db: VectorDB
    job_invoker: JobInvoker
