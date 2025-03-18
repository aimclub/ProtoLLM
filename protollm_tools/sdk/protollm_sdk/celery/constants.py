from protollm_sdk.celery.job import (
    TextEmbedderJob,
    LLMAPIJob,
    ResultStorageJob,
    VectorDBJob,
    OuterLLMAPIJob,
)

JOBS = {TextEmbedderJob, LLMAPIJob, ResultStorageJob, VectorDBJob, OuterLLMAPIJob}

JOB_NAME2CLASS = {cls.__name__: cls for cls in JOBS}
