import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Config:
    llm_api_host = os.environ.get("LLM_API_HOST", "localhost")
    llm_api_port = os.environ.get("LLM_API_PORT", "6672")

    outer_llm_key = os.environ.get("OUTER_LLM_KEY", "sk-or-vv-c49f40fdb086053ec32c6ae2723b8d222cb7767f3b98527e7ae282986e7d33ed")

    redis_host = os.environ.get("REDIS_HOST", "localhost")
    redis_port = os.environ.get("REDIS_PORT", "6379")

    rabbit_mq_host = os.environ.get("RABBIT_HOST", "localhost")
    rabbit_mq_port = os.environ.get("RABBIT_PORT", "5672")
    rabbit_mq_login = os.environ.get("RABBIT_MQ_LOGIN", "admin")
    rabbit_mq_password = os.environ.get("RABBIT_MQ_PASSWORD", "admin")

    mongodb_port = int(os.environ.get("MONGO_DB_PORT", "27017"))
    mongodb_host = os.environ.get("MONGO_DB_HOST", "localhost")
    mongodb_user = os.environ.get("MONGODB_USER", "admin")
    mongodb_password = os.environ.get("MONGODB_PASS", "admin")
    web_mongodb_port = int(os.environ.get("WEB_MONGO_DB_PORT", "8081"))
    mongoexpress_user = os.environ.get("MONGOEXPRESS_USER", "admin")
    mongoexpress_password = os.environ.get("MONGOEXPRESS_PASS", "admin")

    text_embedder_host = os.environ.get("TEXT_EMB_HOST", "localhost")
    text_embedder_port = os.environ.get("TEXT_EMB_PORT", "9942")

    vector_bd_host = os.environ.get("VECTOR_HOST", "localhost")
    vector_db_port = os.environ.get("VECTOR_PORT", "9941")

    job_invocation_type = os.environ.get("JOB_INVOCATION_TYPE", "worker")

    celery_queue_name = os.environ.get("CELERY_QUEUE_NAME", "celery")

    @classmethod
    def reload_invocation_type(cls):
        cls.job_invocation_type = os.environ.get("JOB_INVOCATION_TYPE", "worker")
