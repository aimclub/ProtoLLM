import logging

from celery import Celery
from kombu.serialization import registry

from protollm_sdk.celery.config import CeleryConfig
from protollm_sdk.celery.constants import JOB_NAME2CLASS
from protollm_sdk.jobs.job import Job
from protollm_sdk.jobs.utility import construct_job_context


def init_celery(celery_config: CeleryConfig) -> Celery:
    init_args, init_kwargs = celery_config.init_args

    celery = Celery(*init_args, **init_kwargs)

    if celery_config.conf_update:
        celery.conf.update(**celery_config.conf_update)

    if celery_config.formats:
        for f in celery_config.formats:
            registry.enable(f)

    return celery


celery_config = CeleryConfig()
celery = init_celery(celery_config)


logger = logging.getLogger(__name__)


@celery.task(**celery_config.task_kwargs)
def task_test(task_class: str, task_id: str, **kwargs):  # noqa
    ctx = construct_job_context(task_class, abstract_task)
    if job_class := JOB_NAME2CLASS.get(task_class):
        job = job_class()
    else:
        msg = f"Error in task '{task_id}'. Unknown job class: '{task_class}'."
        logger.error(msg)
        raise Exception(msg)

    logger.info(f"Starting task '{task_id}'. Job '{task_class}'.")

    forecast = job.run(task_id=task_id, ctx=ctx, **kwargs)
    return forecast


@celery.task(**celery_config.task_kwargs)
def abstract_task(task_class: type[Job], task_id: str, **kwargs):
    class_name = (
        task_class.__name__
        if isinstance(task_class, type)
        else task_class.__class__.__name__
    )
    ctx = construct_job_context(class_name, abstract_task)
    logger.info(f"Starting task '{task_id}'. Job '{class_name}'.")
    job_object = task_class() if isinstance(task_class, type) else task_class
    job_object.run(task_id=task_id, ctx=ctx, **kwargs)
