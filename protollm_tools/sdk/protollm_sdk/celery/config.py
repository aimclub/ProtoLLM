from typing import Any

from protollm_sdk.config import Config
from protollm_sdk.utils.singleton import Singleton


__all__ = ["CeleryConfig"]

_ARG_UNSET = object()


class _Defaults:
    conf_update = dict(
        task_serializer="pickle",
        accept_content=["pickle"],
        result_serializer="pickle",
    )

    formats = ["json", "application/json"]


class CeleryConfig(metaclass=Singleton):
    def __init__(
        self,
        queue: str | None = None,
        mq_usr: str | None = None,
        mq_pwd: str | None = None,
        mq_host: str | None = None,
        mq_port: int | None | object = _ARG_UNSET,
        redis_host: str | None = None,
        redis_port: int | None | object = _ARG_UNSET,
        celery_init_kwargs: dict | None = None,
        conf_update: dict[str, Any] | None | object = _ARG_UNSET,
        formats: list[str] | None | object = _ARG_UNSET,
        task_kwargs: dict[str, Any] | None | object = _ARG_UNSET,
    ):
        self.queue = queue if queue is not None else Config.celery_queue_name

        self.mq_usr = mq_usr if mq_usr is not None else Config.rabbit_mq_login
        self.mq_pwd = mq_pwd if mq_pwd is not None else Config.rabbit_mq_password
        self.mq_host = mq_host if mq_host is not None else Config.rabbit_mq_host
        self.mq_port = mq_port if mq_port is not _ARG_UNSET else Config.rabbit_mq_port

        self.redis_host = redis_host if redis_host is not None else Config.redis_host
        self.redis_port = (
            redis_port if redis_port is not _ARG_UNSET else Config.redis_port
        )

        self.celery_init_kwargs = celery_init_kwargs

        self.conf_update = (
            conf_update if conf_update is not _ARG_UNSET else _Defaults.conf_update
        )
        self.formats = formats if formats is not _ARG_UNSET else _Defaults.formats

        self.task_kwargs = (
            (task_kwargs or {})
            if task_kwargs is not _ARG_UNSET
            else dict(
                queue=self.queue,
            )
        )

    @property
    def init_args(self) -> tuple[tuple[str], dict[str, str]]:
        static_kwargs = dict(broker=self.mq_url, backend=self.redis_url)

        return (
            (self.queue,),
            (
                static_kwargs
                if self.celery_init_kwargs is None
                else dict(static_kwargs, **self.celery_init_kwargs)
            ),
        )

    @property
    def mq_url(self) -> str:
        schema = "amqp"
        creds = f"{self.mq_usr}:{self.mq_pwd}"
        base = (
            f"{self.mq_host}:{self.mq_port}"
            if self.mq_port is not None
            else self.mq_host
        )

        return f"{schema}://{creds}@{base}/"

    @property
    def redis_url(self) -> str:
        schema = "redis"
        base = (
            f"{self.redis_host}:{self.redis_port}"
            if self.redis_port is not None
            else self.redis_host
        )

        return f"{schema}://{base}"
