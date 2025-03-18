import json
import logging
from contextlib import contextmanager

import redis

logger = logging.getLogger(__name__)


class ResultStorage:
    """
    Class provides an object interface to a temporary (cache) fast (in memory)
    key-value storage of results - Redis.
    It writes the results to the storage by key, which consists of the name of the agent job_name and
    the identifier of the current task (job_id)
    """

    def __init__(
        self,
        redis_host: str,
        redis_port: str | int | None = None,
        prefix: str | None = None,
    ):
        """
        Initialize ResultStorage

        :param redis_host: host of the redis
        :type redis_host: str
        :param redis_port: port of the redis
        :type redis_port: str | int | None
        :param prefix: prefix for the key
        :type prefix: str | None
        """
        self.url = (
            f"redis://{redis_host}:{redis_port}"
            if redis_port is not None
            else f"redis://{redis_host}"
        )
        self.prefix = prefix or ""

    def for_job(self, job_name: str) -> "ResultStorage":
        """
        Clones the connection to redis, but for another job
        (useful if you call one job from another, and the key prefix in the storage should be different)

        :param job_name: job name
        :type job_name: str
        :return: ResultStorage
        """
        return ResultStorage(redis_host=self.url, redis_port=None, prefix=job_name)

    @contextmanager
    def _get_redis(self):
        """
        Get redis connection
        """
        rd = redis.Redis.from_url(self.url)
        try:
            yield rd
        finally:
            rd.close()

    def _get_key(self, job_id: str) -> str:
        """
        Format the string to get the key for the agent
        for which the storage instance was created, and for the current job_id

        :param job_id: job id
        :type job_id: str
        :return: Formatted key
        """
        return self.build_key(job_id, self.prefix)

    def save_dict(self, job_id: str, result: dict):
        """
        Save the dictionary to the storage by key

        :param job_id: job id
        :type job_id: str
        :param result: true dictionary, not a dictionary serialized to a byte string
        :type result: dict
        :return: None
        """
        key = self._get_key(job_id)
        try:
            with self._get_redis() as rs:
                rs.set(key, json.dumps(result))
                logger.info(
                    f"The result with the key {key} has been successfully written to the redis"
                )
        except Exception as ex:
            msg = f"Saving the result with the key {key} has been interrupted. Error: {ex}."
            logger.info(msg)
            raise Exception(msg) from ex

    def load_dict(self, job_id: str) -> dict:
        """
        Load the dictionary from the storage by key

        :param job_id: job id
        :type job_id: str
        :return: true dictionary, not a dictionary serialized to a byte string
        """
        key = self._get_key(job_id)
        try:
            with self._get_redis() as rs:
                result = rs.get(key)
                logger.info(f"Data by key {key} has been successfully read from redis.")
                if isinstance(result, (bytes, str)):
                    return json.loads(result)
                return result
        except Exception as ex:
            msg = (
                f"Failed to load the result with the key {key} from redis. Error: {ex}."
            )
            logger.error(msg)
            raise Exception(msg) from ex

    @staticmethod
    def build_key(job_id: str, prefix: str or None) -> str:
        """
        Format any suffix and prefix according to the key rules

        :param job_id: job id
        :type job_id: str
        :param prefix: prefix
        :type prefix: str or None
        :return: str
        """
        return job_id if prefix is None else f"{prefix}:{job_id}"
