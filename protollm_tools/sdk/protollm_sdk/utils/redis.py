from protollm_sdk.config import Config
from protollm_sdk.jobs.result_storage import ResultStorage
from protollm_sdk.object_interface import RedisWrapper


def get_reddis_wrapper():
    """
    Create RedisWrapper object for working with Redis

    :return: RedisWrapper object
    :rtype: RedisWrapper
    """
    return RedisWrapper(
        redis_host=Config.redis_host,
        redis_port=Config.redis_port
    )


def load_result(rd: RedisWrapper, job_id: str, prefix: str or None) -> bytes:
    """
    Load result from Redis by job_id and prefix (job_name).
    The code returns bytes. If you want to convert these bytes into a ResponseModel,
    first decode the bytes into a string using `.decode()`, then create the model
    like this:
    model_text = byte_data.decode()
    response_model = ResponseModel.model_validate_json(model_text)

    :param rd: RedisWrapper object
    :type rd: RedisWrapper
    :param job_id: uuid of the job
    :type job_id: str
    :param prefix: prefix of type of job or jon_name
    :type prefix: str or None
    :return: result
    :rtype: bytes
    """
    resp = rd.get_item(ResultStorage.build_key(job_id, prefix))
    return resp
