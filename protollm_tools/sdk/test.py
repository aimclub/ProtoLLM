import asyncio
import logging

import uuid
from time import sleep

from protollm_sdk.object_interface.result_storage import RedisResultStorage, JobStatusType


async def correct_job_lifecycle(job_id: str, rrs: RedisResultStorage):
    """
    This function simulates a work lifecycle by sleeping for 1 second.
    """
    rrs.create_job_status(job_id)
    job = rrs.get_job_status(job_id)
    print(f"Job {job_id} status: {job.status}")
    sleep(1)
    rrs.update_job_status(job_id, JobStatusType.IN_PROGRESS)
    job = rrs.get_job_status(job_id)
    print(f"Job {job_id} status: {job.status}")
    sleep(1)
    rrs.complete_job(job_id, result="Success")
    job = rrs.get_job_status(job_id)
    print(f"Job {job_id} status: {job.status}")

async def job_finish_subscriber(job_id: str, rrs: RedisResultStorage):
    """
    This function subscribes to job status updates.
    """
    sleep(5)
    job = await rrs.wait_for_completion(job_id)
    print(f"Job {job_id} completed with result: {job.result}")

async def subscriber(job_id: str, rrs: RedisResultStorage):
    ans = ""

    for job_status in rrs.subscribe(job_id):
        ans += f"{job_status.value}\n"

    print(f"Subscribed to {job_id} with result: {ans}")





if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    job_id = str(uuid.uuid4())
    rrs = RedisResultStorage(redis_host="localhost", redis_port=6380)
    # add two tasks asynchronously
    loop = asyncio.get_event_loop()
    loop.run_until_complete(asyncio.gather(
        correct_job_lifecycle(job_id, rrs),
        job_finish_subscriber(job_id, rrs)
    ))