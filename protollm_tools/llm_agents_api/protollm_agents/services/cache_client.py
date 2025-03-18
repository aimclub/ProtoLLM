from redis.asyncio import Redis

cache: Redis | None = None


async def get_cache():
    return cache
