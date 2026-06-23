from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import redis.asyncio as aioredis
from redis.asyncio import ConnectionPool

from app.config import get_settings

logger = logging.getLogger(__name__)

_TRUE_VALUES = {"true", "yes"}
_FALSE_VALUES = {"false", "no"}


def _coerce_value(raw: str) -> Any:
    lowered = raw.strip().lower()

    if lowered in _TRUE_VALUES:
        return True
    if lowered in _FALSE_VALUES:
        return False

    try:
        return int(raw)
    except ValueError:
        pass

    try:
        return float(raw)
    except ValueError:
        pass

    return raw


def _coerce_feature_dict(raw_dict: Dict[bytes, bytes]) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    for k, v in raw_dict.items():
        key_str = k.decode("utf-8") if isinstance(k, bytes) else k
        val_str = v.decode("utf-8") if isinstance(v, bytes) else v
        result[key_str] = _coerce_value(val_str)
    return result


def create_redis_pool() -> ConnectionPool:
    settings = get_settings()
    logger.info(
        "Creating Redis connection pool -> %s:%s",
        settings.redis_host,
        settings.redis_port,
    )
    pool = aioredis.ConnectionPool.from_url(
        settings.redis_url,
        max_connections=50,
        decode_responses=True,
        socket_connect_timeout=5,
        socket_timeout=5,
        retry_on_timeout=True,
    )
    return pool


async def ping_redis(client: aioredis.Redis) -> bool:
    try:
        return await client.ping()
    except Exception as exc:
        logger.error("Redis PING failed: %s", exc)
        return False


async def get_total_user_count(client: aioredis.Redis) -> int:
    settings = get_settings()
    try:
        return await client.scard(settings.all_users_set_key)
    except Exception as exc:
        logger.error("SCARD failed: %s", exc)
        return -1


async def user_exists(client: aioredis.Redis, user_id: str) -> bool:
    settings = get_settings()
    try:
        result = await client.sismember(settings.all_users_set_key, user_id)
        return bool(result)
    except Exception as exc:
        logger.error("SISMEMBER error for user '%s': %s", user_id, exc)
        raise


async def get_user_features(
    client: aioredis.Redis, user_id: str
) -> Optional[Dict[str, Any]]:
    settings = get_settings()

    exists = await user_exists(client, user_id)
    if not exists:
        return None

    feature_key = settings.user_features_key(user_id)
    raw: Dict[str, str] = await client.hgetall(feature_key)

    if not raw:
        return {}

    return {k: _coerce_value(v) for k, v in raw.items()}


async def get_users_features_batch(
    client: aioredis.Redis, user_ids: List[str]
) -> List[Tuple[str, Optional[Dict[str, Any]]]]:
    settings = get_settings()

    if not user_ids:
        return []

    async with client.pipeline(transaction=False) as pipe:
        for uid in user_ids:
            pipe.sismember(settings.all_users_set_key, uid)
        existence_results: List[bool] = await pipe.execute()

    existing_ids = [
        uid for uid, exists in zip(user_ids, existence_results) if exists
    ]

    feature_map: Dict[str, Dict[str, Any]] = {}

    if existing_ids:
        async with client.pipeline(transaction=False) as pipe:
            for uid in existing_ids:
                pipe.hgetall(settings.user_features_key(uid))
            raw_results: List[Dict[str, str]] = await pipe.execute()

        for uid, raw in zip(existing_ids, raw_results):
            feature_map[uid] = {k: _coerce_value(v) for k, v in raw.items()}

    results: List[Tuple[str, Optional[Dict[str, Any]]]] = []
    for uid, exists in zip(user_ids, existence_results):
        if exists:
            results.append((uid, feature_map.get(uid, {})))
        else:
            results.append((uid, None))

    return results
