"""
app/services/redis_client.py
----------------------------
Data Access Layer (DAL) for all Redis interactions in the Feature Store.

This module encapsulates every Redis command behind a clean Python interface.
Route handlers never import redis directly; they call functions defined here.
This separation makes unit testing straightforward — tests mock this module
rather than patching the redis library itself.

Data Model
----------
  Feature Storage:
    Key   → user:{user_id}:features  (Redis Hash)
    Cmd   → HSET  (write)  /  HGETALL  (read)

  User Index:
    Key   → all_users  (Redis Set)
    Cmd   → SADD  (write)  /  SISMEMBER  (existence check)

Using Hashes instead of serialized JSON strings allows:
  • Partial field updates without touching unmodified features.
  • Direct field-level access via HGET without deserializing the full vector.
  • Smaller memory footprint due to Redis's compact hash encoding for small
    hashes (ziplist / listpack under the default hash-max-listpack-entries).
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import redis.asyncio as aioredis
from redis.asyncio import ConnectionPool

from app.config import get_settings

logger = logging.getLogger(__name__)

# ── Type coercion helpers ─────────────────────────────────────────────────────

_TRUE_VALUES = {"true", "yes"}
_FALSE_VALUES = {"false", "no"}


def _coerce_value(raw: str) -> Any:
    """
    Attempt to cast a Redis string value into its most appropriate Python type.

    Redis stores everything as byte strings. When the ingestion layer writes
    an integer like 34 or a float like 0.87, it is round-tripped as the
    string "34" or "0.87". This function restores the original semantics so
    the JSON response contains native integers and floats rather than strings.

    Priority order: int → float → bool → str (fallback)
    """
    lowered = raw.strip().lower()

    # Boolean check before int/float to catch "True"/"False" strings
    if lowered in _TRUE_VALUES:
        return True
    if lowered in _FALSE_VALUES:
        return False

    # Integer
    try:
        return int(raw)
    except ValueError:
        pass

    # Float
    try:
        return float(raw)
    except ValueError:
        pass

    # Fall back to the original string
    return raw


def _coerce_feature_dict(raw_dict: Dict[bytes, bytes]) -> Dict[str, Any]:
    """
    Convert a raw HGETALL result (bytes → bytes) into a typed Python dict.

    Redis returns bytes keys and bytes values when decode_responses=False.
    We decode keys to str and coerce values to their native Python types.
    When decode_responses=True is used on the connection pool, both arrive
    as plain strings — this function handles both cases safely.
    """
    result: Dict[str, Any] = {}
    for k, v in raw_dict.items():
        key_str = k.decode("utf-8") if isinstance(k, bytes) else k
        val_str = v.decode("utf-8") if isinstance(v, bytes) else v
        result[key_str] = _coerce_value(val_str)
    return result


# ── Connection pool factory ───────────────────────────────────────────────────


def create_redis_pool() -> ConnectionPool:
    """
    Instantiate a shared async connection pool on application startup.

    A connection pool reuses persistent TCP connections to Redis, eliminating
    the overhead of establishing a new socket per request. FastAPI's lifespan
    hook calls this once and stores the pool on app.state for sharing across
    all request handlers.
    """
    settings = get_settings()
    logger.info(
        "Creating Redis connection pool -> %s:%s",
        settings.redis_host,
        settings.redis_port,
    )
    pool = aioredis.ConnectionPool.from_url(
        settings.redis_url,
        max_connections=50,
        decode_responses=True,   # Redis returns str instead of bytes
        socket_connect_timeout=5,
        socket_timeout=5,
        retry_on_timeout=True,
    )
    return pool


# ── Data access functions ─────────────────────────────────────────────────────


async def ping_redis(client: aioredis.Redis) -> bool:
    """Return True if Redis responds to PING, False otherwise."""
    try:
        return await client.ping()
    except Exception as exc:
        logger.error("Redis PING failed: %s", exc)
        return False


async def get_total_user_count(client: aioredis.Redis) -> int:
    """Return the cardinality of the all_users Set (total indexed users)."""
    settings = get_settings()
    try:
        return await client.scard(settings.all_users_set_key)
    except Exception as exc:
        logger.error("SCARD failed: %s", exc)
        return -1


async def user_exists(client: aioredis.Redis, user_id: str) -> bool:
    """
    Check whether a user is registered in the global index Set.

    Uses SISMEMBER which runs in O(1) regardless of the number of users
    in the set — a critical property for a high-throughput serving layer.
    """
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
    """
    Retrieve the complete feature vector for a single user.

    Workflow:
      1. SISMEMBER all_users <user_id>  →  O(1) existence check.
      2. Return None immediately if the user is absent (avoids unnecessary HGETALL).
      3. HGETALL user:{user_id}:features  →  fetch all hash fields atomically.
      4. Coerce raw Redis strings to typed Python values.

    Returns:
        A dict of feature name → typed value, or None if the user is not found.
    """
    settings = get_settings()

    exists = await user_exists(client, user_id)
    if not exists:
        return None

    feature_key = settings.user_features_key(user_id)
    raw: Dict[str, str] = await client.hgetall(feature_key)

    if not raw:
        # Edge case: user is in the index set but has no hash entries yet.
        # Treat as found but empty rather than not-found.
        return {}

    # Coerce string values from Redis into typed Python objects
    return {k: _coerce_value(v) for k, v in raw.items()}


async def get_users_features_batch(
    client: aioredis.Redis, user_ids: List[str]
) -> List[Tuple[str, Optional[Dict[str, Any]]]]:
    """
    Retrieve feature vectors for multiple users in a single pipeline round-trip.

    This function uses Redis pipelining to batch all SISMEMBER and HGETALL
    commands into one TCP write, dramatically reducing aggregate latency
    compared to issuing sequential individual calls.

    Strategy (two-phase pipeline):
      Phase 1 — Existence checks:
        Execute SISMEMBER for every requested user_id in a single pipeline.
        Collect the boolean results.

      Phase 2 — Feature retrieval:
        For users that exist, execute HGETALL in a second pipeline.
        Users that do not exist receive an empty dict without hitting Redis again.

    This two-phase approach is slightly more complex than a single pipeline
    but avoids fetching empty HGETALL results for non-existent users, which
    would waste bandwidth and add unnecessary processing at the application layer.

    Returns:
        A list of (user_id, features_dict_or_None) tuples in the same order
        as the input user_ids list.
    """
    settings = get_settings()

    if not user_ids:
        return []

    # ── Phase 1: batch existence checks ──────────────────────────────────────
    async with client.pipeline(transaction=False) as pipe:
        for uid in user_ids:
            pipe.sismember(settings.all_users_set_key, uid)
        existence_results: List[bool] = await pipe.execute()

    # ── Phase 2: batch feature retrieval for existing users ──────────────────
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

    # ── Assemble final ordered result list ────────────────────────────────────
    results: List[Tuple[str, Optional[Dict[str, Any]]]] = []
    for uid, exists in zip(user_ids, existence_results):
        if exists:
            results.append((uid, feature_map.get(uid, {})))
        else:
            results.append((uid, None))

    return results
