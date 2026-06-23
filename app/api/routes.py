"""
app/api/routes.py
-----------------
FastAPI route definitions for the Feature Store serving API.

All route handlers delegate data retrieval to the Redis data access layer
(app.services.redis_client). Route handlers are intentionally thin — they
validate input, invoke the DAL, and shape the response. No Redis commands
appear here directly.

Endpoints
---------
  GET  /health              — Liveness and readiness probe
  GET  /features/{user_id} — Single-user feature vector retrieval
  POST /features/batch      — Multi-user feature vector retrieval (pipelined)
"""

from __future__ import annotations

import logging
from typing import List

import redis.asyncio as aioredis
from fastapi import APIRouter, Depends, HTTPException, Request, status

from app.config import get_settings
from app.models.schemas import (
    BatchFeatureRequest,
    BatchFeatureResponse,
    ErrorResponse,
    FeatureVector,
    HealthResponse,
)
from app.services import redis_client as dal

logger = logging.getLogger(__name__)

router = APIRouter()


# ── Dependency: Redis client from app state ───────────────────────────────────


def get_redis(request: Request) -> aioredis.Redis:
    """
    FastAPI dependency that returns a Redis client bound to the shared pool.

    The connection pool is stored on app.state during the lifespan startup
    hook in main.py. Retrieving a client this way is instantaneous — it
    borrows a connection from the pool rather than opening a new socket.
    """
    pool: aioredis.ConnectionPool = request.app.state.redis_pool
    return aioredis.Redis(connection_pool=pool)


# ── Health endpoint ───────────────────────────────────────────────────────────


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Health and readiness probe",
    tags=["Operations"],
)
async def health_check(redis: aioredis.Redis = Depends(get_redis)):
    """
    Returns the operational status of the API and its Redis connection.

    Intended for use by Docker health checks, Kubernetes liveness probes,
    and load balancers. A response of 200 with redis_connected=True indicates
    the service is fully operational.
    """
    settings = get_settings()
    connected = await dal.ping_redis(redis)
    user_count: int | None = None

    if connected:
        user_count = await dal.get_total_user_count(redis)

    return HealthResponse(
        status="ok" if connected else "degraded",
        redis_connected=connected,
        redis_host=settings.redis_host,
        redis_port=settings.redis_port,
        total_users_indexed=user_count,
    )


# ── Single-user feature retrieval ─────────────────────────────────────────────


@router.get(
    "/features/{user_id}",
    response_model=FeatureVector,
    responses={
        200: {"description": "Feature vector returned successfully."},
        404: {"model": ErrorResponse, "description": "User not found in the feature store."},
        500: {"description": "Internal server error — Redis unavailable."},
    },
    summary="Retrieve features for a single user",
    tags=["Features"],
)
async def get_user_features(
    user_id: str,
    redis: aioredis.Redis = Depends(get_redis),
):
    """
    Returns the complete ML feature vector for the specified user.

    The lookup strategy is a two-step Redis operation:
      1. **SISMEMBER** `all_users` → O(1) existence check against the global
         user index Set. Returns 404 immediately if the user is absent.
      2. **HGETALL** `user:{user_id}:features` → Atomic retrieval of all
         hash fields in a single round-trip.

    Redis string values are coerced back to their native Python types
    (int, float, bool) before the response is serialised to JSON.

    **Path Parameters**
    - `user_id`: The string identifier of the target user (e.g., `user_001`).

    **Responses**
    - `200 OK`: User exists; full feature vector returned.
    - `404 Not Found`: User ID absent from the Redis feature store.
    """
    if not user_id or not user_id.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="user_id path parameter must be a non-empty string.",
        )

    try:
        features = await dal.get_user_features(redis, user_id)
    except Exception as exc:
        logger.exception("Redis error while fetching features for '%s': %s", user_id, exc)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Feature store is temporarily unavailable. Please retry.",
        )

    if features is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"User '{user_id}' was not found in the feature store.",
        )

    logger.debug("GET /features/%s → %d features returned", user_id, len(features))
    return FeatureVector(user_id=user_id, features=features)


# ── Batch feature retrieval ───────────────────────────────────────────────────


@router.post(
    "/features/batch",
    response_model=BatchFeatureResponse,
    responses={
        200: {"description": "Batch feature vectors returned. Missing users have empty features."},
        422: {"description": "Request payload failed schema validation."},
        503: {"description": "Internal server error — Redis unavailable."},
    },
    summary="Retrieve features for multiple users in one request",
    tags=["Features"],
)
async def get_batch_features(
    payload: BatchFeatureRequest,
    redis: aioredis.Redis = Depends(get_redis),
):
    """
    Retrieves feature vectors for up to 100 users in a single pipelined
    Redis operation.

    The batch endpoint is optimised for the hot path in real-time ML
    inference: a recommendation engine might need features for dozens of
    candidate users simultaneously. Issuing one pipelined request instead
    of N sequential requests reduces the cumulative latency by an order
    of magnitude.

    **Partial failures are handled gracefully.** If a user_id in the batch
    is not found, the corresponding entry in the response has an empty
    `features` dict (`{}`). The entire batch never fails because of a
    missing subset.

    **Request Body**
    ```json
    { "user_ids": ["user_001", "user_042", "unknown_xyz"] }
    ```

    **Response**
    ```json
    {
      "results": [
        {"user_id": "user_001", "features": {"age": 34, ...}},
        {"user_id": "user_042", "features": {"age": 27, ...}},
        {"user_id": "unknown_xyz", "features": {}}
      ],
      "total_requested": 3,
      "total_found": 2
    }
    ```
    """
    user_ids: List[str] = payload.user_ids

    try:
        raw_results = await dal.get_users_features_batch(redis, user_ids)
    except Exception as exc:
        logger.exception("Redis error during batch retrieval: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Feature store is temporarily unavailable. Please retry.",
        )

    feature_vectors: List[FeatureVector] = []
    found_count = 0

    for uid, features in raw_results:
        if features is not None:
            feature_vectors.append(FeatureVector(user_id=uid, features=features))
            found_count += 1
        else:
            # User not found — return empty features dict, not an error
            feature_vectors.append(FeatureVector(user_id=uid, features={}))

    logger.debug(
        "POST /features/batch → %d/%d users resolved",
        found_count,
        len(user_ids),
    )

    return BatchFeatureResponse(
        results=feature_vectors,
        total_requested=len(user_ids),
        total_found=found_count,
    )
