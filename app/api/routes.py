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


def get_redis(request: Request) -> aioredis.Redis:
    pool: aioredis.ConnectionPool = request.app.state.redis_pool
    return aioredis.Redis(connection_pool=pool)


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Health and readiness probe",
    tags=["Operations"],
)
async def health_check(redis: aioredis.Redis = Depends(get_redis)):
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


@router.get(
    "/features/{user_id}",
    response_model=FeatureVector,
    responses={
        200: {"description": "Feature vector returned successfully."},
        404: {"model": ErrorResponse, "description": "User not found in the feature store."},
        503: {"description": "Internal server error - Redis unavailable."},
    },
    summary="Retrieve features for a single user",
    tags=["Features"],
)
async def get_user_features(
    user_id: str,
    redis: aioredis.Redis = Depends(get_redis),
):
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

    logger.debug("GET /features/%s -> %d features returned", user_id, len(features))
    return FeatureVector(user_id=user_id, features=features)


@router.post(
    "/features/batch",
    response_model=BatchFeatureResponse,
    responses={
        200: {"description": "Batch feature vectors returned. Missing users have empty features."},
        422: {"description": "Request payload failed schema validation."},
        503: {"description": "Internal server error - Redis unavailable."},
    },
    summary="Retrieve features for multiple users in one request",
    tags=["Features"],
)
async def get_batch_features(
    payload: BatchFeatureRequest,
    redis: aioredis.Redis = Depends(get_redis),
):
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
            feature_vectors.append(FeatureVector(user_id=uid, features={}))

    logger.debug(
        "POST /features/batch -> %d/%d users resolved",
        found_count,
        len(user_ids),
    )

    return BatchFeatureResponse(
        results=feature_vectors,
        total_requested=len(user_ids),
        total_found=found_count,
    )
