from __future__ import annotations

import logging
import sys
from contextlib import asynccontextmanager
from typing import AsyncIterator

import redis.asyncio as aioredis
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse

from app.api.routes import router
from app.config import get_settings
from app.services.redis_client import create_redis_pool

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()
    logger.info(
        "Feature Store API starting up - connecting to Redis at %s:%s",
        settings.redis_host,
        settings.redis_port,
    )

    pool = create_redis_pool()
    app.state.redis_pool = pool

    try:
        client = aioredis.Redis(connection_pool=pool)
        pong = await client.ping()
        if pong:
            user_count = await client.scard(settings.all_users_set_key)
            logger.info(
                "Redis connection established. Users currently indexed: %d",
                user_count,
            )
        await client.aclose()
    except Exception as exc:
        logger.warning(
            "Could not reach Redis on startup: %s - requests will fail until Redis becomes available.",
            exc,
        )

    yield

    logger.info("Feature Store API shutting down - closing Redis connection pool.")
    await pool.disconnect()


def create_app() -> FastAPI:
    app = FastAPI(
        title="Feature Store API",
        description=(
            "Real-time ML Feature Store backend powered by FastAPI and Redis. "
            "Provides sub-millisecond feature vector retrieval for online "
            "model inference with support for single and batch lookups."
        ),
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(router, prefix="/api/v1", tags=["Feature Store"])
    app.include_router(router)

    @app.get("/", include_in_schema=False)
    async def root_redirect():
        return RedirectResponse(url="/docs")

    return app


app = create_app()

if __name__ == "__main__":
    import uvicorn

    settings = get_settings()
    uvicorn.run(
        "app.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=False,
        log_level="info",
    )
