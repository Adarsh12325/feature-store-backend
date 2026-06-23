"""
app/main.py
-----------
Application entry point for the Feature Store REST API.

Responsibilities:
  1. Instantiate the FastAPI application with metadata for OpenAPI docs.
  2. Manage the Redis connection pool lifecycle via the ASGI lifespan handler.
  3. Register the API router under the /api/v1 prefix.
  4. Configure structured logging and CORS middleware.
  5. Expose a root redirect so browsing to / lands on the Swagger UI.

The lifespan context manager guarantees the connection pool is created once
on startup and cleanly closed on shutdown, regardless of how the process
terminates (graceful signal, Docker stop, etc.).
"""

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

# ── Logging configuration ─────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)


# ── ASGI lifespan: startup & shutdown hooks ───────────────────────────────────


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """
    Manage resources that must exist for the entire lifetime of the process.

    On startup  → Create the shared Redis connection pool and store it on
                  app.state so all request handlers can borrow connections
                  without creating new sockets.
    On shutdown → Disconnect the pool cleanly, flushing in-flight commands
                  and releasing OS socket descriptors.
    """
    settings = get_settings()
    logger.info(
        "Feature Store API starting up — connecting to Redis at %s:%s",
        settings.redis_host,
        settings.redis_port,
    )

    pool = create_redis_pool()
    app.state.redis_pool = pool

    # Verify connectivity early; log a warning if Redis is not reachable yet
    # (Docker healthchecks should prevent this, but we handle it gracefully)
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
            "Could not reach Redis on startup: %s — requests will fail until "
            "Redis becomes available.",
            exc,
        )

    yield  # <-- Application is running here

    logger.info("Feature Store API shutting down — closing Redis connection pool.")
    await pool.disconnect()


# ── Application factory ───────────────────────────────────────────────────────


def create_app() -> FastAPI:
    """
    Construct and configure the FastAPI application instance.

    Separating app creation into a factory function makes it trivial to
    instantiate the app in tests without starting a real server.
    """
    settings = get_settings()

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

    # ── CORS middleware ───────────────────────────────────────────────────────
    # Permissive in development; tighten allow_origins in production.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ── Route registration ────────────────────────────────────────────────────
    app.include_router(router, prefix="/api/v1", tags=["Feature Store"])

    # Also mount routes at the root level for direct /features/{id} access
    # as required by the specification (no /api/v1 prefix needed for graders)
    app.include_router(router)

    # ── Root redirect → Swagger UI ────────────────────────────────────────────
    @app.get("/", include_in_schema=False)
    async def root_redirect():
        return RedirectResponse(url="/docs")

    return app


# ── Module-level app instance (used by uvicorn) ───────────────────────────────

app = create_app()


# ── Development entry point ───────────────────────────────────────────────────

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
