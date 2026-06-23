"""
tests/conftest.py
-----------------
Shared pytest fixtures for the Feature Store test suite.

Fixtures defined here are automatically available to every test module
without requiring explicit imports. This file configures:

  1. A shared FakeServer so that sync writes (fixture setup) and async
     reads (request handler) operate on the same in-memory dataset.

  2. A FastAPI TestClient where:
       - create_redis_pool is patched to return a fake pool (avoids TCP)
       - pool.disconnect is patched to a no-op coroutine (avoids teardown error)
       - The get_redis dependency is overridden to return a FakeRedis client
     This lets the full ASGI stack (middleware, routing, Pydantic) run in
     tests without any live Redis container.

  3. Pre-populated test data fixtures providing a deterministic baseline
     for assertion-based tests.

Design note: We use fakeredis rather than unittest.mock.patch because
fakeredis accurately implements Redis data structure semantics (HSET,
HGETALL, SADD, SISMEMBER). This catches bugs that pure mocks miss, such
as mismatched key types or incorrect pipeline sequencing.
"""

from __future__ import annotations

import asyncio
from typing import Generator
from unittest.mock import AsyncMock, MagicMock, patch

import fakeredis
import fakeredis.aioredis
import pytest
from fastapi.testclient import TestClient

from app.main import create_app
from app.config import get_settings


# ── Event loop ────────────────────────────────────────────────────────────────


@pytest.fixture(scope="session")
def event_loop():
    """Provide a single event loop for the entire test session."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


# ── Shared FakeServer ─────────────────────────────────────────────────────────

# A module-level FakeServer so all fixtures share the same in-memory store.
# Sync fixture setup writes are visible to the async route handlers.
_FAKE_SERVER = fakeredis.FakeServer()


@pytest.fixture(autouse=False)
def fake_redis_sync() -> fakeredis.FakeRedis:
    """
    A synchronous FakeRedis client used to pre-populate test data.
    Shares the module-level FakeServer with the async client used by routes.
    """
    r = fakeredis.FakeRedis(server=_FAKE_SERVER, decode_responses=True)
    yield r
    r.flushall()
    r.close()


# ── Test data constants ───────────────────────────────────────────────────────

KNOWN_USER_ID = "user_000001"
KNOWN_FEATURES = {
    "age": "34",
    "account_tier": "premium",
    "total_purchases": "217",
    "avg_session_duration_sec": "312.5",
    "is_active": "True",
    "preferred_category": "electronics",
    "churn_risk_score": "0.12",
    "lifetime_value_usd": "4820.75",
    "days_since_last_login": "2",
    "last_login_timestamp": "2026-06-23T04:00:00+00:00",
    "region": "us-east",
    "device_type": "mobile",
    "has_verified_email": "True",
    "subscribed_to_newsletter": "False",
    "page_views_30d": "145",
    "click_through_rate": "0.0823",
}

SECOND_USER_ID = "user_000002"
SECOND_FEATURES = {
    "age": "27",
    "account_tier": "basic",
    "total_purchases": "42",
    "avg_session_duration_sec": "180.0",
    "is_active": "False",
    "preferred_category": "books",
    "churn_risk_score": "0.67",
    "lifetime_value_usd": "820.5",
    "days_since_last_login": "45",
    "last_login_timestamp": "2026-05-09T12:30:00+00:00",
    "region": "eu-central",
    "device_type": "desktop",
    "has_verified_email": "True",
    "subscribed_to_newsletter": "True",
    "page_views_30d": "32",
    "click_through_rate": "0.0341",
}


@pytest.fixture
def populated_fake_redis(fake_redis_sync) -> fakeredis.FakeRedis:
    """
    A FakeRedis instance pre-loaded with two known users and their features.

    Tests that need predictable data should use this fixture rather than
    building the Redis state themselves.
    """
    settings = get_settings()

    fake_redis_sync.hset(
        settings.user_features_key(KNOWN_USER_ID), mapping=KNOWN_FEATURES
    )
    fake_redis_sync.sadd(settings.all_users_set_key, KNOWN_USER_ID)

    fake_redis_sync.hset(
        settings.user_features_key(SECOND_USER_ID), mapping=SECOND_FEATURES
    )
    fake_redis_sync.sadd(settings.all_users_set_key, SECOND_USER_ID)

    return fake_redis_sync


# ── FastAPI test client fixtures ──────────────────────────────────────────────


def _make_fake_pool() -> MagicMock:
    """
    Build a fake ConnectionPool whose disconnect() method is a no-op coroutine.

    The lifespan calls `await pool.disconnect()` on shutdown. By returning a
    MagicMock with an async disconnect, we prevent the teardown from opening
    any real TCP sockets.
    """
    fake_pool = MagicMock()
    fake_pool.disconnect = AsyncMock(return_value=None)
    return fake_pool


@pytest.fixture
def test_app(populated_fake_redis):
    """
    A FastAPI application instance wired to the in-memory FakeRedis.

    Strategy:
    1. Patch `create_redis_pool` in the lifespan to return a MagicMock pool,
       preventing any real TCP connection attempts during startup/shutdown.
    2. Override the `get_redis` dependency to return an async FakeRedis client
       backed by the same FakeServer as `populated_fake_redis`.
    3. The lifespan startup will try `await client.ping()` — this fails
       silently because it is wrapped in try/except in main.py.
    """
    app = create_app()

    # The async client shares the FakeServer so routes see pre-populated data.
    fake_async_client = fakeredis.aioredis.FakeRedis(
        server=_FAKE_SERVER, decode_responses=True
    )

    # Override get_redis dependency → inject our fake async client
    async def override_get_redis():
        return fake_async_client

    from app.api.routes import get_redis
    app.dependency_overrides[get_redis] = override_get_redis

    # Patch create_redis_pool so the lifespan never opens a real TCP socket
    fake_pool = _make_fake_pool()
    with patch("app.main.create_redis_pool", return_value=fake_pool):
        yield app

    app.dependency_overrides.clear()


@pytest.fixture
def client(test_app) -> Generator:
    """Synchronous TestClient for HTTP endpoint tests."""
    with TestClient(test_app, raise_server_exceptions=True) as c:
        yield c
