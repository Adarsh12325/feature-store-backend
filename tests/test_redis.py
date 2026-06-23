"""
tests/test_redis.py
-------------------
Unit tests for the Redis data access layer (app/services/redis_client.py).

These tests target the DAL functions in isolation, verifying that:
  - _coerce_value correctly parses int, float, bool, and str values.
  - user_exists correctly uses SISMEMBER semantics.
  - get_user_features returns None for absent users and a typed dict for
    present users.
  - get_users_features_batch handles mixed found/not-found lists correctly.

All tests use fakeredis.aioredis to emulate Redis behaviour without
requiring a running container. The async tests use pytest-asyncio.
"""

from __future__ import annotations

import pytest
import fakeredis
import fakeredis.aioredis

from app.config import get_settings
from app.services.redis_client import (
    _coerce_value,
    _coerce_feature_dict,
    get_user_features,
    get_users_features_batch,
    user_exists,
    ping_redis,
    get_total_user_count,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture
async def redis_client():
    """Async FakeRedis client for DAL unit tests."""
    server = fakeredis.FakeServer()
    client = fakeredis.aioredis.FakeRedis(server=server, decode_responses=True)
    yield client
    await client.flushall()
    await client.aclose()


@pytest.fixture
async def populated_redis(redis_client):
    """FakeRedis pre-loaded with two test users."""
    settings = get_settings()

    await redis_client.hset(
        settings.user_features_key("alpha_001"),
        mapping={
            "age": "29",
            "account_tier": "premium",
            "is_active": "True",
            "churn_risk_score": "0.05",
            "lifetime_value_usd": "1500.00",
            "total_purchases": "75",
            "avg_session_duration_sec": "240.5",
            "page_views_30d": "90",
            "days_since_last_login": "1",
            "click_through_rate": "0.15",
            "preferred_category": "electronics",
            "region": "us-east",
            "device_type": "mobile",
            "has_verified_email": "True",
            "subscribed_to_newsletter": "True",
            "last_login_timestamp": "2026-06-23T00:00:00+00:00",
        },
    )
    await redis_client.sadd(settings.all_users_set_key, "alpha_001")

    await redis_client.hset(
        settings.user_features_key("beta_002"),
        mapping={
            "age": "55",
            "account_tier": "free",
            "is_active": "False",
            "churn_risk_score": "0.88",
            "lifetime_value_usd": "120.0",
            "total_purchases": "5",
            "avg_session_duration_sec": "45.0",
            "page_views_30d": "8",
            "days_since_last_login": "180",
            "click_through_rate": "0.02",
            "preferred_category": "groceries",
            "region": "latam",
            "device_type": "desktop",
            "has_verified_email": "False",
            "subscribed_to_newsletter": "False",
            "last_login_timestamp": "2025-12-25T12:00:00+00:00",
        },
    )
    await redis_client.sadd(settings.all_users_set_key, "beta_002")

    return redis_client


# ── _coerce_value unit tests ──────────────────────────────────────────────────


class TestCoerceValue:
    """Validate the type coercion helper independently."""

    def test_integer_string(self):
        assert _coerce_value("42") == 42
        assert isinstance(_coerce_value("42"), int)

    def test_negative_integer(self):
        assert _coerce_value("-7") == -7

    def test_float_string(self):
        result = _coerce_value("3.14")
        assert isinstance(result, float)
        assert abs(result - 3.14) < 1e-9

    def test_true_string_capitalised(self):
        assert _coerce_value("True") is True

    def test_false_string_capitalised(self):
        assert _coerce_value("False") is False

    def test_true_lowercase(self):
        assert _coerce_value("true") is True

    def test_false_lowercase(self):
        assert _coerce_value("false") is False

    def test_one_is_integer_not_bool(self):
        """'1' should coerce to int 1, not bool True (int check runs first)."""
        result = _coerce_value("1")
        assert result == 1
        assert isinstance(result, int)

    def test_plain_string_unchanged(self):
        assert _coerce_value("premium") == "premium"
        assert isinstance(_coerce_value("premium"), str)

    def test_timestamp_string_unchanged(self):
        ts = "2026-06-23T04:00:00+00:00"
        assert _coerce_value(ts) == ts

    def test_empty_string_unchanged(self):
        assert _coerce_value("") == ""

    def test_zero_string(self):
        assert _coerce_value("0") == 0
        assert isinstance(_coerce_value("0"), int)


class TestCoerceFeatureDict:
    def test_dict_keys_are_strings(self):
        raw = {"age": "25", "tier": "free"}
        result = _coerce_feature_dict(raw)
        for k in result:
            assert isinstance(k, str)

    def test_integer_values_coerced(self):
        raw = {"age": "25"}
        result = _coerce_feature_dict(raw)
        assert result["age"] == 25
        assert isinstance(result["age"], int)

    def test_bytes_keys_decoded(self):
        raw = {b"age": b"25", b"tier": b"free"}
        result = _coerce_feature_dict(raw)
        assert "age" in result
        assert result["age"] == 25


# ── user_exists tests ─────────────────────────────────────────────────────────


class TestUserExists:
    @pytest.mark.asyncio
    async def test_existing_user_returns_true(self, populated_redis):
        result = await user_exists(populated_redis, "alpha_001")
        assert result is True

    @pytest.mark.asyncio
    async def test_nonexistent_user_returns_false(self, populated_redis):
        result = await user_exists(populated_redis, "totally_unknown_xyz")
        assert result is False

    @pytest.mark.asyncio
    async def test_case_sensitive(self, populated_redis):
        """Redis Set membership is case-sensitive."""
        result = await user_exists(populated_redis, "ALPHA_001")
        assert result is False


# ── get_user_features tests ───────────────────────────────────────────────────


class TestGetUserFeatures:
    @pytest.mark.asyncio
    async def test_returns_none_for_unknown_user(self, populated_redis):
        result = await get_user_features(populated_redis, "ghost_user")
        assert result is None

    @pytest.mark.asyncio
    async def test_returns_dict_for_known_user(self, populated_redis):
        result = await get_user_features(populated_redis, "alpha_001")
        assert isinstance(result, dict)
        assert len(result) > 0

    @pytest.mark.asyncio
    async def test_age_coerced_to_int(self, populated_redis):
        result = await get_user_features(populated_redis, "alpha_001")
        assert result["age"] == 29
        assert isinstance(result["age"], int)

    @pytest.mark.asyncio
    async def test_churn_score_coerced_to_float(self, populated_redis):
        result = await get_user_features(populated_redis, "alpha_001")
        assert isinstance(result["churn_risk_score"], float)

    @pytest.mark.asyncio
    async def test_is_active_true_coerced_to_bool(self, populated_redis):
        result = await get_user_features(populated_redis, "alpha_001")
        assert result["is_active"] is True

    @pytest.mark.asyncio
    async def test_is_active_false_coerced_to_bool(self, populated_redis):
        result = await get_user_features(populated_redis, "beta_002")
        assert result["is_active"] is False

    @pytest.mark.asyncio
    async def test_account_tier_is_string(self, populated_redis):
        result = await get_user_features(populated_redis, "alpha_001")
        assert result["account_tier"] == "premium"
        assert isinstance(result["account_tier"], str)


# ── get_users_features_batch tests ───────────────────────────────────────────


class TestGetUsersBatch:
    @pytest.mark.asyncio
    async def test_both_users_found(self, populated_redis):
        results = await get_users_features_batch(
            populated_redis, ["alpha_001", "beta_002"]
        )
        assert len(results) == 2
        ids = [r[0] for r in results]
        assert "alpha_001" in ids
        assert "beta_002" in ids

    @pytest.mark.asyncio
    async def test_found_users_have_feature_dict(self, populated_redis):
        results = await get_users_features_batch(populated_redis, ["alpha_001"])
        uid, features = results[0]
        assert uid == "alpha_001"
        assert isinstance(features, dict)
        assert len(features) > 0

    @pytest.mark.asyncio
    async def test_missing_user_returns_none(self, populated_redis):
        results = await get_users_features_batch(populated_redis, ["ghost_xyz"])
        uid, features = results[0]
        assert uid == "ghost_xyz"
        assert features is None

    @pytest.mark.asyncio
    async def test_mixed_batch_partial_none(self, populated_redis):
        results = await get_users_features_batch(
            populated_redis, ["alpha_001", "ghost_xyz", "beta_002"]
        )
        result_map = {uid: feats for uid, feats in results}
        assert result_map["alpha_001"] is not None
        assert result_map["ghost_xyz"] is None
        assert result_map["beta_002"] is not None

    @pytest.mark.asyncio
    async def test_empty_input_returns_empty_list(self, populated_redis):
        results = await get_users_features_batch(populated_redis, [])
        assert results == []

    @pytest.mark.asyncio
    async def test_order_preserved(self, populated_redis):
        """Response order must match input order."""
        input_ids = ["beta_002", "alpha_001"]
        results = await get_users_features_batch(populated_redis, input_ids)
        returned_ids = [r[0] for r in results]
        assert returned_ids == input_ids


# ── ping_redis / get_total_user_count ────────────────────────────────────────


class TestHelperFunctions:
    @pytest.mark.asyncio
    async def test_ping_returns_true(self, redis_client):
        result = await ping_redis(redis_client)
        assert result is True

    @pytest.mark.asyncio
    async def test_user_count_zero_when_empty(self, redis_client):
        count = await get_total_user_count(redis_client)
        assert count == 0

    @pytest.mark.asyncio
    async def test_user_count_matches_sadd(self, redis_client):
        settings = get_settings()
        await redis_client.sadd(settings.all_users_set_key, "u1", "u2", "u3")
        count = await get_total_user_count(redis_client)
        assert count == 3
