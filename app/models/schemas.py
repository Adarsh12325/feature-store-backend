"""
app/models/schemas.py
---------------------
Pydantic data models that govern request validation and response serialization
for the Feature Store API.

All incoming request bodies are validated against these schemas before any
business logic executes. Invalid payloads are rejected immediately with an
HTTP 422 Unprocessable Entity, preventing malformed data from ever reaching
the Redis client.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator


# ── Request Schemas ───────────────────────────────────────────────────────────


class BatchFeatureRequest(BaseModel):
    """
    Payload schema for the POST /features/batch endpoint.

    Constraints:
      - user_ids must be a non-empty list (at least 1 element).
      - user_ids is capped at 100 elements to protect against accidental
        denial-of-service from excessively large batch requests.
      - Each individual user_id must be a non-empty string.
    """

    user_ids: List[str] = Field(
        ...,
        min_length=1,
        max_length=100,
        description="A list of user IDs to fetch features for (1–100 entries).",
    )

    @field_validator("user_ids")
    @classmethod
    def validate_user_ids(cls, v: List[str]) -> List[str]:
        """Ensure no blank or whitespace-only strings slip through."""
        cleaned = [uid.strip() for uid in v]
        if any(uid == "" for uid in cleaned):
            raise ValueError("user_ids must not contain empty or blank strings.")
        if len(cleaned) == 0:
            raise ValueError("user_ids must contain at least one entry.")
        return cleaned

    model_config = {
        "json_schema_extra": {
            "examples": [
                {"user_ids": ["user_001", "user_042", "user_999"]}
            ]
        }
    }


# ── Response Schemas ──────────────────────────────────────────────────────────


class FeatureVector(BaseModel):
    """
    Represents the complete feature vector for a single user.

    Feature values arrive from Redis as raw byte strings and are coerced
    into typed Python objects (int, float, bool) by the data access layer
    before populating this model.
    """

    user_id: str = Field(..., description="The unique identifier of the user.")
    features: Dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Key-value map of feature names to their corresponding values. "
            "An empty dict indicates the user was not found in the store."
        ),
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "user_id": "user_001",
                    "features": {
                        "age": 34,
                        "account_tier": "premium",
                        "total_purchases": 217,
                        "avg_session_duration_sec": 312.5,
                        "is_active": True,
                        "preferred_category": "electronics",
                        "churn_risk_score": 0.12,
                        "lifetime_value_usd": 4820.75,
                        "days_since_last_login": 2,
                        "last_login_timestamp": "2026-06-23T04:00:00.000000",
                    },
                }
            ]
        }
    }


class BatchFeatureResponse(BaseModel):
    """
    Response envelope for POST /features/batch.

    Each requested user_id appears exactly once in the results list.
    Users not found in Redis are represented with an empty features dict
    rather than omitted or causing an error.
    """

    results: List[FeatureVector] = Field(
        ...,
        description="Ordered list of feature vectors, one per requested user_id.",
    )
    total_requested: int = Field(
        ..., description="Total number of user IDs in the request."
    )
    total_found: int = Field(
        ..., description="Number of user IDs that were resolved to stored features."
    )


class HealthResponse(BaseModel):
    """Response schema for the GET /health endpoint."""

    status: str
    redis_connected: bool
    redis_host: str
    redis_port: int
    total_users_indexed: Optional[int] = None


class ErrorResponse(BaseModel):
    """Standard error payload returned on 4xx/5xx responses."""

    detail: str
    user_id: Optional[str] = None
