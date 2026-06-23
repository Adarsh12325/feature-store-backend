from __future__ import annotations

from typing import Annotated, Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator


class BatchFeatureRequest(BaseModel):
    user_ids: Annotated[
        List[str],
        Field(
            min_length=1,
            max_length=100,
            description="A list of user IDs to fetch features for (1-100 entries).",
        ),
    ]

    @field_validator("user_ids")
    @classmethod
    def validate_user_ids(cls, v: List[str]) -> List[str]:
        cleaned = [uid.strip() for uid in v]
        if any(uid == "" for uid in cleaned):
            raise ValueError("user_ids must not contain empty or blank strings.")
        return cleaned

    model_config = {
        "json_schema_extra": {
            "examples": [
                {"user_ids": ["user_001", "user_042", "user_999"]}
            ]
        }
    }


class FeatureVector(BaseModel):
    user_id: str = Field(..., description="The unique identifier of the user.")
    features: Dict[str, Any] = Field(
        default_factory=dict,
        description="Key-value map of feature names to their corresponding values.",
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
    status: str
    redis_connected: bool
    redis_host: str
    redis_port: int
    total_users_indexed: Optional[int] = None


class ErrorResponse(BaseModel):
    detail: str
    user_id: Optional[str] = None
