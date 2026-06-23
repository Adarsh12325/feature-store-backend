"""
app/config.py
-------------
Centralised configuration management for the Feature Store API.

All runtime parameters are sourced exclusively from environment variables,
adhering to the 12-factor app methodology. No configuration value is
hardcoded anywhere in the application logic.
"""

import os
from functools import lru_cache
from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    """
    Application settings loaded from environment variables or a .env file.

    Pydantic-settings automatically reads from the process environment and
    from any .env file present in the working directory.
    """

    # ── Redis connection ──────────────────────────────────────────────────────
    redis_host: str = Field(default="localhost", alias="REDIS_HOST")
    redis_port: int = Field(default=6379, alias="REDIS_PORT")
    redis_db: int = Field(default=0, alias="REDIS_DB")
    redis_password: str | None = Field(default=None, alias="REDIS_PASSWORD")

    # ── API server ────────────────────────────────────────────────────────────
    api_host: str = Field(default="0.0.0.0", alias="API_HOST")
    api_port: int = Field(default=8000, alias="API_PORT")

    # ── Ingestion tuning ──────────────────────────────────────────────────────
    ingestion_batch_size: int = Field(default=500, alias="INGESTION_BATCH_SIZE")
    ingestion_interval_sec: float = Field(default=1.0, alias="INGESTION_INTERVAL_SEC")

    # ── Feature store constants ───────────────────────────────────────────────
    all_users_set_key: str = "all_users"
    user_features_key_prefix: str = "user"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        # Allow population by field name as well as alias
        populate_by_name = True

    @property
    def redis_url(self) -> str:
        """Construct a Redis connection URL from individual parameters."""
        if self.redis_password:
            return f"redis://:{self.redis_password}@{self.redis_host}:{self.redis_port}/{self.redis_db}"
        return f"redis://{self.redis_host}:{self.redis_port}/{self.redis_db}"

    def user_features_key(self, user_id: str) -> str:
        """Return the namespaced Redis Hash key for a given user's features."""
        return f"{self.user_features_key_prefix}:{user_id}:features"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """
    Return the cached application settings singleton.

    Using lru_cache ensures the settings object is created exactly once
    per process, avoiding repeated disk I/O for .env file parsing.
    """
    return Settings()
