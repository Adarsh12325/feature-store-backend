import os
from functools import lru_cache
from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    redis_host: str = Field(default="localhost", alias="REDIS_HOST")
    redis_port: int = Field(default=6379, alias="REDIS_PORT")
    redis_db: int = Field(default=0, alias="REDIS_DB")
    redis_password: str | None = Field(default=None, alias="REDIS_PASSWORD")

    api_host: str = Field(default="0.0.0.0", alias="API_HOST")
    api_port: int = Field(default=8000, alias="API_PORT")

    ingestion_batch_size: int = Field(default=500, alias="INGESTION_BATCH_SIZE")
    ingestion_interval_sec: float = Field(default=1.0, alias="INGESTION_INTERVAL_SEC")

    all_users_set_key: str = "all_users"
    user_features_key_prefix: str = "user"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        populate_by_name = True

    @property
    def redis_url(self) -> str:
        if self.redis_password:
            return f"redis://:{self.redis_password}@{self.redis_host}:{self.redis_port}/{self.redis_db}"
        return f"redis://{self.redis_host}:{self.redis_port}/{self.redis_db}"

    def user_features_key(self, user_id: str) -> str:
        return f"{self.user_features_key_prefix}:{user_id}:features"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
