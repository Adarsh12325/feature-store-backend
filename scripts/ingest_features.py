#!/usr/bin/env python3

from __future__ import annotations

import logging
import os
import random
import sys
import time
from datetime import datetime, timezone

import redis

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)

REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
REDIS_DB = int(os.getenv("REDIS_DB", "0"))

BATCH_SIZE = int(os.getenv("INGESTION_BATCH_SIZE", "500"))
INTERVAL_SEC = float(os.getenv("INGESTION_INTERVAL_SEC", "1.0"))

USER_POPULATION = 100_000
ALL_USERS_SET = "all_users"
FEATURE_KEY_PREFIX = "user"

ACCOUNT_TIERS = ["free", "basic", "premium", "enterprise"]
PREFERRED_CATEGORIES = [
    "electronics",
    "fashion",
    "home_garden",
    "sports",
    "books",
    "beauty",
    "automotive",
    "groceries",
]
REGIONS = ["us-east", "us-west", "eu-central", "ap-south", "ap-east", "latam"]
DEVICE_TYPES = ["mobile", "desktop", "tablet"]


def generate_user_id() -> str:
    return f"user_{random.randint(1, USER_POPULATION):06d}"


def generate_user_features() -> dict:
    age = random.randint(18, 80)
    total_purchases = random.randint(0, 2000)
    days_since_last_login = random.randint(0, 365)
    base_churn = min(0.95, days_since_last_login / 400.0 + random.uniform(0.0, 0.25))
    lifetime_value = round(total_purchases * random.uniform(12.5, 85.0), 2)

    return {
        "age": age,
        "account_tier": random.choice(ACCOUNT_TIERS),
        "region": random.choice(REGIONS),
        "device_type": random.choice(DEVICE_TYPES),
        "total_purchases": total_purchases,
        "avg_session_duration_sec": round(random.uniform(30.0, 900.0), 2),
        "page_views_30d": random.randint(0, 500),
        "click_through_rate": round(random.uniform(0.01, 0.35), 4),
        "days_since_last_login": days_since_last_login,
        "is_active": str(days_since_last_login < 30),
        "has_verified_email": str(random.random() > 0.15),
        "subscribed_to_newsletter": str(random.random() > 0.4),
        "preferred_category": random.choice(PREFERRED_CATEGORIES),
        "churn_risk_score": round(base_churn, 4),
        "lifetime_value_usd": lifetime_value,
        "last_login_timestamp": datetime.now(timezone.utc).isoformat(),
    }


def create_redis_client() -> redis.Redis:
    logger.info("Connecting to Redis at %s:%s", REDIS_HOST, REDIS_PORT)
    client = redis.Redis(
        host=REDIS_HOST,
        port=REDIS_PORT,
        db=REDIS_DB,
        decode_responses=True,
        socket_connect_timeout=10,
        socket_timeout=10,
        retry_on_timeout=True,
    )
    return client


def wait_for_redis(client: redis.Redis, max_retries: int = 15, delay: float = 2.0) -> None:
    for attempt in range(1, max_retries + 1):
        try:
            client.ping()
            logger.info("Redis is ready.")
            return
        except (redis.ConnectionError, redis.TimeoutError) as exc:
            logger.warning(
                "Redis not ready (attempt %d/%d): %s - retrying in %.1fs",
                attempt,
                max_retries,
                exc,
                delay,
            )
            time.sleep(delay)

    logger.error("Could not connect to Redis after %d attempts. Exiting.", max_retries)
    sys.exit(1)


def ingest_batch(client: redis.Redis, batch_size: int) -> tuple[int, float]:
    start = time.perf_counter()
    pipe = client.pipeline(transaction=False)

    for _ in range(batch_size):
        user_id = generate_user_id()
        features = generate_user_features()
        feature_key = f"{FEATURE_KEY_PREFIX}:{user_id}:features"

        pipe.hset(feature_key, mapping=features)
        pipe.sadd(ALL_USERS_SET, user_id)

    pipe.execute()
    elapsed = time.perf_counter() - start
    return batch_size, elapsed


def run_ingestion(client: redis.Redis) -> None:
    total_written = 0
    batch_num = 0
    error_backoff = 1.0

    logger.info(
        "Starting ingestion loop - batch_size=%d, interval=%.1fs, user_population=%d",
        BATCH_SIZE,
        INTERVAL_SEC,
        USER_POPULATION,
    )

    while True:
        try:
            written, elapsed = ingest_batch(client, BATCH_SIZE)
            total_written += written
            batch_num += 1
            error_backoff = 1.0

            if batch_num % 10 == 0:
                throughput = written / elapsed if elapsed > 0 else float("inf")
                indexed = client.scard(ALL_USERS_SET)
                logger.info(
                    "Batch #%d | written=%d | total=%d | throughput=%.0f rec/s | unique_users_indexed=%d",
                    batch_num,
                    written,
                    total_written,
                    throughput,
                    indexed,
                )

            time.sleep(INTERVAL_SEC)

        except redis.RedisError as exc:
            logger.error(
                "Redis error during batch #%d: %s - backing off %.1fs",
                batch_num,
                exc,
                error_backoff,
            )
            time.sleep(error_backoff)
            error_backoff = min(error_backoff * 2, 30.0)

        except KeyboardInterrupt:
            logger.info(
                "Ingestion interrupted. Total records written this session: %d",
                total_written,
            )
            break

        except Exception as exc:
            logger.exception("Unexpected error in ingestion loop: %s", exc)
            time.sleep(error_backoff)
            error_backoff = min(error_backoff * 2, 30.0)


def main() -> None:
    client = create_redis_client()
    wait_for_redis(client)

    current_count = client.scard(ALL_USERS_SET)
    if current_count < USER_POPULATION:
        needed = USER_POPULATION - current_count
        warmup_batches = (needed + BATCH_SIZE - 1) // BATCH_SIZE
        logger.info(
            "Pre-populating feature store: %d users needed, running %d warmup batches...",
            needed,
            warmup_batches,
        )
        for i in range(warmup_batches):
            ingest_batch(client, BATCH_SIZE)
            if (i + 1) % 20 == 0:
                progress = client.scard(ALL_USERS_SET)
                logger.info(
                    "Warmup progress: %d / %d unique users indexed",
                    progress,
                    USER_POPULATION,
                )
        logger.info(
            "Warmup complete. Unique users indexed: %d", client.scard(ALL_USERS_SET)
        )

    run_ingestion(client)


if __name__ == "__main__":
    main()
