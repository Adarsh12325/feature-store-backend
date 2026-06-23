#!/usr/bin/env python3
"""
scripts/ingest_features.py
--------------------------
Continuous feature ingestion pipeline that simulates a real-time upstream
data stream writing into the Feature Store's Redis backend.

In a production MLOps system this script would be replaced by a Kafka
consumer, a Flink job, or an Airflow DAG that processes real user events.
Here, it generates statistically realistic synthetic user profiles and
pushes them to Redis via high-throughput pipelining.

Data Model Written
------------------
  Hash  →  user:{user_id}:features     (HSET with mapping)
  Set   →  all_users                   (SADD with user_id)

Usage
-----
  # Standalone (requires a running Redis instance)
  REDIS_HOST=localhost REDIS_PORT=6379 python scripts/ingest_features.py

  # Via Docker Compose (handled automatically)
  docker-compose up ingestion-worker
"""

from __future__ import annotations

import logging
import os
import random
import sys
import time
from datetime import datetime, timezone

import redis

# ── Logging ───────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)

# ── Configuration (from environment or defaults) ──────────────────────────────

REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
REDIS_DB = int(os.getenv("REDIS_DB", "0"))

BATCH_SIZE = int(os.getenv("INGESTION_BATCH_SIZE", "500"))
INTERVAL_SEC = float(os.getenv("INGESTION_INTERVAL_SEC", "1.0"))

# Total synthetic user population (IDs drawn from this range)
USER_POPULATION = 100_000
ALL_USERS_SET = "all_users"
FEATURE_KEY_PREFIX = "user"

# ── Synthetic data generation ─────────────────────────────────────────────────

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
    """
    Draw a random user ID from the synthetic user population pool.

    Using a fixed population size (100,000) ensures that repeated ingestion
    runs update existing records rather than endlessly growing the keyspace,
    which mirrors the idempotent behaviour of a real feature update pipeline.
    """
    return f"user_{random.randint(1, USER_POPULATION):06d}"


def generate_user_features() -> dict:
    """
    Generate a statistically varied ML feature payload for a synthetic user.

    Features span multiple data types to exercise the type-coercion logic
    in the API's data access layer:
      - Integer   : age, total_purchases, days_since_last_login, page_views_30d
      - Float     : avg_session_duration_sec, churn_risk_score, lifetime_value_usd,
                    click_through_rate
      - Boolean   : is_active, has_verified_email, subscribed_to_newsletter
      - Categorical: account_tier, preferred_category, region, device_type
      - Timestamp : last_login_timestamp

    Note: Redis stores all hash values as strings. The API layer is responsible
    for coercing them back to native Python types on read.
    """
    age = random.randint(18, 80)
    total_purchases = random.randint(0, 2000)
    days_since_last_login = random.randint(0, 365)
    # Churn risk increases with days_since_last_login, capped at 0.95
    base_churn = min(0.95, days_since_last_login / 400.0 + random.uniform(0.0, 0.25))
    lifetime_value = round(total_purchases * random.uniform(12.5, 85.0), 2)

    return {
        # Demographic / account
        "age": age,
        "account_tier": random.choice(ACCOUNT_TIERS),
        "region": random.choice(REGIONS),
        "device_type": random.choice(DEVICE_TYPES),
        # Engagement
        "total_purchases": total_purchases,
        "avg_session_duration_sec": round(random.uniform(30.0, 900.0), 2),
        "page_views_30d": random.randint(0, 500),
        "click_through_rate": round(random.uniform(0.01, 0.35), 4),
        "days_since_last_login": days_since_last_login,
        # User state
        "is_active": str(days_since_last_login < 30),         # cast bool → str for Redis
        "has_verified_email": str(random.random() > 0.15),
        "subscribed_to_newsletter": str(random.random() > 0.4),
        # Preferences
        "preferred_category": random.choice(PREFERRED_CATEGORIES),
        # ML-ready signals
        "churn_risk_score": round(base_churn, 4),
        "lifetime_value_usd": lifetime_value,
        # Temporal
        "last_login_timestamp": (
            datetime.now(timezone.utc).isoformat()
        ),
    }


# ── Core ingestion logic ──────────────────────────────────────────────────────


def create_redis_client() -> redis.Redis:
    """
    Build a synchronous Redis client with retry-friendly settings.

    The ingestion script uses the synchronous client (not async) because it
    runs as a standalone long-lived process, not inside an ASGI event loop.
    """
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
    """
    Block until Redis responds to PING, retrying up to max_retries times.

    This compensates for container startup sequencing even when Docker's
    healthcheck and depends_on are configured, as Redis may need a moment
    to load its AOF file from disk before accepting commands.
    """
    for attempt in range(1, max_retries + 1):
        try:
            client.ping()
            logger.info("Redis is ready.")
            return
        except (redis.ConnectionError, redis.TimeoutError) as exc:
            logger.warning(
                "Redis not ready (attempt %d/%d): %s — retrying in %.1fs",
                attempt,
                max_retries,
                exc,
                delay,
            )
            time.sleep(delay)

    logger.error("Could not connect to Redis after %d attempts. Exiting.", max_retries)
    sys.exit(1)


def ingest_batch(client: redis.Redis, batch_size: int) -> tuple[int, float]:
    """
    Write a single batch of synthetic user features to Redis via pipelining.

    A pipeline batches all HSET and SADD commands into one TCP write,
    reducing the per-batch latency from O(batch_size * RTT) to O(1 RTT).
    This is the critical performance optimisation that enables the script
    to sustain >50,000 writes/second on typical hardware.

    Returns:
        (records_written, elapsed_seconds) — for throughput calculation.
    """
    start = time.perf_counter()
    pipe = client.pipeline(transaction=False)

    for _ in range(batch_size):
        user_id = generate_user_id()
        features = generate_user_features()
        feature_key = f"{FEATURE_KEY_PREFIX}:{user_id}:features"

        # Command 1: Write or update the user's feature hash
        pipe.hset(feature_key, mapping=features)

        # Command 2: Register the user in the global index set
        pipe.sadd(ALL_USERS_SET, user_id)

    pipe.execute()
    elapsed = time.perf_counter() - start
    return batch_size, elapsed


def run_ingestion(client: redis.Redis) -> None:
    """
    Main ingestion loop — runs indefinitely, simulating a live event stream.

    Operational characteristics:
      - Writes BATCH_SIZE records per iteration.
      - Sleeps INTERVAL_SEC between batches for flow control.
      - Logs throughput and cumulative record count every 10 batches.
      - Implements exponential backoff on transient Redis errors to avoid
        hammering a temporarily overloaded Redis instance.
    """
    total_written = 0
    batch_num = 0
    error_backoff = 1.0

    logger.info(
        "Starting ingestion loop — batch_size=%d, interval=%.1fs, "
        "user_population=%d",
        BATCH_SIZE,
        INTERVAL_SEC,
        USER_POPULATION,
    )

    while True:
        try:
            written, elapsed = ingest_batch(client, BATCH_SIZE)
            total_written += written
            batch_num += 1
            error_backoff = 1.0  # reset on success

            if batch_num % 10 == 0:
                throughput = written / elapsed if elapsed > 0 else float("inf")
                indexed = client.scard(ALL_USERS_SET)
                logger.info(
                    "Batch #%d | written=%d | total=%d | "
                    "throughput=%.0f rec/s | unique_users_indexed=%d",
                    batch_num,
                    written,
                    total_written,
                    throughput,
                    indexed,
                )

            time.sleep(INTERVAL_SEC)

        except redis.RedisError as exc:
            logger.error(
                "Redis error during batch #%d: %s — backing off %.1fs",
                batch_num,
                exc,
                error_backoff,
            )
            time.sleep(error_backoff)
            error_backoff = min(error_backoff * 2, 30.0)  # cap at 30s

        except KeyboardInterrupt:
            logger.info(
                "Ingestion interrupted. Total records written this session: %d",
                total_written,
            )
            break

        except Exception as exc:  # noqa: BLE001
            logger.exception("Unexpected error in ingestion loop: %s", exc)
            time.sleep(error_backoff)
            error_backoff = min(error_backoff * 2, 30.0)


# ── Entrypoint ────────────────────────────────────────────────────────────────


def main() -> None:
    client = create_redis_client()
    wait_for_redis(client)

    # Warm up: ensure at least 100,000 users are indexed before the main loop
    # starts. This pre-populates the dataset so the API has data to serve
    # immediately after all containers are up.
    current_count = client.scard(ALL_USERS_SET)
    if current_count < USER_POPULATION:
        needed = USER_POPULATION - current_count
        warmup_batches = (needed + BATCH_SIZE - 1) // BATCH_SIZE
        logger.info(
            "Pre-populating feature store: %d users needed, "
            "running %d warmup batches...",
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

    # Enter the continuous streaming loop
    run_ingestion(client)


if __name__ == "__main__":
    main()
