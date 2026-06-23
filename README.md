# Feature Store Backend

A production-grade, real-time **ML Feature Store** built with **FastAPI**, **Redis**, and **Docker**. Designed to serve machine learning model features at sub-millisecond latency, this system demonstrates the complete online-serving component of an MLOps Feature Store pipeline — from continuous data ingestion to ultra-fast, pipelined API retrieval.

---

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Why Redis Hashes and Sets?](#why-redis-hashes-and-sets)
3. [Project Structure](#project-structure)
4. [Quick Start](#quick-start)
5. [API Reference](#api-reference)
6. [Running Tests](#running-tests)
7. [Performance Benchmark](#performance-benchmark)
8. [Environment Variables](#environment-variables)
9. [Troubleshooting](#troubleshooting)
10. [Design Decisions and Assumptions](#design-decisions-and-assumptions)

---

## Architecture Overview

The system is composed of three containerised microservices orchestrated via Docker Compose, each with a clearly scoped responsibility:

```
┌─────────────────────────────────────────────────────────────────────────┐
│  Data Ingestion Layer                                                   │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │  ingestion-worker  (scripts/ingest_features.py)                  │  │
│  │  • Generates 100,000+ synthetic user feature records             │  │
│  │  • Pushes via Redis pipelining: HSET + SADD per user             │  │
│  └──────────────────────┬───────────────────────────────────────────┘  │
│                         │ 1. HSET user:{id}:features  (Redis Hash)     │
│                         │ 2. SADD all_users  (Redis Set)               │
└─────────────────────────┼───────────────────────────────────────────────┘
                          ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  Storage Layer                                                          │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │  Redis 7 (redis:alpine)                                          │  │
│  │  • In-memory data store with AOF persistence (--appendonly yes)  │  │
│  │  • Named Docker volume → /data (survives restarts)               │  │
│  └──────────────────────┬───────────────────────────────────────────┘  │
│                         │ 3. SISMEMBER + HGETALL (pipelined)           │
└─────────────────────────┼───────────────────────────────────────────────┘
                          ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  Serving Layer                                                          │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │  api-service  (FastAPI + uvicorn, port 8000)                     │  │
│  │  • GET  /features/{user_id}  → single-user retrieval (HGETALL)   │  │
│  │  • POST /features/batch      → multi-user batch (pipelined)      │  │
│  │  • GET  /health              → liveness + Redis connectivity      │  │
│  └──────────────────────┬───────────────────────────────────────────┘  │
│                         │ 4. JSON response (typed feature vector)      │
└─────────────────────────┼───────────────────────────────────────────────┘
                          ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  Client Layer (simulated by curl / benchmark script / ML model)         │
│  GET /features/user_042 → {"user_id": "user_042", "features": {...}}   │
└─────────────────────────────────────────────────────────────────────────┘
```

**Data flow summary:**

1. The **ingestion worker** continuously generates synthetic user profiles and writes them to Redis using `HSET` (feature hash) and `SADD` (global user index).
2. **Redis** stores feature hashes under `user:{user_id}:features` and tracks all user IDs in the `all_users` Set.
3. The **API service** receives HTTP requests, performs `SISMEMBER` for O(1) existence checks, then `HGETALL` (or a batched pipeline) to retrieve feature vectors.
4. The **ML inference client** (simulated by `curl` or the benchmark script) receives a typed JSON payload for immediate use in model scoring.

---

## Why Redis Hashes and Sets?

This is the most important architectural decision in the system. Two common approaches exist for storing feature vectors in Redis:

###  Anti-pattern: Serialised JSON Strings

```
SET user:123:features '{"age": 25, "is_active": true, "tier": "premium"}'
```

**Problems:**
- Updating a single field requires deserialising the entire JSON blob, modifying it in the application, and re-serialising it back — wasting CPU cycles proportional to the feature vector size.
- The entire string must be transferred over the network on every read, even if only one field is needed.
- No atomic partial updates; concurrent writes can cause race conditions.

###  Correct approach: Redis Hashes

```
HSET user:123:features age 25 is_active True tier premium
```

**Advantages:**
- **Partial updates**: `HSET user:123:features age 26` updates only the `age` field without reading or rewriting any other field.
- **Atomic retrieval**: `HGETALL user:123:features` fetches the complete feature vector in a single network round-trip.
- **Memory efficiency**: Redis uses a compact **listpack** (previously ziplist) encoding for hashes with fewer than 128 fields, significantly reducing memory overhead compared to a raw string key per field.
- **Idempotency**: `HSET` is naturally idempotent — re-running the ingestion pipeline with the same data leaves the database in an identical, correct state.

###  Correct approach: Redis Sets for Indexing

```
SADD all_users 123
SISMEMBER all_users 123  → 1 (O(1))
```

Attempting `HGETALL` on a non-existent key returns an empty dict, which is **indistinguishable** from a user who exists but has no features. Maintaining a separate `all_users` Set provides an O(1) definitive existence check (`SISMEMBER`) before querying the hash, enabling precise 404 responses.

---

## Project Structure

```
feature-store-backend/
├── app/                         # FastAPI application
│   ├── __init__.py
│   ├── main.py                  # Entry point, lifespan hooks, CORS
│   ├── config.py                # Pydantic-settings config (12-factor)
│   ├── api/
│   │   ├── __init__.py
│   │   └── routes.py            # GET /features/{id}, POST /features/batch
│   ├── models/
│   │   ├── __init__.py
│   │   └── schemas.py           # Pydantic request/response schemas
│   └── services/
│       ├── __init__.py
│       └── redis_client.py      # Data access layer (HSET, HGETALL, pipeline)
├── scripts/
│   ├── ingest_features.py       # Continuous ingestion pipeline
│   └── benchmark.py             # p90 latency benchmarking tool
├── tests/
│   ├── conftest.py              # Shared fixtures (FakeRedis, TestClient)
│   ├── test_api.py              # HTTP endpoint tests (200/404/422)
│   └── test_redis.py            # DAL unit tests (coercion, pipeline)
├── .env.example                 # Environment variable template
├── .gitignore
├── docker-compose.yml           # Three-service orchestration
├── Dockerfile                   # Python 3.10-slim image
├── pytest.ini                   # asyncio_mode=auto configuration
├── requirements.txt             # Pinned Python dependencies
└── README.md
```

---

## Quick Start

### Prerequisites

- [Docker](https://docs.docker.com/get-docker/) ≥ 24.0
- [Docker Compose](https://docs.docker.com/compose/install/) ≥ 2.20 (included with Docker Desktop)

### One-command setup

```bash
# Clone the repository
git clone https://github.com/Adarsh12325/feature-store-backend.git
cd feature-store-backend

# Build images and start all three services in detached mode
docker-compose up --build -d
```

This single command:
1. Builds the Python application image from the `Dockerfile`.
2. Starts the **Redis** container with AOF persistence enabled.
3. Waits for the Redis healthcheck to pass before starting dependent services.
4. Launches the **API service** on `http://localhost:8000`.
5. Launches the **ingestion worker**, which pre-populates 100,000 synthetic user records.

### Verify the system is running

```bash
# Check container status
docker-compose ps

# Follow live logs from all services
docker-compose logs -f

# Check API health
curl http://localhost:8000/health
```

Expected health response:
```json
{
  "status": "ok",
  "redis_connected": true,
  "redis_host": "redis",
  "redis_port": 6379,
  "total_users_indexed": 100000
}
```

### Tear down

```bash
# Stop containers (data volume preserved)
docker-compose down

# Stop containers and delete all data
docker-compose down -v
```

---

## API Reference

### GET `/health`

Returns the operational status of the API and its Redis connection.

```bash
curl http://localhost:8000/health
```

```json
{
  "status": "ok",
  "redis_connected": true,
  "redis_host": "redis",
  "redis_port": 6379,
  "total_users_indexed": 100000
}
```

---

### GET `/features/{user_id}`

Retrieves the complete ML feature vector for a single user.

**Path Parameters**
| Parameter | Type   | Description                            |
|-----------|--------|----------------------------------------|
| `user_id` | string | The unique identifier of the target user (e.g., `user_000042`) |

**Success Response — 200 OK**

```bash
curl http://localhost:8000/features/user_000042
```

```json
{
  "user_id": "user_000042",
  "features": {
    "age": 34,
    "account_tier": "premium",
    "total_purchases": 217,
    "avg_session_duration_sec": 312.5,
    "is_active": true,
    "preferred_category": "electronics",
    "churn_risk_score": 0.12,
    "lifetime_value_usd": 4820.75,
    "days_since_last_login": 2,
    "last_login_timestamp": "2026-06-23T04:00:00+00:00",
    "region": "us-east",
    "device_type": "mobile",
    "has_verified_email": true,
    "subscribed_to_newsletter": false,
    "page_views_30d": 145,
    "click_through_rate": 0.0823
  }
}
```

**User Not Found — 404 Not Found**

```bash
curl -i http://localhost:8000/features/unknown_user_xyz
```

```
HTTP/1.1 404 Not Found

{
  "detail": "User 'unknown_user_xyz' was not found in the feature store."
}
```

---

### POST `/features/batch`

Retrieves feature vectors for multiple users in a single pipelined Redis request. Ideal for ML inference where a model needs features for a set of candidate users simultaneously.

**Request Body**
| Field      | Type          | Constraints          | Description                     |
|------------|---------------|----------------------|---------------------------------|
| `user_ids` | `List[string]`| min: 1, max: 100     | List of user IDs to look up     |

**Success Response — 200 OK**

Users not found in the store are included with an empty `features` object rather than causing an error.

```bash
curl -X POST http://localhost:8000/features/batch \
  -H "Content-Type: application/json" \
  -d '{"user_ids": ["user_000001", "user_000002", "unknown_ghost_user"]}'
```

```json
{
  "results": [
    {
      "user_id": "user_000001",
      "features": {
        "age": 29,
        "account_tier": "premium",
        "churn_risk_score": 0.05,
        "is_active": true
      }
    },
    {
      "user_id": "user_000002",
      "features": {
        "age": 55,
        "account_tier": "free",
        "churn_risk_score": 0.88,
        "is_active": false
      }
    },
    {
      "user_id": "unknown_ghost_user",
      "features": {}
    }
  ],
  "total_requested": 3,
  "total_found": 2
}
```

**Validation Error — 422 Unprocessable Entity**

```bash
# Empty list is rejected
curl -X POST http://localhost:8000/features/batch \
  -H "Content-Type: application/json" \
  -d '{"user_ids": []}'
```

```json
{
  "detail": [
    {
      "type": "too_short",
      "loc": ["body", "user_ids"],
      "msg": "List should have at least 1 item after validation, not 0",
      "input": []
    }
  ]
}
```

**Interactive API documentation** is available at `http://localhost:8000/docs` (Swagger UI) and `http://localhost:8000/redoc` (ReDoc).

---

## Running Tests

The test suite uses `pytest` with `fakeredis` for in-process Redis emulation — no Docker container is required.

### Setup (local virtual environment)

```bash
# Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate          # Linux/macOS
.venv\Scripts\activate             # Windows PowerShell

# Install dependencies
pip install -r requirements.txt
```

### Run all tests

```bash
pytest tests/ -v
```

### Run tests inside the running Docker container

```bash
docker-compose exec api-service pytest tests/ -v
```

### Expected output

```
tests/test_api.py::TestHealthEndpoint::test_health_returns_200 PASSED
tests/test_api.py::TestHealthEndpoint::test_health_redis_connected PASSED
tests/test_api.py::TestSingleUserRetrieval::test_known_user_returns_200 PASSED
tests/test_api.py::TestSingleUserRetrieval::test_unknown_user_returns_404 PASSED
tests/test_api.py::TestSingleUserRetrieval::test_known_user_age_is_integer PASSED
tests/test_api.py::TestBatchRetrieval::test_missing_user_in_batch_gets_empty_features PASSED
tests/test_api.py::TestBatchRetrieval::test_empty_user_ids_list_returns_422 PASSED
tests/test_api.py::TestBatchRetrieval::test_oversized_batch_returns_422 PASSED
tests/test_redis.py::TestCoerceValue::test_integer_string PASSED
tests/test_redis.py::TestCoerceValue::test_true_string_capitalised PASSED
tests/test_redis.py::TestGetUserFeatures::test_returns_none_for_unknown_user PASSED
tests/test_redis.py::TestGetUsersBatch::test_mixed_batch_partial_none PASSED
...

============================== 40 passed in 1.23s ==============================
```

---

## Performance Benchmark

### Methodology

To verify the core real-time serving requirement — that individual feature retrieval completes in **under 50 milliseconds at the 90th percentile (p90)** — a dedicated benchmarking script (`scripts/benchmark.py`) was developed.

**Procedure:**

1. Ensured the ingestion worker had fully populated the feature store with **100,000 unique synthetic user records**.
2. Verified API health via `GET /health` to confirm Redis connectivity and user count.
3. Executed **1,000 sequential `GET /features/{user_id}` requests** using randomly selected user IDs drawn uniformly from the 100,000-user population.
4. Measured wall-clock response time (`time.perf_counter()`) for each request, capturing only the application-level HTTP round-trip (not DNS or TCP setup, which is amortised across requests via connection reuse).
5. Sorted the recorded latency distribution and extracted key percentile values.

**Sequential mode** (one request at a time, `CONCURRENCY=1`) was used to measure the latency experienced by a single calling service, which represents the most honest p90 metric for a single-threaded ML inference loop. Concurrent mode is available via `BENCH_CONCURRENCY` for throughput testing.

### Running the Benchmark

```bash
# Ensure the stack is fully up and ingestion has completed
docker-compose up -d
docker-compose logs ingestion-worker | tail -20  # confirm warmup complete

# Run benchmark against the live API (from host machine)
python scripts/benchmark.py

# Optional: run 2,000 requests with 4 concurrent threads
NUM_REQUESTS=2000 BENCH_CONCURRENCY=4 python scripts/benchmark.py
```

### Results

The following results were recorded on a standard development machine
(Docker Desktop on Linux/macOS, Redis and API co-located in Docker containers
on localhost, no network hop):

```
============================================================
  Feature Store Latency Benchmark
============================================================
  Target      : http://localhost:8000
  Requests    : 1,000
  Concurrency : 1 (sequential)
  P90 Budget  : 50.0 ms
------------------------------------------------------------
  API Status  : ok
  Users in DB : 100,000
------------------------------------------------------------
  Running 1000 requests...
============================================================
  RESULTS
============================================================
  Total requests    : 1,000
  Successful        : 1,000
  Errors            : 0
------------------------------------------------------------
  Min latency       : 1.84 ms
  Mean latency      : 3.21 ms
  Std deviation     : 1.47 ms
  P50 (median)      : 2.97 ms
  P90               : 4.83 ms  ← SLA target: < 50 ms
  P95               : 5.61 ms
  P99               : 8.24 ms
  Max latency       : 22.31 ms
============================================================

  ✅  P90 PASS — 4.83 ms ≤ 50.0 ms SLA target.
```

**p90 Latency: 4.83 ms** — well within the 50 ms budget.

The result demonstrates that Redis's in-memory architecture, combined with the async FastAPI/uvicorn stack and connection pooling, delivers feature retrieval latency that is **an order of magnitude faster** than the stated SLA. This headroom accommodates real-world factors not present in local testing: network distance between services in a cloud environment, TLS overhead, and load balancer latency, all of which add roughly 5–20 ms in production VPC deployments.

---

## Environment Variables

| Variable                 | Default      | Description                                              |
|--------------------------|--------------|----------------------------------------------------------|
| `REDIS_HOST`             | `redis`      | Hostname of the Redis instance (service name in Docker)  |
| `REDIS_PORT`             | `6379`       | Redis server port                                        |
| `REDIS_DB`               | `0`          | Redis logical database index                             |
| `REDIS_PASSWORD`         | *(unset)*    | Redis AUTH password (optional)                           |
| `API_HOST`               | `0.0.0.0`    | Interface the API server binds to                        |
| `API_PORT`               | `8000`       | Port the API server listens on                           |
| `INGESTION_BATCH_SIZE`   | `500`        | Number of records per pipeline flush                     |
| `INGESTION_INTERVAL_SEC` | `1.0`        | Sleep duration between ingestion batches (seconds)       |

For local development outside Docker, copy `.env.example` to `.env` and set `REDIS_HOST=localhost`.

---

## Troubleshooting

### `api-service` fails to connect to Redis on startup

**Symptom:** API container exits immediately with `ConnectionRefusedError`.

**Cause:** The API started before Redis passed its healthcheck, despite the `depends_on: condition: service_healthy` setting. This can happen if Docker's healthcheck interval is longer than the API's connection timeout.

**Fix:** Restart the API container after Redis is healthy:
```bash
docker-compose restart api-service
```
Or rebuild with a longer Redis start period:
```yaml
healthcheck:
  start_period: 20s   # increase from 10s
```

---

### Ingestion worker shows `0 unique users indexed` after warmup

**Symptom:** The ingestion log shows batches executing but `SCARD all_users` returns 0.

**Cause:** The worker may be connecting to a different Redis DB index, or the `all_users` key was flushed.

**Fix:**
```bash
# Connect to Redis CLI inside the container
docker-compose exec redis redis-cli
127.0.0.1:6379> SCARD all_users
(integer) 100000   # Expected after warmup
127.0.0.1:6379> HGETALL user:000001:features
```

---

### Tests fail with `ModuleNotFoundError: No module named 'app'`

**Cause:** The `PYTHONPATH` is not set when running pytest from outside the project root.

**Fix:**
```bash
# Run from the repository root
cd feature-store-backend
pytest tests/ -v

# Or explicitly set PYTHONPATH
PYTHONPATH=. pytest tests/ -v
```

---

### `docker-compose up` fails with `port 6379 already in use`

**Cause:** A Redis server is already running on the host.

**Fix:** Either stop the local Redis service, or change the host port mapping in `docker-compose.yml`:
```yaml
ports:
  - "6380:6379"   # map to 6380 on host instead
```

---

### Benchmark p90 exceeds 50 ms on Windows Docker Desktop

**Cause:** Docker Desktop on Windows uses a Linux VM with a virtual network bridge, adding 5–15 ms compared to native Linux. This is a known Docker Desktop overhead, not an application issue.

**Fix:** Run the benchmark from inside the API container to exclude Docker's virtual network latency:
```bash
docker-compose exec api-service python scripts/benchmark.py
```

---

## Design Decisions and Assumptions

1. **FastAPI over Flask**: FastAPI's native async support (via `asyncio`) allows the API to handle concurrent requests without blocking on I/O. This is essential for a feature store where the bottleneck is always network I/O to Redis, not CPU computation.

2. **Connection pooling**: A shared `ConnectionPool` is instantiated once during application startup and reused across all requests. This eliminates the TCP and socket overhead that would occur if a new connection were opened per request.

3. **Two-phase batch pipeline**: The batch endpoint uses two sequential pipeline round-trips — first SISMEMBER for all requested users, then HGETALL only for existing users. This avoids fetching empty HGETALL results for non-existent users, reducing unnecessary network bytes and Redis CPU cycles.

4. **Type coercion in the DAL**: Redis stores all hash values as byte strings. The `_coerce_value` function in the data access layer restores integer, float, and boolean types before the response is serialised to JSON. This keeps the API schema clean and ensures clients receive properly typed data without extra parsing.

5. **Synchronous ingestion client**: The ingestion script uses the synchronous `redis.Redis` client rather than the async variant because it runs as a standalone blocking process outside any event loop. Mixing `asyncio` event loops with long-running blocking processes is an anti-pattern in Python.

6. **Synthetic user population capped at 100,000**: The `USER_POPULATION` constant ensures that repeated ingestion runs update existing records (idempotent behaviour) rather than endlessly growing the keyspace. This mirrors how a real feature pipeline refreshes stale features for an existing user base.

7. **AOF persistence with `appendfsync everysec`**: This configuration provides a balance between durability and throughput. At most one second of writes can be lost on a crash, which is acceptable for a feature store where features can be regenerated from the upstream data source.
