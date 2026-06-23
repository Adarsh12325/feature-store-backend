#!/usr/bin/env python3

from __future__ import annotations

import os
import random
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List

import requests

API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")
NUM_REQUESTS = int(os.getenv("NUM_REQUESTS", "1000"))
CONCURRENCY = int(os.getenv("BENCH_CONCURRENCY", "1"))
USER_POPULATION = 100_000
P90_BUDGET_MS = 50.0


def build_session() -> requests.Session:
    session = requests.Session()
    adapter = requests.adapters.HTTPAdapter(
        pool_connections=CONCURRENCY,
        pool_maxsize=CONCURRENCY,
    )
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session


def fetch_single(session: requests.Session, user_id: str) -> float:
    url = f"{API_BASE_URL}/features/{user_id}"
    start = time.perf_counter()
    try:
        resp = session.get(url, timeout=10)
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        if resp.status_code in (200, 404):
            return elapsed_ms
        print(f"  Unexpected status {resp.status_code} for {user_id}", file=sys.stderr)
        return -1.0
    except Exception as exc:
        print(f"  Request error for {user_id}: {exc}", file=sys.stderr)
        return -1.0


def run_benchmark() -> None:
    print("=" * 60)
    print("  Feature Store Latency Benchmark")
    print("=" * 60)
    print(f"  Target      : {API_BASE_URL}")
    print(f"  Requests    : {NUM_REQUESTS}")
    print(f"  Concurrency : {CONCURRENCY} (1 = sequential)")
    print(f"  P90 Budget  : {P90_BUDGET_MS} ms")
    print("-" * 60)

    try:
        health = requests.get(f"{API_BASE_URL}/health", timeout=5)
        health.raise_for_status()
        info = health.json()
        print(f"  API Status  : {info.get('status', 'unknown')}")
        print(f"  Users in DB : {info.get('total_users_indexed', 'unknown')}")
    except Exception as exc:
        print(f"\n  ERROR: Cannot reach API at {API_BASE_URL}: {exc}")
        print("  Ensure docker-compose is running and ingestion has completed.")
        sys.exit(1)

    print("-" * 60)
    print(f"  Running {NUM_REQUESTS} requests...\n")

    session = build_session()
    latencies: List[float] = []
    errors = 0

    user_ids = [f"user_{random.randint(1, USER_POPULATION):06d}" for _ in range(NUM_REQUESTS)]

    if CONCURRENCY == 1:
        for i, uid in enumerate(user_ids):
            ms = fetch_single(session, uid)
            if ms >= 0:
                latencies.append(ms)
            else:
                errors += 1
            if (i + 1) % 100 == 0:
                print(f"  Progress: {i + 1}/{NUM_REQUESTS} requests completed...")
    else:
        with ThreadPoolExecutor(max_workers=CONCURRENCY) as pool:
            futures = {pool.submit(fetch_single, session, uid): uid for uid in user_ids}
            completed = 0
            for future in as_completed(futures):
                ms = future.result()
                if ms >= 0:
                    latencies.append(ms)
                else:
                    errors += 1
                completed += 1
                if completed % 100 == 0:
                    print(f"  Progress: {completed}/{NUM_REQUESTS} requests completed...")

    if not latencies:
        print("\n  ERROR: No successful responses recorded. Check API connectivity.")
        sys.exit(1)

    latencies.sort()
    n = len(latencies)

    def percentile(sorted_list: List[float], pct: float) -> float:
        idx = max(0, int(pct / 100.0 * len(sorted_list)) - 1)
        return sorted_list[idx]

    p50 = percentile(latencies, 50)
    p90 = percentile(latencies, 90)
    p95 = percentile(latencies, 95)
    p99 = percentile(latencies, 99)
    mean = statistics.mean(latencies)
    stdev = statistics.stdev(latencies) if n > 1 else 0.0
    min_ms = latencies[0]
    max_ms = latencies[-1]

    print("\n" + "=" * 60)
    print("  RESULTS")
    print("=" * 60)
    print(f"  Total requests    : {NUM_REQUESTS}")
    print(f"  Successful        : {n}")
    print(f"  Errors            : {errors}")
    print("-" * 60)
    print(f"  Min latency       : {min_ms:.2f} ms")
    print(f"  Mean latency      : {mean:.2f} ms")
    print(f"  Std deviation     : {stdev:.2f} ms")
    print(f"  P50 (median)      : {p50:.2f} ms")
    print(f"  P90               : {p90:.2f} ms  <- SLA target: < {P90_BUDGET_MS} ms")
    print(f"  P95               : {p95:.2f} ms")
    print(f"  P99               : {p99:.2f} ms")
    print(f"  Max latency       : {max_ms:.2f} ms")
    print("=" * 60)

    sla_pass = p90 <= P90_BUDGET_MS
    if sla_pass:
        print(f"\n  [PASS] P90 SLA: {p90:.2f} ms <= {P90_BUDGET_MS} ms SLA target.")
    else:
        print(f"\n  [FAIL] P90 SLA: {p90:.2f} ms > {P90_BUDGET_MS} ms SLA target.")
        print("      Investigate Redis connection latency or API overhead.")

    print()
    sys.exit(0 if sla_pass else 1)


if __name__ == "__main__":
    run_benchmark()
