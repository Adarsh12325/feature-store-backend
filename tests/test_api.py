"""
tests/test_api.py
-----------------
HTTP-level integration tests for the Feature Store API endpoints.

These tests use FastAPI's TestClient (backed by httpx) to send real HTTP
requests through the full ASGI stack — including middleware, routing, and
Pydantic validation — but with a FakeRedis instance in place of a live
Redis server.

Test Coverage
-------------
  GET /health                         — Service liveness
  GET /features/{user_id}             — Single user: success (200)
  GET /features/{user_id}             — Single user: not found (404)
  GET /features/{user_id}             — Single user: empty user_id path param
  POST /features/batch                — Batch: all users found (200)
  POST /features/batch                — Batch: mix of found / not-found
  POST /features/batch                — Batch: empty user_ids list (422)
  POST /features/batch                — Batch: missing key in body (422)
  POST /features/batch                — Batch: exceeds max 100 items (422)
  Feature coercion                    — Integers and floats come back typed
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from tests.conftest import KNOWN_USER_ID, SECOND_USER_ID


# ── Health endpoint ───────────────────────────────────────────────────────────


class TestHealthEndpoint:
    def test_health_returns_200(self, client: TestClient):
        response = client.get("/health")
        assert response.status_code == 200

    def test_health_payload_shape(self, client: TestClient):
        data = client.get("/health").json()
        assert "status" in data
        assert "redis_connected" in data
        assert "redis_host" in data
        assert "redis_port" in data

    def test_health_redis_connected(self, client: TestClient):
        data = client.get("/health").json()
        assert data["redis_connected"] is True


# ── Single-user retrieval: GET /features/{user_id} ───────────────────────────


class TestSingleUserRetrieval:

    def test_known_user_returns_200(self, client: TestClient):
        response = client.get(f"/features/{KNOWN_USER_ID}")
        assert response.status_code == 200

    def test_known_user_payload_contains_user_id(self, client: TestClient):
        data = client.get(f"/features/{KNOWN_USER_ID}").json()
        assert data["user_id"] == KNOWN_USER_ID

    def test_known_user_features_not_empty(self, client: TestClient):
        data = client.get(f"/features/{KNOWN_USER_ID}").json()
        assert isinstance(data["features"], dict)
        assert len(data["features"]) > 0

    def test_known_user_age_is_integer(self, client: TestClient):
        """Redis stores everything as strings; the API must coerce age → int."""
        data = client.get(f"/features/{KNOWN_USER_ID}").json()
        assert isinstance(data["features"]["age"], int)
        assert data["features"]["age"] == 34

    def test_known_user_churn_risk_is_float(self, client: TestClient):
        """Coercion: churn_risk_score should arrive as float, not string."""
        data = client.get(f"/features/{KNOWN_USER_ID}").json()
        assert isinstance(data["features"]["churn_risk_score"], float)

    def test_known_user_is_active_is_bool(self, client: TestClient):
        """Coercion: 'True' string in Redis should become Python True."""
        data = client.get(f"/features/{KNOWN_USER_ID}").json()
        assert isinstance(data["features"]["is_active"], bool)
        assert data["features"]["is_active"] is True

    def test_unknown_user_returns_404(self, client: TestClient):
        response = client.get("/features/nonexistent_user_xyz_999")
        assert response.status_code == 404

    def test_unknown_user_error_payload_has_detail(self, client: TestClient):
        data = client.get("/features/nonexistent_user_xyz_999").json()
        assert "detail" in data
        assert "nonexistent_user_xyz_999" in data["detail"]

    def test_second_user_returns_200(self, client: TestClient):
        response = client.get(f"/features/{SECOND_USER_ID}")
        assert response.status_code == 200

    def test_second_user_is_active_false(self, client: TestClient):
        data = client.get(f"/features/{SECOND_USER_ID}").json()
        assert data["features"]["is_active"] is False

    def test_response_content_type_is_json(self, client: TestClient):
        response = client.get(f"/features/{KNOWN_USER_ID}")
        assert "application/json" in response.headers["content-type"]


# ── Batch retrieval: POST /features/batch ────────────────────────────────────


class TestBatchRetrieval:

    def test_both_known_users_returns_200(self, client: TestClient):
        payload = {"user_ids": [KNOWN_USER_ID, SECOND_USER_ID]}
        response = client.post("/features/batch", json=payload)
        assert response.status_code == 200

    def test_batch_response_has_results_list(self, client: TestClient):
        payload = {"user_ids": [KNOWN_USER_ID]}
        data = client.post("/features/batch", json=payload).json()
        assert "results" in data
        assert isinstance(data["results"], list)

    def test_batch_result_count_matches_request(self, client: TestClient):
        payload = {"user_ids": [KNOWN_USER_ID, SECOND_USER_ID]}
        data = client.post("/features/batch", json=payload).json()
        assert len(data["results"]) == 2

    def test_batch_total_requested_field(self, client: TestClient):
        payload = {"user_ids": [KNOWN_USER_ID, SECOND_USER_ID]}
        data = client.post("/features/batch", json=payload).json()
        assert data["total_requested"] == 2

    def test_batch_total_found_field(self, client: TestClient):
        payload = {"user_ids": [KNOWN_USER_ID, SECOND_USER_ID]}
        data = client.post("/features/batch", json=payload).json()
        assert data["total_found"] == 2

    def test_missing_user_in_batch_gets_empty_features(self, client: TestClient):
        """
        A user not in the store must appear in the results with features={}
        rather than causing a 500 or being omitted from the response.
        """
        payload = {"user_ids": [KNOWN_USER_ID, "ghost_user_does_not_exist"]}
        data = client.post("/features/batch", json=payload).json()

        assert data["total_requested"] == 2
        assert data["total_found"] == 1

        # Find the ghost user in results
        ghost = next(
            r for r in data["results"] if r["user_id"] == "ghost_user_does_not_exist"
        )
        assert ghost["features"] == {}

    def test_all_missing_users_returns_200_not_404(self, client: TestClient):
        """Batch endpoint must return 200 even when zero users are found."""
        payload = {"user_ids": ["ghost_a", "ghost_b", "ghost_c"]}
        response = client.post("/features/batch", json=payload)
        assert response.status_code == 200

    def test_all_missing_batch_total_found_is_zero(self, client: TestClient):
        payload = {"user_ids": ["ghost_a", "ghost_b"]}
        data = client.post("/features/batch", json=payload).json()
        assert data["total_found"] == 0

    # ── Validation error cases ────────────────────────────────────────────────

    def test_empty_user_ids_list_returns_422(self, client: TestClient):
        """An empty list violates min_length=1 and must return 422."""
        payload = {"user_ids": []}
        response = client.post("/features/batch", json=payload)
        assert response.status_code == 422

    def test_missing_user_ids_key_returns_422(self, client: TestClient):
        """A body without the user_ids key must return 422."""
        payload = {"wrong_key": ["user_001"]}
        response = client.post("/features/batch", json=payload)
        assert response.status_code == 422

    def test_oversized_batch_returns_422(self, client: TestClient):
        """More than 100 user_ids must be rejected with 422."""
        payload = {"user_ids": [f"user_{i}" for i in range(101)]}
        response = client.post("/features/batch", json=payload)
        assert response.status_code == 422

    def test_non_list_user_ids_returns_422(self, client: TestClient):
        """Passing a string instead of a list must return 422."""
        payload = {"user_ids": "not_a_list"}
        response = client.post("/features/batch", json=payload)
        assert response.status_code == 422

    def test_null_body_returns_422(self, client: TestClient):
        response = client.post(
            "/features/batch",
            content=b"null",
            headers={"Content-Type": "application/json"},
        )
        assert response.status_code == 422

    # ── Feature coercion in batch context ─────────────────────────────────────

    def test_batch_age_is_integer(self, client: TestClient):
        payload = {"user_ids": [KNOWN_USER_ID]}
        data = client.post("/features/batch", json=payload).json()
        found = data["results"][0]["features"]
        assert isinstance(found["age"], int)

    def test_batch_churn_risk_is_float(self, client: TestClient):
        payload = {"user_ids": [KNOWN_USER_ID]}
        data = client.post("/features/batch", json=payload).json()
        found = data["results"][0]["features"]
        assert isinstance(found["churn_risk_score"], float)


# ── OpenAPI / docs endpoints ──────────────────────────────────────────────────


class TestDocumentationEndpoints:
    def test_openapi_schema_accessible(self, client: TestClient):
        response = client.get("/openapi.json")
        assert response.status_code == 200

    def test_swagger_ui_accessible(self, client: TestClient):
        response = client.get("/docs")
        assert response.status_code == 200
