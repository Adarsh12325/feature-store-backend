# syntax=docker/dockerfile:1
# ============================================================================
# Dockerfile — Feature Store API Service
# ============================================================================
#
# Build strategy:
#   - Use python:3.10-slim as the base to minimise image size (~130 MB)
#     while retaining pip and standard library completeness.
#   - Copy requirements.txt first to exploit Docker layer caching:
#     if only application code changes, the expensive pip install layer
#     is not rebuilt.
#   - The CMD is intentionally left generic; docker-compose.yml overrides
#     it per service (uvicorn for the API, python for the ingestion worker).
#
# Build:
#   docker build -t feature-store-api .
#
# Run (API, standalone):
#   docker run -p 8000:8000 -e REDIS_HOST=localhost feature-store-api
# ============================================================================

FROM python:3.10-slim

# Metadata labels (OCI standard)
LABEL maintainer="Feature Store Team"
LABEL description="Real-time ML Feature Store API powered by FastAPI and Redis"
LABEL version="1.0.0"

# Prevent Python from writing .pyc files and enable unbuffered stdout/stderr
# so log lines appear immediately in docker logs without buffering.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

# Set the working directory inside the container
WORKDIR /app

# ── System dependencies ────────────────────────────────────────────────────────
# curl is needed for the Docker healthcheck on the api-service.
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

# ── Python dependencies ────────────────────────────────────────────────────────
# Copy only the requirements manifest first. Docker will cache this layer
# separately so that changes to application code do not trigger a full
# pip reinstall.
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# ── Application code ───────────────────────────────────────────────────────────
COPY app/ ./app/
COPY scripts/ ./scripts/

# ── Expose API port ────────────────────────────────────────────────────────────
EXPOSE 8000

# ── Default command (overridden by docker-compose per service) ─────────────────
# Runs the FastAPI application via uvicorn with a single worker.
# For production, increase --workers to match the CPU core count.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
