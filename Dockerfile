# =============================================================================
# Credit Risk Intelligence Platform
#
# Multi-stage build. The builder stage compiles wheels (LightGBM and pyarrow
# both pull a toolchain); the runtime stage installs those wheels into a clean
# slim image, so gcc, g++ and cmake never ship to production. Roughly 1.6 GB
# saved over a single-stage build.
#
#   docker compose up          -> builds artifacts if absent, then serves the UI
# =============================================================================

# ------------------------------- builder ------------------------------------
FROM python:3.11-slim AS builder

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        cmake \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /wheels
COPY requirements.txt .
RUN pip wheel --wheel-dir=/wheels -r requirements.txt


# ------------------------------- runtime ------------------------------------
FROM python:3.11-slim AS runtime

LABEL org.opencontainers.image.title="Credit Risk Intelligence Platform" \
      org.opencontainers.image.description="Explainable, agentic credit-risk decisioning on the Home Credit dataset"

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    RUNNING_IN_DOCKER=1 \
    MPLBACKEND=Agg \
    PYTHONPATH=/app

# libgomp1 is LightGBM's OpenMP runtime - required at inference, not just build.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 \
        curl \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /wheels /wheels
COPY requirements.txt .
RUN pip install --no-index --find-links=/wheels -r requirements.txt \
    && rm -rf /wheels requirements.txt

WORKDIR /app

# Source first, then the entrypoint: application code changes far more often
# than the dependency layers above, which stay cached across rebuilds.
COPY src/ ./src/
COPY app/ ./app/
COPY sql/ ./sql/
COPY evaluation/ ./evaluation/
COPY notebooks/ ./notebooks/
COPY tests/ ./tests/
COPY docker/entrypoint.sh /usr/local/bin/entrypoint.sh

RUN chmod +x /usr/local/bin/entrypoint.sh \
    && mkdir -p /app/data/raw /app/data/processed /app/models /app/reports/figures \
    && useradd --create-home --uid 1000 appuser \
    && chown -R appuser:appuser /app
USER appuser

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=180s --retries=3 \
    CMD curl -fsS http://localhost:8501/_stcore/health || exit 1

ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
CMD ["streamlit", "run", "app/ui.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0", \
     "--server.headless=true", \
     "--browser.gatherUsageStats=false"]
