# ───────────────────────────────────────────────────────────────
# PortOps-LLM Dockerfile
# Lightweight Python 3.11-slim image for Hugging Face Spaces.
# Exposes the OpenEnv FastAPI server on port 7860.
# ───────────────────────────────────────────────────────────────

FROM python:3.11-slim

# Metadata labels
LABEL maintainer="PortOps Team" \
      description="PortOps-LLM: Container Relocation Problem OpenEnv" \
      version="1.0.0"

# ── System dependencies ──────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
        curl \
    && rm -rf /var/lib/apt/lists/*

# ── Create non-root user (security best practice) ───────────────
RUN useradd -m -u 1000 appuser

# ── Set working directory ────────────────────────────────────────
WORKDIR /app

# ── Copy and install dependencies first (Docker layer caching) ───
COPY requirements.txt ./
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# ── Copy application source ──────────────────────────────────────
COPY --chown=appuser:appuser . .

# ── Switch to non-root user ──────────────────────────────────────
USER appuser

# ── Expose the OpenEnv / Hugging Face Spaces port ───────────────
EXPOSE 7860

# ── Health check ─────────────────────────────────────────────────
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:7860/health || exit 1

# ── Launch the FastAPI server via Uvicorn ────────────────────────
# Single worker — environment state is in-process.
# For multi-user hackathon evaluation, use a process manager
# or mount state externally. Single-worker ensures consistency.
CMD ["uvicorn", "server.app:app", \
     "--host", "0.0.0.0", \
     "--port", "7860", \
     "--workers", "1", \
     "--log-level", "info", \
     "--access-log"]
