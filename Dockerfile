# ─────────────────────────────────────────────────────────────
# Trade System v2 — Multi-Stage Production Dockerfile
# ─────────────────────────────────────────────────────────────
# Build:  docker build -t trade-system .
# Run:    docker compose up -d --build
# ─────────────────────────────────────────────────────────────

# ═════════════════════════════════════════════════════════════
# Stage 1: Builder — Install Python dependencies in isolation
# ═════════════════════════════════════════════════════════════
FROM python:3.11-slim AS builder

WORKDIR /build

# Install build tools for native extension compilation
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --prefix=/install --no-cache-dir -r requirements.txt

# ═════════════════════════════════════════════════════════════
# Stage 2: Runtime — Lean, secure final image
# ═════════════════════════════════════════════════════════════
FROM python:3.11-slim AS runtime

LABEL maintainer="Pratham <support@example.com>"
LABEL description="Trade System v2 — Institutional Algorithmic Trading Platform"

# Runtime-only system packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    sqlite3 \
    tzdata \
    cron \
    curl \
    && rm -rf /var/lib/apt/lists/*

# ─── Timezone — IST is critical for Indian market schedules ──
ENV TZ="Asia/Kolkata"
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# ─── Copy installed Python packages from builder ─────────────
COPY --from=builder /install /usr/local

# ─── Application setup ───────────────────────────────────────
WORKDIR /app

# Python environment
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app/src

# Copy application code (respects .dockerignore)
COPY . /app/

# ─── Security: Non-root user ─────────────────────────────────
RUN groupadd -r appuser && useradd -r -g appuser -d /app appuser

# Create required directories with correct permissions
RUN mkdir -p /app/logs /app/data /app/data/option_chain_data /app/data/backups \
    && chown -R appuser:appuser /app

USER appuser

# ─── Ports ────────────────────────────────────────────────────
# Streamlit (8502) + Health probe (9090)
EXPOSE 8502 9090

# ─── Health check ─────────────────────────────────────────────
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:9090/health || exit 1

# ─── Default entrypoint (overridden by docker-compose) ────────
CMD ["python", "scripts/run_live_trading.py"]
