# ─────────────────────────────────────────────────────────────
# Stage 1: Builder — Install dependencies in isolation
# ─────────────────────────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /build

# Install build tools needed for some native python packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --prefix=/install --no-cache-dir -r requirements.txt

# ─────────────────────────────────────────────────────────────
# Stage 2: Runtime — Lean final image
# ─────────────────────────────────────────────────────────────
FROM python:3.11-slim AS runtime

# System-level packages only (runtime, not build)
RUN apt-get update && apt-get install -y --no-install-recommends \
    sqlite3 \
    tzdata \
    cron \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set timezone to IST — critical for Indian market schedules
ENV TZ="Asia/Kolkata"
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# Copy installed Python packages from builder
COPY --from=builder /install /usr/local

# Set working directory
WORKDIR /app

# Set Python environment
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app/src

# Copy application code
COPY . /app/

# Create required directories with correct permissions
RUN mkdir -p /app/logs /app/data /app/data/option_chain_data

# Expose Streamlit port
EXPOSE 8502

# Default command (overridden by docker-compose)
CMD ["python", "scripts/run_live_trading.py"]
