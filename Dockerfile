FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy configuration files and source code
COPY pyproject.toml README.md ./
COPY src/ ./src/

# Install the application
RUN pip install --no-cache-dir -e ".[api,agent]"

# Ensure directories exist
RUN mkdir -p /app/data /app/logs /app/.secrets

# Create non-root user
RUN useradd -m appuser && chown -R appuser:appuser /app
USER appuser

# Expose API port
EXPOSE 8000

# Default command (API server)
CMD ["trade-api"]
