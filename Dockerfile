# Multi-stage build for Waveseer pattern detection system

# Build stage - installs dependencies and builds package
FROM python:3.10-slim AS builder

# Set working directory for build
WORKDIR /build

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better layer caching
COPY requirements.txt requirements-dev.txt ./

# Install dependencies
RUN pip install --no-cache-dir --user -r requirements.txt \
    && pip install --no-cache-dir --user -r requirements-dev.txt

# Copy package files
COPY pyproject.toml setup.py README.md ./

# Copy source code
COPY wave/ ./wave/

# Build package wheel
RUN pip install --no-cache-dir --user build && \
    python -m build --wheel

# Production stage
FROM python:3.10-slim AS production

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    WAVESEER_MODELS_DIR=/app/models

# Create non-root user for security
RUN groupadd -g 1000 waveseer && \
    useradd -u 1000 -g waveseer -s /bin/bash -m waveseer

# Set working directory
WORKDIR /app

# Create directory structure and set permissions
RUN mkdir -p /app/models /app/data /app/logs \
    && chown -R waveseer:waveseer /app

# Install runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    tini \
    && rm -rf /var/lib/apt/lists/*

# Copy built package from builder stage
COPY --from=builder /build/dist/*.whl /tmp/

# Install the package
RUN pip install --no-cache-dir /tmp/*.whl && rm /tmp/*.whl

# Copy configuration
COPY config.yml /app/

# Port for the API
EXPOSE 9000

# Switch to non-root user
USER waveseer

# Use tini as init system to handle signals properly
ENTRYPOINT ["/usr/bin/tini", "--"]

# Default command: run the API server
CMD ["waveseer-api", "serve", "--host", "0.0.0.0", "--port", "9000", "--workers", "1"]
