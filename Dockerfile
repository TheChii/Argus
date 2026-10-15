# syntax=docker/dockerfile:1
# ==============================================================================
# Argus Quant — Institutional Trading Terminal Container
# Multi-stage production build with C++ OpenMP acceleration and CPU-optimized PyTorch
# ==============================================================================

# ------------------------------------------------------------------------------
# Stage 1: Build & Compilation
# ------------------------------------------------------------------------------
FROM python:3.12-slim-bookworm AS builder

# Install C++ compiler, OpenMP development headers, and build tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy dependency specifications first to leverage Docker layer caching
COPY requirements.txt .

# Install CPU-only PyTorch first to prevent CUDA runtime bloat (~2GB reduction),
# followed by core quantitative, machine learning, and server dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch>=2.0.0 --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements.txt

# Copy source repository
COPY . .

# Compile high-performance C++ feature extraction shared library with OpenMP
RUN make -C src/features clean && \
    make -C src/features CXXFLAGS="-O3 -fopenmp -fPIC -std=c++17 -Wall -Wextra"

# ------------------------------------------------------------------------------
# Stage 2: Production Runtime
# ------------------------------------------------------------------------------
FROM python:3.12-slim-bookworm AS runner

# Install OpenMP runtime library and curl for healthcheck telemetry
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create unprivileged system user and group
RUN groupadd -g 1000 argus && \
    useradd -u 1000 -g argus -s /bin/bash -m argus

WORKDIR /app

# Copy installed Python site-packages and compiled binaries from builder stage
COPY --from=builder /usr/local /usr/local
COPY --from=builder /app /app

# Ensure data directories exist and assign permissions to unprivileged user
RUN mkdir -p data/prices data/macro data/news/archive data/fundamentals data/logs && \
    chown -R argus:argus /app

USER argus

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app \
    DATA_DIR=/app/data

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8501/api/health || exit 1

ENTRYPOINT ["python", "main.py"]
CMD ["dashboard", "--host", "0.0.0.0", "--port", "8501"]
