# Trading Bot v9.1 Enhanced Dockerfile with Hybrid Ultra-Diagnostics
# Multi-stage build for optimized production image

# Build stage
FROM python:3.11-slim as builder

# Set build arguments
ARG BUILD_DATE
ARG VERSION=9.1
ARG VCS_REF

# Labels for metadata
LABEL maintainer="Trading Bot v9.1 with Hybrid Ultra-Diagnostics" \
      version="${VERSION}" \
      description="Enhanced Trading Bot with Signal Intelligence and Diagnostics" \
      build-date="${BUILD_DATE}" \
      vcs-ref="${VCS_REF}"

# Install system dependencies for building
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    wget \
    curl \
    git \
    pkg-config \
    libffi-dev \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies with diagnostic libraries
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt

# Production stage
FROM python:3.11-slim as production

# Install runtime dependencies including diagnostic tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    wget \
    curl \
    sqlite3 \
    ca-certificates \
    tzdata \
    htop \
    iotop \
    procps \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Create non-root user for security
RUN groupadd -r botuser && useradd -r -g botuser botuser

# Set working directory
WORKDIR /app

# Copy Python packages from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Create necessary directories including diagnostic directories
RUN mkdir -p /app/logs /app/data /app/ml_models /app/backups \
             /app/diagnostics /app/patterns /app/health_reports \
             /app/decision_logs /app/performance_data && \
    chown -R botuser:botuser /app

# Copy application code
COPY --chown=botuser:botuser . .

# Set environment variables including diagnostic settings
ENV PYTHONPATH=/app \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONIOENCODING=utf-8 \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    TZ=UTC \
    BOT_VERSION=9.1 \
    DIAGNOSTICS_ENABLED=true \
    DIAGNOSTICS_LEVEL=FULL \
    PERFORMANCE_MONITORING_ENABLED=true \
    PATTERN_DETECTION_ENABLED=true \
    DIAGNOSTIC_DASHBOARD_ENABLED=true

# Switch to non-root user
USER botuser

# Expose ports (main app + diagnostic dashboard)
EXPOSE 5000 8080

# Enhanced health check with diagnostic endpoints
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD wget --no-verbose --tries=1 --spider http://localhost:5000/health && \
        wget --no-verbose --tries=1 --spider http://localhost:8080/diagnostic-health || exit 1

# Default command
CMD ["python", "main.py"]

# Development stage (optional) with diagnostic tools
FROM production as development

USER root

# Install development and diagnostic tools
RUN pip install --no-cache-dir \
    pytest \
    pytest-asyncio \
    pytest-cov \
    black \
    isort \
    flake8 \
    mypy \
    jupyter \
    matplotlib \
    seaborn \
    plotly \
    dash \
    streamlit

# Install system diagnostic tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    strace \
    tcpdump \
    netstat-nat \
    lsof \
    && rm -rf /var/lib/apt/lists/*

# Switch back to botuser
USER botuser

# Override command for development
CMD ["python", "main.py", "--dev"]

# Diagnostic stage for monitoring and analysis
FROM production as diagnostic

USER root

# Install additional diagnostic and monitoring tools
RUN pip install --no-cache-dir \
    prometheus-client \
    grafana-api \
    influxdb-client \
    psutil \
    memory-profiler \
    py-spy

USER botuser

# Expose additional monitoring ports
EXPOSE 5000 8080 9090 3000

# Command with full diagnostic monitoring
CMD ["python", "main.py", "--diagnostic-mode"]