# Multi-stage Dockerfile for Remake RPC
# Stage 1: Build React frontend
# Stage 2: Python backend + built frontend

# ============================================================================
# Stage 1: Build Frontend
# ============================================================================
FROM node:24-alpine AS frontend-builder

WORKDIR /app/frontend

# Copy package files
COPY frontend/package*.json ./

# Debug: Show what files we have and Node/npm versions
RUN echo "=== DEBUG INFO ===" && \
    node --version && \
    npm --version && \
    ls -la && \
    cat package.json && \
    echo "=== END DEBUG ==="

# Install dependencies
RUN npm ci --legacy-peer-deps || (echo "npm ci failed!" && cat /root/.npm/_logs/*.log 2>/dev/null && exit 1)

# Copy frontend source
COPY frontend/ ./

# Build production frontend
RUN npm run build

# ============================================================================
# Stage 2: Python Backend
# ============================================================================
FROM python:3.11-slim

WORKDIR /app

# Ensure logs stream in real-time on the platform
ENV PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy Python dependencies
COPY backend/requirements.txt ./

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy backend code
COPY backend/ ./

# Copy built frontend from stage 1
COPY --from=frontend-builder /app/frontend/dist ./static

# Run as non-root user
RUN useradd -m appuser
USER appuser

# Expose port (dynamic via environment variable)
EXPOSE 8080

# Health check — Traefik checks /health
HEALTHCHECK --interval=10s --timeout=3s --start-period=30s \
    CMD curl -f http://localhost:${PORT:-8080}/health || exit 1

# Start server
# Note: PORT is set by Remake Platform, defaults to 8080
CMD ["sh", "-c", "python main.py"]
