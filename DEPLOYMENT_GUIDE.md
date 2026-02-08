# Remake Platform - App Deployment Guide

**Based on**: remake-rpc deployment experience & lessons learned
**Last Updated**: 2026-02-07
**Applies To**: All Remake Platform apps (CLI & Dashboard deployment)

---

## Table of Contents

1. [Overview](#overview)
2. [Lessons Learned from remake-rpc](#lessons-learned-from-remake-rpc)
3. [Pre-Deployment Checklist](#pre-deployment-checklist)
4. [Docker Configuration Best Practices](#docker-configuration-best-practices)
5. [Environment Variables Setup](#environment-variables-setup)
6. [CLI Deployment Process](#cli-deployment-process)
7. [Dashboard Deployment Process](#dashboard-deployment-process)
8. [Testing & Validation](#testing--validation)
9. [Troubleshooting Common Issues](#troubleshooting-common-issues)
10. [Platform Requirements](#platform-requirements)

---

## Overview

This guide ensures your Remake Platform app deploys successfully without the common issues we encountered with remake-rpc:

- Node version mismatches
- npm ci failures
- Missing package-lock.json
- Docker build cache problems
- Environment variable issues
- Health check failures

---

## Lessons Learned from remake-rpc

### Git Commit History Analysis

Here's what went wrong and how to avoid it:

```
dd15bf5 fix: commit package-lock.json for reproducible npm ci builds
687566e debug: add verbose output to Dockerfile to diagnose npm ci failure
d8afe9a chore: bump version to bust Docker build cache
4c9af51 fix: update Node version from 20 to 24 to match package-lock.json
c89744e Initial commit: Remake RPC app
```

### Issue #1: Node Version Mismatch
**Problem**: Dockerfile used Node 20, but package-lock.json was generated with Node 24
**Error**: `npm ci` failed with lockfile version mismatch
**Fix**: Always match Node versions

### Issue #2: Missing package-lock.json
**Problem**: package-lock.json not committed to git
**Error**: Non-reproducible builds, dependency conflicts
**Fix**: Always commit package-lock.json

### Issue #3: Docker Build Cache
**Problem**: Docker cached old dependencies, causing stale builds
**Error**: Changes not reflected in deployment
**Fix**: Use proper cache busting strategies

---

## Pre-Deployment Checklist

Use this checklist before deploying ANY app to Remake Platform:

### 📦 Repository Setup

- [ ] **package-lock.json committed** (if using npm)
  ```bash
  git add package-lock.json
  git commit -m "chore: add package-lock.json for reproducible builds"
  ```

- [ ] **requirements.txt pinned** (if using Python)
  ```
  # ✅ Good - pinned versions
  fastapi==0.109.0
  uvicorn==0.27.0

  # ❌ Bad - unpinned versions
  fastapi>=0.109.0
  uvicorn>=0.27.0
  ```

- [ ] **Node version documented**
  ```json
  // package.json
  {
    "engines": {
      "node": "24.x",
      "npm": "10.x"
    }
  }
  ```

- [ ] **.dockerignore configured**
  ```
  node_modules/
  __pycache__/
  *.pyc
  .env
  .git/
  *.md
  .vscode/
  .DS_Store
  ```

- [ ] **.gitignore configured**
  ```
  node_modules/
  __pycache__/
  dist/
  build/
  .env
  *.log
  .vscode/
  ```

### 🐳 Docker Configuration

- [ ] **Multi-stage build** (if using frontend + backend)
  ```dockerfile
  # Stage 1: Build frontend
  FROM node:24-alpine AS frontend-builder

  # Stage 2: Backend + built frontend
  FROM python:3.11-slim
  ```

- [ ] **Node version matches package-lock.json**
  ```bash
  # Check lockfileVersion in package-lock.json
  cat frontend/package-lock.json | grep lockfileVersion
  # "lockfileVersion": 3  → Use Node 16+
  # "lockfileVersion": 2  → Use Node 14-15
  ```

- [ ] **Health check configured**
  ```dockerfile
  ENV PORT=8080
  EXPOSE ${PORT}
  HEALTHCHECK --interval=10s --timeout=3s --start-period=30s \
      CMD curl -f http://localhost:${PORT}/api/health || exit 1
  ```

- [ ] **Non-root user** (security best practice)
  ```dockerfile
  RUN adduser --disabled-password --gecos '' appuser && \
      chown -R appuser:appuser /app
  USER appuser
  CMD ["python", "main.py"]
  ```

- [ ] **npm ci instead of npm install** (reproducible builds)
  ```dockerfile
  # ✅ Good - uses lockfile exactly
  RUN npm ci --legacy-peer-deps

  # ❌ Bad - may update dependencies
  RUN npm install
  ```

### 🔒 Security

- [ ] **No hardcoded secrets**
  ```python
  # ❌ Bad
  APP_SECRET = os.getenv("APP_SECRET", "default-secret")

  # ✅ Good
  APP_SECRET = os.getenv("APP_SECRET")
  if not APP_SECRET:
      raise ValueError("APP_SECRET must be set")
  ```

- [ ] **CORS properly configured**
  ```python
  # ❌ Bad - allows any origin
  allow_origins=["*"]

  # ✅ Good - specific origins
  ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "").split(",")
  allow_origins=ALLOWED_ORIGINS
  ```

- [ ] **Input validation on commands**
  ```python
  # ✅ Add bounds checking
  MAX_LINEAR_VELOCITY = 2.0
  MAX_ANGULAR_VELOCITY = 3.0

  if abs(linear_x) > MAX_LINEAR_VELOCITY:
      raise ValueError("Velocity exceeds limit")
  ```

- [ ] **Environment variables validated at startup**
  ```python
  def validate_config():
      errors = []
      if not Config.APP_SECRET or len(Config.APP_SECRET) < 32:
          errors.append("APP_SECRET must be >= 32 characters")
      if not Config.APP_ID:
          errors.append("APP_ID is required")
      if errors:
          for error in errors:
              logger.error(f"Config error: {error}")
          sys.exit(1)

  validate_config()  # Call at startup
  ```

### 📡 Remake Platform Integration

- [ ] **Health endpoint implemented**
  ```python
  @app.get("/api/health")
  async def health():
      return {
          "status": "healthy",
          "timestamp": datetime.utcnow().isoformat()
      }
  ```

- [ ] **WebSocket endpoints correct**
  ```python
  # For Socket.IO
  /sessions/{sessionId}/robot   # Robot connection
  /                             # UI connection

  # For plain WebSocket
  /robot                        # Robot connection
  /ui                           # UI connection
  ```

- [ ] **Three-phase protocol implemented**
  ```python
  # Phase 1: Send app_signature
  # Phase 2: Handle setup_app_cmd
  # Phase 3: Handle enable_remote_control_response
  ```

- [ ] **Environment variables documented**
  ```bash
  # .env.example
  APP_ID=your-app-id
  APP_SECRET=your-secret-key-min-32-chars
  APPSTORE_URL=https://apps.remake.ai
  ALLOWED_ORIGINS=https://your-app.remake.ai
  PORT=8080
  ```

---

## Docker Configuration Best Practices

### 1. Node Version Consistency

**Always match Node version across:**
- Dockerfile
- package.json engines
- Local development
- CI/CD pipeline

```dockerfile
# Check your package-lock.json version first!
# lockfileVersion 3 = Node 16+
# lockfileVersion 2 = Node 14-15

# ✅ Correct - matches package-lock.json
FROM node:24-alpine AS frontend-builder
```

**Verify locally:**
```bash
node --version  # Should match Dockerfile
npm --version
```

### 2. Reproducible npm Builds

```dockerfile
# ✅ Best Practice
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci --legacy-peer-deps --verbose

# ❌ Avoid
COPY frontend/package.json ./
RUN npm install  # No lockfile = non-reproducible
```

**Why `npm ci`?**
- Uses exact versions from package-lock.json
- Fails if package.json and lockfile are out of sync
- Faster than `npm install` in CI/CD
- Removes node_modules before installing (clean slate)

**Why `--legacy-peer-deps`?**
- Required for React 18+ with some older peer dependencies
- Avoids peer dependency conflicts

### 3. Docker Build Cache Strategy

```dockerfile
# ✅ Good - Copy dependencies first (cached layer)
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci --legacy-peer-deps

# Then copy source (changes frequently)
COPY frontend/ ./
RUN npm run build

# ❌ Bad - Rebuilds dependencies on any file change
COPY frontend/ ./
RUN npm ci --legacy-peer-deps && npm run build
```

**Manual cache busting if needed:**
```bash
# Option 1: No cache
docker build --no-cache -t my-app .

# Option 2: Bust specific stage
docker build --build-arg CACHEBUST=$(date +%s) -t my-app .
```

### 4. Health Check Configuration

```dockerfile
# ✅ Correct - Uses ENV variable
ENV PORT=8080
EXPOSE ${PORT}

HEALTHCHECK --interval=10s \
            --timeout=3s \
            --start-period=30s \
            --retries=3 \
    CMD curl -f http://localhost:${PORT}/api/health || exit 1
```

**Health Check Parameters:**
- `--interval=10s`: Check every 10 seconds
- `--timeout=3s`: Fail if no response in 3 seconds
- `--start-period=30s`: Grace period for app startup
- `--retries=3`: Fail after 3 consecutive failures

### 5. Multi-Stage Build Template

```dockerfile
# ==================================
# Stage 1: Build Frontend
# ==================================
FROM node:24-alpine AS frontend-builder

WORKDIR /app/frontend

# Install dependencies (cached layer)
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci --legacy-peer-deps --verbose

# Build frontend
COPY frontend/ ./
RUN npm run build

# ==================================
# Stage 2: Backend + Static Files
# ==================================
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies (cached layer)
COPY backend/requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy backend code
COPY backend/ ./

# Copy built frontend from stage 1
COPY --from=frontend-builder /app/frontend/dist ./static

# Create non-root user
RUN adduser --disabled-password --gecos '' appuser && \
    chown -R appuser:appuser /app

USER appuser

# Configuration
ENV PORT=8080
EXPOSE ${PORT}

# Health check
HEALTHCHECK --interval=10s --timeout=3s --start-period=30s \
    CMD curl -f http://localhost:${PORT}/api/health || exit 1

# Start application
CMD ["python", "main.py"]
```

---

## Environment Variables Setup

### Required Variables

Every Remake Platform app needs these:

```bash
# .env (DO NOT COMMIT)
APP_ID=my-robot-app
APP_SECRET=abc123def456ghi789jkl012mno345pq  # Min 32 chars
APPSTORE_URL=https://apps.remake.ai
ALLOWED_ORIGINS=https://my-robot-app.remake.ai,https://apps.remake.ai
PORT=8080
```

### .env.example Template

```bash
# .env.example (COMMIT THIS)
# Copy this to .env and fill in your values

# ==================================
# Required Configuration
# ==================================

# Your app identifier (must match Appstore registration)
APP_ID=your-app-id

# HMAC secret for signature verification (min 32 characters)
# Generate with: openssl rand -hex 32
APP_SECRET=your-secret-key-here

# Appstore backend URL
APPSTORE_URL=https://apps.remake.ai

# ==================================
# CORS Configuration
# ==================================

# Comma-separated list of allowed origins
# Development: http://localhost:3000,http://localhost:5173
# Production: https://your-app.remake.ai,https://apps.remake.ai
ALLOWED_ORIGINS=http://localhost:3000

# ==================================
# Server Configuration
# ==================================

# Server bind address (0.0.0.0 for Docker, 127.0.0.1 for local)
HOST=0.0.0.0

# Server port
PORT=8080

# Session timeout in seconds
SESSION_TIMEOUT=3600

# ==================================
# Optional Configuration
# ==================================

# Log level (DEBUG, INFO, WARNING, ERROR)
LOG_LEVEL=INFO

# Sentry DSN for error tracking (optional)
# SENTRY_DSN=https://...@sentry.io/...
```

### Generating Secure Secrets

```bash
# Generate 32-character secret
openssl rand -hex 32

# Or use Python
python -c "import secrets; print(secrets.token_hex(32))"

# Or use Node
node -e "console.log(require('crypto').randomBytes(32).toString('hex'))"
```

### Config Validation Code

```python
# backend/config.py
import os
import sys
import logging
from typing import List

logger = logging.getLogger(__name__)

class Config:
    """Application configuration with validation"""

    # Required
    APP_ID: str = os.getenv("APP_ID", "")
    APP_SECRET: str = os.getenv("APP_SECRET", "")
    APPSTORE_URL: str = os.getenv("APPSTORE_URL", "")

    # CORS
    ALLOWED_ORIGINS_STR: str = os.getenv("ALLOWED_ORIGINS", "")
    ALLOWED_ORIGINS: List[str] = [
        origin.strip()
        for origin in ALLOWED_ORIGINS_STR.split(",")
        if origin.strip()
    ]

    # Server
    HOST: str = os.getenv("HOST", "0.0.0.0")
    PORT: int = int(os.getenv("PORT", "8080"))

    # Optional
    SESSION_TIMEOUT: int = int(os.getenv("SESSION_TIMEOUT", "3600"))
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")

    @classmethod
    def validate(cls) -> None:
        """Validate configuration at startup"""
        errors = []

        # Required fields
        if not cls.APP_ID:
            errors.append("APP_ID environment variable is required")

        if not cls.APP_SECRET:
            errors.append("APP_SECRET environment variable is required")
        elif len(cls.APP_SECRET) < 32:
            errors.append("APP_SECRET must be at least 32 characters")

        if not cls.APPSTORE_URL:
            errors.append("APPSTORE_URL environment variable is required")
        elif not cls.APPSTORE_URL.startswith(("http://", "https://")):
            errors.append("APPSTORE_URL must be a valid HTTP(S) URL")

        if not cls.ALLOWED_ORIGINS:
            errors.append("ALLOWED_ORIGINS must be configured (comma-separated list)")

        # Log errors and exit
        if errors:
            logger.error("Configuration validation failed:")
            for error in errors:
                logger.error(f"  - {error}")
            sys.exit(1)

        logger.info("Configuration validated successfully")

# Call at startup (in main.py)
if __name__ == "__main__":
    Config.validate()
    # ... rest of startup
```

---

## CLI Deployment Process

### 1. Install Remake CLI

```bash
# Install globally
npm install -g @remake/cli

# Or use locally
npx @remake/cli

# Verify installation
remake --version
```

### 2. Login to Remake Platform

```bash
remake login

# You'll be prompted for:
# - Email
# - Password
# - 2FA code (if enabled)
```

### 3. Create App (First Time)

```bash
# Create new app on platform
remake create

# You'll be prompted for:
# - App name: my-robot-app
# - Description: Robot control application
# - App type: robot-control
```

**This generates:**
- `APP_ID` (save this!)
- Initial configuration on platform

### 4. Configure Environment Variables

```bash
# Set required environment variables
remake env:set APP_SECRET=$(openssl rand -hex 32)
remake env:set APPSTORE_URL=https://apps.remake.ai
remake env:set ALLOWED_ORIGINS=https://my-robot-app.remake.ai,https://apps.remake.ai

# View all env vars
remake env:list

# View specific env var (value hidden)
remake env:get APP_SECRET
```

### 5. Deploy from Local Files

```bash
# Deploy current directory
remake deploy

# Deploy specific directory
remake deploy --path ./my-app

# Deploy with specific Dockerfile
remake deploy --dockerfile ./Dockerfile.prod

# Deploy with build args
remake deploy --build-arg NODE_ENV=production
```

**Deployment Process:**
1. CLI packages your code
2. Uploads to Remake build service
3. Build service runs Docker build
4. Pushes image to registry
5. Deploys to platform
6. Runs health checks

### 6. Deploy from Git Repository

```bash
# Deploy from GitHub
remake deploy:git \
  --repo https://github.com/username/my-robot-app \
  --branch main

# Deploy specific commit
remake deploy:git \
  --repo https://github.com/username/my-robot-app \
  --commit abc123

# Deploy with submodules
remake deploy:git \
  --repo https://github.com/username/my-robot-app \
  --branch main \
  --submodules
```

### 7. Monitor Deployment

```bash
# View deployment status
remake status

# View real-time logs
remake logs --follow

# View last 100 lines
remake logs --tail 100

# View logs for specific session
remake logs --session abc-123-def

# View build logs
remake logs:build
```

### 8. Scale Application

```bash
# Scale to 2 instances (if supported)
remake scale 2

# Scale to 0 (suspend)
remake scale 0

# Resume (scale to 1)
remake scale 1
```

### 9. Manage App Lifecycle

```bash
# Restart app
remake restart

# Stop app
remake stop

# Start app
remake start

# Delete app (WARNING: Permanent!)
remake destroy
```

---

## Dashboard Deployment Process

### 1. Access Remake Dashboard

Navigate to: `https://dashboard.remake.ai`

Login with your credentials.

### 2. Create New App

1. Click **"Create New App"**
2. Fill in details:
   - **App Name**: my-robot-app
   - **Description**: Robot control application
   - **App Type**: robot-control
   - **Icon**: Upload app icon (PNG, 512x512)
3. Click **"Create App"**

### 3. Configure App Settings

#### General Settings
- **App ID**: `my-robot-app` (auto-generated)
- **Display Name**: My Robot App
- **Version**: 1.0.0
- **Visibility**: Private / Public

#### Environment Variables
Click **"Environment"** tab:

| Variable | Value | Type |
|----------|-------|------|
| APP_SECRET | `******` | Secret |
| APPSTORE_URL | `https://apps.remake.ai` | Config |
| ALLOWED_ORIGINS | `https://my-robot-app.remake.ai` | Config |
| PORT | `8080` | Config |

Click **"Add Variable"** for each.

#### Deployment Settings
- **Build Method**: Docker
- **Dockerfile Path**: `./Dockerfile`
- **Build Args**: (optional)
  ```
  NODE_ENV=production
  ```
- **Health Check Path**: `/api/health`
- **Port**: `8080`

### 4. Deploy from Git

#### Connect Repository
1. Click **"Deployment"** tab
2. Click **"Connect Git Repository"**
3. Choose provider: GitHub / GitLab / Bitbucket
4. Authorize Remake Platform
5. Select repository: `username/my-robot-app`
6. Select branch: `main`

#### Configure Auto-Deploy
- **Auto-deploy**: Enabled
- **Deploy on push**: `main` branch only
- **Deploy on PR**: Disabled (or enable for preview)
- **Build notifications**: Email / Slack

#### Manual Deploy
1. Click **"Deploy Now"**
2. Select commit or use latest
3. Click **"Start Deployment"**

### 5. Monitor Deployment

Dashboard shows:
- **Build Status**: Building / Success / Failed
- **Build Logs**: Real-time streaming
- **Deployment Time**: Duration
- **Health Status**: Healthy / Unhealthy

### 6. View Application

Once deployed:
- **URL**: `https://my-robot-app.remake.ai`
- **Status**: Running / Stopped / Error
- **Uptime**: Last 30 days
- **Metrics**: CPU, Memory, Network

### 7. Manage Scaling

#### Resources
- **CPU**: 0.5 vCPU (adjust as needed)
- **Memory**: 512 MB (adjust as needed)
- **Instances**: 1 (for single-instance apps)

#### Auto-Scaling (if available)
- **Min Instances**: 1
- **Max Instances**: 3
- **Scale on**: CPU > 80% for 5 minutes

### 8. Configure Domains

#### Custom Domain
1. Click **"Domains"** tab
2. Click **"Add Custom Domain"**
3. Enter: `robot.mycompany.com`
4. Add DNS records (shown in dashboard):
   ```
   CNAME  robot  my-robot-app.remake.ai
   ```
5. Click **"Verify Domain"**
6. SSL certificate auto-provisioned

### 9. View Logs & Metrics

#### Logs
- **Application Logs**: stdout/stderr
- **Access Logs**: HTTP requests
- **Error Logs**: Exceptions and errors
- **Filters**: Level, timestamp, search

#### Metrics
- **Request Rate**: Requests per second
- **Response Time**: P50, P95, P99
- **Error Rate**: 4xx, 5xx errors
- **Resource Usage**: CPU, Memory

### 10. Configure Alerts

1. Click **"Alerts"** tab
2. Click **"Add Alert"**
3. Configure:
   - **Metric**: Response time > 1s
   - **Duration**: For 5 minutes
   - **Notify**: Email / Slack / PagerDuty

---

## Testing & Validation

### Pre-Deployment Testing

#### 1. Local Docker Build Test

```bash
# Build Docker image locally
docker build -t my-app:test .

# Check image size
docker images my-app:test

# Run container locally
docker run -p 8080:8080 \
  -e APP_SECRET="test-secret-32-characters-long" \
  -e APP_ID="my-app" \
  -e APPSTORE_URL="https://apps.remake.ai" \
  -e ALLOWED_ORIGINS="http://localhost:3000" \
  my-app:test

# Test health endpoint
curl http://localhost:8080/api/health
```

#### 2. Health Check Test

```bash
# Check if health endpoint returns 200 OK
curl -f http://localhost:8080/api/health || echo "Health check failed!"

# Check response format
curl -s http://localhost:8080/api/health | jq
# Expected: {"status": "healthy", "timestamp": "..."}
```

#### 3. WebSocket Connection Test

```javascript
// test-connection.js
const io = require('socket.io-client');

const socket = io('http://localhost:8080', {
  transports: ['websocket']
});

socket.on('connect', () => {
  console.log('✅ Connected to WebSocket');

  socket.emit('join_session', { session_id: 'test-123' });
});

socket.on('connect_error', (error) => {
  console.error('❌ Connection failed:', error);
});
```

#### 4. Environment Variable Test

```python
# test_config.py
import pytest
import os

def test_required_env_vars():
    """Test that all required env vars are set"""
    required = ['APP_SECRET', 'APP_ID', 'APPSTORE_URL', 'ALLOWED_ORIGINS']

    for var in required:
        assert os.getenv(var), f"{var} is not set"

def test_app_secret_length():
    """Test that APP_SECRET is at least 32 characters"""
    secret = os.getenv('APP_SECRET')
    assert len(secret) >= 32, "APP_SECRET must be >= 32 characters"
```

### Post-Deployment Validation

#### 1. Deployment Checklist

- [ ] **App is accessible** at `https://your-app.remake.ai`
- [ ] **Health check passes** (`/api/health` returns 200)
- [ ] **WebSocket connects** successfully
- [ ] **Static files load** (frontend UI visible)
- [ ] **Environment variables** are set correctly
- [ ] **Logs are streaming** in dashboard
- [ ] **Metrics are collecting** (CPU, memory, requests)

#### 2. Robot Connection Test

```bash
# Test with real robot (ROS2 + kaiaai)
ros2 launch turtlebot3_gazebo turtlebot3_world.launch.py

# Run robot client
python -m kaiaai.appstore_robot_client

# Check logs for connection
remake logs --follow | grep "Robot Connected"
```

#### 3. End-to-End Test

1. Open app UI: `https://your-app.remake.ai`
2. Launch app from Appstore dashboard
3. Verify three-phase connection:
   - Phase 1: Signature sent ✓
   - Phase 2: Setup complete ✓
   - Phase 3: Remote control enabled ✓
4. Send movement command (twist)
5. Verify robot receives command
6. Check sensor data (LiDAR, battery, pose)
7. Verify object detection works

#### 4. Load Test (Optional)

```bash
# Install k6
brew install k6

# Create load test script
cat > load-test.js << 'EOF'
import http from 'k6/http';
import { check } from 'k6';

export let options = {
  vus: 10,  // 10 virtual users
  duration: '30s',
};

export default function () {
  let res = http.get('https://your-app.remake.ai/api/health');
  check(res, {
    'status is 200': (r) => r.status === 200,
  });
}
EOF

# Run load test
k6 run load-test.js
```

---

## Troubleshooting Common Issues

### Issue: Docker Build Fails - npm ci

**Error:**
```
npm ERR! `npm ci` can only install packages when your package.json and package-lock.json or npm-shrinkwrap.json are in sync
```

**Causes:**
1. package-lock.json not committed
2. package.json modified without running `npm install`
3. Node version mismatch

**Fix:**
```bash
# Regenerate package-lock.json
cd frontend
rm package-lock.json
npm install

# Commit it
git add package-lock.json
git commit -m "fix: regenerate package-lock.json"

# Verify Node version matches Dockerfile
node --version  # Should be 24.x
```

---

### Issue: Docker Build Fails - Node Version

**Error:**
```
npm ERR! Unsupported engine
npm ERR! Required: {"node":"24.x"}
npm ERR! Actual: {"node":"v20.11.0"}
```

**Fix:**
```dockerfile
# Update Dockerfile to match package.json engines
FROM node:24-alpine AS frontend-builder
#         ^^ Update this
```

---

### Issue: Health Check Failing

**Error:**
```
Health check failed: curl: (7) Failed to connect to localhost port 8080
```

**Causes:**
1. App not listening on `0.0.0.0` (listening on `127.0.0.1`)
2. Wrong port in health check
3. App crashed before health check started

**Fix:**
```python
# Ensure app binds to 0.0.0.0 in Docker
uvicorn.run(
    app,
    host="0.0.0.0",  # Not "127.0.0.1"
    port=int(os.getenv("PORT", "8080"))
)
```

```dockerfile
# Ensure PORT matches
ENV PORT=8080
HEALTHCHECK CMD curl -f http://localhost:${PORT}/api/health || exit 1
```

---

### Issue: CORS Errors in Browser

**Error:**
```
Access to XMLHttpRequest blocked by CORS policy:
No 'Access-Control-Allow-Origin' header
```

**Fix:**
```python
# Add your production URL to ALLOWED_ORIGINS
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "").split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,  # Not ["*"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

```bash
# Set in platform
remake env:set ALLOWED_ORIGINS=https://your-app.remake.ai,https://apps.remake.ai
```

---

### Issue: WebSocket Connection Refused

**Error:**
```
WebSocket connection to 'wss://your-app.remake.ai/socket.io/' failed
```

**Causes:**
1. CORS not configured for Socket.IO
2. App not listening on correct path
3. Traefik not configured for WebSocket upgrade

**Fix:**
```python
# Ensure Socket.IO CORS matches
sio = socketio.AsyncServer(
    async_mode='asgi',
    cors_allowed_origins=ALLOWED_ORIGINS  # Not "*"
)
```

```javascript
// Frontend: Use same origin
const socket = io('/', {  // Not 'http://localhost:8080'
  transports: ['websocket']
});
```

---

### Issue: Environment Variable Not Set

**Error:**
```
ValueError: APP_SECRET environment variable must be set
```

**Fix:**
```bash
# CLI: Set env var
remake env:set APP_SECRET=$(openssl rand -hex 32)

# Dashboard: Add in Environment tab
APP_SECRET = [paste 32+ char secret]

# Verify
remake env:list
```

---

### Issue: Docker Cache Not Busting

**Error:**
Changes in code not reflected in deployment

**Fix:**
```bash
# Force rebuild without cache
remake deploy --no-cache

# Or locally
docker build --no-cache -t my-app .
```

---

### Issue: App Crashes on Startup

**Check logs:**
```bash
# CLI
remake logs --tail 100

# Dashboard
Deployment → Logs → Application Logs
```

**Common causes:**
1. Missing environment variables
2. Port already in use
3. Database connection failed
4. Import errors

**Fix:**
Add better error logging at startup:
```python
if __name__ == "__main__":
    try:
        Config.validate()
        logger.info("Starting application...")
        uvicorn.run(app, host="0.0.0.0", port=Config.PORT)
    except Exception as e:
        logger.error(f"Failed to start application: {e}", exc_info=True)
        sys.exit(1)
```

---

### Issue: Large Image Size

**Error:**
Docker image is 2+ GB

**Fix:**
```dockerfile
# Use slim/alpine base images
FROM python:3.11-slim  # Not python:3.11
FROM node:24-alpine    # Not node:24

# Clean up in same layer
RUN apt-get update && apt-get install -y curl && \
    rm -rf /var/lib/apt/lists/*

# Use multi-stage build to exclude build tools
FROM node:24-alpine AS builder
# ... build frontend ...

FROM python:3.11-slim
COPY --from=builder /app/frontend/dist ./static
# Build tools not included in final image
```

---

## Platform Requirements

### Mandatory Requirements

All Remake Platform apps MUST have:

1. **Health Endpoint**: `GET /api/health` returns `{"status": "healthy"}`
2. **Docker Support**: Valid `Dockerfile` in repository root
3. **Environment Variables**: No hardcoded secrets, all config externalized
4. **Port Configuration**: Configurable via `PORT` env var
5. **CORS Configuration**: Proper CORS setup for production domains

### Recommended

1. **Multi-stage Build**: Separate build and runtime stages
2. **Non-root User**: Don't run as root in container
3. **Graceful Shutdown**: Handle SIGTERM signal
4. **Structured Logging**: JSON logs for easy parsing
5. **Metrics Endpoint**: `/metrics` for Prometheus (optional)

### Robot Control Apps

Apps that control robots MUST also have:

1. **Three-Phase Protocol**: Implement full three-phase connection
2. **WebSocket Support**: Socket.IO or plain WebSocket
3. **Message Format**: Follow standard message formats
4. **Input Validation**: Bounds checking on movement commands
5. **Session Management**: Track robot connections per session

### Performance Requirements

1. **Startup Time**: < 30 seconds
2. **Health Check**: < 1 second response time
3. **Memory Usage**: < 1 GB per instance (typical)
4. **CPU Usage**: < 1 vCPU per instance (typical)

---

## Quick Reference

### Common Commands

```bash
# CLI
remake login
remake create
remake deploy
remake logs --follow
remake env:set KEY=value
remake scale 1
remake restart

# Docker
docker build -t my-app .
docker run -p 8080:8080 my-app
docker logs -f container_id

# Git
git add package-lock.json requirements.txt
git commit -m "chore: add lockfiles"
git push origin main

# Testing
curl -f http://localhost:8080/api/health
npm test
pytest

# Debugging
remake logs --tail 100 | grep ERROR
docker exec -it container_id /bin/sh
```

### File Checklist

```
your-app/
├── Dockerfile              ✓ Multi-stage, health check, non-root
├── .dockerignore           ✓ Exclude node_modules, .env
├── .gitignore              ✓ Exclude node_modules, .env
├── README.md               ✓ Setup and deployment instructions
├── .env.example            ✓ All required variables documented
├── package.json            ✓ Engines specified
├── package-lock.json       ✓ COMMITTED to git
├── requirements.txt        ✓ Pinned versions
└── backend/
    ├── main.py             ✓ Health endpoint, config validation
    └── config.py           ✓ Environment variable validation
```

---

## Summary

**Before Deployment:**
1. ✅ Commit `package-lock.json` and pin dependency versions
2. ✅ Match Node version in Dockerfile to package.json
3. ✅ Use `npm ci --legacy-peer-deps` not `npm install`
4. ✅ Configure health check with correct port
5. ✅ Validate all environment variables at startup
6. ✅ Remove hardcoded secrets
7. ✅ Configure CORS properly
8. ✅ Add input validation on commands
9. ✅ Test Docker build locally first
10. ✅ Review ANALYSIS_REPORT.md for security issues

**Deploy via CLI:**
```bash
remake login
remake create
remake env:set APP_SECRET=$(openssl rand -hex 32)
remake env:set APPSTORE_URL=https://apps.remake.ai
remake env:set ALLOWED_ORIGINS=https://your-app.remake.ai
remake deploy
```

**Deploy via Dashboard:**
1. Create app → Configure → Connect Git → Deploy

**After Deployment:**
1. ✅ Test health endpoint
2. ✅ Test WebSocket connection
3. ✅ Test with real robot
4. ✅ Monitor logs for errors
5. ✅ Check metrics (CPU, memory)

---

**For Questions**: Refer to `ANALYSIS_REPORT.md` for detailed architecture review

**Last Updated**: 2026-02-07 based on remake-rpc deployment experience