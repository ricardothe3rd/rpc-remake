# Remake Platform - New App Requirements

**Quick Start**: Everything you need to deploy a new app successfully

---

## Core Requirements (Non-Negotiable)

Every app MUST have these to deploy on Remake Platform:

### 1. Project Structure

```
your-app/
├── Dockerfile              # ✅ REQUIRED - Platform builds from this
├── .dockerignore           # ✅ REQUIRED - Exclude unnecessary files
├── .gitignore              # ✅ REQUIRED - Don't commit secrets
├── README.md               # ✅ REQUIRED - Setup instructions
├── .env.example            # ✅ REQUIRED - Document all env vars
└── backend/
    └── main.py (or server.js)  # ✅ Must have /api/health endpoint
```

### 2. Health Endpoint

**REQUIRED**: Your app MUST respond to health checks

```python
# Python (FastAPI)
@app.get("/api/health")
async def health():
    return {"status": "healthy"}
```

```javascript
// Node.js (Express)
app.get('/api/health', (req, res) => {
  res.json({ status: 'healthy' });
});
```

**Test it locally:**
```bash
curl http://localhost:8080/api/health
# Expected: {"status":"healthy"}
```

### 3. Dockerfile

**REQUIRED**: Must have working Dockerfile

**Minimum requirements:**
- Exposes a port (usually 8080)
- Has HEALTHCHECK
- Runs as non-root user (recommended)
- Uses PORT environment variable

**Template:**
```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt ./
RUN pip install -r requirements.txt

# Copy code
COPY . .

# Non-root user (security)
RUN adduser --disabled-password appuser && \
    chown -R appuser:appuser /app
USER appuser

# Port config
ENV PORT=8080
EXPOSE ${PORT}

# Health check
HEALTHCHECK --interval=10s --timeout=3s --start-period=30s \
    CMD curl -f http://localhost:${PORT}/api/health || exit 1

# Start
CMD ["python", "main.py"]
```

### 4. Environment Variables

**REQUIRED**: No hardcoded secrets, all config externalized

**Create `.env.example`:**
```bash
# .env.example (COMMIT THIS)

# Required
APP_ID=your-app-id
APP_SECRET=generate-with-openssl-rand-hex-32
APPSTORE_URL=https://apps.remake.ai
ALLOWED_ORIGINS=https://your-app.remake.ai

# Optional
PORT=8080
HOST=0.0.0.0
LOG_LEVEL=INFO
```

**Validate at startup:**
```python
# main.py
import os
import sys

# Required vars
APP_SECRET = os.getenv("APP_SECRET")
if not APP_SECRET or len(APP_SECRET) < 32:
    print("ERROR: APP_SECRET must be set (>= 32 chars)")
    sys.exit(1)

APP_ID = os.getenv("APP_ID")
if not APP_ID:
    print("ERROR: APP_ID must be set")
    sys.exit(1)

# ... rest of your app
```

### 5. Dependency Lockfiles

**REQUIRED**: Commit lockfiles for reproducible builds

**For Node.js:**
```bash
# Generate and commit
npm install
git add package-lock.json
git commit -m "chore: add package-lock.json"
```

**For Python:**
```bash
# Pin all versions in requirements.txt
fastapi==0.109.0
uvicorn==0.27.0
# NOT: fastapi>=0.109.0  ❌
```

### 6. Port Configuration

**REQUIRED**: App must listen on configurable port

```python
# Python
PORT = int(os.getenv("PORT", "8080"))
uvicorn.run(app, host="0.0.0.0", port=PORT)
```

```javascript
// Node.js
const PORT = process.env.PORT || 8080;
app.listen(PORT, '0.0.0.0');
```

**IMPORTANT**: Listen on `0.0.0.0` NOT `127.0.0.1` (Docker requirement)

### 7. CORS Configuration

**REQUIRED**: Proper CORS for production

```python
# Python (FastAPI)
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "").split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,  # NOT ["*"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

```javascript
// Node.js (Express)
const cors = require('cors');
const ALLOWED_ORIGINS = process.env.ALLOWED_ORIGINS.split(',');

app.use(cors({
  origin: ALLOWED_ORIGINS,  // NOT '*'
  credentials: true
}));
```

---

## Robot Control App Requirements

If your app controls robots, add these:

### 8. WebSocket Support

**REQUIRED**: Socket.IO or plain WebSocket

```python
# Socket.IO (recommended - same as Appstore)
import socketio

sio = socketio.AsyncServer(
    async_mode='asgi',
    cors_allowed_origins=ALLOWED_ORIGINS
)

# Namespaces
# /sessions/{sessionId}/robot  - For robot connection
# /                            - For UI connection
```

### 9. Three-Phase Connection Protocol

**REQUIRED**: Implement all 3 phases

```python
# Phase 1: Send app_signature (authentication)
await sio.emit('app_signature', {
    'app_signature': hmac_signature,
    'timestamp': timestamp,
    'app_id': APP_ID
})

# Phase 2: Handle setup_app_cmd (create robot app instance)
@sio.on('setup_app_cmd')
async def on_setup_app_cmd(sid, data):
    # Create your robot app instance
    robot_app = MyRobotApp(...)
    robot_app.start()

    # Respond
    await sio.emit('setup_app_response', {
        'session_id': session_id,
        'status': 'ready'
    })

# Phase 3: Handle enable_remote_control_response (activate commands)
@sio.on('enable_remote_control_response')
async def on_enable_control(sid, data):
    if data['status'] == 'enabled':
        # Movement commands now allowed
        pass
```

### 10. Input Validation

**REQUIRED**: Validate all robot commands

```python
# Add safety limits
MAX_LINEAR_VELOCITY = 2.0   # m/s
MAX_ANGULAR_VELOCITY = 3.0  # rad/s

@sio.on('move_cmd')
async def on_move_cmd(sid, data):
    linear_x = data.get('linear_x', 0.0)
    angular_z = data.get('angular_z', 0.0)

    # Validate
    if abs(linear_x) > MAX_LINEAR_VELOCITY:
        await sio.emit('error', {'message': 'Velocity too high'}, room=sid)
        return

    if abs(angular_z) > MAX_ANGULAR_VELOCITY:
        await sio.emit('error', {'message': 'Angular velocity too high'}, room=sid)
        return

    # Send to robot
    # ...
```

---

## Frontend Requirements (If Applicable)

If your app has a React/Vue/etc frontend:

### 11. Build Process

**REQUIRED**: Use multi-stage Docker build

```dockerfile
# Stage 1: Build frontend
FROM node:24-alpine AS frontend-builder

WORKDIR /app/frontend

# Install dependencies
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci --legacy-peer-deps

# Build
COPY frontend/ ./
RUN npm run build

# Stage 2: Backend + static files
FROM python:3.11-slim

WORKDIR /app

# ... backend setup ...

# Copy built frontend
COPY --from=frontend-builder /app/frontend/dist ./static

# Serve static files
# (FastAPI example - mount static directory)
```

### 12. Node Version Match

**REQUIRED**: Node version MUST match everywhere

```json
// package.json
{
  "engines": {
    "node": "24.x",
    "npm": "10.x"
  }
}
```

```dockerfile
# Dockerfile - MUST MATCH package.json
FROM node:24-alpine AS frontend-builder
#         ^^ This number
```

**Check locally:**
```bash
node --version  # Should be v24.x.x
```

---

## Security Requirements

### 13. No Hardcoded Secrets

**REQUIRED**: All secrets from environment

```python
# ❌ BAD
APP_SECRET = "my-secret-key"

# ❌ ALSO BAD
APP_SECRET = os.getenv("APP_SECRET", "default-secret")

# ✅ GOOD
APP_SECRET = os.getenv("APP_SECRET")
if not APP_SECRET:
    raise ValueError("APP_SECRET must be set")
```

### 14. Generate Strong Secrets

**REQUIRED**: Use cryptographically secure secrets

```bash
# Generate 32-character secret
openssl rand -hex 32

# Output: abc123def456...  (64 characters = 32 bytes)
```

### 15. .gitignore

**REQUIRED**: Never commit secrets

```
# .gitignore
.env
*.env
!.env.example
node_modules/
__pycache__/
*.pyc
.vscode/
.DS_Store
```

---

## File Requirements Checklist

Before deploying, ensure you have:

```
your-app/
├── Dockerfile              ✅ Working, with HEALTHCHECK
├── .dockerignore           ✅ Excludes node_modules, .env
├── .gitignore              ✅ Excludes .env, node_modules
├── README.md               ✅ Setup and run instructions
├── .env.example            ✅ All variables documented
├── package.json            ✅ Node version specified (if using Node)
├── package-lock.json       ✅ COMMITTED (if using npm)
├── requirements.txt        ✅ Pinned versions (if using Python)
└── backend/
    ├── main.py             ✅ Has /api/health endpoint
    └── config.py           ✅ Validates environment variables
```

---

## Quick Deployment Steps

Once you have all requirements:

### Step 1: Test Locally

```bash
# Build Docker image
docker build -t my-app:test .

# Run container
docker run -p 8080:8080 \
  -e APP_SECRET="test-secret-32-chars-long-here" \
  -e APP_ID="my-app" \
  -e APPSTORE_URL="https://apps.remake.ai" \
  -e ALLOWED_ORIGINS="http://localhost:3000" \
  my-app:test

# Test health endpoint
curl http://localhost:8080/api/health
```

### Step 2: Deploy via CLI

```bash
# Login
remake login

# Create app (first time only)
remake create

# Set environment variables
remake env:set APP_SECRET=$(openssl rand -hex 32)
remake env:set APP_ID=my-app
remake env:set APPSTORE_URL=https://apps.remake.ai
remake env:set ALLOWED_ORIGINS=https://my-app.remake.ai,https://apps.remake.ai

# Deploy
remake deploy

# Monitor
remake logs --follow
```

### Step 3: Verify Deployment

```bash
# Check status
remake status

# Test health endpoint
curl https://my-app.remake.ai/api/health

# Check logs for errors
remake logs --tail 100 | grep ERROR
```

---

## Common Mistakes to Avoid

### ❌ Don't Do This

1. **Hardcode secrets**
   ```python
   APP_SECRET = "my-secret"  # ❌ Never do this
   ```

2. **Use wildcard CORS in production**
   ```python
   allow_origins=["*"]  # ❌ Security risk
   ```

3. **Listen on 127.0.0.1**
   ```python
   app.listen(PORT, '127.0.0.1')  # ❌ Won't work in Docker
   ```

4. **Forget health check**
   ```dockerfile
   # ❌ No HEALTHCHECK = platform can't monitor
   ```

5. **Use npm install instead of npm ci**
   ```dockerfile
   RUN npm install  # ❌ Non-reproducible
   ```

6. **Node version mismatch**
   ```dockerfile
   FROM node:20-alpine  # ❌ But package-lock.json is Node 24
   ```

7. **Don't commit lockfiles**
   ```bash
   # ❌ package-lock.json not in git
   ```

8. **Unpinned Python dependencies**
   ```
   fastapi>=0.109.0  # ❌ Can break in future
   ```

### ✅ Do This Instead

1. **Use environment variables**
   ```python
   APP_SECRET = os.getenv("APP_SECRET")
   ```

2. **Specific CORS origins**
   ```python
   allow_origins=["https://my-app.remake.ai"]
   ```

3. **Listen on 0.0.0.0**
   ```python
   app.listen(PORT, '0.0.0.0')
   ```

4. **Add health check**
   ```dockerfile
   HEALTHCHECK CMD curl -f http://localhost:${PORT}/api/health || exit 1
   ```

5. **Use npm ci**
   ```dockerfile
   RUN npm ci --legacy-peer-deps
   ```

6. **Match Node versions**
   ```dockerfile
   FROM node:24-alpine  # ✅ Matches package-lock.json
   ```

7. **Commit lockfiles**
   ```bash
   git add package-lock.json requirements.txt
   git commit -m "chore: add lockfiles"
   ```

8. **Pin Python dependencies**
   ```
   fastapi==0.109.0
   ```

---

## Testing Checklist

Before deploying to production:

- [ ] Docker build succeeds locally
- [ ] Health endpoint returns 200 OK
- [ ] App starts without errors
- [ ] Environment variables validated
- [ ] CORS configured correctly
- [ ] WebSocket connects (if applicable)
- [ ] Robot commands work (if applicable)
- [ ] No secrets in git history
- [ ] Lockfiles committed
- [ ] README has setup instructions

---

## Minimum Working Example

Here's the absolute minimum for a deployable app:

**File: main.py**
```python
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import os
import sys

# Validate config
APP_SECRET = os.getenv("APP_SECRET")
if not APP_SECRET or len(APP_SECRET) < 32:
    print("ERROR: APP_SECRET must be set (>= 32 chars)")
    sys.exit(1)

ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "").split(",")
PORT = int(os.getenv("PORT", "8080"))

# Create app
app = FastAPI()

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Health endpoint (REQUIRED)
@app.get("/api/health")
async def health():
    return {"status": "healthy"}

# Your endpoints here
@app.get("/")
async def root():
    return {"message": "Hello Remake Platform!"}

# Start server
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=PORT)
```

**File: Dockerfile**
```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install dependencies
RUN pip install fastapi uvicorn[standard]

# Copy code
COPY main.py .

# Non-root user
RUN adduser --disabled-password appuser && \
    chown -R appuser:appuser /app
USER appuser

# Config
ENV PORT=8080
EXPOSE ${PORT}

# Health check
HEALTHCHECK --interval=10s --timeout=3s --start-period=30s \
    CMD curl -f http://localhost:${PORT}/api/health || exit 1

# Start
CMD ["python", "main.py"]
```

**File: .env.example**
```bash
APP_SECRET=your-secret-here-min-32-chars
APP_ID=my-app
APPSTORE_URL=https://apps.remake.ai
ALLOWED_ORIGINS=https://my-app.remake.ai
PORT=8080
```

**File: .dockerignore**
```
.env
*.pyc
__pycache__/
.git/
.vscode/
README.md
```

**File: .gitignore**
```
.env
__pycache__/
*.pyc
.vscode/
```

**Deploy:**
```bash
docker build -t my-app .
docker run -p 8080:8080 \
  -e APP_SECRET="abc123def456ghi789jkl012mno345pqr" \
  -e APP_ID="my-app" \
  -e APPSTORE_URL="https://apps.remake.ai" \
  -e ALLOWED_ORIGINS="http://localhost:3000" \
  my-app

# Test
curl http://localhost:8080/api/health
```

---

## Summary

**Must Have (Non-Negotiable):**
1. ✅ Dockerfile with HEALTHCHECK
2. ✅ `/api/health` endpoint
3. ✅ No hardcoded secrets
4. ✅ Configurable PORT (env var)
5. ✅ Listen on 0.0.0.0
6. ✅ CORS configured
7. ✅ Lockfiles committed

**For Robot Apps, Also Need:**
8. ✅ WebSocket support
9. ✅ Three-phase protocol
10. ✅ Input validation

**Best Practices:**
11. ✅ Multi-stage build (if frontend)
12. ✅ Non-root user
13. ✅ Config validation at startup
14. ✅ Strong secrets (32+ chars)
15. ✅ .dockerignore and .gitignore

**Test Before Deploy:**
```bash
docker build -t test .
docker run -p 8080:8080 -e APP_SECRET="..." test
curl http://localhost:8080/api/health
```

**Deploy:**
```bash
remake login
remake create
remake env:set APP_SECRET=$(openssl rand -hex 32)
remake deploy
```

**That's it!** Follow these requirements and your app will deploy without issues.

---

**For More Details**: See `DEPLOYMENT_GUIDE.md` for troubleshooting and advanced topics.