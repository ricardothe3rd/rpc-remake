# Remake RPC - Implementation Plan

**Status**: Ready to deploy
**Issue**: App configured with wrong GitHub repository
**Date**: February 8, 2026

---

## Executive Summary

The remake-rpc app is currently broken in production because it's configured to deploy from the **wrong GitHub repository** (ricardothe3rd/remake-spin-the-robot instead of ricardothe3rd/remake-rpc). This causes a 502 Bad Gateway error.

**Solution**: Delete the misconfigured app, recreate it with the correct repository, and deploy.

---

## What is Remake RPC?

Remake RPC is an advanced robot control application with:

- **Object Detection**: RANSAC V4 + ML 1D CNN for line/corner/arc detection
- **Chat Interface**: AppstoreBridge integration for dashboard commands
- **Real-Time Sensors**: LiDAR, pose, battery, map, camera visualization
- **Modern Stack**: FastAPI + Socket.IO backend, React + TypeScript + Vite frontend
- **Three-Phase Protocol**: Full Remake Platform app launch protocol implementation

**Repository**: https://github.com/ricardothe3rd/remake-rpc

---

## The Problem

```
Current Configuration (WRONG):
  App Name:     remake-rpc
  Repository:   ricardothe3rd/remake-spin-the-robot  ❌
  Status:       502 Bad Gateway

Correct Configuration:
  App Name:     remake-rpc
  Repository:   ricardothe3rd/remake-rpc  ✅
  Status:       Should work once fixed
```

The platform is trying to run spin-the-robot code as if it were remake-rpc, which fails.

---

## Implementation Steps

### Phase 1: Commit Local Changes ✅

The local repository has uncommitted work that needs to be pushed:

**Modified files:**
- `Dockerfile` - Health check and build improvements
- `README.md` - Updated deployment instructions
- `backend/main.py` - Configuration updates
- `frontend/vite.config.ts` - Production settings

**New documentation:**
- `ANALYSIS_REPORT.md` - Technical analysis
- `APP_REQUIREMENTS.md` - Platform requirements
- `DEPLOYMENT_GUIDE.md` - Deployment instructions
- `IMPLEMENTATION_PLAN.md` - This file

**Commands:**
```bash
cd "C:\Users\Ricardo\Documents\Remake Ai\remake-rpc"

# Stage all changes
git add .

# Commit with descriptive message
git commit -m "docs: add deployment guides and update app configuration

- Add comprehensive deployment guide
- Add app requirements documentation
- Add analysis report
- Add implementation plan
- Update Dockerfile with health check improvements
- Update README with deployment instructions
- Update vite config for production

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"

# Push to GitHub
git push origin main
```

**Verification:**
```bash
# Check GitHub to confirm push
git log origin/main -1
```

---

### Phase 2: Delete Broken App ✅

Remove the misconfigured app from the platform:

**Command:**
```bash
remake destroy remake-rpc
```

**Expected Output:**
```
Destroying app remake-rpc...
App destroyed successfully.
```

**Why this is necessary:**
The app is configured with the wrong repository. Deleting and recreating ensures a clean slate.

---

### Phase 3: Create New App ✅

Create the app with the correct repository:

**Command:**
```bash
remake create remake-rpc --github ricardothe3rd/remake-rpc
```

**Expected Output:**
```
Creating app remake-rpc...
App created successfully.
App ID: [new-app-id]
```

**Verification:**
```bash
remake status remake-rpc
```

Should show:
```
Name: remake-rpc
Repository: ricardothe3rd/remake-rpc  ✅ CORRECT
State: created
```

---

### Phase 4: Environment Variables ✅

The platform should auto-detect environment variables from `.env.example`:

**Auto-detected variables:**
```bash
APP_SECRET=<64-char-hex-generated-automatically>
APP_ID=remake-rpc-001
APPSTORE_URL=https://apps.remake.ai
ALLOWED_ORIGINS=*
HOST=0.0.0.0
PORT=8080
SESSION_TIMEOUT=3600
```

**Manual override (if needed):**
```bash
# Only if auto-detection fails
remake env:set APP_SECRET=$(python3 -c "import secrets; print(secrets.token_hex(32))")
remake env:set APP_ID=remake-rpc-001
remake env:set APPSTORE_URL=https://apps.remake.ai
remake env:set ALLOWED_ORIGINS=https://remake-rpc.apps.remake.ai,https://apps.remake.ai
```

**Verification:**
```bash
remake env remake-rpc
```

---

### Phase 5: Deploy ✅

Deploy the application:

**Command:**
```bash
remake deploy remake-rpc
```

**Monitor deployment:**
```bash
# Watch live logs
remake logs remake-rpc --follow

# Or check build logs
remake logs remake-rpc --build
```

**Expected build output:**
```
Building Docker image...
Stage 1: Building frontend (Node 24)
  - npm ci --legacy-peer-deps
  - npm run build
  - dist/ created successfully
Stage 2: Building backend (Python 3.11)
  - Installing requirements.txt
  - Copying backend code
  - Copying frontend dist/ to static/
Build completed successfully
Starting container...
Health check passed
Deployment successful
```

**Typical deployment time:** 3-5 minutes

---

### Phase 6: Verify Deployment ✅

Test that the app is working:

**1. Check status:**
```bash
remake status remake-rpc
```

Expected:
```
Name: remake-rpc
State: running  ✅
URL: https://remake-rpc.apps.remake.ai
Repository: ricardothe3rd/remake-rpc
```

**2. Test health endpoint:**
```bash
curl https://remake-rpc.apps.remake.ai/api/health
```

Expected response:
```json
{"status":"healthy"}
```

**3. Test frontend loads:**
```bash
curl -I https://remake-rpc.apps.remake.ai/
```

Expected:
```
HTTP/2 200 OK
content-type: text/html
```

**4. Check for errors:**
```bash
remake logs remake-rpc --tail 100 | grep ERROR
```

Should return no errors (or only benign warnings).

---

### Phase 7: Appstore Database Registration

For robots to connect to this app, it must be registered in the appstore database.

**SQL Command:**
```sql
-- Connect to appstore database
-- Then run:

INSERT INTO robot_apps (
  app_id,
  app_secret,
  app_name,
  app_ws_url,
  developer_id,
  status
) VALUES (
  'remake-rpc-001',                                    -- Must match APP_ID env var
  '<GET-FROM-ENV-VARS>',                               -- Must match APP_SECRET exactly
  'Remake RPC',
  'https://remake-rpc.apps.remake.ai',                 -- URL from deployment
  (SELECT id FROM users WHERE email = 'your-email'),   -- Your developer ID
  'active'
);
```

**CRITICAL:** The `app_secret` in the database MUST match the `APP_SECRET` environment variable exactly. This is used for HMAC signature verification during the three-phase protocol.

**How to get APP_SECRET:**
```bash
# Get the auto-generated secret
remake env remake-rpc | grep APP_SECRET
```

**Verification:**
```sql
SELECT app_id, app_name, app_ws_url, status
FROM robot_apps
WHERE app_id = 'remake-rpc-001';
```

Expected:
```
app_id          | remake-rpc-001
app_name        | Remake RPC
app_ws_url      | https://remake-rpc.apps.remake.ai
status          | active
```

---

## Technical Architecture

### Build Process

**Multi-stage Dockerfile:**

```
Stage 1: Frontend Builder
  Base: node:24-alpine
  Process:
    1. Copy package.json, package-lock.json
    2. npm ci --legacy-peer-deps
    3. Copy frontend source
    4. npm run build
    5. Output: dist/ directory

Stage 2: Backend + Static Files
  Base: python:3.11-slim
  Process:
    1. Install curl (for health checks)
    2. Install Python dependencies (requirements.txt)
    3. Copy backend code
    4. Copy dist/ from Stage 1 to static/
    5. Configure health check
    6. Start server on PORT (default 8080)
```

### Health Check

```dockerfile
HEALTHCHECK --interval=10s --timeout=3s --start-period=30s \
    CMD curl -f http://localhost:${PORT:-8080}/api/health || exit 1
```

Platform monitors this endpoint every 10 seconds. If it fails, the app is marked as unhealthy.

### Environment Configuration

**Required variables:**
- `APP_SECRET` - HMAC secret for signature generation (min 32 chars, auto-generated)
- `APP_ID` - App identifier (must match appstore registration)
- `APPSTORE_URL` - Appstore backend URL for chat integration

**Optional variables:**
- `HOST` - Bind address (default: 0.0.0.0)
- `PORT` - Server port (default: 8080, platform may override)
- `ALLOWED_ORIGINS` - CORS origins (default: *, should set in production)
- `SESSION_TIMEOUT` - Session timeout seconds (default: 3600)

### Dependencies

**Backend (Python):**
- `fastapi==0.109.0` - Web framework
- `uvicorn==0.27.0` - ASGI server
- `python-socketio==5.11.0` - Socket.IO for robot/UI connections
- `numpy, scipy, scikit-learn` - Object detection
- `torch>=2.0.0` - ML model inference

**Frontend (React):**
- Vite - Build tool
- TypeScript - Type safety
- Node 24 - Runtime (MUST match package-lock.json)

---

## API Endpoints

### REST Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api` | GET | Basic health check |
| `/api/health` | GET | Detailed health with active sessions |
| `/api/sessions` | GET | List all active sessions |
| `/api/sessions/{id}` | GET | Get session details |
| `/api/object_detection/objects` | GET | Get detected objects (query: session_id) |
| `/api/robot/status` | GET | Get robot connection status (query: session_id) |
| `/` | GET | Serves React frontend |

### Socket.IO Namespaces

**`/sessions/{sessionId}/robot` - Robot Connection**

From Robot:
- `laser_scan` - LiDAR data (360 ranges)
- `robot_pose` - Position (x, y, yaw)
- `battery` - Battery status
- `map` - Occupancy grid
- `camera` - Camera frames (JPEG)
- `navigation_status` - Nav2 status
- `navigation_feedback` - Nav2 feedback

To Robot:
- `app_signature` - Phase 1 authentication
- `setup_app_response` - Phase 2 ready signal
- `move_cmd` - Twist velocity command
- `stop_cmd` - Stop movement
- `navigate_cmd` - Nav2 goal

**`/` - UI Connection**

From UI:
- `join_session` - Join a robot session
- `move_cmd` - Send movement command
- `stop_cmd` - Stop robot
- `navigate_cmd` - Send navigation goal

To UI:
- `session_state` - Current session state
- `laser_scan` - Forwarded sensor data
- `robot_pose` - Forwarded pose
- `battery` - Forwarded battery
- `map` - Forwarded map
- `remote_control_status` - Control enabled/disabled
- `robot_disconnected` - Robot disconnected

---

## Chat Commands

Available via AppstoreBridge integration:

| Command | Action |
|---------|--------|
| `forward` | Move forward 50cm |
| `backward` | Move backward 50cm |
| `spin` | Rotate 360 degrees |
| `status` | Show robot status (model, battery) |
| `help` | Show available commands |

---

## Three-Phase Connection Protocol

**Phase 1: Authentication**
1. Robot connects to `/sessions/{sessionId}/robot`
2. App sends `app_signature` with HMAC-SHA256 signature
3. Robot verifies signature using APP_SECRET
4. Robot proceeds to Phase 2 if valid

**Phase 2: Setup**
1. Robot sends `setup_app_cmd` with session_token
2. App creates SocketIOWrapper for robot transport
3. App instantiates MyRobotApp with robot transport
4. App connects to Appstore via AppstoreBridge (chat integration)
5. App sends `setup_app_response` with status='ready'
6. Robot proceeds to Phase 3

**Phase 3: Enable Control**
1. Robot sends `enable_remote_control_response` with status='enabled'
2. Movement commands now allowed
3. Sensor data starts flowing
4. Chat interface becomes active

---

## Troubleshooting

### Issue: 502 Bad Gateway

**Cause**: App not running or health check failing
**Solution**: Check logs, verify health endpoint responds

```bash
remake logs remake-rpc --tail 100
curl https://remake-rpc.apps.remake.ai/api/health
```

### Issue: App builds but won't start

**Cause**: Missing or invalid environment variables
**Solution**: Check APP_SECRET and APP_ID are set

```bash
remake env remake-rpc
remake logs remake-rpc --tail 50
```

### Issue: npm ci fails during build

**Cause**: Node version mismatch
**Solution**: Already fixed - Dockerfile uses Node 24, package-lock.json committed

### Issue: Robot can't connect

**Cause 1**: App not registered in appstore database
**Solution**: Run the INSERT INTO robot_apps SQL command

**Cause 2**: APP_SECRET mismatch
**Solution**: Ensure database app_secret matches env var exactly

```bash
# Get the secret from env vars
remake env remake-rpc | grep APP_SECRET

# Update database to match
UPDATE robot_apps
SET app_secret = '<value-from-env-vars>'
WHERE app_id = 'remake-rpc-001';
```

### Issue: CORS errors in browser

**Cause**: ALLOWED_ORIGINS not configured
**Solution**: Set to include app domain and appstore domain

```bash
remake env:set ALLOWED_ORIGINS=https://remake-rpc.apps.remake.ai,https://apps.remake.ai
remake deploy remake-rpc
```

### Issue: Chat not working

**Cause**: AppstoreBridge not connecting
**Solution**: Check APPSTORE_URL is correct, check logs for connection errors

```bash
remake logs remake-rpc | grep -i appstore
```

---

## Testing Plan

### Local Testing (Optional)

```bash
cd "C:\Users\Ricardo\Documents\Remake Ai\remake-rpc"

# Build Docker image
docker build -t remake-rpc:test .

# Run container
docker run -p 8080:8080 \
  -e APP_SECRET="test-secret-32-chars-long-here-abc123" \
  -e APP_ID="remake-rpc-001" \
  -e APPSTORE_URL="https://apps.remake.ai" \
  -e ALLOWED_ORIGINS="http://localhost:3002,https://apps.remake.ai" \
  remake-rpc:test

# Test health endpoint
curl http://localhost:8080/api/health

# Test frontend
curl http://localhost:8080/
```

### Production Testing

```bash
# 1. Check deployment status
remake status remake-rpc

# 2. Test health endpoint
curl https://remake-rpc.apps.remake.ai/api/health

# 3. Test frontend loads
curl -I https://remake-rpc.apps.remake.ai/

# 4. Check logs for errors
remake logs remake-rpc --tail 100 | grep -E "(ERROR|Exception)"

# 5. Test WebSocket connection (from browser console)
const socket = io('https://remake-rpc.apps.remake.ai');
socket.on('connect', () => console.log('Connected!'));
```

### Robot Connection Testing

**Prerequisites:**
- ROS2 + Gazebo with TurtleBot3
- kaiaai robot client
- Robot registered in appstore
- remake-rpc registered in appstore

**Steps:**
1. Launch Gazebo:
   ```bash
   ros2 launch turtlebot3_gazebo turtlebot3_world.launch.py
   ```

2. Run robot client:
   ```bash
   cd kaiaai
   python -m kaiaai.appstore_robot_client \
     --robot-id your-robot-id \
     --robot-secret your-robot-secret \
     --appstore-url https://apps.remake.ai
   ```

3. Login to appstore:
   - Go to https://apps.remake.ai
   - Login with your account

4. Launch app:
   - Go to "My Robots"
   - Verify robot shows "online"
   - Click "Launch App" → Select "Remake RPC"

5. Verify three-phase protocol:
   - Check robot logs for: "Phase 1: Received app_signature"
   - Check robot logs for: "Phase 2: Sending setup_app_cmd"
   - Check robot logs for: "Phase 3: Remote control enabled"

6. Test chat commands:
   - Type `help` in chat
   - Type `forward` - robot should move forward
   - Type `spin` - robot should rotate 360°
   - Type `status` - should show robot model and battery

7. Verify sensor visualization:
   - LiDAR scan should display in real-time
   - Robot pose should update as robot moves
   - Battery percentage should display
   - Map should show explored area
   - Camera feed should show (if camera available)

8. Test object detection:
   - LiDAR should detect lines, corners, arcs
   - Detected objects should highlight in UI

**Expected behavior:**
- All phases complete successfully
- Chat commands execute
- Sensors display in real-time
- No errors in logs
- Robot responds to commands

---

## Success Criteria

The implementation is successful when:

- ✅ App deploys without errors
- ✅ Health endpoint returns 200 OK
- ✅ Frontend loads in browser
- ✅ App registered in appstore database
- ✅ Robot can connect via three-phase protocol
- ✅ Chat commands work
- ✅ Sensor data displays in UI
- ✅ No errors in production logs

---

## Rollback Plan

If deployment fails:

```bash
# 1. Check what went wrong
remake logs remake-rpc --tail 200

# 2. If issue is with code, fix locally and redeploy
cd "C:\Users\Ricardo\Documents\Remake Ai\remake-rpc"
# Make fixes
git add .
git commit -m "fix: issue description"
git push origin main
remake deploy remake-rpc

# 3. If issue is with environment vars
remake env remake-rpc
remake env:set VARIABLE_NAME=correct-value
remake deploy remake-rpc

# 4. If completely broken, destroy and recreate
remake destroy remake-rpc
# Fix issues locally first
remake create remake-rpc --github ricardothe3rd/remake-rpc
remake deploy remake-rpc
```

---

## Post-Deployment

After successful deployment:

1. **Monitor logs** for the first 24 hours:
   ```bash
   remake logs remake-rpc --follow
   ```

2. **Test with multiple robots** to ensure scalability

3. **Update documentation** if any issues discovered

4. **Create backup** of working configuration:
   ```bash
   remake env remake-rpc > remake-rpc-env-backup.txt
   ```

5. **Document APP_SECRET** in secure password manager (needed for appstore database)

---

## Timeline Estimate

| Phase | Task | Est. Time |
|-------|------|-----------|
| 1 | Commit and push changes | 2 min |
| 2 | Delete broken app | 1 min |
| 3 | Create new app | 1 min |
| 4 | Verify environment variables | 2 min |
| 5 | Deploy | 5 min |
| 6 | Verify deployment | 3 min |
| 7 | Appstore registration | 5 min |
| **Total** | | **~20 min** |

Testing with robot: +15 minutes

---

## Conclusion

The remake-rpc app is ready to deploy. The only issue is the misconfigured repository setting in the platform database. Following this plan will result in a working production deployment.

**Next step:** Commit changes and begin Phase 1.

---

**Document Version**: 1.0
**Last Updated**: February 8, 2026
**Author**: Claude Code + Ricardo
