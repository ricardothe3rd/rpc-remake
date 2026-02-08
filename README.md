# Remake RPC

Robot control application with object detection, chat interface, and sensor visualization. Built for the Remake Platform.

## Features

- **Three-Phase Protocol**: Implements Remake Platform's app launch protocol for secure robot connections
- **Object Detection**: RANSAC V4 + ML-based detection for lines, corners, and arcs
- **Chat Interface**: AppstoreBridge integration for dashboard chat commands
- **Sensor Visualization**: Real-time display of LiDAR, pose, battery, map, and camera data
- **React UI**: Modern TypeScript React frontend with Vite

## Architecture

```
Robot ←Socket.IO→ Remake RPC ←Socket.IO→ Frontend UI
   (via Appstore)     (this app)
```

### Three-Phase Connection Protocol

1. **Phase 1**: Robot connects → App sends `app_signature` for verification
2. **Phase 2**: Robot sends `setup_app_cmd` → App prepares resources → Instantiates `MyRobotApp`
3. **Phase 3**: Robot sends `enable_remote_control_response` → Commands and sensors active

## Project Structure

```
remake-rpc/
├── backend/                          # Python FastAPI + Socket.IO
│   ├── main.py                      # Hybrid server (Socket.IO + REST)
│   ├── my_robot_app.py              # App logic with chat integration
│   ├── robot_app.py                 # Base class
│   ├── robot_proxy.py               # Robot command interface
│   ├── ui_proxy.py                  # UI message interface
│   ├── websocket_wrapper.py         # Transport layer
│   ├── message_callback_mixin.py    # Pub/sub pattern
│   ├── appstore_bridge.py           # Chat integration
│   ├── object_detector.py           # RANSAC V4 detection
│   ├── object_detector_ml.py        # ML 1D CNN detection
│   ├── ml/                          # Trained models
│   │   └── best_model.pth
│   └── requirements.txt
├── frontend/                         # React + TypeScript + Vite
│   ├── src/
│   │   ├── App.tsx
│   │   └── ...
│   ├── vite.config.ts
│   └── package.json
├── Dockerfile                        # Multi-stage build
├── .dockerignore
├── .gitignore
└── README.md
```

## Local Development

### Prerequisites

- Python 3.11+
- Node.js 20+
- Git

### Backend Setup

```bash
cd backend
pip install -r requirements.txt

# Create .env file
cp .env.example .env

# Edit .env and set:
# - APP_SECRET (min 32 characters)
# - APP_ID (from appstore registration)
# - APPSTORE_URL (http://localhost:5000 for local)

# Run backend
python main.py
```

Backend runs on `http://localhost:8080`

### Frontend Setup

```bash
cd frontend
npm install
npm run dev
```

Frontend runs on `http://localhost:3002`

## Deployment to Remake Platform

### 1. Push to GitHub

```bash
git add .
git commit -m "Initial commit: remake-rpc app"
git remote add origin https://github.com/YOUR_USERNAME/remake-rpc.git
git push -u origin main
```

### 2. Deploy via Remake Platform CLI

```bash
# Login
python3 -m remake_cli.main login --token YOUR_API_TOKEN

# Create and deploy app
python3 -m remake_cli.main create remake-rpc --github YOUR_USERNAME/remake-rpc

# Watch deployment logs
python3 -m remake_cli.main logs remake-rpc --build

# Check status
python3 -m remake_cli.main status remake-rpc
```

### 3. Register in Appstore

Once deployed, register the app in the Appstore database:

```sql
INSERT INTO robot_apps (
  app_id,
  app_secret,
  app_name,
  app_ws_url,
  developer_id,
  status
) VALUES (
  'remake-rpc-001',                              -- Must match APP_ID in env vars
  'your-secret-key-matching-env-var',             -- Must match APP_SECRET
  'Remake RPC',
  'https://remake-rpc-XXXXX.app.remake.ai',      -- URL from deployment
  1,                                              -- Your developer ID
  'active'
);
```

## Environment Variables

### Required (Production)

| Variable | Description | Example |
|----------|-------------|---------|
| `APP_SECRET` | HMAC secret for signature generation (min 32 chars) | `your-production-secret-32-chars-min` |
| `APP_ID` | App identifier (must match appstore registration) | `remake-rpc-001` |
| `APPSTORE_URL` | Appstore backend URL | `https://apps.remake.ai` |

### Optional

| Variable | Description | Default |
|----------|-------------|---------|
| `HOST` | Server bind address | `0.0.0.0` |
| `PORT` | Server port | `8080` |
| `ALLOWED_ORIGINS` | CORS allowed origins | `*` |
| `SESSION_TIMEOUT` | Session timeout in seconds | `3600` |

## Testing with Gazebo

### 1. Launch Gazebo

```bash
ros2 launch turtlebot3_gazebo turtlebot3_world.launch.py
```

### 2. Run Robot Client

```bash
cd kaiaai
python -m kaiaai.appstore_robot_client \
  --robot-id your-robot-id \
  --robot-secret your-robot-secret \
  --appstore-url https://apps.remake.ai
```

### 3. Launch App from Appstore

1. Login to https://apps.remake.ai
2. Go to "My Robots"
3. See your robot "online"
4. Click "Launch App" → Select "Remake RPC"
5. Three-phase protocol executes
6. Chat interface appears

### 4. Control Robot via Chat

Type commands in the chat:
- `forward` - Move forward 50cm
- `backward` - Move backward 50cm
- `spin` - Rotate 360 degrees
- `status` - Check robot status
- `help` - Show available commands

Or click the button options for instant execution.

## API Endpoints

### Health & Status

- `GET /api` - Basic health check
- `GET /api/health` - Detailed health with active sessions
- `GET /api/sessions` - List all active sessions
- `GET /api/sessions/{session_id}` - Get session details

### Object Detection

- `GET /api/object_detection/objects?session_id=X` - Get detected objects
- `GET /api/robot/status?session_id=X` - Get robot connection status

### Frontend

- `GET /` - Serves the React frontend UI (built from `frontend/dist`)

### Socket.IO Events

#### Robot Namespace: `/sessions/{sessionId}/robot`

**From Robot:**
- `laser_scan` - LiDAR data
- `robot_pose` - Robot position
- `battery` - Battery status
- `map` - Occupancy grid
- `camera` - Camera frames
- `navigation_status` - Nav2 status
- `navigation_feedback` - Nav2 feedback

**To Robot:**
- `app_signature` - Phase 1 signature
- `setup_app_response` - Phase 2 response
- `move_cmd` - Twist velocity command
- `stop_cmd` - Stop movement
- `navigate_cmd` - Nav2 goal

#### UI Namespace: `/` (default)

**From UI:**
- `join_session` - Join a robot session
- `move_cmd` - Send movement command
- `stop_cmd` - Stop robot
- `navigate_cmd` - Send navigation goal

**To UI:**
- `session_state` - Current session state
- `laser_scan` - Forwarded sensor data
- `robot_pose` - Forwarded pose
- `battery` - Forwarded battery
- `map` - Forwarded map
- `remote_control_status` - Control enabled/disabled
- `robot_disconnected` - Robot disconnected

## Chat Commands

The app supports these chat commands via AppstoreBridge:

| Command | Action |
|---------|--------|
| `forward` | Move forward 50cm |
| `backward` | Move backward 50cm |
| `spin` | Rotate 360 degrees |
| `status` | Show robot status (model, battery) |
| `help` | Show available commands |

## Development Notes

### Key Files

- **`backend/main.py`**: Combines Socket.IO dynamic namespaces (robot-app-basic) with RPC_v2 functionality
- **`backend/my_robot_app.py`**: Modified to accept pre-connected transports instead of creating WebSocket connections
- **`backend/appstore_bridge.py`**: Handles chat integration with Appstore dashboard
- **`backend/object_detector.py`**: RANSAC V4 feature detection
- **`Dockerfile`**: Multi-stage build (React → Python → combined image)

### Socket.IO Wrappers

The `SocketIOWrapper` and `SocketIOUIWrapper` classes bridge Socket.IO connections to the `WebSocketWrapper` interface expected by `MyRobotApp`. This allows reusing all RPC_v2 code without modification.

### Robot Connection Flow

1. Robot connects to `/sessions/{sessionId}/robot` namespace
2. App sends `app_signature` (Phase 1)
3. Robot sends `setup_app_cmd` with `session_token` (Phase 2)
4. App creates `SocketIOWrapper` for robot transport
5. App instantiates `MyRobotApp` with robot transport
6. App connects to Appstore via `AppstoreBridge`
7. Robot sends `enable_remote_control_response` (Phase 3)
8. Commands and sensor data flow

## Troubleshooting

### App won't connect to robot

- Check `APP_SECRET` matches between app env vars and appstore database
- Check `APP_ID` matches between code and appstore registration
- Check `app_ws_url` in appstore database points to deployed URL

### Chat not working

- Check `APPSTORE_URL` is set correctly
- Check `AppstoreBridge` connected successfully (see logs)
- Check session credentials are passed correctly in `setup_app_cmd`

### Build fails

- Ensure `frontend/` has `dist/` directory after build
- Check Node.js version (needs 20+)
- Check all Python dependencies installed

## License

MIT

## Author

Remake AI
