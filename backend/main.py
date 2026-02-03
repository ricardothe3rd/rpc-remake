"""
Remake RPC - Robot Control Application
========================================
FastAPI + Socket.IO server combining RPC_v2 functionality with three-phase protocol.

This app implements:
- Three-phase connection protocol for Appstore integration
- Object detection (RANSAC V4 + ML)
- Chat interface via AppstoreBridge
- Full React UI with sensor visualization

Architecture:
    Robot ←Socket.IO→ App Backend ←Socket.IO→ Frontend UI
                      (this file)

WebSocket Namespaces:
    /sessions/{sessionId}/robot - Robot connection (dynamic namespace)
    / - Frontend UI connection

Author: Remake AI
Version: 1.0.0
"""

import os
import asyncio
import hashlib
import hmac
import logging
from typing import Dict, Optional, Set
from datetime import datetime
from pathlib import Path

import socketio
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import uvicorn

from my_robot_app import MyRobotApp
from robot_proxy import RobotProxy
from ui_proxy import UIProxy
from websocket_wrapper import WebSocketWrapper

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================================
# Configuration
# ============================================================================

class Config:
    """Application configuration"""

    # Server
    HOST = os.getenv("HOST", "0.0.0.0")
    PORT = int(os.getenv("PORT", "8080"))

    # Security (REQUIRED - set in production!)
    APP_SECRET = os.getenv("APP_SECRET", "your-app-secret-key-here")
    APP_ID = os.getenv("APP_ID", "remake-rpc-001")

    # Appstore
    APPSTORE_URL = os.getenv("APPSTORE_URL", "http://localhost:5000")

    # CORS
    ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "*").split(",")

    # Session
    SESSION_TIMEOUT = int(os.getenv("SESSION_TIMEOUT", "3600"))  # 1 hour


# ============================================================================
# FastAPI Application
# ============================================================================

app = FastAPI(
    title="Remake RPC",
    description="Robot control application with object detection and chat",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================================
# Socket.IO Server
# ============================================================================

# Create Socket.IO server with CORS
sio = socketio.AsyncServer(
    async_mode='asgi',
    cors_allowed_origins="*",
    logger=True,
    engineio_logger=False
)

# Wrap with ASGI app
socket_app = socketio.ASGIApp(
    socketio_server=sio,
    other_asgi_app=app,
    socketio_path='/socket.io'
)


# ============================================================================
# Session State Management
# ============================================================================

class SessionState:
    """Tracks state for active robot sessions"""

    def __init__(self):
        self.sessions: Dict[str, Dict] = {}  # session_id -> session data
        self.robot_sids: Dict[str, str] = {}  # session_id -> robot socket_id
        self.ui_clients: Dict[str, Set[str]] = {}  # session_id -> set of ui socket_ids
        self.robot_apps: Dict[str, MyRobotApp] = {}  # session_id -> MyRobotApp instance

    def create_session(self, session_id: str, robot_sid: str) -> None:
        """Create new session when robot connects"""
        self.sessions[session_id] = {
            'session_id': session_id,
            'robot_sid': robot_sid,
            'connected_at': datetime.utcnow().isoformat(),
            'phase': 1,  # Phase 1: establishing
            'setup_complete': False,
            'remote_control_enabled': False,
            'sensor_data': {},
            'session_token': None,  # Will be set during setup
        }
        self.robot_sids[session_id] = robot_sid
        self.ui_clients[session_id] = set()
        logger.info(f"[Session] Created session {session_id}")

    def add_ui_client(self, session_id: str, ui_sid: str) -> None:
        """Add UI client to session"""
        if session_id in self.ui_clients:
            self.ui_clients[session_id].add(ui_sid)
            logger.info(f"[Session] Added UI client {ui_sid} to session {session_id}")

    def remove_ui_client(self, session_id: str, ui_sid: str) -> None:
        """Remove UI client from session"""
        if session_id in self.ui_clients and ui_sid in self.ui_clients[session_id]:
            self.ui_clients[session_id].remove(ui_sid)
            logger.info(f"[Session] Removed UI client {ui_sid} from session {session_id}")

    def end_session(self, session_id: str) -> None:
        """Clean up session"""
        # Stop robot app if exists
        if session_id in self.robot_apps:
            try:
                self.robot_apps[session_id].stop()
            except Exception as e:
                logger.error(f"Error stopping robot app: {e}")
            del self.robot_apps[session_id]

        if session_id in self.sessions:
            del self.sessions[session_id]
        if session_id in self.robot_sids:
            del self.robot_sids[session_id]
        if session_id in self.ui_clients:
            del self.ui_clients[session_id]
        logger.info(f"[Session] Ended session {session_id}")

    def get_session(self, session_id: str) -> Optional[Dict]:
        """Get session data"""
        return self.sessions.get(session_id)

    def get_robot_sid(self, session_id: str) -> Optional[str]:
        """Get robot socket ID for session"""
        return self.robot_sids.get(session_id)

    def get_ui_clients(self, session_id: str) -> Set[str]:
        """Get all UI client socket IDs for session"""
        return self.ui_clients.get(session_id, set())

    def get_robot_app(self, session_id: str) -> Optional[MyRobotApp]:
        """Get robot app instance for session"""
        return self.robot_apps.get(session_id)

    def set_robot_app(self, session_id: str, robot_app: MyRobotApp) -> None:
        """Store robot app instance for session"""
        self.robot_apps[session_id] = robot_app


# Global session state
session_state = SessionState()


# ============================================================================
# Utility Functions
# ============================================================================

def generate_app_signature(session_id: str, timestamp: str) -> str:
    """
    Generate HMAC-SHA256 signature for robot authentication.

    Args:
        session_id: Unique session identifier
        timestamp: ISO 8601 timestamp

    Returns:
        Hex-encoded HMAC signature
    """
    message = f"{Config.APP_ID}:{session_id}:{timestamp}"
    signature = hmac.new(
        Config.APP_SECRET.encode('utf-8'),
        message.encode('utf-8'),
        hashlib.sha256
    ).hexdigest()
    return signature


async def broadcast_to_ui(session_id: str, event: str, data: Dict) -> None:
    """
    Broadcast event to all UI clients connected to a session.

    Args:
        session_id: Session identifier
        event: Event name
        data: Event payload
    """
    ui_clients = session_state.get_ui_clients(session_id)
    for ui_sid in ui_clients:
        try:
            await sio.emit(event, data, room=ui_sid)
        except Exception as e:
            logger.error(f"[UI Broadcast] Error sending to {ui_sid}: {e}")


# ============================================================================
# Dynamic Namespace Handler for Robot Connections
# ============================================================================

class RobotNamespace(socketio.AsyncNamespace):
    """
    Dynamic namespace handler for robot connections.
    Handles pattern: /sessions/{sessionId}/robot
    """

    def __init__(self, namespace_pattern: str):
        super().__init__(namespace_pattern)
        self.namespace_pattern = namespace_pattern

    async def on_connect(self, sid: str, environ: Dict):
        """
        Robot connected to session namespace.

        Phase 1: Send app signature for platform verification.
        """
        # Extract session_id from namespace
        namespace = environ.get('asgi.scope', {}).get('path', '').replace('/socket.io/', '')
        session_id = self._extract_session_id(namespace)

        if not session_id:
            logger.error(f"[Robot] Invalid namespace: {namespace}")
            return False

        logger.info(f"[Robot] Connected: sid={sid}, session={session_id}")

        # Create session
        session_state.create_session(session_id, sid)

        # Phase 1: Send signature for verification
        timestamp = datetime.utcnow().isoformat()
        signature = generate_app_signature(session_id, timestamp)

        await self.emit('app_signature', {
            'app_signature': signature,
            'timestamp': timestamp,
            'app_id': Config.APP_ID
        }, room=sid)

        logger.info(f"[Phase 1] Sent signature for session {session_id}")
        return True

    async def on_disconnect(self, sid: str):
        """Robot disconnected"""
        # Find session by robot sid
        session_id = None
        for sess_id, robot_sid in session_state.robot_sids.items():
            if robot_sid == sid:
                session_id = sess_id
                break

        if session_id:
            logger.info(f"[Robot] Disconnected: session={session_id}")

            # Notify UI clients
            await broadcast_to_ui(session_id, 'robot_disconnected', {
                'session_id': session_id,
                'timestamp': datetime.utcnow().isoformat()
            })

            # Clean up session
            session_state.end_session(session_id)

    async def on_setup_app_cmd(self, sid: str, data: Dict):
        """
        Phase 2: Handle setup command from robot.

        This is where we instantiate MyRobotApp with the robot connection.
        """
        session_id = data.get('session_id')
        session_token = data.get('session_token')  # From platform
        logger.info(f"[Phase 2] Setup command received for session {session_id}")

        session = session_state.get_session(session_id)
        if not session:
            logger.error(f"[Phase 2] Session not found: {session_id}")
            return

        # Update session state
        session['phase'] = 2
        session['session_token'] = session_token

        try:
            # Create WebSocketWrapper for robot communication
            # This wraps the Socket.IO connection to look like a WebSocket
            robot_transport = SocketIOWrapper(sio, sid, f'/sessions/{session_id}/robot')

            # Create UI transport (will be used when UI clients connect)
            ui_transport = SocketIOUIWrapper(sio, session_id)

            # Instantiate MyRobotApp with the connected robot
            robot_app = MyRobotApp(
                robot_transport=robot_transport,
                ui_transport=ui_transport,
                session_id=session_id,
                session_token=session_token,
                appstore_url=Config.APPSTORE_URL
            )

            # Store the robot app instance
            session_state.set_robot_app(session_id, robot_app)

            # Start the robot app
            robot_app.start()

            # Mark setup complete
            session['setup_complete'] = True
            session['phase'] = 3

            # Send success response to robot
            await self.emit('setup_app_response', {
                'session_id': session_id,
                'status': 'success',
                'message': 'App setup completed successfully'
            }, room=sid)

            logger.info(f"[Phase 2] Setup complete for session {session_id}, MyRobotApp started")

            # Notify UI
            await broadcast_to_ui(session_id, 'setup_complete', {
                'session_id': session_id
            })

        except Exception as e:
            logger.error(f"[Phase 2] Error during setup: {e}")
            await self.emit('setup_app_response', {
                'session_id': session_id,
                'status': 'error',
                'message': str(e)
            }, room=sid)

    async def on_enable_remote_control_response(self, sid: str, data: Dict):
        """
        Phase 3: Remote control enabled by robot.

        Commands and sensor streaming now active.
        """
        session_id = data.get('session_id')
        status = data.get('status')

        logger.info(f"[Phase 3] Remote control response: session={session_id}, status={status}")

        session = session_state.get_session(session_id)
        if session:
            session['remote_control_enabled'] = (status == 'enabled')

            # Notify UI
            await broadcast_to_ui(session_id, 'remote_control_status', {
                'enabled': session['remote_control_enabled']
            })

    # ========================================================================
    # Sensor Data Handlers (Robot → App → UI)
    # ========================================================================

    async def on_laser_scan(self, sid: str, data: Dict):
        """Receive LiDAR scan data from robot"""
        session_id = data.get('session_id')

        # Update session state
        session = session_state.get_session(session_id)
        if session:
            session['sensor_data']['laser_scan'] = data

        # Forward to robot app for object detection
        robot_app = session_state.get_robot_app(session_id)
        if robot_app:
            # Inject message into robot transport
            robot_app.robot.get_transport().inject_message(data)

        # Forward to UI clients
        await broadcast_to_ui(session_id, 'laser_scan', data)

    async def on_robot_pose(self, sid: str, data: Dict):
        """Receive robot pose from robot"""
        session_id = data.get('session_id')

        session = session_state.get_session(session_id)
        if session:
            session['sensor_data']['pose'] = data

        # Forward to robot app
        robot_app = session_state.get_robot_app(session_id)
        if robot_app:
            robot_app.robot.get_transport().inject_message(data)

        await broadcast_to_ui(session_id, 'robot_pose', data)

    async def on_battery(self, sid: str, data: Dict):
        """Receive battery data from robot"""
        session_id = data.get('session_id')

        session = session_state.get_session(session_id)
        if session:
            session['sensor_data']['battery'] = data

        robot_app = session_state.get_robot_app(session_id)
        if robot_app:
            robot_app.robot.get_transport().inject_message(data)

        await broadcast_to_ui(session_id, 'battery', data)

    async def on_map(self, sid: str, data: Dict):
        """Receive map data from robot"""
        session_id = data.get('session_id')

        session = session_state.get_session(session_id)
        if session:
            session['sensor_data']['map'] = data

        robot_app = session_state.get_robot_app(session_id)
        if robot_app:
            robot_app.robot.get_transport().inject_message(data)

        await broadcast_to_ui(session_id, 'map', data)

    async def on_camera(self, sid: str, data: Dict):
        """Receive camera frame from robot"""
        session_id = data.get('session_id')

        session = session_state.get_session(session_id)
        if session:
            session['sensor_data']['camera'] = data

        await broadcast_to_ui(session_id, 'camera', data)

    async def on_navigation_status(self, sid: str, data: Dict):
        """Receive navigation status from robot"""
        session_id = data.get('session_id')
        await broadcast_to_ui(session_id, 'navigation_status', data)

    async def on_navigation_feedback(self, sid: str, data: Dict):
        """Receive navigation feedback from robot"""
        session_id = data.get('session_id')
        await broadcast_to_ui(session_id, 'navigation_feedback', data)

    def _extract_session_id(self, namespace: str) -> Optional[str]:
        """Extract session ID from namespace path"""
        # Pattern: /sessions/{sessionId}/robot
        parts = namespace.strip('/').split('/')
        if len(parts) >= 2 and parts[0] == 'sessions':
            return parts[1]
        return None


# ============================================================================
# Socket.IO Wrappers for MyRobotApp
# ============================================================================

class SocketIOWrapper:
    """
    Wraps Socket.IO connection to look like WebSocketWrapper for MyRobotApp.
    """

    def __init__(self, sio_server: socketio.AsyncServer, robot_sid: str, namespace: str):
        self.sio = sio_server
        self.robot_sid = robot_sid
        self.namespace = namespace
        self.message_callbacks = []
        self._connected = True

    async def send_message(self, message: Dict) -> None:
        """Send message to robot via Socket.IO"""
        event_type = message.get('type')
        await self.sio.emit(event_type, message, room=self.robot_sid, namespace=self.namespace)

    def send_message_async(self, message: Dict) -> None:
        """Synchronous wrapper for send_message"""
        asyncio.create_task(self.send_message(message))

    def inject_message(self, message: Dict) -> None:
        """Inject message from robot into callback chain"""
        for callback in self.message_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    asyncio.create_task(callback(message))
                else:
                    callback(message)
            except Exception as e:
                logger.error(f"Error in message callback: {e}")

    def add_message_callback(self, callback: callable) -> None:
        """Add message callback"""
        self.message_callbacks.append(callback)

    def remove_message_callback(self, callback: callable) -> None:
        """Remove message callback"""
        if callback in self.message_callbacks:
            self.message_callbacks.remove(callback)

    def is_connected(self) -> bool:
        """Check if connected"""
        return self._connected

    async def run_message_loop(self):
        """Dummy message loop (handled by Socket.IO)"""
        pass


class SocketIOUIWrapper:
    """
    Wraps Socket.IO UI broadcast to look like WebSocketWrapper for MyRobotApp.
    """

    def __init__(self, sio_server: socketio.AsyncServer, session_id: str):
        self.sio = sio_server
        self.session_id = session_id
        self.message_callbacks = []

    async def send_message(self, message: Dict) -> None:
        """Broadcast message to all UI clients"""
        event_type = message.get('type', 'message')
        await broadcast_to_ui(self.session_id, event_type, message)

    def send_message_async(self, message: Dict) -> None:
        """Synchronous wrapper for send_message"""
        asyncio.create_task(self.send_message(message))

    def inject_message(self, message: Dict) -> None:
        """Inject message from UI"""
        for callback in self.message_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    asyncio.create_task(callback(message))
                else:
                    callback(message)
            except Exception as e:
                logger.error(f"Error in UI callback: {e}")

    def add_message_callback(self, callback: callable) -> None:
        """Add message callback"""
        self.message_callbacks.append(callback)

    def is_connected(self) -> bool:
        """Always return True since UI clients can join anytime"""
        return len(session_state.get_ui_clients(self.session_id)) > 0


# Register robot namespace handler
robot_ns = RobotNamespace('/sessions')
sio.register_namespace(robot_ns)


# ============================================================================
# UI WebSocket Handlers (Default Namespace)
# ============================================================================

@sio.on('connect')
async def ui_connect(sid: str, environ: Dict):
    """UI client connected"""
    logger.info(f"[UI] Client connected: {sid}")
    return True


@sio.on('disconnect')
async def ui_disconnect(sid: str):
    """UI client disconnected"""
    logger.info(f"[UI] Client disconnected: {sid}")

    # Remove from all sessions
    for session_id in list(session_state.ui_clients.keys()):
        session_state.remove_ui_client(session_id, sid)


@sio.on('join_session')
async def ui_join_session(sid: str, data: Dict):
    """
    UI client joins a robot session to receive sensor data.

    Args:
        data: {'session_id': str}
    """
    session_id = data.get('session_id')

    if not session_id:
        await sio.emit('error', {'message': 'session_id required'}, room=sid)
        return

    session = session_state.get_session(session_id)
    if not session:
        await sio.emit('error', {'message': f'Session {session_id} not found'}, room=sid)
        return

    # Add UI client to session
    session_state.add_ui_client(session_id, sid)

    # Send current session state
    await sio.emit('session_state', {
        'session_id': session_id,
        'phase': session.get('phase'),
        'setup_complete': session.get('setup_complete'),
        'remote_control_enabled': session.get('remote_control_enabled'),
        'sensor_data': session.get('sensor_data', {})
    }, room=sid)

    logger.info(f"[UI] Client {sid} joined session {session_id}")


@sio.on('move_cmd')
async def ui_move_command(sid: str, data: Dict):
    """UI sends movement command to robot"""
    session_id = data.get('session_id')

    session = session_state.get_session(session_id)
    if not session or not session.get('remote_control_enabled'):
        await sio.emit('error', {'message': 'Remote control not enabled'}, room=sid)
        return

    robot_sid = session_state.get_robot_sid(session_id)
    if not robot_sid:
        await sio.emit('error', {'message': 'Robot not connected'}, room=sid)
        return

    # Forward to robot
    namespace = f'/sessions/{session_id}/robot'
    await sio.emit('move_cmd', data, room=robot_sid, namespace=namespace)

    logger.info(f"[Command] Move command sent to robot: linear_x={data.get('linear_x')}, angular_z={data.get('angular_z')}")


@sio.on('stop_cmd')
async def ui_stop_command(sid: str, data: Dict):
    """UI sends stop command to robot"""
    session_id = data.get('session_id')

    session = session_state.get_session(session_id)
    if not session:
        await sio.emit('error', {'message': 'Session not found'}, room=sid)
        return

    robot_sid = session_state.get_robot_sid(session_id)
    if not robot_sid:
        await sio.emit('error', {'message': 'Robot not connected'}, room=sid)
        return

    # Forward to robot
    namespace = f'/sessions/{session_id}/robot'
    await sio.emit('stop_cmd', data, room=robot_sid, namespace=namespace)

    logger.info(f"[Command] Stop command sent to robot")


# ============================================================================
# REST API Endpoints (from RPC_v2)
# ============================================================================

class NavigateToPoseCommand(BaseModel):
    session_id: str
    x: float
    y: float
    z: float = 0.0
    qx: float = 0.0
    qy: float = 0.0
    qz: float = 0.0
    qw: float = 1.0
    frame_id: str = "map"
    relative: bool = False


@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "app": "Remake RPC",
        "version": "1.0.0",
        "status": "running"
    }


@app.get("/health")
async def health():
    """Detailed health check for platform"""
    return {
        "status": "healthy",
        "active_sessions": len(session_state.sessions),
        "timestamp": datetime.utcnow().isoformat()
    }


@app.get("/sessions")
async def list_sessions():
    """List all active sessions"""
    sessions = []
    for session_id, session_data in session_state.sessions.items():
        sessions.append({
            'session_id': session_id,
            'connected_at': session_data.get('connected_at'),
            'phase': session_data.get('phase'),
            'setup_complete': session_data.get('setup_complete'),
            'remote_control_enabled': session_data.get('remote_control_enabled'),
            'ui_clients': len(session_state.get_ui_clients(session_id))
        })
    return {'sessions': sessions}


@app.get("/sessions/{session_id}")
async def get_session(session_id: str):
    """Get session details"""
    session = session_state.get_session(session_id)
    if not session:
        return JSONResponse(
            status_code=404,
            content={'error': f'Session {session_id} not found'}
        )

    return {
        'session': session,
        'ui_clients': len(session_state.get_ui_clients(session_id))
    }


@app.get("/object_detection/objects")
async def get_detected_objects(session_id: str):
    """Get list of currently detected objects"""
    robot_app = session_state.get_robot_app(session_id)
    if robot_app:
        return {
            "objects": robot_app.object_detector.get_detected_objects(),
            "summary": robot_app.object_detector.get_objects_summary(),
            "features": robot_app.object_detector.get_features_for_ui()
        }
    return {"objects": [], "summary": {"total_objects": 0}, "features": []}


@app.get("/robot/status")
async def robot_status(session_id: str):
    """Get robot connection status"""
    session = session_state.get_session(session_id)
    return {
        "connected": session is not None,
        "ui_clients": len(session_state.get_ui_clients(session_id)) if session else 0,
        "remote_control_enabled": session.get('remote_control_enabled') if session else False
    }


# Serve frontend static files (if available)
static_path = Path(__file__).parent / "static"
if static_path.exists():
    app.mount("/assets", StaticFiles(directory=static_path / "assets"), name="assets")

    @app.get("/{path:path}")
    async def serve_frontend(path: str):
        """Serve frontend files"""
        if path == "" or not (static_path / path).exists():
            return FileResponse(static_path / "index.html")
        return FileResponse(static_path / path)


# ============================================================================
# Main Entry Point
# ============================================================================

if __name__ == "__main__":
    logger.info(f"""
+==============================================================+
|                    Remake RPC Server                         |
|                                                              |
|  Port: {Config.PORT}                                              |
|  App ID: {Config.APP_ID}                                   |
|  Appstore: {Config.APPSTORE_URL}          |
|                                                              |
|  WebSocket Endpoints:                                        |
|    /sessions/{{sessionId}}/robot  - Robot connection         |
|    /                              - UI clients               |
|                                                              |
|  Features:                                                   |
|    - Three-phase protocol                                    |
|    - Object detection (RANSAC V4 + ML)                       |
|    - Chat interface (AppstoreBridge)                         |
|    - Sensor visualization                                    |
+==============================================================+
    """)

    uvicorn.run(
        socket_app,
        host=Config.HOST,
        port=Config.PORT,
        log_level="info",
        ws_ping_interval=20,
        ws_ping_timeout=60,
        timeout_keep_alive=75
    )
