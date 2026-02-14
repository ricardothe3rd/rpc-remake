"""
Remake RPC - Robot Control Application
========================================
FastAPI + Socket.IO server implementing the Remake Platform App Robot Protocol.

Protocol Flow:
    1. Robot connects with auth={type:'robot', session_id, session_token}
    2. Robot emits start_protocol
    3. App emits app_signature → Robot responds signature_verified
    4. App emits setup_app_cmd → Robot responds setup_app_response
    5. Robot sends enable_remote_control_response → App emits robot_ready to UI

Architecture:
    Robot ←Socket.IO (root /)→ App Backend ←Socket.IO (root /)→ Frontend UI

Author: Remake AI
Version: 2.0.0
Protocol: APP_ROBOT_PROTOCOL v1.0
"""

import os
import asyncio
import hashlib
import hmac
import logging
from dataclasses import dataclass, field
from typing import Dict, Optional
from datetime import datetime
from pathlib import Path

import socketio
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
import uvicorn

from my_robot_app import MyRobotApp

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

    # Security
    APP_SECRET = os.getenv("APP_SECRET", "your-app-secret-key-here")
    APP_ID = os.getenv("APP_ID", "remake-rpc-001")

    # Appstore
    APPSTORE_URL = os.getenv("APPSTORE_URL", "http://localhost:5000")

    # CORS
    ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "*").split(",")

    # Session
    SESSION_TIMEOUT = int(os.getenv("SESSION_TIMEOUT", "3600"))


# ============================================================================
# FastAPI Application
# ============================================================================

app = FastAPI(
    title="Remake RPC",
    description="Robot control application with object detection and chat",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================================
# Socket.IO Server (Root Namespace Only)
# ============================================================================

sio = socketio.AsyncServer(
    async_mode='asgi',
    cors_allowed_origins="*",
    logger=True,
    engineio_logger=False,
    ping_timeout=60,
    ping_interval=25
)

socket_app = socketio.ASGIApp(
    socketio_server=sio,
    other_asgi_app=app,
    socketio_path='/socket.io'
)


# ============================================================================
# Session Model
# ============================================================================

@dataclass
class Session:
    """Represents an active robot-app session"""
    session_id: str
    robot_sid: Optional[str] = None
    ui_sids: list = field(default_factory=list)
    state: str = 'waiting'  # waiting -> protocol_started -> setup_complete -> active
    session_token: Optional[str] = None
    robot_app: Optional[MyRobotApp] = None
    sensor_data: dict = field(default_factory=dict)
    connected_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())

    def is_active(self) -> bool:
        return self.state == 'active' and self.robot_sid is not None


# Module-level lookup maps
sessions: Dict[str, Session] = {}
robot_sid_to_session: Dict[str, str] = {}
ui_sid_to_session: Dict[str, str] = {}


# ============================================================================
# Utility Functions
# ============================================================================

def generate_app_signature(session_id: str, timestamp: str) -> str:
    """Generate HMAC-SHA256 signature for robot authentication."""
    message = f"{Config.APP_ID}:{session_id}:{timestamp}"
    signature = hmac.new(
        Config.APP_SECRET.encode('utf-8'),
        message.encode('utf-8'),
        hashlib.sha256
    ).hexdigest()
    return signature


def _get_session_for_robot(sid: str) -> Optional[Session]:
    """Get session for a robot SID via reverse lookup."""
    session_id = robot_sid_to_session.get(sid)
    if session_id:
        return sessions.get(session_id)
    return None


async def broadcast_to_ui(session_id: str, event: str, data: Dict) -> None:
    """Broadcast event to all UI clients connected to a session."""
    session = sessions.get(session_id)
    if not session:
        return
    for ui_sid in session.ui_sids:
        try:
            await sio.emit(event, data, room=ui_sid)
        except Exception as e:
            logger.error(f"[UI Broadcast] Error sending to {ui_sid}: {e}")


# ============================================================================
# Socket.IO Wrappers for MyRobotApp
# ============================================================================

class SocketIOWrapper:
    """
    Wraps Socket.IO connection to look like WebSocketWrapper for MyRobotApp.
    Emits on root namespace (no namespace parameter).
    """

    def __init__(self, sio_server: socketio.AsyncServer, robot_sid: str):
        self.sio = sio_server
        self.robot_sid = robot_sid
        self.message_callbacks = []
        self._connected = True

    async def send_message(self, message: Dict) -> None:
        """Send message to robot via Socket.IO on root namespace"""
        event_type = message.get('type')
        await self.sio.emit(event_type, message, room=self.robot_sid)

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
        self.message_callbacks.append(callback)

    def remove_message_callback(self, callback: callable) -> None:
        if callback in self.message_callbacks:
            self.message_callbacks.remove(callback)

    def is_connected(self) -> bool:
        return self._connected

    async def run_message_loop(self):
        pass  # Handled by Socket.IO


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
        self.message_callbacks.append(callback)

    def is_connected(self) -> bool:
        session = sessions.get(self.session_id)
        return session is not None and len(session.ui_sids) > 0


# ============================================================================
# Connection Handlers (Root Namespace)
# ============================================================================

@sio.on('connect')
async def handle_connect(sid: str, environ: Dict, auth: Dict = None):
    """
    Unified connect handler for both robot and UI clients.
    Robot: auth = {type: 'robot', session_id: ..., session_token: ...}
    UI: no auth or auth without type='robot'
    """
    if auth and isinstance(auth, dict) and auth.get('type') == 'robot':
        # --- Robot connection ---
        session_id = auth.get('session_id')
        session_token = auth.get('session_token')

        if not session_id:
            logger.error(f"[Robot] Connect rejected: no session_id in auth")
            return False

        logger.info(f"[Robot] Connected: sid={sid}, session={session_id}")

        # Create or update session
        if session_id not in sessions:
            sessions[session_id] = Session(session_id=session_id)

        session = sessions[session_id]
        session.robot_sid = sid
        session.session_token = session_token
        session.state = 'waiting'

        # Register reverse lookup
        robot_sid_to_session[sid] = session_id

        # DO NOT emit anything here — wait for start_protocol
        return True
    else:
        # --- UI connection ---
        logger.info(f"[UI] Client connected: {sid}")
        return True


@sio.on('disconnect')
async def handle_disconnect(sid: str):
    """Unified disconnect handler for robot and UI clients."""

    # Check if this was a robot
    if sid in robot_sid_to_session:
        session_id = robot_sid_to_session.pop(sid)
        session = sessions.get(session_id)

        if session:
            logger.info(f"[Robot] Disconnected: session={session_id}")

            # Stop robot app
            if session.robot_app:
                try:
                    session.robot_app.stop()
                except Exception as e:
                    logger.error(f"Error stopping robot app: {e}")

            # Notify all UI clients
            for ui_sid in session.ui_sids:
                try:
                    await sio.emit('robot_disconnected', {
                        'session_id': session_id,
                        'timestamp': datetime.utcnow().isoformat()
                    }, room=ui_sid)
                except Exception as e:
                    logger.error(f"Error notifying UI {ui_sid}: {e}")

            # Clean up session
            for ui_sid in session.ui_sids:
                ui_sid_to_session.pop(ui_sid, None)
            del sessions[session_id]
        return

    # Check if this was a UI client
    if sid in ui_sid_to_session:
        session_id = ui_sid_to_session.pop(sid)
        session = sessions.get(session_id)
        if session and sid in session.ui_sids:
            session.ui_sids.remove(sid)
            logger.info(f"[UI] Client {sid} left session {session_id}")
        return

    logger.info(f"[Socket] Unknown client disconnected: {sid}")


# ============================================================================
# Three-Phase Protocol Handlers
# ============================================================================

@sio.on('start_protocol')
async def handle_start_protocol(sid: str, data: Dict = None):
    """
    Robot signals it's ready to receive events.
    Phase 1: Generate and send app_signature.
    """
    session_id = robot_sid_to_session.get(sid)
    if not session_id:
        logger.error(f"[Protocol] start_protocol from unknown sid: {sid}")
        return

    session = sessions.get(session_id)
    if not session:
        return

    logger.info(f"[Phase 1] Robot ready, sending app_signature for session {session_id}")
    session.state = 'protocol_started'

    # Generate and send HMAC signature
    timestamp = datetime.utcnow().isoformat()
    signature = generate_app_signature(session_id, timestamp)

    await sio.emit('app_signature', {
        'app_signature': signature,
        'timestamp': timestamp,
        'app_id': Config.APP_ID,
        'session_id': session_id
    }, room=sid)

    logger.info(f"[Phase 1] Sent app_signature for session {session_id}")


@sio.on('signature_verified')
async def handle_signature_verified(sid: str, data: Dict = None):
    """
    Phase 1 complete: Robot verified our signature.
    Phase 2 start: Send setup_app_cmd TO the robot.
    """
    session_id = robot_sid_to_session.get(sid)
    if not session_id:
        return

    session = sessions.get(session_id)
    if not session:
        return

    verified = data.get('verified', True) if data else True

    if not verified:
        logger.error(f"[Phase 1] Signature verification FAILED for session {session_id}")
        await sio.disconnect(sid)
        return

    logger.info(f"[Phase 2] Signature verified, sending setup_app_cmd for session {session_id}")
    session.state = 'signature_verified'

    # App sends setup command TO the robot
    await sio.emit('setup_app_cmd', {
        'session_id': session_id
    }, room=sid)


@sio.on('setup_app_response')
async def handle_setup_app_response(sid: str, data: Dict = None):
    """
    Phase 2 complete: Robot confirms setup.
    Instantiate MyRobotApp with the robot connection.
    """
    session_id = robot_sid_to_session.get(sid)
    if not session_id:
        return

    session = sessions.get(session_id)
    if not session:
        return

    status = data.get('status') if data else 'success'
    logger.info(f"[Phase 2] Setup response: session={session_id}, status={status}")

    if status != 'success':
        logger.error(f"[Phase 2] Setup failed: {data.get('message') if data else 'unknown'}")
        return

    try:
        # Create transport wrappers (no namespace — root namespace)
        robot_transport = SocketIOWrapper(sio, sid)
        ui_transport = SocketIOUIWrapper(sio, session_id)

        # Instantiate MyRobotApp
        robot_app = MyRobotApp(
            robot_transport=robot_transport,
            ui_transport=ui_transport,
            session_id=session_id,
            session_token=session.session_token,
            appstore_url=Config.APPSTORE_URL
        )

        session.robot_app = robot_app
        session.state = 'setup_complete'
        robot_app.start()

        logger.info(f"[Phase 2] Setup complete, MyRobotApp started for session {session_id}")

        # Notify UI
        await broadcast_to_ui(session_id, 'setup_complete', {
            'session_id': session_id
        })

    except Exception as e:
        logger.error(f"[Phase 2] Error during setup: {e}")


@sio.on('enable_remote_control_response')
async def handle_enable_remote_control_response(sid: str, data: Dict = None):
    """
    Phase 3 complete: Robot confirms remote control is enabled.
    Emit robot_ready to all UI clients in the session.
    """
    session_id = robot_sid_to_session.get(sid)
    if not session_id:
        return

    session = sessions.get(session_id)
    if not session:
        return

    status = data.get('status') if data else 'enabled'
    logger.info(f"[Phase 3] Remote control response: session={session_id}, status={status}")

    if status == 'enabled':
        session.state = 'active'

        # Emit robot_ready to ALL UI clients
        await broadcast_to_ui(session_id, 'robot_ready', {
            'session_id': session_id,
            'timestamp': datetime.utcnow().isoformat()
        })

        logger.info(f"[Phase 3] robot_ready emitted to {len(session.ui_sids)} UI clients")


# ============================================================================
# Sensor Data Handlers (Robot → App → UI)
# ============================================================================

@sio.on('laser_scan')
async def handle_laser_scan(sid: str, data: Dict):
    """Receive LiDAR scan data from robot"""
    session = _get_session_for_robot(sid)
    if not session:
        return

    session.sensor_data['laser_scan'] = data

    # Forward to robot app for object detection
    if session.robot_app:
        session.robot_app.robot.get_transport().inject_message(data)

    await broadcast_to_ui(session.session_id, 'laser_scan', data)


@sio.on('robot_pose')
async def handle_robot_pose(sid: str, data: Dict):
    """Receive robot pose from robot"""
    session = _get_session_for_robot(sid)
    if not session:
        return

    session.sensor_data['pose'] = data

    if session.robot_app:
        session.robot_app.robot.get_transport().inject_message(data)

    await broadcast_to_ui(session.session_id, 'robot_pose', data)


@sio.on('battery')
async def handle_battery(sid: str, data: Dict):
    """Receive battery data from robot"""
    session = _get_session_for_robot(sid)
    if not session:
        return

    session.sensor_data['battery'] = data

    if session.robot_app:
        session.robot_app.robot.get_transport().inject_message(data)

    await broadcast_to_ui(session.session_id, 'battery', data)


@sio.on('map')
async def handle_map(sid: str, data: Dict):
    """Receive map data from robot"""
    session = _get_session_for_robot(sid)
    if not session:
        return

    session.sensor_data['map'] = data

    if session.robot_app:
        session.robot_app.robot.get_transport().inject_message(data)

    await broadcast_to_ui(session.session_id, 'map', data)


@sio.on('camera')
async def handle_camera(sid: str, data: Dict):
    """Receive camera frame from robot"""
    session = _get_session_for_robot(sid)
    if not session:
        return

    session.sensor_data['camera'] = data
    await broadcast_to_ui(session.session_id, 'camera', data)


@sio.on('navigation_status')
async def handle_navigation_status(sid: str, data: Dict):
    """Receive navigation status from robot"""
    session = _get_session_for_robot(sid)
    if not session:
        return
    await broadcast_to_ui(session.session_id, 'navigation_status', data)


@sio.on('navigation_feedback')
async def handle_navigation_feedback(sid: str, data: Dict):
    """Receive navigation feedback from robot"""
    session = _get_session_for_robot(sid)
    if not session:
        return
    await broadcast_to_ui(session.session_id, 'navigation_feedback', data)


@sio.on('wifi')
async def handle_wifi(sid: str, data: Dict):
    """Receive WiFi signal data from robot"""
    session = _get_session_for_robot(sid)
    if not session:
        return

    session.sensor_data['wifi'] = data
    await broadcast_to_ui(session.session_id, 'wifi', data)


@sio.on('cmd_vel')
async def handle_cmd_vel(sid: str, data: Dict):
    """Receive current velocity from robot"""
    session = _get_session_for_robot(sid)
    if not session:
        return
    await broadcast_to_ui(session.session_id, 'cmd_vel', data)


@sio.on('robot_spec')
async def handle_robot_spec(sid: str, data: Dict):
    """Receive robot specification from robot"""
    session = _get_session_for_robot(sid)
    if not session:
        return
    await broadcast_to_ui(session.session_id, 'robot_spec', data)


@sio.on('robot_specification')
async def handle_robot_specification(sid: str, data: Dict):
    """Receive robot specification (alternate event name)"""
    session = _get_session_for_robot(sid)
    if not session:
        return
    await broadcast_to_ui(session.session_id, 'robot_specification', data)


@sio.on('object_detection')
async def handle_object_detection(sid: str, data: Dict):
    """Receive object detection results"""
    session = _get_session_for_robot(sid)
    if not session:
        return
    await broadcast_to_ui(session.session_id, 'object_detection', data)


@sio.on('console')
async def handle_console(sid: str, data: Dict):
    """Receive console output"""
    session = _get_session_for_robot(sid)
    if not session:
        return
    await broadcast_to_ui(session.session_id, 'console', data)


# ============================================================================
# UI Event Handlers
# ============================================================================

@sio.on('join_session')
async def handle_join_session(sid: str, data: Dict):
    """
    UI client joins a robot session.
    Creates session if it doesn't exist (UI may connect before robot).
    Sends robot_ready immediately if session is already active.
    """
    session_id = data.get('session_id') if data else None

    if not session_id:
        await sio.emit('error', {'message': 'session_id required'}, room=sid)
        return

    # Create session if it doesn't exist yet
    if session_id not in sessions:
        sessions[session_id] = Session(session_id=session_id)

    session = sessions[session_id]

    # Add UI client
    if sid not in session.ui_sids:
        session.ui_sids.append(sid)
    ui_sid_to_session[sid] = session_id

    logger.info(f"[UI] Client {sid} joined session {session_id} (state: {session.state})")

    # Send current session state
    await sio.emit('session_state', {
        'session_id': session_id,
        'state': session.state,
        'robot_connected': session.robot_sid is not None,
        'sensor_data': session.sensor_data
    }, room=sid)

    # If session is already active, send robot_ready immediately
    if session.is_active():
        await sio.emit('robot_ready', {
            'session_id': session_id,
            'timestamp': datetime.utcnow().isoformat()
        }, room=sid)


@sio.on('move_command')
async def handle_move_command(sid: str, data: Dict):
    """UI sends movement command to robot."""
    session_id = ui_sid_to_session.get(sid)
    if not session_id:
        await sio.emit('error', {'message': 'Not in a session'}, room=sid)
        return

    session = sessions.get(session_id)
    if not session or session.state != 'active':
        await sio.emit('error', {'message': 'Session not active'}, room=sid)
        return

    if not session.robot_sid:
        await sio.emit('error', {'message': 'Robot not connected'}, room=sid)
        return

    # Forward to robot (root namespace, no namespace param)
    await sio.emit('move_cmd', data, room=session.robot_sid)


@sio.on('move_cmd')
async def handle_move_cmd(sid: str, data: Dict):
    """UI sends move_cmd (alternate event name)."""
    session_id = ui_sid_to_session.get(sid)
    if not session_id:
        # Could be from robot — check robot lookup
        session = _get_session_for_robot(sid)
        if session:
            # Robot is echoing back — ignore or forward to UI
            return
        return

    session = sessions.get(session_id)
    if not session or not session.robot_sid:
        return

    await sio.emit('move_cmd', data, room=session.robot_sid)


@sio.on('stop_command')
async def handle_stop_command(sid: str, data: Dict = None):
    """UI sends stop command to robot."""
    session_id = ui_sid_to_session.get(sid)
    if not session_id:
        return

    session = sessions.get(session_id)
    if not session or not session.robot_sid:
        return

    await sio.emit('move_cmd', {
        'type': 'twist',
        'linear_x': 0.0,
        'angular_z': 0.0
    }, room=session.robot_sid)


@sio.on('stop_cmd')
async def handle_stop_cmd(sid: str, data: Dict = None):
    """UI sends stop_cmd (alternate event name)."""
    session_id = ui_sid_to_session.get(sid)
    if not session_id:
        return

    session = sessions.get(session_id)
    if not session or not session.robot_sid:
        return

    await sio.emit('stop_cmd', data or {}, room=session.robot_sid)


@sio.on('twist_command')
async def handle_twist_command(sid: str, data: Dict):
    """Forward twist command from UI to robot."""
    session_id = ui_sid_to_session.get(sid)
    if not session_id:
        return

    session = sessions.get(session_id)
    if not session or not session.robot_sid:
        return

    await sio.emit('twist_command', data, room=session.robot_sid)


@sio.on('navigate_to_pose')
async def handle_navigate_to_pose(sid: str, data: Dict):
    """Forward navigation command from UI to robot."""
    session_id = ui_sid_to_session.get(sid)
    if not session_id:
        return

    session = sessions.get(session_id)
    if not session or not session.robot_sid:
        return

    await sio.emit('navigate_to_pose', data, room=session.robot_sid)


@sio.on('cancel_navigation')
async def handle_cancel_navigation(sid: str, data: Dict = None):
    """Forward cancel navigation from UI to robot."""
    session_id = ui_sid_to_session.get(sid)
    if not session_id:
        return

    session = sessions.get(session_id)
    if not session or not session.robot_sid:
        return

    await sio.emit('cancel_navigation', data or {}, room=session.robot_sid)


@sio.on('test_command')
async def handle_test_command(sid: str, data: Dict = None):
    """Forward test command from UI to robot."""
    session_id = ui_sid_to_session.get(sid)
    if not session_id:
        return

    session = sessions.get(session_id)
    if not session or not session.robot_sid:
        return

    await sio.emit('test_command', data or {}, room=session.robot_sid)


@sio.on('test_ws')
async def handle_test_ws(sid: str, data: Dict = None):
    """Forward test WS message to robot app's UI handler."""
    session_id = ui_sid_to_session.get(sid)
    if not session_id:
        return

    session = sessions.get(session_id)
    if session and session.robot_app:
        session.robot_app.ui.get_transport().inject_message(data or {})


@sio.on('occupancy_grid')
async def handle_occupancy_grid(sid: str, data: Dict):
    """Forward occupancy grid from UI to robot."""
    session_id = ui_sid_to_session.get(sid)
    if not session_id:
        return

    session = sessions.get(session_id)
    if not session or not session.robot_sid:
        return

    await sio.emit('occupancy_grid', data, room=session.robot_sid)


# ============================================================================
# REST API Endpoints
# ============================================================================

@app.get("/health")
async def health_check():
    """Traefik health check endpoint — required by platform"""
    return {"status": "ok"}


@app.get("/api")
async def root():
    """App info endpoint"""
    return {
        "app": "Remake RPC",
        "version": "2.0.0",
        "protocol": "APP_ROBOT_PROTOCOL v1.0",
        "status": "running"
    }


@app.get("/api/health")
async def health():
    """Detailed health check for platform"""
    return {
        "status": "healthy",
        "active_sessions": len(sessions),
        "timestamp": datetime.utcnow().isoformat()
    }


@app.get("/api/sessions")
async def list_sessions():
    """List all active sessions"""
    result = []
    for session_id, session in sessions.items():
        result.append({
            'session_id': session_id,
            'connected_at': session.connected_at,
            'state': session.state,
            'robot_connected': session.robot_sid is not None,
            'ui_clients': len(session.ui_sids)
        })
    return {'sessions': result}


@app.get("/api/sessions/{session_id}")
async def get_session(session_id: str):
    """Get session details"""
    session = sessions.get(session_id)
    if not session:
        return JSONResponse(
            status_code=404,
            content={'error': f'Session {session_id} not found'}
        )

    return {
        'session_id': session.session_id,
        'state': session.state,
        'robot_connected': session.robot_sid is not None,
        'ui_clients': len(session.ui_sids),
        'connected_at': session.connected_at
    }


@app.get("/api/object_detection/objects")
async def get_detected_objects(session_id: str):
    """Get list of currently detected objects"""
    session = sessions.get(session_id)
    if session and session.robot_app:
        return {
            "objects": session.robot_app.object_detector.get_detected_objects(),
            "summary": session.robot_app.object_detector.get_objects_summary(),
            "features": session.robot_app.object_detector.get_features_for_ui()
        }
    return {"objects": [], "summary": {"total_objects": 0}, "features": []}


@app.get("/api/robot/status")
async def robot_status(session_id: str):
    """Get robot connection status"""
    session = sessions.get(session_id)
    return {
        "connected": session is not None and session.robot_sid is not None,
        "ui_clients": len(session.ui_sids) if session else 0,
        "state": session.state if session else None
    }


# ============================================================================
# Static File Serving
# ============================================================================

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
|                    Remake RPC Server v2.0                     |
|                                                              |
|  Port: {Config.PORT}                                              |
|  App ID: {Config.APP_ID}                                   |
|  Appstore: {Config.APPSTORE_URL}          |
|                                                              |
|  Protocol: APP_ROBOT_PROTOCOL v1.0                           |
|  WebSocket: Root namespace /                                 |
|    Robot: auth={{type:'robot', session_id, session_token}}    |
|    UI:    join_session {{session_id}}                         |
|                                                              |
|  Flow: start_protocol -> app_signature ->                    |
|    signature_verified -> setup_app_cmd ->                     |
|    setup_app_response -> enable_remote_control_response ->   |
|    robot_ready                                               |
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
