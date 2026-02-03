"""
AppstoreBridge - Socket.IO client for connecting RPC_v1 to appstore messaging system

This module provides bidirectional communication between RPC_v1 robot control system
and the appstore platform, enabling users to chat with robots while controlling them.
"""

import socketio
import asyncio
import logging
import os
from typing import Optional, Callable, List, Dict, Any

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AppstoreBridge:
    """
    Bridge for connecting RPC_v1 backend to appstore Socket.IO namespace

    Handles:
    - Connection to appstore /sessions/{sessionId}/robot namespace
    - Authentication with session token
    - Listening for user_message events from appstore
    - Sending app_message events to appstore
    - Clean async/await interface for MyRobotApp
    """

    def __init__(self, session_id: str, session_token: str, appstore_url: str = None):
        """
        Initialize the bridge.

        Args:
            session_id: Session ID from appstore
            session_token: Session token for authentication
            appstore_url: Base URL of appstore backend (defaults to APPSTORE_URL env var or localhost)
        """
        self.session_id = session_id
        self.session_token = session_token
        self.appstore_url = appstore_url or os.getenv('APPSTORE_URL', 'http://localhost:5000')
        self.namespace = f'/sessions/{session_id}/robot'

        # Socket.IO client
        self.sio = socketio.AsyncClient(
            reconnection=True,
            reconnection_delay=1,
            reconnection_delay_max=5,
            reconnection_attempts=5
        )

        # Callback for user messages (text or button clicks)
        self.user_message_callback: Optional[Callable] = None

        # Track connection state
        self._connected = False

        logger.info(f"[AppstoreBridge] Initialized for session {session_id}")

    async def connect(self) -> bool:
        """
        Connect to appstore Socket.IO namespace.

        Returns:
            True if connection successful, False otherwise
        """
        try:
            # Event to signal when server confirms connection
            connection_confirmed = asyncio.Event()

            # Register event handlers BEFORE connecting
            @self.sio.event(namespace=self.namespace)
            async def connected(data=None):
                """Called when connected to appstore namespace"""
                self._connected = True
                logger.info(f"[AppstoreBridge] Connected to appstore session {self.session_id}")
                connection_confirmed.set()  # Signal that connection is confirmed
                await self._handle_connected()

            @self.sio.event(namespace=self.namespace)
            async def user_message(data):
                """Called when user sends a text or button click message"""
                logger.info(f"[AppstoreBridge] Received user_message: {data.get('content')}, option={data.get('selected_option_id')}")
                await self._handle_user_message(data)

            @self.sio.event(namespace=self.namespace)
            async def disconnect():
                """Called when disconnected from appstore"""
                self._connected = False
                logger.warning(f"[AppstoreBridge] Disconnected from appstore session {self.session_id}")

            # Connect to appstore with authentication
            logger.info(f"[AppstoreBridge] Connecting to {self.appstore_url}{self.namespace}")
            await self.sio.connect(
                self.appstore_url,
                namespaces=[self.namespace],
                auth={'token': self.session_token},
                wait_timeout=10
            )

            # Wait for server to confirm connection (max 10 seconds)
            try:
                await asyncio.wait_for(connection_confirmed.wait(), timeout=10.0)
                logger.info(f"[AppstoreBridge] Connection confirmed, ready to send messages")
                return True
            except asyncio.TimeoutError:
                logger.error(f"[AppstoreBridge] Server did not confirm connection within 10 seconds")
                return False

        except Exception as e:
            logger.error(f"[AppstoreBridge] Connection failed: {e}")
            self._connected = False
            return False

    async def send_message(self, content: str, metadata: Optional[Dict[str, Any]] = None) -> bool:
        """
        Send a simple text message to user via appstore.

        Args:
            content: Message text
            metadata: Optional metadata dict (stored with message)

        Returns:
            True if sent successfully, False otherwise
        """
        if not self._connected:
            logger.warning("[AppstoreBridge] Cannot send message - not connected to appstore")
            return False

        try:
            await self.sio.emit('app_message', {
                'content': content,
                'metadata': metadata
            }, namespace=self.namespace)
            logger.info(f"[AppstoreBridge] Sent message: {content[:50]}...")
            return True
        except Exception as e:
            logger.error(f"[AppstoreBridge] Failed to send message: {e}")
            return False

    async def send_options(self,
                          content: str,
                          options: List[Dict[str, str]],
                          timeout_seconds: int = 300,
                          default_option: Optional[str] = None) -> bool:
        """
        Send a message with selectable button options to user.

        Args:
            content: Message text
            options: List of option dicts with 'id' and 'label' keys
                    Example: [{'id': 'turn', 'label': 'Turn toward object'}, ...]
            timeout_seconds: Seconds until auto-select default_option
            default_option: Option ID to auto-select if no response (can be None)

        Returns:
            True if sent successfully, False otherwise
        """
        if not self._connected:
            logger.warning("[AppstoreBridge] Cannot send options - not connected to appstore")
            return False

        try:
            await self.sio.emit('app_message_with_options', {
                'content': content,
                'options': options,
                'timeout_seconds': timeout_seconds,
                'default_option': default_option
            }, namespace=self.namespace)
            logger.info(f"[AppstoreBridge] Sent message with {len(options)} options")
            return True
        except Exception as e:
            logger.error(f"[AppstoreBridge] Failed to send options: {e}")
            return False

    def on_user_message(self, callback: Callable) -> None:
        """
        Register a callback function to handle user messages (text or button clicks).

        The callback will be called with a dict containing:
        - message_id: str
        - content: str (the user's message text or button label)
        - selected_option_id: str (optional, present when user clicked a button)
        - timestamp: str

        Args:
            callback: Async function that takes (data: dict) argument
        """
        self.user_message_callback = callback
        logger.info("[AppstoreBridge] User message callback registered")

    async def _handle_user_message(self, data: Dict[str, Any]) -> None:
        """
        Internal handler for user_message events from appstore.

        Called when user sends a text message OR clicks an option button.
        Invokes the registered callback if one exists.
        """
        if self.user_message_callback:
            try:
                await self.user_message_callback(data)
            except Exception as e:
                logger.error(f"[AppstoreBridge] Error in user message callback: {e}")
        else:
            logger.warning("[AppstoreBridge] Received user_message but no callback registered")

    async def _handle_connected(self) -> None:
        """Internal handler for connection confirmation from appstore."""
        logger.info("[AppstoreBridge] Appstore confirmed connection")

    async def disconnect(self) -> None:
        """Disconnect from appstore namespace."""
        try:
            await self.sio.disconnect()
            self._connected = False
            logger.info("[AppstoreBridge] Disconnected from appstore")
        except Exception as e:
            logger.error(f"[AppstoreBridge] Error during disconnect: {e}")

    def is_connected(self) -> bool:
        """Check if currently connected to appstore."""
        return self._connected

    async def send_navigate_to_pose(self, x: float, y: float, yaw: float = 0.0, relative: bool = True) -> bool:
        """
        Send navigation command to robot via AppStore session namespace.

        Args:
            x: X position (meters)
            y: Y position (meters)
            yaw: Rotation in radians
            relative: If True, move relative to current position

        Returns:
            True if sent successfully
        """
        if not self._connected:
            logger.warning("[AppstoreBridge] Cannot send navigate - not connected")
            return False

        try:
            # Use 'navigate_cmd' event name (matches AppStore server.js)
            await self.sio.emit('navigate_cmd', {
                'x': x,
                'y': y,
                'yaw': yaw,
                'relative': relative
            }, namespace=self.namespace)
            logger.info(f"[AppstoreBridge] Sent navigate_cmd: x={x}, y={y}, yaw={yaw}, relative={relative}")
            return True
        except Exception as e:
            logger.error(f"[AppstoreBridge] Failed to send navigate: {e}")
            return False

    async def send_twist_command(self, linear_x: float, angular_z: float) -> bool:
        """
        Send twist (velocity) command to robot via AppStore session namespace.

        Args:
            linear_x: Forward/backward velocity (m/s)
            angular_z: Rotation velocity (rad/s)

        Returns:
            True if sent successfully
        """
        if not self._connected:
            logger.warning("[AppstoreBridge] Cannot send twist - not connected")
            return False

        try:
            await self.sio.emit('twist_command', {
                'linear_x': linear_x,
                'angular_z': angular_z
            }, namespace=self.namespace)
            logger.info(f"[AppstoreBridge] Sent twist_command: linear={linear_x}, angular={angular_z}")
            return True
        except Exception as e:
            logger.error(f"[AppstoreBridge] Failed to send twist: {e}")
            return False

    async def wait_for_connection(self, timeout: float = 10.0) -> bool:
        """
        Wait for connection to be established.

        Args:
            timeout: Seconds to wait before giving up

        Returns:
            True if connected within timeout, False otherwise
        """
        start_time = asyncio.get_event_loop().time()
        while not self._connected:
            if asyncio.get_event_loop().time() - start_time > timeout:
                logger.warning(f"[AppstoreBridge] Connection timeout after {timeout}s")
                return False
            await asyncio.sleep(0.1)
        return True
