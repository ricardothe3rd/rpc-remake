from fastapi import WebSocket, WebSocketDisconnect
import json
import logging
import asyncio
from typing import Callable, List, Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class WebSocketWrapper:
    """Handles websocket transport management with callback support"""

    def __init__(self, websocket: Optional[WebSocket] = None):
        self.message_callbacks: List[Callable] = []
        self.disconnect_callbacks: List[Callable] = []
        self.connect_callbacks: List[Callable] = []
        self._message_loop_task = None
        self.websocket = None
        self.set_websocket(websocket)

    def set_websocket(self, websocket: Optional[WebSocket]):
        """Set the websocket (for cases where it's not available at construction)"""
        if websocket is None and self.websocket is not None:
            # Websocket is being cleared, call disconnect callbacks
            logger.info("Websocket disconnected")
            self._call_callbacks(self.disconnect_callbacks)

        self.websocket = websocket
        if websocket:
            self._call_callbacks(self.connect_callbacks)

    def add_message_callback(self, callback: Callable):
        """Register a callback to be called for each received message"""
        self.message_callbacks.append(callback)

    def remove_message_callback(self, callback: Callable):
        """Remove a registered callback"""
        if callback in self.message_callbacks:
            self.message_callbacks.remove(callback)

    def add_disconnect_callback(self, callback: Callable):
        """Register a callback to be called when the websocket disconnects"""
        self.disconnect_callbacks.append(callback)

    def remove_disconnect_callback(self, callback: Callable):
        """Remove a registered disconnect callback"""
        if callback in self.disconnect_callbacks:
            self.disconnect_callbacks.remove(callback)

    def add_connect_callback(self, callback: Callable):
        """Register a callback to be called when the websocket connects"""
        self.connect_callbacks.append(callback)

    def remove_connect_callback(self, callback: Callable):
        """Remove a registered connect callback"""
        if callback in self.connect_callbacks:
            self.connect_callbacks.remove(callback)

    def _call_callbacks(self, callbacks: List[Callable], message: dict = None):
        """Call all callbacks in the provided list"""
        for callback in callbacks:
            try:
                if message is not None:
                    result = callback(message)
                else:
                    result = callback()

                # If the callback returns a coroutine, schedule it as a task
                if asyncio.iscoroutine(result):
                    asyncio.create_task(result)
            except Exception as e:
                logger.error(f"Error in callback: {e}")

    def inject_message(self, message: dict):
        """Inject a message to all registered message callbacks"""
        self._call_callbacks(self.message_callbacks, message)

    async def run_message_loop(self):
        """Run the websocket message loop"""
        logger.debug("Starting websocket message loop")
        try:
            while True:
                data = await self.websocket.receive_text()
                try:
                    message = json.loads(data)
                    logger.debug(f"Received from websocket: {message.get('type', 'unknown')}")

                    # Call all registered callbacks
                    self.inject_message(message)

                except json.JSONDecodeError:
                    logger.error(f"Invalid JSON from websocket: {data}")

        except WebSocketDisconnect:
            logger.info("WebSocket disconnected normally")
            self.set_websocket(None)

    async def send_message(self, message):
        """Send a message or list of messages through the websocket"""
        if not self.is_connected():
            # Silently ignore if not connected (prevents race conditions)
            logger.debug("Cannot send message: WebSocket not connected")
            return

        try:
            if isinstance(message, list):
                # Send multiple messages
                for msg in message:
                    await self.websocket.send_text(json.dumps(msg))
            else:
                # Send single message
                await self.websocket.send_text(json.dumps(message))
        except Exception as e:
            logger.warning(f"Error sending message: {e}")
            # Mark as disconnected if send fails
            self.set_websocket(None)

    def send_message_async(self, message: dict):
        """Send a message asynchronously as a background task"""
        # Always create the task - send_message will handle disconnection gracefully
        try:
            asyncio.create_task(self.send_message(message))
        except Exception as e:
            logger.debug(f"Could not create send task: {e}")

    def get_websocket(self) -> Optional[WebSocket]:
        """Get the connected websocket"""
        return self.websocket

    def is_connected(self) -> bool:
        """Check if websocket is connected"""
        return self.websocket is not None

