import json
import asyncio
import logging
from typing import Dict, Any, Optional, Callable
from websocket_wrapper import WebSocketWrapper
from message_callback_mixin import MessageCallbackMixin

logger = logging.getLogger(__name__)


class UIProxy(MessageCallbackMixin):
    def __init__(self, transport: WebSocketWrapper):
        super().__init__()  # Initialize MessageCallbackMixin
        self.transport = transport

        # Register all known UI message types for callbacks and latest message storage
        known_message_types = [
            'test_ws',
            'console',
            'button_press',
            'ui_command'
        ]
        for message_type in known_message_types:
            self.register_message_type(message_type)

        # Register inject_message as a message callback so data updates happen automatically
        self.transport.add_message_callback(self.inject_message)

        logger.info("UIProxy initialized")

    def get_transport(self) -> WebSocketWrapper:
        """Get the underlying transport"""
        return self.transport

    @staticmethod
    def _create_console_message(content: str) -> dict:
        """Create a console message dict"""
        return {
            "type": "console",
            "content": content,
            "timestamp": asyncio.get_event_loop().time()
        }

    async def send_console_message(self, content: str):
        """Send a console message to the UI"""
        logger.info(f"Console: {content}")
        message = self._create_console_message(content)
        try:
            await self.transport.send_message(message)
        except RuntimeError as e:
            if "not connected" in str(e):
                logger.debug(f"UI not connected, skipping message: {content}")
            else:
                raise

    def send_console_message_async(self, content: str):
        """Send a message to the GUI console"""
        logger.info(f"Console: {content}")
        message = self._create_console_message(content)
        self.transport.send_message_async(message)

    async def wait_for_connection(self):
        """Wait until at least one GUI is connected"""
        await self.send_console_message("Waiting for GUI to connect...")
        while not self.transport.is_connected():
            await asyncio.sleep(0.1)
        await self.send_console_message("GUI connected! Starting workflow...")