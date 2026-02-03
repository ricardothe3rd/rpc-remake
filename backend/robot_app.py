import asyncio
import logging
import math
from typing import Optional, Callable
from robot_proxy import RobotProxy
from ui_proxy import UIProxy

logger = logging.getLogger(__name__)

class RobotApp:
    """Base class for robot applications"""

    def __init__(self, robot_transport, ui_transport):
        self.robot = RobotProxy(robot_transport)
        self.ui = UIProxy(ui_transport)
        self._task: Optional[asyncio.Task] = None
        self.app_stop_callback: Optional[Callable[[], None]] = None

        # Register disconnect callback to trigger app stop
        self.robot.get_transport().add_disconnect_callback(self._handle_transport_disconnect)

        logger.info("RobotApp initialized")

    def get_robot(self) -> RobotProxy:
        """Get the robot instance"""
        return self.robot

    def send_initial_data_to_ui(self):
        """Send initial data to UI client"""
        latest_data = self.robot.get_all_latest_messages()
        messages = [data for data_type, data in latest_data.items() if data]
        if messages:
            self.ui.get_transport().send_message_async(messages)

    def is_running(self) -> bool:
        """Check if the robot application is currently running"""
        return self._task is not None and not self._task.done()

    def set_app_stop_callback(self, callback: Callable[[], None]):
        """Set callback for when the app stops"""
        self.app_stop_callback = callback

    def start(self):
        """Start the robot application"""
        if self._task is None or self._task.done():
            self._task = asyncio.create_task(self.run())
            logger.info("RobotApp started")
        return self._task

    async def run(self):
        """Main robot workflow - to be implemented by subclasses"""
        pass

    def _handle_transport_disconnect(self):
        """Handle transport disconnect by stopping the app"""
        logger.info("Transport disconnected, stopping app")
        self.stop()

    def stop(self):
        """Stop the robot application"""
        # Only disconnect if we're still running and connected
        if self.is_running() and self.robot.get_transport().is_connected():
            asyncio.create_task(self.robot.disconnect())
        if self.is_running():
            self._task.cancel()
        logger.info("RobotApp stopped")
        if self.app_stop_callback:
            self.app_stop_callback()