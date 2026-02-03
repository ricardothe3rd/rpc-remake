import asyncio
import logging
import math
import json
import os
from typing import Optional, Callable
from robot_proxy import RobotProxy
from ui_proxy import UIProxy
from object_detector import ObjectDetector
from robot_app import RobotApp

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MyRobotApp(RobotApp):
    """
    Robot control application with object detection and chat integration.

    Modified for remake-rpc: Accepts pre-connected robot_transport instead of
    creating WebSocket connection.
    """

    def __init__(self, robot_transport, ui_transport,
                 session_id: Optional[str] = None,
                 session_token: Optional[str] = None,
                 appstore_url: Optional[str] = None):
        """
        Initialize robot app with pre-connected transports.

        Args:
            robot_transport: WebSocketWrapper or SocketIOWrapper for robot connection
            ui_transport: WebSocketWrapper or SocketIOWrapper for UI connection
            session_id: Session ID from platform (for AppstoreBridge)
            session_token: Session token from platform (for AppstoreBridge)
            appstore_url: Appstore URL (for AppstoreBridge)
        """
        super().__init__(robot_transport, ui_transport)
        self.object_detector = ObjectDetector(self.robot)

        # Store session credentials
        self.session_id = session_id
        self.session_token = session_token
        self.appstore_url = appstore_url

        # Initialize appstore bridge if credentials provided
        self.appstore = None
        if session_id and session_token and appstore_url:
            from appstore_bridge import AppstoreBridge
            self.appstore = AppstoreBridge(
                session_id=session_id,
                session_token=session_token,
                appstore_url=appstore_url
            )
            self.appstore.on_user_message(self.handle_user_chat_message)
            logger.info(f"AppstoreBridge initialized for session {session_id}")
        else:
            logger.info("Running in standalone mode (no appstore connection)")

        logger.info("MyRobotApp initialized")

    def stop(self):
        """Stop the robot application and clean up object detector"""
        # Clean up object detector first
        if hasattr(self, 'object_detector') and self.object_detector:
            self.object_detector.cleanup()

        # Call parent stop method
        super().stop()

    def handle_test_websocket(self, message):
        """Handle test websocket message from UI"""
        content = message.get('content', 'No content')
        logger.debug(f"🌐 Test WebSocket message received from UI: {content}")
        self.ui.send_console_message_async(f"Test WS received: {content}")

    async def handle_user_chat_message(self, data):
        """
        Handle chat message from user via appstore.

        Called when user sends a text message OR clicks an option button.
        When user clicks a button, 'selected_option_id' is included in data.
        """
        content = data.get('content', '').lower()
        message_id = data.get('message_id')
        selected_option_id = data.get('selected_option_id')

        if not self.appstore:
            logger.warning("[Chat] No appstore connection available")
            return

        try:
            # Check if user clicked a button (has selected_option_id)
            if selected_option_id:
                logger.info(f"[Chat] User clicked button: option_id='{selected_option_id}'")
                await self._execute_option(selected_option_id)
                return

            # Otherwise it's a text message - parse commands
            logger.info(f"[Chat] User text message: content='{content}'")

            if 'forward' in content:
                await self.execute_move_forward()
            elif 'backward' in content or 'back' in content:
                await self.execute_move_backward()
            elif 'spin' in content or 'rotate' in content:
                await self.execute_full_spin()
            elif 'status' in content:
                await self.execute_status()
            elif 'help' in content:
                await self.send_intro_message()
            elif content.strip():
                # Unrecognized command - show help
                await self.appstore.send_message(
                    "ℹ️ I didn't understand that. Let me show you what I can do:"
                )
                await self.send_intro_message()
            else:
                # Empty message
                logger.debug("[Chat] Empty message received")

        except Exception as e:
            logger.error(f"[Chat] Error handling user message: {e}")
            await self.appstore.send_message(f"❌ Error: {str(e)}")

    async def _execute_option(self, option_id: str):
        """Execute action for a clicked option button."""
        if option_id == 'forward':
            await self.execute_move_forward()
        elif option_id == 'backward':
            await self.execute_move_backward()
        elif option_id == 'spin':
            await self.execute_full_spin()
        elif option_id == 'status':
            await self.execute_status()
        elif option_id == 'help':
            await self.send_intro_message()
        else:
            logger.warning(f"[Chat] Unknown option: {option_id}")
            await self.appstore.send_message(f"❓ Unknown command: {option_id}")

    async def send_intro_message(self):
        """Send the intro message with action buttons"""
        await self.appstore.send_options(
            "Hi there! 👋 How can I help you today?",
            [
                {'id': 'forward', 'label': '⬆️ Move forward 50cm'},
                {'id': 'backward', 'label': '⬇️ Move backward 50cm'},
                {'id': 'spin', 'label': '🔄 Rotate 1 full spin'},
                {'id': 'status', 'label': '📊 Check status'}
            ],
            timeout_seconds=300,
            default_option=None
        )

    async def execute_move_forward(self):
        """Move robot forward 50cm"""
        await self.appstore.send_message("⬆️ Moving forward 50cm...")
        result = await self.robot.move_relative(0.5, 0.0)  # 0.5 meters forward
        if result.get("success"):
            await self.appstore.send_message("✅ Moved forward 50cm!")
        else:
            await self.appstore.send_message(f"❌ Move failed: {result.get('message', 'Unknown error')}")
        # Show options again
        await self.send_intro_message()

    async def execute_move_backward(self):
        """Move robot backward 50cm"""
        await self.appstore.send_message("⬇️ Moving backward 50cm...")
        result = await self.robot.move_relative(-0.5, 0.0)  # 0.5 meters backward
        if result.get("success"):
            await self.appstore.send_message("✅ Moved backward 50cm!")
        else:
            await self.appstore.send_message(f"❌ Move failed: {result.get('message', 'Unknown error')}")
        # Show options again
        await self.send_intro_message()

    async def execute_full_spin(self):
        """Rotate robot 360 degrees"""
        await self.appstore.send_message("🔄 Rotating 360 degrees...")
        # Full rotation = 2*pi radians = ~6.28 rad
        result = await self.robot.turn_relative(math.pi * 2)
        if result.get("success"):
            await self.appstore.send_message("✅ Completed full rotation!")
        else:
            await self.appstore.send_message(f"❌ Rotation failed: {result.get('message', 'Unknown error')}")
        # Show options again
        await self.send_intro_message()

    async def execute_status(self):
        """Show robot status"""
        try:
            spec = await self.robot.get_specification()
            battery_msg = self.robot.get_latest_message('battery')
            battery_level = battery_msg.get('level', 'unknown') if battery_msg else 'unknown'

            status_text = f"🤖 Robot: {spec.get('model_name', 'Unknown')}\n📊 Battery: {battery_level}%"
            await self.appstore.send_message(status_text)
        except Exception as e:
            await self.appstore.send_message(f"❌ Could not get status: {str(e)}")
        # Show options again
        await self.send_intro_message()

    # Helper methods for simple linear programming

    async def turn_toward_closest_object(self):
        """Turn toward the closest detected object"""
        closest_object = self.object_detector.find_closest_object()
        if not closest_object:
            await self.ui.send_console_message("No objects detected - cannot turn")
            return False

        await self.ui.send_console_message(f"Turning toward {closest_object.object_type} at {math.degrees(closest_object.angle):.1f}°")

        # Use robot_proxy turn_toward method
        result = await self.robot.turn_toward(closest_object)

        if result.get("success"):
            await self.ui.send_console_message("Turn command completed successfully")
            return True
        else:
            await self.ui.send_console_message(f"Turn command failed: {result.get('message', 'Unknown error')}")
            return False

    async def run(self):
        """
        Main robot workflow.

        This version connects to appstore and handles chat commands.
        The robot connection is already established via three-phase protocol.
        """
        self.robot.get_transport().add_message_callback(self.ui.get_transport().send_message_async)
        self.ui.add_message_callback('test_ws', self.handle_test_websocket)
        logger.info("MyRobotApp running")

        try:
            # Step 0: Connect to appstore if available
            if self.appstore:
                await self.appstore.connect()
                # Send intro message with action buttons
                await self.send_intro_message()
                logger.info("Connected to appstore")

            # Step 1: Get robot specification
            robot_spec = await self.robot.get_specification()
            logger.info("Specification received")
            model_name = self.robot.get_property('model_name', 'Unknown')
            manufacturer = self.robot.get_property('manufacturer', 'Unknown')
            await self.ui.send_console_message(f"Robot: {model_name} ({manufacturer})")

            # Step 1.1: Wait for GUI to connect
            await self.ui.wait_for_connection()

            # Step 1.5: Send initial data to connected UI
            self.send_initial_data_to_ui()

            # Keep running until stopped
            # Chat commands are handled via appstore callbacks
            while self.is_running:
                await asyncio.sleep(1.0)

        except asyncio.CancelledError:
            logger.info("Workflow cancelled due to robot disconnect")
            await self.ui.send_console_message("Robot disconnected - workflow stopped")
            raise  # Re-raise to properly handle task cancellation
        except Exception as e:
            logger.error(f"Error in workflow: {e}")
            await self.ui.send_console_message(f"Error: {e}")
        finally:
            # Disconnect from appstore if connected
            if self.appstore and self.appstore.is_connected():
                await self.appstore.disconnect()
                logger.info("Disconnected from appstore")
            self.stop()
