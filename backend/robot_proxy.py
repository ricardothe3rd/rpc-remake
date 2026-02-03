import json
import asyncio
import logging
import math
from typing import Dict, Any, Optional, Callable
from websocket_wrapper import WebSocketWrapper
from message_callback_mixin import MessageCallbackMixin

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RobotProxy(MessageCallbackMixin):
    def __init__(self, transport: WebSocketWrapper):
        super().__init__()  # Initialize MessageCallbackMixin
        self.transport = transport

        # Register all known message types for callbacks and latest message storage
        known_message_types = [
            'laser_scan', 'battery', 'wifi', 'cmd_vel', 'map', 'camera',
            'navigation_status', 'navigation_feedback', 'robot_pose', 'robot_specification'
        ]
        for message_type in known_message_types:
            self.register_message_type(message_type)

        # Register inject_message as a message callback so data updates happen automatically
        self.transport.add_message_callback(self.inject_message)

    def get_transport(self):
        """Get the transport layer for direct access to transport methods"""
        return self.transport

    async def send_twist_command(self, linear_x: float, angular_z: float):
        message = {"type": "twist", "linear_x": linear_x, "angular_z": angular_z}
        await self.transport.send_message(message)
        logger.info(f"Sent twist command: linear_x={linear_x}, angular_z={angular_z}")
        return {"status": "sent", "command": message}

    async def send_occupancy_grid(self, grid_data: dict):
        message = {"type": "occupancy_grid", "grid_data": grid_data}
        await self.transport.send_message(message)
        logger.debug(f"Sent OccupancyGrid command: {grid_data.get('width', 'unknown')}x{grid_data.get('height', 'unknown')}")
        return {"status": "sent", "command": message}

    async def send_navigate_to_pose(self, x: float, y: float, z: float = 0.0,
                                  qx: float = 0.0, qy: float = 0.0, qz: float = 0.0,
                                  qw: float = 1.0, frame_id: str = "map", relative: bool = False):
        pose_data = {
            "x": x, "y": y, "z": z,
            "qx": qx, "qy": qy, "qz": qz, "qw": qw,
            "frame_id": frame_id
        }
        message = {"type": "navigate_to_pose", "pose": pose_data, "relative": relative}
        await self.transport.send_message(message)
        navigation_type = "relative" if relative else "absolute"
        logger.debug(f"Sent {navigation_type} navigate_to_pose command: ({x}, {y})")
        return {"status": "sent", "command": message}

    async def cancel_navigation(self):
        message = {"type": "cancel_navigation"}
        await self.transport.send_message(message)
        logger.debug("Sent cancel_navigation command")
        return {"status": "sent", "command": message}

    async def send_get_robot_specification_command(self):
        message = {"type": "get_robot_specification"}
        await self.transport.send_message(message)
        logger.debug("Sent get_robot_specification command")
        return {"status": "sent", "command": message}

    async def send_end_session_command(self):
        message = {"type": "end_session"}
        await self.transport.send_message(message)
        logger.debug("Sent end_session command")
        return {"status": "sent", "command": message}

    async def disconnect(self):
        """Disconnect the robot"""
        try:
            result = await self.send_end_session_command()
            logger.info(f"Disconnect result: {result}")
        except RuntimeError as e:
            logger.warning(f"Could not send disconnect command (connection already closed): {e}")
        await asyncio.sleep(1.0)

    async def get_specification(self):
        """Request and wait for robot specification"""
        # Send the command to get robot specification
        result = await self.send_get_robot_specification_command()
        logger.debug(f"Sent get_robot_specification command: {result}")

        # Wait for robot_specification message
        return await self.wait_for_message('robot_specification')

    def get_property(self, property_path: str, default=None):
        """
        Get a property from the robot specification using dot notation.

        Args:
            property_path: Dot-separated path like 'lidar_sensor.position.x_m'
            default: Default value if property is not found

        Returns:
            The property value or default if not found

        Example:
            get_property('model_name') -> 'CleanBot Pro 3000'
            get_property('lidar_sensor.position.x_m') -> 0.0
        """
        robot_spec = self.get_latest_message('robot_specification')
        if not robot_spec:
            return default

        # Split the path and traverse the nested structure
        keys = property_path.split('.')
        current = robot_spec

        for key in keys:
            if isinstance(current, dict) and key in current:
                current = current[key]
            else:
                return default

        return current

    async def turn_toward(self, closest_object):
        """Turn toward a detected object"""
        # Get turn angle
        angle = closest_object.angle

        # Use turn_relative to turn in place (yaw=angle)
        result = await self.turn_relative(yaw=angle)

        return result

    async def approach(self, obj, stop_distance: float, timeout_seconds: float = 10.0):
        """Approach a detected object and stop at specified distance

        Args:
            obj: Object with distance and angle properties
            stop_distance: Distance to maintain from the object (meters)
            timeout_seconds: Maximum time to wait for navigation completion (default: 10.0)

        Returns:
            dict: Result with 'success' (bool), 'status' (str), and 'message' (str)
        """
        # Calculate how far to move (object distance minus desired stop distance)
        move_distance = obj.distance - stop_distance

        # Calculate x and y components based on object angle
        x = move_distance * math.cos(obj.angle)
        y = move_distance * math.sin(obj.angle)

        # Use move_relative to move towards the object and face it
        result = await self.move_relative(x, y, yaw=obj.angle, timeout_seconds=timeout_seconds)

        return result

    async def turn_relative(self, yaw: float, timeout_seconds: float = 10.0):
        """Turn relative to current orientation

        Args:
            yaw: Angle to turn in radians (positive=counter-clockwise, negative=clockwise)
            timeout_seconds: Maximum time to wait for navigation completion (default: 10.0)

        Returns:
            dict: Result with 'success' (bool), 'status' (str), and 'message' (str)
        """
        # Use move_relative to turn in place (x=0, y=0, yaw=angle)
        result = await self.move_relative(0.0, 0.0, yaw=yaw, timeout_seconds=timeout_seconds)

        return result

    async def move_relative(self, x: float, y: float, yaw: float = 0.0, timeout_seconds: float = 10.0):
        """Move relative to current position

        Args:
            x: Distance to move in X direction (meters, positive=forward, negative=backward)
            y: Distance to move in Y direction (meters, positive=left, negative=right)
            yaw: Angle to turn in radians (positive=counter-clockwise, negative=clockwise, default: 0.0)
            timeout_seconds: Maximum time to wait for navigation completion (default: 10.0)

        Returns:
            dict: Result with 'success' (bool), 'status' (str), and 'message' (str)
        """
        # Calculate quaternion from yaw angle (similar to turn_toward_closest_object)
        qz = math.sin(yaw / 2)
        qw = math.cos(yaw / 2)

        logger.info(f"Moving relative: x={x}m, y={y}m, yaw={math.degrees(yaw):.1f}°")

        result = await self.send_navigate_to_pose(
            x=x, y=y,
            qx=0.0, qy=0.0, qz=qz, qw=qw,
            frame_id="base_link",
            relative=True
        )

        if result.get("status") == "sent":
            logger.info("Relative movement command sent, waiting for completion...")

            # Wait for navigation status with specified timeout
            nav_status = await self.wait_for_message('navigation_status', timeout_seconds)

            if nav_status:
                status = nav_status.get('status', 'unknown')
                message = nav_status.get('message', '')

                if status == 'succeeded':
                    logger.info("Relative movement completed successfully!")
                    return {"success": True, "status": status, "message": "Relative movement completed successfully!"}
                elif status == 'failed':
                    logger.warning(f"Relative movement failed: {message}")
                    return {"success": False, "status": status, "message": f"Relative movement failed: {message}"}
                elif status == 'cancelled':
                    logger.warning("Relative movement was cancelled")
                    return {"success": False, "status": status, "message": "Relative movement was cancelled"}
                else:
                    logger.info(f"Navigation status: {status}")
                    # For 'navigating' or other statuses, wait a bit more
                    await asyncio.sleep(2.0)
                    return {"success": True, "status": status, "message": f"Navigation status: {status}"}
            else:
                logger.warning("Timeout waiting for navigation status")
                return {"success": False, "status": "timeout", "message": "Timeout waiting for navigation status"}
        else:
            logger.error("Failed to send relative movement command")
            return {"success": False, "status": "command_failed", "message": "Failed to send relative movement command"}