import math
import time
import base64
from typing import List, Dict, Any, Optional, TYPE_CHECKING, Tuple
import logging
import numpy as np
from collections import deque
from dataclasses import dataclass
from scipy import spatial
from sklearn.cluster import DBSCAN

if TYPE_CHECKING:
    from robot_proxy import RobotProxy

# Set up logger with INFO level for this module
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)  # Set to INFO to see detection progress

@dataclass
class GeometricFeature:
    """Base class for geometric features detected in laser scans"""
    feature_type: str  # 'line', 'circle', 'corner'
    points: List[Tuple[float, float]]  # cartesian points
    confidence: float
    parameters: Dict[str, Any]  # feature-specific parameters

@dataclass
class LineFeature(GeometricFeature):
    """Line feature with start/end points and equation"""
    def __post_init__(self):
        self.feature_type = 'line'
        if len(self.points) >= 2:
            self.parameters['start'] = self.points[0]
            self.parameters['end'] = self.points[-1]
            self.parameters['length'] = self._calculate_length()
            self.parameters['angle'] = self._calculate_angle()

    def _calculate_length(self) -> float:
        start, end = self.points[0], self.points[-1]
        return math.sqrt((end[0] - start[0])**2 + (end[1] - start[1])**2)

    def _calculate_angle(self) -> float:
        start, end = self.points[0], self.points[-1]
        return math.atan2(end[1] - start[1], end[0] - start[0])

@dataclass
class CircleFeature(GeometricFeature):
    """Circle feature with center and radius"""
    def __post_init__(self):
        self.feature_type = 'circle'
        if len(self.points) >= 3:
            center, radius = self._fit_circle()
            self.parameters['center'] = center
            self.parameters['radius'] = radius
            self.parameters['arc_length'] = len(self.points)

    def _fit_circle(self) -> Tuple[Tuple[float, float], float]:
        # Simple circle fitting using least squares
        points = np.array(self.points)
        x, y = points[:, 0], points[:, 1]

        # Set up system of equations: (x-a)^2 + (y-b)^2 = r^2
        A = np.column_stack([2*x, 2*y, np.ones(len(x))])
        b = x**2 + y**2

        try:
            solution = np.linalg.lstsq(A, b, rcond=None)[0]
            center = (solution[0], solution[1])
            radius = math.sqrt(solution[2] + solution[0]**2 + solution[1]**2)
            return center, radius
        except:
            # Fallback to centroid and average distance
            center = (np.mean(x), np.mean(y))
            radius = np.mean(np.sqrt((x - center[0])**2 + (y - center[1])**2))
            return center, radius

@dataclass
class CornerFeature(GeometricFeature):
    """Corner feature formed by two intersecting lines"""
    def __post_init__(self):
        self.feature_type = 'corner'
        if len(self.points) >= 3:
            corner_point, angle = self._find_corner()
            self.parameters['corner_point'] = corner_point
            self.parameters['interior_angle'] = angle

    def _find_corner(self) -> Tuple[Tuple[float, float], float]:
        points = np.array(self.points)
        # Use the middle point as corner and calculate angle
        if len(points) >= 3:
            mid_idx = len(points) // 2
            corner = tuple(points[mid_idx])

            # Calculate vectors from corner to nearby points
            vec1 = points[0] - points[mid_idx]
            vec2 = points[-1] - points[mid_idx]

            # Calculate angle between vectors
            dot_product = np.dot(vec1, vec2)
            norms = np.linalg.norm(vec1) * np.linalg.norm(vec2)
            if norms > 0:
                angle = math.acos(np.clip(dot_product / norms, -1.0, 1.0))
            else:
                angle = 0

            return corner, angle
        return (0, 0), 0

class DetectedObject:
    """Represents a detected object from laser scan data"""
    def __init__(self, object_type: str, features: List[GeometricFeature],
                 confidence: float, position: Optional[Tuple[float, float]] = None):
        self.object_type = object_type
        self.features = features
        self.confidence = confidence
        self.timestamp = time.time()
        self.position = position or self._calculate_center_position()

        # For backward compatibility
        self.distance = math.sqrt(self.position[0]**2 + self.position[1]**2)
        self.angle = math.atan2(self.position[1], self.position[0])
        self.x, self.y = self.position

    def _calculate_center_position(self) -> Tuple[float, float]:
        """Calculate the center position from all feature points"""
        all_points = []
        for feature in self.features:
            all_points.extend(feature.points)

        if all_points:
            x_coords = [p[0] for p in all_points]
            y_coords = [p[1] for p in all_points]
            return (sum(x_coords) / len(x_coords), sum(y_coords) / len(y_coords))
        return (0, 0)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            "type": self.object_type,
            "distance": round(self.distance, 2),
            "angle_rad": round(self.angle, 3),
            "angle_deg": round(math.degrees(self.angle), 1),
            "x": round(self.x, 2),
            "y": round(self.y, 2),
            "confidence": round(self.confidence, 2),
            "timestamp": time.strftime("%H:%M:%S", time.localtime(self.timestamp)),
            "features": [{
                "type": f.feature_type,
                "confidence": f.confidence,
                "parameters": f.parameters,
                "point_count": len(f.points)
            } for f in self.features]
        }

class ObjectDetector:
    """Real object detector that processes laser scan data using geometric feature detection"""

    def __init__(self, robot: "RobotProxy", laser_scan_message_type: str = 'laser_scan'):
        self.robot = robot
        self.laser_scan_message_type = laser_scan_message_type
        self.latest_scan_data = None
        self.detected_objects: List[DetectedObject] = []
        self.detected_features: List[GeometricFeature] = []
        self.processing_enabled = True

        # Geometric feature detection parameters
        self.line_fitting_threshold = 0.05  # meters
        self.min_line_length = 0.2  # meters (reduced from 0.3 to detect smaller features)
        self.circle_fitting_threshold = 0.1  # meters
        self.min_circle_points = 8
        self.corner_angle_threshold = math.pi / 6  # 30 degrees

        # Temporal analysis for motion detection
        self.scan_history = deque(maxlen=10)  # Store last 10 scans
        self.motion_threshold = 0.2  # meters

        # Map data
        self.latest_map_data = None
        self.latest_robot_pose = None

        # Subscribe to messages from the robot
        self.robot.add_message_callback(laser_scan_message_type, self.process_scan_data)
        self.robot.add_message_callback('map', self.update_map_data)
        self.robot.add_message_callback('robot_pose', self.update_robot_pose)

        logger.info("Real ObjectDetector initialized with geometric feature detection")

    def cleanup(self) -> None:
        """Clean up the object detector by unregistering callbacks and disabling processing"""
        logger.info("Cleaning up ObjectDetector")

        # Disable processing first
        self.processing_enabled = False

        # Remove all registered callbacks
        self.robot.remove_message_callback(self.laser_scan_message_type, self.process_scan_data)
        self.robot.remove_message_callback('map', self.update_map_data)
        self.robot.remove_message_callback('robot_pose', self.update_robot_pose)

        # Clear any stored data
        self.detected_objects.clear()
        self.detected_features.clear()
        self.scan_history.clear()

        logger.info("ObjectDetector cleanup completed")

    def update_map_data(self, map_data: Dict[str, Any]) -> None:
        """Update stored map data"""
        self.latest_map_data = map_data
        logger.debug("Updated map data")

    def update_robot_pose(self, pose_data: Dict[str, Any]) -> None:
        """Update stored robot pose"""
        self.latest_robot_pose = pose_data
        logger.debug("Updated robot pose")

    def _decode_binary_ranges(self, base64_string: str) -> List[float]:
        """Decode binary ranges from base64 encoded float32 array"""
        try:
            binary_string = base64.b64decode(base64_string)
            floats = np.frombuffer(binary_string, dtype=np.float32)
            return floats.tolist()
        except Exception as e:
            logger.error(f"Error decoding binary ranges: {e}")
            return []

    def process_scan_data(self, scan_data: Dict[str, Any]) -> None:
        """Process laser scan data and detect real objects using geometric features"""
        if not self.processing_enabled:
            return

        try:
            self.latest_scan_data = scan_data

            # Handle both binary and plain ranges format
            ranges = scan_data.get('ranges', [])
            if not ranges and 'ranges_binary' in scan_data:
                # Decode binary ranges if present
                ranges = self._decode_binary_ranges(scan_data['ranges_binary'])
                logger.debug(f"Decoded {len(ranges)} ranges from binary format")

            angle_min = scan_data.get('angle_min', -math.pi)
            angle_increment = scan_data.get('angle_increment', math.pi / 180)

            logger.debug(f"Processing scan with {len(ranges)} range readings")

            if not ranges:
                logger.warning("Received scan data with no ranges")
                return

            # Convert to cartesian coordinates and filter valid points
            points = self._polar_to_cartesian(ranges, angle_min, angle_increment)
            logger.debug(f"Converted to {len(points)} valid cartesian points")
            if not points:
                logger.warning("No valid points after polar to cartesian conversion")
                return

            # Store scan for temporal analysis
            self.scan_history.append({
                'timestamp': time.time(),
                'points': points,
                'robot_pose': self.latest_robot_pose
            })

            # Clear previous detections
            self.detected_objects.clear()

            # Detect geometric features
            features = self._detect_geometric_features(points)
            self.detected_features = features

            # Classify objects based on features and context
            self._classify_objects(features)

            logger.info(f"Detected {len(self.detected_objects)} objects from {len(features)} geometric features")

        except Exception as e:
            logger.error(f"Error processing scan data: {e}")

    def _polar_to_cartesian(self, ranges: List[float], angle_min: float, angle_increment: float) -> List[Tuple[float, float, float]]:
        """Convert polar laser scan data to cartesian coordinates with angle information"""
        points = []
        max_range = 8.0  # ignore points beyond this range
        min_range = 0.1  # ignore points closer than this

        for i, range_val in enumerate(ranges):
            if (math.isnan(range_val) or math.isinf(range_val) or
                range_val > max_range or range_val < min_range):
                continue

            angle = angle_min + i * angle_increment
            x = range_val * math.cos(angle)
            y = range_val * math.sin(angle)
            points.append((x, y, angle))  # Include angle for better corner detection

        return points

    def _detect_geometric_features(self, points: List[Tuple[float, float, float]]) -> List[GeometricFeature]:
        """Detect geometric features (lines, circles, corners) from point cloud"""
        if len(points) < 3:
            return []

        features = []

        # Use DBSCAN clustering to group nearby points (use only x, y for clustering)
        points_xy = np.array([(p[0], p[1]) for p in points])
        clustering = DBSCAN(eps=0.15, min_samples=3).fit(points_xy)

        unique_labels = set(clustering.labels_)
        for label in unique_labels:
            if label == -1:  # noise points
                continue

            cluster_mask = clustering.labels_ == label
            cluster_points = [points[i] for i, mask in enumerate(cluster_mask) if mask]

            if len(cluster_points) < 3:
                continue

            # Try to fit all three geometric primitives and compare fit errors
            line_feature, line_error = self._fit_line_with_error(cluster_points)
            circle_feature, circle_error = self._fit_circle_with_error(cluster_points)
            corner_feature, corner_error = self._fit_corner_with_error(cluster_points)

            # Choose the feature with the lowest error
            best_feature = None
            best_error = float('inf')

            if line_feature and line_error < best_error:
                best_feature = line_feature
                best_error = line_error

            if circle_feature and circle_error < best_error:
                best_feature = circle_feature
                best_error = circle_error

            if corner_feature and corner_error < best_error:
                best_feature = corner_feature
                best_error = corner_error

            if best_feature:
                features.append(best_feature)

        return features

    def _fit_line(self, points: List[Tuple[float, float]]) -> Optional[LineFeature]:
        """Fit a line to a set of points using RANSAC-like approach"""
        if len(points) < 3:
            return None

        points_array = np.array(points)

        # Use SVD to fit line
        centroid = np.mean(points_array, axis=0)
        centered_points = points_array - centroid

        try:
            _, _, v = np.linalg.svd(centered_points)
            line_direction = v[0]  # First principal component

            # Calculate distances from points to line
            distances = []
            for point in points_array:
                point_to_centroid = point - centroid
                # Distance from point to line
                distance = np.linalg.norm(point_to_centroid - np.dot(point_to_centroid, line_direction) * line_direction)
                distances.append(distance)

            # Check if points are close enough to be considered a line
            avg_distance = np.mean(distances)
            if avg_distance > self.line_fitting_threshold:
                return None

            # Check minimum line length
            line_extent = np.max(np.dot(centered_points, line_direction)) - np.min(np.dot(centered_points, line_direction))
            if line_extent < self.min_line_length:
                return None

            # Sort points along the line direction for better representation
            projections = np.dot(centered_points, line_direction)
            sorted_indices = np.argsort(projections)
            sorted_points = [tuple(points_array[i]) for i in sorted_indices]

            confidence = max(0.1, 1.0 - avg_distance / self.line_fitting_threshold)

            return LineFeature(
                feature_type='line',
                points=sorted_points,
                confidence=confidence,
                parameters={}
            )

        except np.linalg.LinAlgError:
            return None

    def _fit_circle(self, points: List[Tuple[float, float]]) -> Optional[CircleFeature]:
        """Fit a circle to a set of points"""
        if len(points) < self.min_circle_points:
            return None

        # Create CircleFeature and let it fit itself
        circle = CircleFeature(
            feature_type='circle',
            points=points,
            confidence=0.5,
            parameters={}
        )

        # Validate the circle fit
        center = circle.parameters.get('center', (0, 0))
        radius = circle.parameters.get('radius', 0)

        if radius < 0.1 or radius > 2.0:  # Reasonable radius range for indoor objects
            return None

        # Calculate fit quality
        distances_to_center = []
        for point in points:
            dist = math.sqrt((point[0] - center[0])**2 + (point[1] - center[1])**2)
            distances_to_center.append(abs(dist - radius))

        avg_error = np.mean(distances_to_center)
        if avg_error > self.circle_fitting_threshold:
            return None

        circle.confidence = max(0.1, 1.0 - avg_error / self.circle_fitting_threshold)
        return circle

    def _calculate_line_fit_error(self, points: List[Tuple[float, float, float]], centroid: np.ndarray, line_direction: np.ndarray) -> float:
        """Calculate average distance from points to fitted line"""
        points_xy = np.array([(p[0], p[1]) for p in points])
        distances = []
        for point in points_xy:
            point_to_centroid = point - centroid
            distance = np.linalg.norm(point_to_centroid - np.dot(point_to_centroid, line_direction) * line_direction)
            distances.append(distance)
        return np.mean(distances) if distances else float('inf')

    def _calculate_circle_fit_error(self, points: List[Tuple[float, float, float]], center: Tuple[float, float], radius: float) -> float:
        """Calculate average distance from points to fitted circle"""
        distances = []
        for point in points:
            dist_to_center = math.sqrt((point[0] - center[0])**2 + (point[1] - center[1])**2)
            distances.append(abs(dist_to_center - radius))
        return np.mean(distances) if distances else float('inf')

    def _fit_line_with_error(self, points: List[Tuple[float, float, float]]) -> Tuple[Optional[LineFeature], float]:
        """Fit a line and return both the feature and fit error"""
        if len(points) < 3:
            return None, float('inf')

        points_xy = np.array([(p[0], p[1]) for p in points])

        # Use SVD to fit line
        centroid = np.mean(points_xy, axis=0)
        centered_points = points_xy - centroid

        try:
            _, _, v = np.linalg.svd(centered_points)
            line_direction = v[0]

            # Calculate fit error
            avg_distance = self._calculate_line_fit_error(points, centroid, line_direction)

            # Check if points are close enough to be considered a line
            if avg_distance > self.line_fitting_threshold:
                return None, float('inf')

            # Check minimum line length
            line_extent = np.max(np.dot(centered_points, line_direction)) - np.min(np.dot(centered_points, line_direction))
            if line_extent < self.min_line_length:
                return None, float('inf')

            # Sort points along the line direction for better representation
            projections = np.dot(centered_points, line_direction)
            sorted_indices = np.argsort(projections)
            sorted_points = [(points[i][0], points[i][1]) for i in sorted_indices]

            confidence = max(0.1, 1.0 - avg_distance / self.line_fitting_threshold)

            feature = LineFeature(
                feature_type='line',
                points=sorted_points,
                confidence=confidence,
                parameters={}
            )
            return feature, avg_distance

        except np.linalg.LinAlgError:
            return None, float('inf')

    def _fit_circle_with_error(self, points: List[Tuple[float, float, float]]) -> Tuple[Optional[CircleFeature], float]:
        """Fit a circle and return both the feature and fit error"""
        if len(points) < self.min_circle_points:
            return None, float('inf')

        points_xy = [(p[0], p[1]) for p in points]

        # Create CircleFeature and let it fit itself
        circle = CircleFeature(
            feature_type='circle',
            points=points_xy,
            confidence=0.5,
            parameters={}
        )

        # Validate the circle fit
        center = circle.parameters.get('center', (0, 0))
        radius = circle.parameters.get('radius', 0)

        if radius < 0.1 or radius > 2.0:
            return None, float('inf')

        # Calculate fit error
        avg_error = self._calculate_circle_fit_error(points, center, radius)

        if avg_error > self.circle_fitting_threshold:
            return None, float('inf')

        circle.confidence = max(0.1, 1.0 - avg_error / self.circle_fitting_threshold)
        return circle, avg_error

    def _fit_corner_with_error(self, points: List[Tuple[float, float, float]]) -> Tuple[Optional[CornerFeature], float]:
        """Fit a corner and return both the feature and fit error"""
        if len(points) < 5:
            return None, float('inf')

        # Sort points by their scan angle
        sorted_points = sorted(points, key=lambda p: p[2])  # Sort by angle

        best_corner = None
        best_error = float('inf')

        # Try different split points to find the best corner
        for split_idx in range(2, len(sorted_points) - 2):
            # Split points: first segment from min angle to split, second from split to max angle
            segment1_points = sorted_points[:split_idx + 1]  # Include split point in first segment
            segment2_points = sorted_points[split_idx:]      # Include split point in second segment

            # Fit lines to each segment
            line1_feature, line1_error = self._fit_line_with_error(segment1_points)
            line2_feature, line2_error = self._fit_line_with_error(segment2_points)

            if line1_feature and line2_feature:
                # Calculate angle between lines
                angle1 = line1_feature.parameters.get('angle', 0)
                angle2 = line2_feature.parameters.get('angle', 0)
                angle_diff = abs(angle1 - angle2)

                # Normalize angle difference
                if angle_diff > math.pi:
                    angle_diff = 2 * math.pi - angle_diff

                # Check if it's close to 90 degrees (corner)
                if abs(angle_diff - math.pi/2) < self.corner_angle_threshold:
                    # Average error of both line fits
                    combined_error = (line1_error + line2_error) / 2

                    if combined_error < best_error:
                        best_error = combined_error
                        points_xy = [(p[0], p[1]) for p in sorted_points]

                        confidence = (line1_feature.confidence + line2_feature.confidence) / 2
                        best_corner = CornerFeature(
                            feature_type='corner',
                            points=points_xy,
                            confidence=confidence,
                            parameters={}
                        )

        if best_corner:
            return best_corner, best_error
        return None, float('inf')

    def _fit_corner(self, points: List[Tuple[float, float]]) -> Optional[CornerFeature]:
        """Detect corners formed by intersecting lines"""
        if len(points) < 5:
            return None

        # Try to find a corner by fitting two line segments
        points_array = np.array(points)

        # Sort points by distance from origin to get consistent ordering
        distances = np.linalg.norm(points_array, axis=1)
        sorted_indices = np.argsort(distances)
        sorted_points = points_array[sorted_indices]

        best_corner = None
        best_score = 0

        # Try different split points to find the best corner
        for split_idx in range(2, len(sorted_points) - 2):
            segment1 = sorted_points[:split_idx]
            segment2 = sorted_points[split_idx:]

            # Fit lines to each segment
            line1_fit = self._fit_line([tuple(p) for p in segment1])
            line2_fit = self._fit_line([tuple(p) for p in segment2])

            if line1_fit and line2_fit:
                # Calculate angle between lines
                angle1 = line1_fit.parameters.get('angle', 0)
                angle2 = line2_fit.parameters.get('angle', 0)
                angle_diff = abs(angle1 - angle2)

                # Normalize angle difference
                if angle_diff > math.pi:
                    angle_diff = 2 * math.pi - angle_diff

                # Check if it's close to 90 degrees (corner)
                if abs(angle_diff - math.pi/2) < self.corner_angle_threshold:
                    score = (line1_fit.confidence + line2_fit.confidence) / 2
                    if score > best_score:
                        best_score = score
                        best_corner = CornerFeature(
                            feature_type='corner',
                            points=[tuple(p) for p in sorted_points],
                            confidence=score,
                            parameters={}
                        )

        return best_corner

    def _classify_objects(self, features: List[GeometricFeature]) -> None:
        """Classify detected geometric features into meaningful objects"""
        logger.debug(f"Classifying {len(features)} features")
        for feature in features:
            if feature.feature_type == 'line':
                obj = self._classify_line_feature(feature)
            elif feature.feature_type == 'circle':
                obj = self._classify_circle_feature(feature)
            elif feature.feature_type == 'corner':
                obj = self._classify_corner_feature(feature)
            else:
                continue

            if obj:
                # Check for motion by comparing with previous scans
                if self._is_moving_object(obj):
                    obj.object_type = 'moving_' + obj.object_type

                self.detected_objects.append(obj)
                logger.debug(f"Added object: {obj.object_type} at ({obj.x:.2f}, {obj.y:.2f})")
            else:
                logger.debug(f"Feature {feature.feature_type} did not classify to an object")

    def _classify_line_feature(self, feature: LineFeature) -> Optional[DetectedObject]:
        """Classify line features into walls, doors, furniture edges, etc."""
        length = feature.parameters.get('length', 0)

        # Long lines are likely walls
        if length > 1.5:  # reduced from 2.0 to detect smaller walls
            # Check against map to see if this is a known wall
            if self._matches_map_wall(feature):
                return DetectedObject('wall', [feature], feature.confidence * 0.9)
            else:
                # Could be a new wall or large furniture
                return DetectedObject('large_furniture', [feature], feature.confidence * 0.7)

        # Medium lines could be furniture edges or door frames
        elif 0.3 < length <= 1.5:  # reduced from 0.5 to detect smaller features
            # Check if it's vertical (door frame) or horizontal (furniture)
            angle = feature.parameters.get('angle', 0)
            if abs(angle) < math.pi/4 or abs(angle - math.pi) < math.pi/4:
                return DetectedObject('furniture_edge', [feature], feature.confidence * 0.6)
            else:
                return DetectedObject('door_frame', [feature], feature.confidence * 0.5)

        # Short lines might be small objects or noise
        else:
            # Even short lines are now classified as objects
            return DetectedObject('small_object', [feature], feature.confidence * 0.4)

    def _classify_circle_feature(self, feature: CircleFeature) -> Optional[DetectedObject]:
        """Classify circular features into table legs, people, pets, etc."""
        radius = feature.parameters.get('radius', 0)
        center = feature.parameters.get('center', (0, 0))
        distance_from_robot = math.sqrt(center[0]**2 + center[1]**2)

        # Large circles are likely round tables or large objects
        if radius > 0.3:
            return DetectedObject('round_table', [feature], feature.confidence * 0.8)

        # Medium circles could be people, pets, or cylindrical objects
        elif 0.1 < radius <= 0.3:
            if distance_from_robot < 3.0:  # Close objects more likely to be people/pets
                return DetectedObject('person_or_pet', [feature], feature.confidence * 0.7)
            else:
                return DetectedObject('cylindrical_object', [feature], feature.confidence * 0.6)

        # Small circles are likely table legs, stools, or small objects
        else:
            return DetectedObject('table_leg', [feature], feature.confidence * 0.5)

    def _classify_corner_feature(self, feature: CornerFeature) -> Optional[DetectedObject]:
        """Classify corner features into wall corners, furniture corners, etc."""
        corner_point = feature.parameters.get('corner_point', (0, 0))
        angle = feature.parameters.get('interior_angle', 0)

        # Check against map to see if this is a known corner
        if self._matches_map_corner(feature):
            return DetectedObject('wall_corner', [feature], feature.confidence * 0.9)

        # 90-degree corners are likely furniture or structural corners
        if abs(angle - math.pi/2) < math.pi/6:  # Within 30 degrees of 90
            return DetectedObject('furniture_corner', [feature], feature.confidence * 0.7)

        # Other angles might be furniture with different geometry
        return DetectedObject('angled_corner', [feature], feature.confidence * 0.5)

    def _matches_map_wall(self, feature: LineFeature) -> bool:
        """Check if a line feature matches a known wall in the map"""
        if not self.latest_map_data or not self.latest_robot_pose:
            return False

        # This would require map processing logic
        # For now, return False as a placeholder
        return False

    def _matches_map_corner(self, feature: CornerFeature) -> bool:
        """Check if a corner feature matches a known corner in the map"""
        if not self.latest_map_data or not self.latest_robot_pose:
            return False

        # This would require map processing logic
        # For now, return False as a placeholder
        return False

    def _is_moving_object(self, obj: DetectedObject) -> bool:
        """Detect if an object is moving by comparing with previous scans"""
        if len(self.scan_history) < 2:
            return False

        current_pos = obj.position

        # Look for similar objects in previous scans
        for prev_scan in list(self.scan_history)[-3:-1]:  # Check last 2 scans
            prev_timestamp = prev_scan['timestamp']
            time_diff = time.time() - prev_timestamp

            if time_diff > 2.0:  # Only consider recent scans
                continue

            # Find closest object in previous scan (simplified)
            # In a real implementation, this would be more sophisticated
            for prev_point in prev_scan['points']:
                distance = math.sqrt((current_pos[0] - prev_point[0])**2 +
                                   (current_pos[1] - prev_point[1])**2)

                if distance > self.motion_threshold:
                    return True

        return False

    def get_detected_objects(self) -> List[Dict[str, Any]]:
        """Get list of currently detected objects"""
        return [obj.to_dict() for obj in self.detected_objects]

    def get_objects_summary(self) -> Dict[str, Any]:
        """Get summary of detected objects"""
        object_counts = {}
        feature_counts = {}

        for obj in self.detected_objects:
            obj_type = obj.object_type
            object_counts[obj_type] = object_counts.get(obj_type, 0) + 1

            for feature in obj.features:
                f_type = feature.feature_type
                feature_counts[f_type] = feature_counts.get(f_type, 0) + 1

        last_timestamp = None
        if self.detected_objects:
            last_timestamp = max(obj.timestamp for obj in self.detected_objects)
            last_timestamp = time.strftime("%H:%M:%S", time.localtime(last_timestamp))

        return {
            "total_objects": len(self.detected_objects),
            "object_counts": object_counts,
            "feature_counts": feature_counts,
            "last_update": last_timestamp,
            "scan_available": self.latest_scan_data is not None,
            "map_available": self.latest_map_data is not None,
            "pose_available": self.latest_robot_pose is not None
        }

    def set_processing_enabled(self, enabled: bool) -> None:
        """Enable or disable object detection processing"""
        self.processing_enabled = enabled
        logger.info(f"Object detection processing {'enabled' if enabled else 'disabled'}")

    def find_closest_object(self):
        """Find the closest detected object"""
        if not self.detected_objects:
            return None
        return min(self.detected_objects, key=lambda obj: obj.distance)

    def find_closest_objects(self):
        """Find all detected objects sorted by distance (closest first)"""
        return sorted(self.detected_objects, key=lambda obj: obj.distance)

    def find_objects_by_type(self, object_type: str) -> List[DetectedObject]:
        """Find all objects of a specific type"""
        return [obj for obj in self.detected_objects if obj.object_type == object_type]

    def find_moving_objects(self) -> List[DetectedObject]:
        """Find all objects that are currently moving"""
        return [obj for obj in self.detected_objects if obj.object_type.startswith('moving_')]

    def get_geometric_features(self) -> List[GeometricFeature]:
        """Get all detected geometric features"""
        features = []
        for obj in self.detected_objects:
            features.extend(obj.features)
        return features

    def get_features_for_ui(self) -> List[Dict[str, Any]]:
        """Get detected features formatted for UI visualization"""
        features_data = []
        for feature in self.detected_features:
            feature_dict = {
                "type": feature.feature_type,
                "points": feature.points,
                "confidence": feature.confidence,
                "parameters": feature.parameters
            }
            features_data.append(feature_dict)
        return features_data

    def clear_detections(self) -> None:
        """Clear all detected objects"""
        self.detected_objects.clear()
        logger.info("Cleared all object detections")

    def set_detection_parameters(self, **kwargs) -> None:
        """Update detection parameters for fine-tuning"""
        if 'line_fitting_threshold' in kwargs:
            self.line_fitting_threshold = kwargs['line_fitting_threshold']
        if 'min_line_length' in kwargs:
            self.min_line_length = kwargs['min_line_length']
        if 'circle_fitting_threshold' in kwargs:
            self.circle_fitting_threshold = kwargs['circle_fitting_threshold']
        if 'min_circle_points' in kwargs:
            self.min_circle_points = kwargs['min_circle_points']
        if 'corner_angle_threshold' in kwargs:
            self.corner_angle_threshold = kwargs['corner_angle_threshold']
        if 'motion_threshold' in kwargs:
            self.motion_threshold = kwargs['motion_threshold']

        logger.info(f"Updated detection parameters: {kwargs}")

    def get_detection_parameters(self) -> Dict[str, Any]:
        """Get current detection parameters"""
        return {
            'line_fitting_threshold': self.line_fitting_threshold,
            'min_line_length': self.min_line_length,
            'circle_fitting_threshold': self.circle_fitting_threshold,
            'min_circle_points': self.min_circle_points,
            'corner_angle_threshold': self.corner_angle_threshold,
            'motion_threshold': self.motion_threshold
        }