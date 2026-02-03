"""
Machine Learning-based Object Detector for Lidar Data

This module implements an ML-based approach to detecting geometric features
(lines, corners, arcs) from Lidar scans using neural networks trained on
synthetic data.
"""

import math
import time
import base64
import logging
from typing import List, Dict, Any, Optional, Tuple, TYPE_CHECKING
from dataclasses import dataclass
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque

if TYPE_CHECKING:
    from robot_proxy import RobotProxy

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


# ============================================================================
# Data Structures
# ============================================================================

@dataclass
class MLDetectedFeature:
    """ML-detected geometric feature with parameters"""
    feature_type: str  # 'line', 'corner', 'arc'
    confidence: float
    parameters: Dict[str, Any]
    points: List[Tuple[float, float]]  # Associated scan points


class DetectedObject:
    """Detected object from ML-based feature detection"""
    def __init__(self, object_type: str, features: List[MLDetectedFeature],
                 confidence: float, position: Optional[Tuple[float, float]] = None):
        self.object_type = object_type
        self.features = features
        self.confidence = confidence
        self.timestamp = time.time()
        self.position = position or self._calculate_center_position()

        # Backward compatibility
        self.distance = math.sqrt(self.position[0]**2 + self.position[1]**2)
        self.angle = math.atan2(self.position[1], self.position[0])
        self.x, self.y = self.position

    def _calculate_center_position(self) -> Tuple[float, float]:
        """Calculate center position from all feature points"""
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


# ============================================================================
# Synthetic Scene Generation
# ============================================================================

class SyntheticSceneGenerator:
    """Generates random scenes with lines, corners, and arcs for training"""

    def __init__(self, scene_size: float = 10.0, seed: Optional[int] = None):
        """
        Args:
            scene_size: Size of the scene in meters
            seed: Random seed for reproducibility
        """
        self.scene_size = scene_size
        self.rng = np.random.RandomState(seed)

        # Create a LidarSimulator instance for checking scan hits
        self._lidar_sim = LidarSimulator(n_rays=360, max_range=8.0)

    def generate_scene(self, n_features: int = 5) -> List[Dict[str, Any]]:
        """
        Generate a random scene with geometric features

        Args:
            n_features: Number of features to generate

        Returns:
            List of feature dictionaries with type and parameters
        """
        features = []
        feature_types = ['line', 'corner', 'arc']

        for _ in range(n_features):
            feature_type = self.rng.choice(feature_types)

            if feature_type == 'line':
                features.append(self._generate_line())
            elif feature_type == 'corner':
                features.append(self._generate_corner())
            elif feature_type == 'arc':
                features.append(self._generate_arc())

        return features

    def _generate_line(self) -> Dict[str, Any]:
        """Generate a random line segment with minimum scan hit requirement"""
        min_hits = 4  # Minimum scan points that must hit the line
        max_attempts = 100

        for attempt in range(max_attempts):
            # Random position closer to origin for better visibility
            x1 = self.rng.uniform(-3.0, 3.0)  # Reduced from scene_size/2
            y1 = self.rng.uniform(-3.0, 3.0)

            # Random length and angle
            length = self.rng.uniform(0.8, 2.5)  # Slightly longer minimum
            angle = self.rng.uniform(0, 2 * np.pi)

            x2 = x1 + length * np.cos(angle)
            y2 = y1 + length * np.sin(angle)

            line = {
                'type': 'line',
                'start': (float(x1), float(y1)),
                'end': (float(x2), float(y2)),
                'length': float(length),
                'angle': float(angle)
            }

            # Check if line receives enough scan hits
            hits = self._count_scan_hits(line)
            if hits >= min_hits:
                return line

        # Fallback: return last generated line if max attempts reached
        logger.warning(f"Line generation: max attempts ({max_attempts}) reached, using line with {hits} hits")
        return line

    def _generate_corner(self) -> Dict[str, Any]:
        """Generate a random corner with occlusion checking and minimum scan hits

        Generates corner point C and arm1 end point A, then tries different
        positions for arm2 end point B until no occlusion is detected and
        each arm receives at least 4 scan hits.
        If 30 attempts fail, regenerates C and A and tries again.
        """
        min_hits_per_arm = 4  # Minimum scan points per arm
        max_outer_attempts = 100  # Try regenerating C and A
        max_inner_attempts = 30   # Try different B positions for each C,A pair

        for outer_attempt in range(max_outer_attempts):
            # Generate corner point C and arm1 (points C and A)
            cx = self.rng.uniform(-3.0, 3.0)
            cy = self.rng.uniform(-3.0, 3.0)

            angle1 = self.rng.uniform(0, 2 * np.pi)
            length1 = self.rng.uniform(0.8, 1.8)

            x1 = cx + length1 * np.cos(angle1)
            y1 = cy + length1 * np.sin(angle1)

            # Try different positions for arm2 (point B) until no occlusion
            for inner_attempt in range(max_inner_attempts):
                # Generate random arm2
                interior_angle = self.rng.uniform(np.pi/6, 5*np.pi/6)  # 30 to 150 degrees
                angle2 = angle1 + interior_angle
                length2 = self.rng.uniform(0.8, 1.8)

                x2 = cx + length2 * np.cos(angle2)
                y2 = cy + length2 * np.sin(angle2)

                corner = {
                    'type': 'corner',
                    'corner_point': (float(cx), float(cy)),
                    'arm1_end': (float(x1), float(y1)),
                    'arm2_end': (float(x2), float(y2)),
                    'interior_angle': float(interior_angle),
                    'arm1_length': float(length1),
                    'arm2_length': float(length2)
                }

                # Check if both arms are visible (no occlusion)
                if not self._check_corner_visibility(corner):
                    continue

                # Check if each arm receives enough scan hits
                arm1_hits, arm2_hits = self._count_corner_arm_hits(corner)
                if arm1_hits >= min_hits_per_arm and arm2_hits >= min_hits_per_arm:
                    return corner

        logger.error(f"Corner generation: max attempts ({max_outer_attempts * max_inner_attempts}) reached, failed to generate valid corner")
        exit()

    def _check_corner_visibility(self, corner: Dict[str, Any]) -> bool:
        """Check if corner has self-occlusion using geometric proof

        Corner is defined by points: C (corner point), A (arm1 end), B (arm2 end)
        Robot is at R = (0, 0)

        Check four occlusion cases:
        1. Does line RB intersect segment AC? (arm2 blocks arm1)
        2. Does ray RA intersect segment CB AND is intersection farther than B? (arm1 blocks arm2)
        3. Does line RA intersect segment BC? (arm1 blocks arm2 - symmetric check)
        4. Does ray RB intersect segment CA AND is intersection farther than A? (arm2 blocks arm1 - symmetric check)

        Returns True if no occlusion, False if occluded.
        """

        # Extract points: C (corner), A (arm1 end), B (arm2 end), R (robot at origin)
        cx, cy = corner['corner_point']  # Point C
        ax, ay = corner['arm1_end']      # Point A
        bx, by = corner['arm2_end']      # Point B
        rx, ry = 0.0, 0.0                # Point R (robot)

        # Check 1: Does ray RB intersect segment CA before reaching B?
        # If yes, segment CA blocks view of point B
        intersection_dist = self._ray_intersects_segment_with_distance(rx, ry, bx, by, cx, cy, ax, ay)

        if intersection_dist is not None:
            dist_rb = np.sqrt((bx - rx)**2 + (by - ry)**2)
            # Occlusion if intersection happens before reaching B
            if intersection_dist < dist_rb:
                return False  # Self-occlusion detected

        # Check 2: Does ray RA intersect segment CB AND is intersection farther than A?
        # If yes, arm1 partially occludes arm2
        intersection_dist = self._ray_intersects_segment_with_distance(rx, ry, ax, ay, cx, cy, bx, by)
        if intersection_dist is not None:
            dist_ra = np.sqrt((ax - rx)**2 + (ay - ry)**2)
            # Occlusion if intersection is farther than A (arm1 partially occludes arm2)
            if intersection_dist > dist_ra:
                return False  # Self-occlusion detected

        # Check 3: Does ray RA intersect segment CB before reaching A?
        # If yes, segment CB blocks view of point A
        intersection_dist = self._ray_intersects_segment_with_distance(rx, ry, ax, ay, bx, by, cx, cy)
        if intersection_dist is not None:
            dist_ra = np.sqrt((ax - rx)**2 + (ay - ry)**2)
            # Occlusion if intersection happens before reaching A
            if intersection_dist < dist_ra:
                return False  # Self-occlusion detected

        # Check 4: Does ray RB intersect segment CA AND is intersection farther than B?
        # If yes, arm2 partially occludes arm1
        intersection_dist = self._ray_intersects_segment_with_distance(rx, ry, bx, by, cx, cy, ax, ay)
        if intersection_dist is not None:
            dist_ra = np.sqrt((bx - rx)**2 + (by - ry)**2)
            # Occlusion if intersection is farther than A (arm2 extends past and blocks arm1)
            if intersection_dist > dist_ra:
                return False  # Self-occlusion detected

        return True  # No occlusion, corner is visible

    def _line_intersects_segment(self, lx1: float, ly1: float, lx2: float, ly2: float,
                                  sx1: float, sy1: float, sx2: float, sy2: float) -> bool:
        """Check if line from (lx1,ly1) through (lx2,ly2) intersects segment (sx1,sy1)-(sx2,sy2)

        Args:
            lx1, ly1: First point on the line (start point)
            lx2, ly2: Second point on the line (defines direction)
            sx1, sy1: Segment start point
            sx2, sy2: Segment end point

        Returns:
            True if line intersects the segment, False otherwise
        """
        # Line direction vector
        ldx = lx2 - lx1
        ldy = ly2 - ly1

        # Segment direction vector
        sdx = sx2 - sx1
        sdy = sy2 - sy1

        # Check if lines are parallel
        denom = ldx * sdy - ldy * sdx
        if abs(denom) < 1e-10:
            return False  # Parallel or collinear

        # Calculate intersection parameters
        # t is parameter for the line (any value works)
        # u is parameter for the segment (must be in [0, 1])
        dx = sx1 - lx1
        dy = sy1 - ly1

        u = (ldx * dy - ldy * dx) / denom

        # Check if intersection point is on the segment
        if 0 <= u <= 1:
            return True

        return False

    def _ray_intersects_segment(self, rx: float, ry: float, rdx: float, rdy: float,
                                 sx1: float, sy1: float, sx2: float, sy2: float) -> bool:
        """Check if ray from (rx,ry) through (rdx,rdy) intersects segment (sx1,sy1)-(sx2,sy2)

        Args:
            rx, ry: Ray origin
            rdx, rdy: Point that ray passes through (defines direction)
            sx1, sy1: Segment start point
            sx2, sy2: Segment end point

        Returns:
            True if ray intersects the segment, False otherwise
        """
        # Ray direction vector
        ray_dx = rdx - rx
        ray_dy = rdy - ry

        # Segment direction vector
        sdx = sx2 - sx1
        sdy = sy2 - sy1

        # Check if ray and segment are parallel
        denom = ray_dx * sdy - ray_dy * sdx
        if abs(denom) < 1e-10:
            return False  # Parallel or collinear

        # Calculate intersection parameters
        # t is parameter for the ray (must be >= 0 for ray direction)
        # u is parameter for the segment (must be in [0, 1])
        dx = sx1 - rx
        dy = sy1 - ry

        t = (dx * sdy - dy * sdx) / denom
        u = (dx * ray_dy - dy * ray_dx) / denom

        # Check if intersection is in ray direction (t >= 0) and on segment (0 <= u <= 1)
        if t >= 0 and 0 <= u <= 1:
            return True

        return False

    def _ray_intersects_segment_with_distance(self, rx: float, ry: float, rdx: float, rdy: float,
                                               sx1: float, sy1: float, sx2: float, sy2: float) -> Optional[float]:
        """Check if ray intersects segment and return distance to intersection point

        Args:
            rx, ry: Ray origin
            rdx, rdy: Point that ray passes through (defines direction)
            sx1, sy1: Segment start point
            sx2, sy2: Segment end point

        Returns:
            Distance from ray origin to intersection point, or None if no intersection
        """
        # Ray direction vector
        ray_dx = rdx - rx
        ray_dy = rdy - ry

        # Segment direction vector
        sdx = sx2 - sx1
        sdy = sy2 - sy1

        # Check if ray and segment are parallel
        denom = ray_dx * sdy - ray_dy * sdx
        if abs(denom) < 1e-10:
            return None  # Parallel or collinear

        # Calculate intersection parameters
        # t is parameter for the ray (must be >= 0 for ray direction)
        # u is parameter for the segment (must be in [0, 1])
        dx = sx1 - rx
        dy = sy1 - ry

        t = (dx * sdy - dy * sdx) / denom
        u = (dx * ray_dy - dy * ray_dx) / denom

        # Check if intersection is in ray direction (t >= 0) and on segment (0 <= u <= 1)
        if t >= 0 and 0 <= u <= 1:
            # Calculate actual intersection point
            intersection_x = rx + t * ray_dx
            intersection_y = ry + t * ray_dy

            # Return distance from ray origin to intersection
            distance = np.sqrt((intersection_x - rx)**2 + (intersection_y - ry)**2)
            return distance

        return None

    def _line_intersection_helper(self, x1: float, y1: float, x2: float, y2: float,
                                   x3: float, y3: float, x4: float, y4: float) -> Optional[float]:
        """Helper for line-ray intersection (same as LidarSimulator)"""
        denom = (x2 * (y4 - y3) - y2 * (x4 - x3))

        if abs(denom) < 1e-10:
            return None

        t = ((x3 - x1) * (y4 - y3) - (y3 - y1) * (x4 - x3)) / denom
        u = ((x3 - x1) * y2 - (y3 - y1) * x2) / denom

        if t > 0 and 0 <= u <= 1:
            return t

        return None

    def _generate_arc(self) -> Dict[str, Any]:
        """Generate arc using two methods for guaranteed visibility and minimum scan hits

        1. Tangent-based (robot outside circle): Arc between tangent points
        2. Robot inside circle: Any arc segment is fully visible
        """
        min_hits = 4  # Minimum scan points that must hit the arc
        max_attempts = 100  # Increased from 30 to account for hit check

        # Decide which type to generate (70% robot inside for more variety, 30% robot outside)
        # Robot-inside arcs can appear both concave and convex, while robot-outside arcs are mostly concave
        generate_robot_inside = self.rng.random() < 0.7

        for attempt in range(max_attempts):
            if generate_robot_inside:
                # Intentionally create arc with robot inside
                # Place center close to origin, use large radius
                cx = self.rng.uniform(-1.5, 1.5)  # Closer to robot
                cy = self.rng.uniform(-1.5, 1.5)
                dist_to_robot = np.sqrt(cx**2 + cy**2)

                # Radius larger than distance to ensure robot is inside
                radius = self.rng.uniform(dist_to_robot + 0.5, 2.5)
            else:
                # Generate arc with robot outside
                # Random center position
                cx = self.rng.uniform(-3.0, 3.0)
                cy = self.rng.uniform(-3.0, 3.0)
                dist_to_robot = np.sqrt(cx**2 + cy**2)

                # Random radius (smaller than distance to ensure robot is outside)
                radius = self.rng.uniform(0.5, min(1.8, dist_to_robot - 0.3))

            # Case 1: Robot INSIDE circle - entire circle is visible!
            if dist_to_robot < radius:
                # Any arc segment is fully visible, pick random arc
                start_angle = self.rng.uniform(-np.pi, np.pi)
                angular_extent = self.rng.uniform(np.pi/3, 4*np.pi/3)  # 60-240 degrees
                end_angle = start_angle + angular_extent

                # Normalize angles
                start_angle = np.arctan2(np.sin(start_angle), np.cos(start_angle))
                end_angle = np.arctan2(np.sin(end_angle), np.cos(end_angle))

                # Calculate angular extent properly
                angular_extent_final = end_angle - start_angle
                if angular_extent_final < 0:
                    angular_extent_final += 2 * np.pi

                arc = {
                    'type': 'arc',
                    'center': (float(cx), float(cy)),
                    'radius': float(radius),
                    'start_angle': float(start_angle),
                    'end_angle': float(end_angle),
                    'angular_extent': float(angular_extent_final)
                }

                # Check if arc receives enough scan hits
                hits = self._count_scan_hits(arc)
                if hits >= min_hits:
                    return arc
                else:
                    continue  # Try again

            # Case 2: Robot OUTSIDE circle - use tangent-based method
            # Add small buffer to avoid floating point issues with arcsin
            if dist_to_robot <= radius * 1.01:
                continue  # Skip if on or too close to the boundary

            # Calculate tangent points from robot (0,0) to circle
            # angle from robot to circle center
            angle_to_center = np.arctan2(cy, cx)

            # Angle between line-to-center and tangent line
            # sin(tangent_angle) = radius / dist_to_robot
            # Clamp the ratio to [-1, 1] to avoid numerical errors in arcsin
            ratio = radius / dist_to_robot
            ratio = np.clip(radius / dist_to_robot, -1.0, 1.0)
            tangent_half_angle = np.arcsin(ratio)

            # Two tangent directions from robot's perspective
            tangent_dir1 = angle_to_center + tangent_half_angle
            tangent_dir2 = angle_to_center - tangent_half_angle

            # Convert to angles in circle's polar frame (from center)
            # The tangent point angle in the circle frame equals the tangent direction from robot
            # because both the robot and circle center are looking at the same point
            alpha1 = tangent_dir1  # Angle from circle center to first tangent point
            alpha2 = tangent_dir2  # Angle from circle center to second tangent point

            # Normalize angles to [-pi, pi]
            alpha1 = np.arctan2(np.sin(alpha1), np.cos(alpha1))
            alpha2 = np.arctan2(np.sin(alpha2), np.cos(alpha2))

            # Make sure alpha1 < alpha2 for the visible arc
            if alpha2 < alpha1:
                alpha1, alpha2 = alpha2, alpha1

            # Calculate angular spans
            # Convex arc (short path): alpha1 → alpha2 (curves toward robot)
            # Concave arc (long path): alpha2 → alpha1 (wrapping around, curves away)
            # The concave arc (alpha1..alpha2) never self-occludes by construction
            # The convex arc (alpha2..alpha1) will self-occlude IFF
            #   any portion of the arc lies outside of alpha1..alpha2
            useful_span = alpha2 - alpha1

            # Randomly choose convex or concave arc (50/50 for good variety)
            use_convex = self.rng.random() < 0.5

            if use_convex:
                # Generate convex arc between alpha1 and alpha2
                min_extent = min(np.pi/3, useful_span * 0.5)
                max_extent = min(useful_span * 0.9, 2*np.pi/3)  # Cap at 120° for safety

                if max_extent > min_extent:
                    trimmed_extent = self.rng.uniform(min_extent, max_extent)
                    max_offset = useful_span - trimmed_extent
                    start_offset = self.rng.uniform(0, max(max_offset, 0))
                    start_angle = alpha1 + start_offset
                    end_angle = start_angle + trimmed_extent
                else:
                    start_angle = alpha1
                    end_angle = alpha2
            else:
                # Generate concave arc between alpha2 and alpha1 (wrapping)
                min_extent = min(np.pi/3, useful_span * 0.4)
                max_extent = min(useful_span * 0.8, 2*np.pi/3)  # Cap at 120° for safety

                if max_extent > min_extent:
                    trimmed_extent = self.rng.uniform(min_extent, max_extent)
                    start_angle = alpha2  # TODO start arc from alpha2 with a random offset
                    end_angle = start_angle + trimmed_extent
                else:
                    start_angle = alpha2
                    end_angle = alpha1 + 2 * np.pi  # Wrap around

            # Normalize final angles
            start_angle = np.arctan2(np.sin(start_angle), np.cos(start_angle))
            end_angle = np.arctan2(np.sin(end_angle), np.cos(end_angle))

            # Calculate angular extent
            angular_extent = end_angle - start_angle
            if angular_extent < 0:
                angular_extent += 2 * np.pi

            arc = {
                'type': 'arc',
                'center': (float(cx), float(cy)),
                'radius': float(radius),
                'start_angle': float(start_angle),
                'end_angle': float(end_angle),
                'angular_extent': float(angular_extent)
            }

            # Check if arc receives enough scan hits
            hits = self._count_scan_hits(arc)

            # Incorrect
            # For arcs, require proportional hits based on angular extent
            # A 180-degree arc should get roughly twice as many hits as a 90-degree arc
            # expected_hits_per_radian = 360 / (2 * np.pi)  # ~57 rays per radian
            # min_expected_hits = max(min_hits, int(angular_extent * expected_hits_per_radian * 0.15))  # At least 15% of theoretical

            if hits >= min_hits:
                return arc
            # Otherwise continue to next attempt

        # Fallback if generation failed - create a simple arc that should get hits
        logger.warning(f"Arc generation: max attempts ({max_attempts}) reached, using fallback arc")
        fallback_arc = {
            'type': 'arc',
            'center': (2.0, 0.0),
            'radius': 0.8,
            'start_angle': -np.pi/4,
            'end_angle': np.pi/4,
            'angular_extent': np.pi/2
        }
        return fallback_arc

    def _check_arc_visibility(self, arc: Dict[str, Any]) -> bool:
        """Check if arc is visible from robot at origin

        Returns True if arc has sufficient visible scan points.
        """
        cx, cy = arc['center']
        radius = arc['radius']
        start_angle = arc['start_angle']
        end_angle = arc['end_angle']

        # Quick scan with 90 rays
        n_rays = 90
        max_range = 8.0
        angles = np.linspace(-np.pi, np.pi, n_rays)

        hits = 0

        # Check ray intersections
        for angle in angles:
            ray_x = np.cos(angle)
            ray_y = np.sin(angle)

            # Ray-circle intersection
            fx = 0 - cx
            fy = 0 - cy

            a = ray_x**2 + ray_y**2
            b = 2 * (fx * ray_x + fy * ray_y)
            c = fx**2 + fy**2 - radius**2

            discriminant = b**2 - 4*a*c

            if discriminant >= 0:
                sqrt_disc = np.sqrt(discriminant)
                t1 = (-b - sqrt_disc) / (2*a)
                t2 = (-b + sqrt_disc) / (2*a)

                # Check both intersections
                for t in [t1, t2]:
                    if t > 0 and t < max_range:
                        # Hit point
                        hit_x = t * ray_x
                        hit_y = t * ray_y

                        # Angle from arc center to hit point
                        hit_angle = np.arctan2(hit_y - cy, hit_x - cx)

                        # Check if within arc range
                        if self._angle_in_arc_range(hit_angle, start_angle, end_angle):
                            hits += 1
                            break  # Count each ray once

        # Need at least 5 visible points
        return hits >= 5

    def _angle_in_arc_range(self, angle: float, start: float, end: float) -> bool:
        """Check if angle is within arc range"""
        # Normalize angles to [-pi, pi]
        angle = np.arctan2(np.sin(angle), np.cos(angle))
        start = np.arctan2(np.sin(start), np.cos(start))
        end = np.arctan2(np.sin(end), np.cos(end))

        if start <= end:
            return start <= angle <= end
        else:
            # Range wraps around -pi/pi boundary
            return angle >= start or angle <= end

    def _count_scan_hits(self, feature: Dict[str, Any]) -> int:
        """Count how many scan points hit a feature

        Args:
            feature: Feature dictionary (line, corner, or arc)

        Returns:
            Number of scan points that hit the feature (not at max range)
        """
        # Simulate scan with only this feature
        scan = self._lidar_sim.simulate_scan([feature], (0.0, 0.0, 0.0))

        # Count hits (scan values less than max_range threshold)
        hits = np.sum(scan < 7.9)

        return int(hits)

    def _count_corner_arm_hits(self, corner: Dict[str, Any]) -> Tuple[int, int]:
        """Count scan hits for each arm of a corner separately

        Args:
            corner: Corner feature dictionary

        Returns:
            Tuple of (arm1_hits, arm2_hits)
        """
        # Create line features for each arm
        cx, cy = corner['corner_point']
        x1, y1 = corner['arm1_end']
        x2, y2 = corner['arm2_end']

        arm1_line = {
            'type': 'line',
            'start': (cx, cy),
            'end': (x1, y1)
        }

        arm2_line = {
            'type': 'line',
            'start': (cx, cy),
            'end': (x2, y2)
        }

        # Count hits for each arm
        arm1_hits = self._count_scan_hits(arm1_line)
        arm2_hits = self._count_scan_hits(arm2_line)

        return arm1_hits, arm2_hits


# ============================================================================
# Lidar Scan Simulator
# ============================================================================

class LidarSimulator:
    """Simulates Lidar scans from synthetic scenes"""

    def __init__(self, n_rays: int = 360, max_range: float = 8.0,
                 angle_min: float = -np.pi, angle_max: float = np.pi,
                 noise_std: float = 0.02):
        """
        Args:
            n_rays: Number of laser rays (scan resolution)
            max_range: Maximum sensing range in meters
            angle_min: Minimum scan angle in radians
            angle_max: Maximum scan angle in radians
            noise_std: Standard deviation of range noise in meters
        """
        self.n_rays = n_rays
        self.max_range = max_range
        self.angle_min = angle_min
        self.angle_max = angle_max
        self.noise_std = noise_std

        # Pre-compute ray angles
        self.ray_angles = np.linspace(angle_min, angle_max, n_rays)

    def simulate_scan(self, features: List[Dict[str, Any]],
                     robot_pose: Tuple[float, float, float] = (0, 0, 0)) -> np.ndarray:
        """
        Simulate a Lidar scan from a scene

        Args:
            features: List of geometric features in the scene
            robot_pose: Robot position (x, y, theta) in meters and radians

        Returns:
            Array of range measurements (one per ray)
        """
        ranges = np.full(self.n_rays, self.max_range)

        robot_x, robot_y, robot_theta = robot_pose

        for feature in features:
            feature_type = feature['type']

            if feature_type == 'line':
                self._raycast_line(ranges, feature, robot_x, robot_y, robot_theta)
            elif feature_type == 'corner':
                self._raycast_corner(ranges, feature, robot_x, robot_y, robot_theta)
            elif feature_type == 'arc':
                self._raycast_arc(ranges, feature, robot_x, robot_y, robot_theta)

        # Add noise
        if self.noise_std > 0:
            noise = np.random.normal(0, self.noise_std, self.n_rays)
            ranges = np.clip(ranges + noise, 0, self.max_range)

        return ranges

    def _raycast_line(self, ranges: np.ndarray, line: Dict[str, Any],
                     robot_x: float, robot_y: float, robot_theta: float):
        """Compute ray intersections with a line segment"""
        x1, y1 = line['start']
        x2, y2 = line['end']

        # Transform to robot frame
        x1 -= robot_x
        y1 -= robot_y
        x2 -= robot_x
        y2 -= robot_y

        # Rotate by -robot_theta
        cos_theta = np.cos(-robot_theta)
        sin_theta = np.sin(-robot_theta)

        x1_rot = x1 * cos_theta - y1 * sin_theta
        y1_rot = x1 * sin_theta + y1 * cos_theta
        x2_rot = x2 * cos_theta - y2 * sin_theta
        y2_rot = x2 * sin_theta + y2 * cos_theta

        # Check each ray for intersection
        for i, angle in enumerate(self.ray_angles):
            ray_x = np.cos(angle)
            ray_y = np.sin(angle)

            # Line-line intersection in 2D
            dist = self._line_intersection(0, 0, ray_x, ray_y,
                                          x1_rot, y1_rot, x2_rot, y2_rot)

            if dist is not None and 0 < dist < ranges[i]:
                ranges[i] = dist

    def _raycast_corner(self, ranges: np.ndarray, corner: Dict[str, Any],
                       robot_x: float, robot_y: float, robot_theta: float):
        """Compute ray intersections with a corner (two line segments)"""
        # Corner consists of two line segments
        cx, cy = corner['corner_point']
        x1, y1 = corner['arm1_end']
        x2, y2 = corner['arm2_end']

        # Raycast both arms
        line1 = {'start': (cx, cy), 'end': (x1, y1)}
        line2 = {'start': (cx, cy), 'end': (x2, y2)}

        self._raycast_line(ranges, line1, robot_x, robot_y, robot_theta)
        self._raycast_line(ranges, line2, robot_x, robot_y, robot_theta)

    def _raycast_arc(self, ranges: np.ndarray, arc: Dict[str, Any],
                    robot_x: float, robot_y: float, robot_theta: float):
        """Compute ray intersections with an arc"""
        cx, cy = arc['center']
        radius = arc['radius']
        start_angle = arc['start_angle']
        end_angle = arc['end_angle']

        # Check each ray for intersection with circle (in world frame)
        for i, ray_angle in enumerate(self.ray_angles):
            # Transform ray to world frame
            ray_world_angle = ray_angle + robot_theta
            ray_world_x = np.cos(ray_world_angle)
            ray_world_y = np.sin(ray_world_angle)

            # Get ALL circle intersections (both near and far side)
            dists = self._ray_circle_intersections_all(robot_x, robot_y, ray_world_x, ray_world_y,
                                                       cx, cy, radius)

            if dists:
                # Check each intersection point
                for dist in dists:
                    if dist is not None and 0 < dist < ranges[i]:
                        # Hit point in world frame
                        hit_world_x = robot_x + dist * ray_world_x
                        hit_world_y = robot_y + dist * ray_world_y

                        # Angle from arc center to hit point (in world frame)
                        hit_angle = np.arctan2(hit_world_y - cy, hit_world_x - cx)

                        # Check if this angle is within the arc's angular range
                        if self._angle_in_range(hit_angle, start_angle, end_angle):
                            ranges[i] = dist
                            break  # Use the first valid hit

    def _line_intersection(self, x1: float, y1: float, x2: float, y2: float,
                          x3: float, y3: float, x4: float, y4: float) -> Optional[float]:
        """
        Compute intersection of ray from (x1,y1) in direction (x2,y2)
        with line segment from (x3,y3) to (x4,y4)

        Returns distance along ray, or None if no intersection
        """
        denom = (x2 * (y4 - y3) - y2 * (x4 - x3))

        if abs(denom) < 1e-10:
            return None

        t = ((x3 - x1) * (y4 - y3) - (y3 - y1) * (x4 - x3)) / denom
        u = ((x3 - x1) * y2 - (y3 - y1) * x2) / denom

        if t > 0 and 0 <= u <= 1:
            return t

        return None

    def _ray_circle_intersection(self, rx: float, ry: float,
                                 rdx: float, rdy: float,
                                 cx: float, cy: float, radius: float) -> Optional[float]:
        """
        Compute intersection of ray from (rx,ry) in direction (rdx,rdy)
        with circle at (cx,cy) with given radius

        Returns distance along ray, or None if no intersection
        """
        # Vector from ray origin to circle center
        fx = rx - cx
        fy = ry - cy

        a = rdx**2 + rdy**2
        b = 2 * (fx * rdx + fy * rdy)
        c = fx**2 + fy**2 - radius**2

        discriminant = b**2 - 4*a*c

        if discriminant < 0:
            return None

        sqrt_disc = np.sqrt(discriminant)
        t1 = (-b - sqrt_disc) / (2*a)
        t2 = (-b + sqrt_disc) / (2*a)

        # Return closest positive intersection
        if t1 > 0:
            return t1
        elif t2 > 0:
            return t2

        return None

    def _ray_circle_intersections_all(self, rx: float, ry: float,
                                      rdx: float, rdy: float,
                                      cx: float, cy: float, radius: float) -> List[float]:
        """
        Compute ALL intersections of ray with circle (both near and far side)

        Returns list of distances along ray (both intersections if they exist)
        """
        # Vector from ray origin to circle center
        fx = rx - cx
        fy = ry - cy

        a = rdx**2 + rdy**2
        b = 2 * (fx * rdx + fy * rdy)
        c = fx**2 + fy**2 - radius**2

        discriminant = b**2 - 4*a*c

        if discriminant < 0:
            return []

        sqrt_disc = np.sqrt(discriminant)
        t1 = (-b - sqrt_disc) / (2*a)
        t2 = (-b + sqrt_disc) / (2*a)

        # Return both positive intersections
        intersections = []
        if t1 > 0:
            intersections.append(t1)
        if t2 > 0 and t2 != t1:  # Avoid duplicates (tangent case)
            intersections.append(t2)

        return intersections

    def _angle_in_range(self, angle: float, start: float, end: float) -> bool:
        """Check if angle is within [start, end] range, handling wrapping"""
        # Normalize all angles to [-pi, pi]
        angle = np.arctan2(np.sin(angle), np.cos(angle))
        start = np.arctan2(np.sin(start), np.cos(start))
        end = np.arctan2(np.sin(end), np.cos(end))

        if start <= end:
            return start <= angle <= end
        else:
            # Range wraps around -pi/pi boundary
            return angle >= start or angle <= end


# ============================================================================
# Neural Network Models
# ============================================================================

class LidarFeatureEncoder(nn.Module):
    """Encodes raw Lidar scans into feature representations"""

    def __init__(self, n_rays: int = 360, embedding_dim: int = 128):
        super().__init__()

        self.n_rays = n_rays
        self.embedding_dim = embedding_dim

        # 1D Convolutional layers for processing scan
        self.conv1 = nn.Conv1d(1, 32, kernel_size=7, padding=3)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=7, padding=3)
        self.conv3 = nn.Conv1d(64, 128, kernel_size=7, padding=3)

        self.pool = nn.MaxPool1d(2)
        self.dropout = nn.Dropout(0.3)

        # Calculate size after convolutions and pooling
        conv_output_size = n_rays // 8  # Three pooling layers

        # Fully connected layers
        self.fc1 = nn.Linear(128 * conv_output_size, 256)
        self.fc2 = nn.Linear(256, embedding_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch_size, n_rays) Lidar scan ranges

        Returns:
            (batch_size, embedding_dim) feature embedding
        """
        # Add channel dimension: (batch, 1, n_rays)
        x = x.unsqueeze(1)

        # Convolutional layers
        x = F.relu(self.conv1(x))
        x = self.pool(x)

        x = F.relu(self.conv2(x))
        x = self.pool(x)

        x = F.relu(self.conv3(x))
        x = self.pool(x)

        # Flatten
        x = x.view(x.size(0), -1)

        # Fully connected
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)

        return x


class FeatureDetectionHead(nn.Module):
    """Detection head for a specific feature type"""

    def __init__(self, embedding_dim: int = 128, n_params: int = 4):
        """
        Args:
            embedding_dim: Size of input feature embedding
            n_params: Number of parameters for this feature type
        """
        super().__init__()

        self.n_params = n_params

        # Presence classifier
        self.presence_fc = nn.Sequential(
            nn.Linear(embedding_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

        # Parameter regressor
        self.params_fc = nn.Sequential(
            nn.Linear(embedding_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, n_params)
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (batch_size, embedding_dim) feature embedding

        Returns:
            confidence: (batch_size, 1) feature presence probability
            params: (batch_size, n_params) feature parameters
        """
        confidence = self.presence_fc(x)
        params = self.params_fc(x)

        return confidence, params


class MLObjectDetectorNetwork(nn.Module):
    """
    Complete neural network for detecting lines, corners, and arcs from Lidar scans
    """

    def __init__(self, n_rays: int = 360, embedding_dim: int = 128):
        super().__init__()

        # Shared encoder
        self.encoder = LidarFeatureEncoder(n_rays, embedding_dim)

        # Detection heads for each feature type
        # Line: start_x, start_y, end_x, end_y
        self.line_head = FeatureDetectionHead(embedding_dim, n_params=4)

        # Corner: corner_x, corner_y, angle1, angle2
        self.corner_head = FeatureDetectionHead(embedding_dim, n_params=4)

        # Arc: center_x, center_y, radius, start_angle, angular_extent
        self.arc_head = FeatureDetectionHead(embedding_dim, n_params=5)

    def forward(self, scan: torch.Tensor) -> Dict[str, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Args:
            scan: (batch_size, n_rays) Lidar scan

        Returns:
            Dictionary with 'line', 'corner', 'arc' keys, each containing
            (confidence, parameters) tuple
        """
        # Encode scan
        features = self.encoder(scan)

        # Detect each feature type
        line_conf, line_params = self.line_head(features)
        corner_conf, corner_params = self.corner_head(features)
        arc_conf, arc_params = self.arc_head(features)

        return {
            'line': (line_conf, line_params),
            'corner': (corner_conf, corner_params),
            'arc': (arc_conf, arc_params)
        }


# ============================================================================
# ML-based Object Detector
# ============================================================================

class MLObjectDetector:
    """
    Machine Learning-based Object Detector using trained neural networks
    """

    def __init__(self, robot: "RobotProxy",
                 model_path: Optional[str] = None,
                 laser_scan_message_type: str = 'laser_scan',
                 device: str = 'cpu'):
        """
        Args:
            robot: Robot proxy for receiving scan data
            model_path: Path to trained model weights (if None, uses untrained model)
            laser_scan_message_type: Message type for laser scans
            device: 'cpu' or 'cuda'
        """
        self.robot = robot
        self.laser_scan_message_type = laser_scan_message_type
        self.device = torch.device(device)

        # Initialize model
        self.model = MLObjectDetectorNetwork(n_rays=360, embedding_dim=128)
        self.model.to(self.device)
        self.model.eval()

        # Load trained weights if provided
        if model_path:
            self.load_model(model_path)
            logger.info(f"Loaded model from {model_path}")
        else:
            logger.warning("No model path provided - using untrained model")

        # Detection parameters
        self.confidence_threshold = 0.5
        self.max_range = 8.0
        self.min_range = 0.1

        # State
        self.latest_scan_data = None
        self.detected_objects: List[DetectedObject] = []
        self.detected_features: List[MLDetectedFeature] = []
        self.processing_enabled = True

        # Subscribe to laser scans
        self.robot.add_message_callback(laser_scan_message_type, self.process_scan_data)

        logger.info("ML ObjectDetector initialized")

    def load_model(self, model_path: str):
        """Load trained model weights"""
        try:
            state_dict = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(state_dict)
            logger.info(f"Successfully loaded model from {model_path}")
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise

    def save_model(self, model_path: str):
        """Save model weights"""
        torch.save(self.model.state_dict(), model_path)
        logger.info(f"Saved model to {model_path}")

    def cleanup(self) -> None:
        """Clean up the detector"""
        logger.info("Cleaning up ML ObjectDetector")
        self.processing_enabled = False
        self.robot.remove_message_callback(self.laser_scan_message_type, self.process_scan_data)
        self.detected_objects.clear()
        self.detected_features.clear()
        logger.info("ML ObjectDetector cleanup completed")

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
        """Process incoming Lidar scan and detect features"""
        if not self.processing_enabled:
            return

        try:
            self.latest_scan_data = scan_data

            # Extract ranges
            ranges = scan_data.get('ranges', [])
            if not ranges and 'ranges_binary' in scan_data:
                ranges = self._decode_binary_ranges(scan_data['ranges_binary'])

            if not ranges:
                logger.warning("Received scan data with no ranges")
                return

            # Preprocess scan
            scan_tensor = self._preprocess_scan(ranges)

            # Run inference
            with torch.no_grad():
                detections = self.model(scan_tensor)

            # Extract features from detections
            features = self._extract_features(detections, ranges, scan_data)

            # Update detected features and objects
            self.detected_features = features
            self.detected_objects.clear()
            self._classify_objects(features)

            logger.info(f"ML detected {len(features)} features, {len(self.detected_objects)} objects")

        except Exception as e:
            logger.error(f"Error processing scan data: {e}")

    def _preprocess_scan(self, ranges: List[float]) -> torch.Tensor:
        """Preprocess raw scan data for neural network"""
        # Convert to numpy array
        ranges_array = np.array(ranges, dtype=np.float32)

        # Handle invalid values
        ranges_array = np.where(np.isnan(ranges_array) | np.isinf(ranges_array),
                               self.max_range, ranges_array)

        # Clip to valid range
        ranges_array = np.clip(ranges_array, self.min_range, self.max_range)

        # Normalize to [0, 1]
        ranges_array = ranges_array / self.max_range

        # Convert to tensor
        scan_tensor = torch.from_numpy(ranges_array).float()
        scan_tensor = scan_tensor.unsqueeze(0).to(self.device)  # Add batch dimension

        return scan_tensor

    def _extract_features(self, detections: Dict[str, Tuple[torch.Tensor, torch.Tensor]],
                         ranges: List[float], scan_data: Dict[str, Any]) -> List[MLDetectedFeature]:
        """Extract feature objects from network detections"""
        features = []

        angle_min = scan_data.get('angle_min', -math.pi)
        angle_increment = scan_data.get('angle_increment', 2*math.pi / len(ranges))

        # Convert scan to cartesian points for feature association
        points = self._polar_to_cartesian(ranges, angle_min, angle_increment)

        # Process each feature type
        for feature_type, (conf_tensor, params_tensor) in detections.items():
            confidence = conf_tensor[0, 0].item()

            if confidence >= self.confidence_threshold:
                params = params_tensor[0].cpu().numpy()

                # Extract feature based on type
                feature = self._create_feature(feature_type, confidence, params, points)
                if feature:
                    features.append(feature)

        return features

    def _create_feature(self, feature_type: str, confidence: float,
                       params: np.ndarray, all_points: List[Tuple[float, float]]) -> Optional[MLDetectedFeature]:
        """Create a feature object from network output"""

        if feature_type == 'line':
            # params: [start_x, start_y, end_x, end_y]
            start = (float(params[0]), float(params[1]))
            end = (float(params[2]), float(params[3]))

            # Find scan points near this line
            feature_points = self._find_points_near_line(all_points, start, end)

            length = math.sqrt((end[0] - start[0])**2 + (end[1] - start[1])**2)
            angle = math.atan2(end[1] - start[1], end[0] - start[0])

            return MLDetectedFeature(
                feature_type='line',
                confidence=confidence,
                parameters={
                    'start': start,
                    'end': end,
                    'length': length,
                    'angle': angle
                },
                points=feature_points
            )

        elif feature_type == 'corner':
            # params: [corner_x, corner_y, angle1, angle2]
            corner_point = (float(params[0]), float(params[1]))
            angle1 = float(params[2])
            angle2 = float(params[3])

            # Find scan points near this corner
            feature_points = self._find_points_near_corner(all_points, corner_point, angle1, angle2)

            interior_angle = abs(angle2 - angle1)

            return MLDetectedFeature(
                feature_type='corner',
                confidence=confidence,
                parameters={
                    'corner_point': corner_point,
                    'angle1': angle1,
                    'angle2': angle2,
                    'interior_angle': interior_angle
                },
                points=feature_points
            )

        elif feature_type == 'arc':
            # params: [center_x, center_y, radius, angular_extent]
            center = (float(params[0]), float(params[1]))
            radius = float(params[2])
            angular_extent = float(params[3])

            # Find scan points near this arc
            feature_points = self._find_points_near_arc(all_points, center, radius)

            return MLDetectedFeature(
                feature_type='arc',
                confidence=confidence,
                parameters={
                    'center': center,
                    'radius': radius,
                    'angular_extent': angular_extent,
                    'arc_length': len(feature_points)
                },
                points=feature_points
            )

        return None

    def _polar_to_cartesian(self, ranges: List[float], angle_min: float,
                           angle_increment: float) -> List[Tuple[float, float]]:
        """Convert polar scan to cartesian points"""
        points = []
        for i, range_val in enumerate(ranges):
            if (math.isnan(range_val) or math.isinf(range_val) or
                range_val > self.max_range or range_val < self.min_range):
                continue

            angle = angle_min + i * angle_increment
            x = range_val * math.cos(angle)
            y = range_val * math.sin(angle)
            points.append((x, y))

        return points

    def _find_points_near_line(self, points: List[Tuple[float, float]],
                              start: Tuple[float, float], end: Tuple[float, float],
                              threshold: float = 0.1) -> List[Tuple[float, float]]:
        """Find scan points near a line segment"""
        near_points = []

        for point in points:
            dist = self._point_to_line_distance(point, start, end)
            if dist < threshold:
                near_points.append(point)

        return near_points

    def _find_points_near_corner(self, points: List[Tuple[float, float]],
                                corner: Tuple[float, float],
                                angle1: float, angle2: float,
                                threshold: float = 0.2) -> List[Tuple[float, float]]:
        """Find scan points near a corner"""
        near_points = []

        for point in points:
            dist = math.sqrt((point[0] - corner[0])**2 + (point[1] - corner[1])**2)
            if dist < threshold:
                near_points.append(point)

        return near_points

    def _find_points_near_arc(self, points: List[Tuple[float, float]],
                             center: Tuple[float, float], radius: float,
                             threshold: float = 0.1) -> List[Tuple[float, float]]:
        """Find scan points near an arc"""
        near_points = []

        for point in points:
            dist_to_center = math.sqrt((point[0] - center[0])**2 + (point[1] - center[1])**2)
            if abs(dist_to_center - radius) < threshold:
                near_points.append(point)

        return near_points

    def _point_to_line_distance(self, point: Tuple[float, float],
                               line_start: Tuple[float, float],
                               line_end: Tuple[float, float]) -> float:
        """Calculate perpendicular distance from point to line segment"""
        px, py = point
        x1, y1 = line_start
        x2, y2 = line_end

        # Line vector
        dx = x2 - x1
        dy = y2 - y1

        if dx == 0 and dy == 0:
            return math.sqrt((px - x1)**2 + (py - y1)**2)

        # Parameter t for projection
        t = max(0, min(1, ((px - x1) * dx + (py - y1) * dy) / (dx**2 + dy**2)))

        # Closest point on line
        closest_x = x1 + t * dx
        closest_y = y1 + t * dy

        return math.sqrt((px - closest_x)**2 + (py - closest_y)**2)

    def _classify_objects(self, features: List[MLDetectedFeature]) -> None:
        """Classify detected features into objects"""
        for feature in features:
            if feature.feature_type == 'line':
                obj = self._classify_line_feature(feature)
            elif feature.feature_type == 'corner':
                obj = self._classify_corner_feature(feature)
            elif feature.feature_type == 'arc':
                obj = self._classify_arc_feature(feature)
            else:
                continue

            if obj:
                self.detected_objects.append(obj)

    def _classify_line_feature(self, feature: MLDetectedFeature) -> Optional[DetectedObject]:
        """Classify a line feature"""
        length = feature.parameters.get('length', 0)

        if length > 1.5:
            return DetectedObject('wall', [feature], feature.confidence * 0.9)
        elif length > 0.5:
            return DetectedObject('furniture_edge', [feature], feature.confidence * 0.7)
        else:
            return DetectedObject('small_object', [feature], feature.confidence * 0.5)

    def _classify_corner_feature(self, feature: MLDetectedFeature) -> Optional[DetectedObject]:
        """Classify a corner feature"""
        return DetectedObject('corner', [feature], feature.confidence * 0.8)

    def _classify_arc_feature(self, feature: MLDetectedFeature) -> Optional[DetectedObject]:
        """Classify an arc feature"""
        radius = feature.parameters.get('radius', 0)

        if radius > 0.3:
            return DetectedObject('round_object', [feature], feature.confidence * 0.8)
        else:
            return DetectedObject('cylindrical_object', [feature], feature.confidence * 0.7)

    # ========================================================================
    # Public API (compatible with original ObjectDetector)
    # ========================================================================

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
            "processing_enabled": self.processing_enabled
        }

    def set_processing_enabled(self, enabled: bool) -> None:
        """Enable or disable processing"""
        self.processing_enabled = enabled
        logger.info(f"ML object detection processing {'enabled' if enabled else 'disabled'}")

    def find_closest_object(self) -> Optional[DetectedObject]:
        """Find the closest detected object"""
        if not self.detected_objects:
            return None
        return min(self.detected_objects, key=lambda obj: obj.distance)

    def find_closest_objects(self) -> List[DetectedObject]:
        """Find all detected objects sorted by distance"""
        return sorted(self.detected_objects, key=lambda obj: obj.distance)

    def find_objects_by_type(self, object_type: str) -> List[DetectedObject]:
        """Find all objects of a specific type"""
        return [obj for obj in self.detected_objects if obj.object_type == object_type]

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
        self.detected_features.clear()
        logger.info("Cleared all ML detections")

    def set_confidence_threshold(self, threshold: float) -> None:
        """Set detection confidence threshold"""
        self.confidence_threshold = max(0.0, min(1.0, threshold))
        logger.info(f"Set confidence threshold to {self.confidence_threshold}")


# ============================================================================
# Shared Visualization Functions
# ============================================================================

def visualize_scene_with_scan(feature: Dict[str, Any], scan: np.ndarray,
                             title: str = "", save_path: Optional[str] = None) -> None:
    """
    Visualize a synthetic scene feature with its Lidar scan

    Args:
        feature: Feature dictionary from SyntheticSceneGenerator
        scan: Lidar scan array (360 ranges)
        title: Plot title
        save_path: Path to save image (if None, displays instead)
    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 8))

    # Convert scan to cartesian
    angles = np.linspace(-np.pi, np.pi, len(scan))

    # Plot feature FIRST (so scan points appear on top)
    if feature['type'] == 'line':
        x1, y1 = feature['start']
        x2, y2 = feature['end']
        ax.plot([x1, x2], [y1, y2], 'g-', linewidth=8, alpha=0.6,
               label='Line feature', zorder=1)
    elif feature['type'] == 'corner':
        cx, cy = feature['corner_point']
        x1, y1 = feature['arm1_end']
        x2, y2 = feature['arm2_end']
        # Draw arm1 in orange, arm2 in darker brown/orange
        ax.plot([cx, x1], [cy, y1], 'orange', linewidth=8, alpha=0.6,
               label='Corner (arm1)', zorder=1)
        ax.plot([cx, x2], [cy, y2], 'darkorange', linewidth=8, alpha=0.8,
               label='Corner (arm2)', zorder=1)
        ax.scatter(cx, cy, c='orange', s=200, marker='x', zorder=2)
    elif feature['type'] == 'arc':
        cx, cy = feature['center']
        radius = feature['radius']
        start_angle = feature['start_angle']
        end_angle = feature['end_angle']

        # Handle angle wrapping: if end < start, the arc wraps around
        if end_angle < start_angle:
            # Arc wraps around -pi/pi boundary
            # Draw from start to pi, then from -pi to end
            arc_angles1 = np.linspace(start_angle, np.pi, 50)
            arc_angles2 = np.linspace(-np.pi, end_angle, 50)
            arc_angles = np.concatenate([arc_angles1, arc_angles2])
        else:
            # Normal arc from start to end
            arc_angles = np.linspace(start_angle, end_angle, 100)

        arc_xs = cx + radius * np.cos(arc_angles)
        arc_ys = cy + radius * np.sin(arc_angles)
        ax.plot(arc_xs, arc_ys, 'purple', linewidth=8, alpha=0.6,
               label='Arc feature', zorder=1)

        # Draw full circle outline for reference (light)
        circle_angles = np.linspace(0, 2*np.pi, 100)
        circle_xs = cx + radius * np.cos(circle_angles)
        circle_ys = cy + radius * np.sin(circle_angles)
        ax.plot(circle_xs, circle_ys, 'purple', linewidth=1, alpha=0.2,
               linestyle='--', zorder=1)

        # Mark arc center
        ax.scatter(cx, cy, c='purple', s=100, marker='+', zorder=2, label='Arc center')

    # Plot all rays (including max range) - behind everything
    xs_all = scan * np.cos(angles)
    ys_all = scan * np.sin(angles)
    ax.scatter(xs_all, ys_all, c='lightgray', s=1, alpha=0.3,
              label='Max range', zorder=0)

    # Plot feature hits ON TOP with high zorder
    hit_mask = scan < 7.9
    xs_hit = scan[hit_mask] * np.cos(angles[hit_mask])
    ys_hit = scan[hit_mask] * np.sin(angles[hit_mask])
    ax.scatter(xs_hit, ys_hit, c='blue', s=30, alpha=0.9,
              edgecolors='darkblue', linewidths=0.5,
              label='Feature hits', zorder=5)

    # Plot robot on top
    ax.scatter(0, 0, c='red', s=200, marker='o', label='Robot',
              zorder=10, edgecolors='darkred', linewidths=2)

    hits = np.sum(hit_mask)
    ax.set_aspect('equal')
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_title(title, fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_xlim(-8, 8)
    ax.set_ylim(-8, 8)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        logger.info(f"Saved visualization to {save_path}")
    else:
        plt.show()

    plt.close()
