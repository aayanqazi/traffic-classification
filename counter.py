"""
Vehicle Counting Module
Handles counting vehicles using line-crossing logic to avoid double counting.
"""

import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional
from loguru import logger


class VehicleCounter:
    """Counts vehicles crossing a configurable line."""

    def __init__(self, line_points: Optional[List[Tuple[int, int]]] = None):
        """
        Initialize the vehicle counter.

        Args:
            line_points: List of (x, y) points defining the counting line
        """
        # Default will be set dynamically based on frame size in draw_counting_line
        # This is just a placeholder - actual position calculated from frame dimensions
        self.counting_line = line_points or [(50, 300), (750, 300)]  # Placeholder, will be updated dynamically
        
        self.counts = {
            'car': 0,
            'motorcycle': 0,
            'bus': 0,
            'truck': 0,
            'total': 0
        }

        # Directional counts for two-way traffic
        self.direction_counts = {
            'up': 0,      # Moving upward (top direction)
            'down': 0     # Moving downward (bottom direction)
        }

        # Track which tracks have already been counted
        self.counted_tracks = set()

        # Store previous positions for direction calculation
        self.previous_positions = {}  # track_id -> (x, y)

        logger.info(f"Vehicle counter initialized with line: {self.counting_line}")
        logger.info("Two-way traffic counting enabled")

    def set_counting_line(self, points: List[Tuple[int, int]]):
        """
        Update the counting line.

        Args:
            points: List of (x, y) points defining the new line
        """
        if len(points) >= 2:
            self.counting_line = points
            logger.info(f"Counting line updated: {points}")
        else:
            logger.warning("Need at least 2 points to define a counting line")

    def set_counting_line_from_frame(self, frame: np.ndarray):
        """
        Set counting line position based on frame dimensions (centered horizontally).

        Args:
            frame: Input frame to get dimensions from
        """
        frame_height, frame_width = frame.shape[:2]
        center_y = frame_height // 2
        margin = 50  # Margin from edges
        start_x = margin
        end_x = frame_width - margin
        self.counting_line = [(start_x, center_y), (end_x, center_y)]
        logger.info(f"Counting line set to center: {self.counting_line} (frame: {frame_width}x{frame_height})")

    def _point_to_line_distance(self, point: Tuple[float, float],
                               line_start: Tuple[float, float],
                               line_end: Tuple[float, float]) -> float:
        """
        Calculate perpendicular distance from point to line.

        Args:
            point: (x, y) position
            line_start: (x, y) start of line
            line_end: (x, y) end of line

        Returns:
            Distance from point to line
        """
        px, py = point
        x1, y1 = line_start
        x2, y2 = line_end

        # Vector from line start to end
        dx = x2 - x1
        dy = y2 - y1

        # Vector from line start to point
        dxp = px - x1
        dyp = py - y1

        # Length of line segment
        line_length = np.sqrt(dx**2 + dy**2)
        if line_length == 0:
            return np.sqrt(dxp**2 + dyp**2)

        # Project point onto line
        t = max(0, min(1, (dxp * dx + dyp * dy) / (line_length ** 2)))

        # Closest point on line
        closest_x = x1 + t * dx
        closest_y = y1 + t * dy

        # Distance to closest point
        distance = np.sqrt((px - closest_x)**2 + (py - closest_y)**2)

        return distance

    def _check_line_crossing(self, track_id: int, current_pos: Tuple[float, float]) -> tuple:
        """
        Check if a track has crossed the counting line.

        Args:
            track_id: Track ID
            current_pos: Current (x, y) position

        Returns:
            Tuple of (crossed: bool, direction: str) where direction is 'down', 'up', 'right', 'left', or None
        """
        if track_id not in self.previous_positions:
            return False, None

        prev_pos = self.previous_positions[track_id]

        # For simplicity, we'll use a horizontal line crossing logic
        # Check if the line is mostly horizontal or vertical
        line_start, line_end = self.counting_line[0], self.counting_line[-1]

        if abs(line_end[1] - line_start[1]) < abs(line_end[0] - line_start[0]):
            # Mostly horizontal line - check y crossing
            prev_y = prev_pos[1]
            curr_y = current_pos[1]
            line_y = line_start[1]

            logger.debug(f"Track {track_id}: prev_y={prev_y:.1f}, curr_y={curr_y:.1f}, line_y={line_y}")

            # Crossed from above to below (downward)
            if prev_y < line_y and curr_y >= line_y:
                logger.info(f"🎯 Track {track_id} crossed horizontal line DOWNWARD at y={line_y}")
                return True, 'down'
            # Crossed from below to above (upward)
            elif prev_y > line_y and curr_y <= line_y:
                logger.info(f"🎯 Track {track_id} crossed horizontal line UPWARD at y={line_y}")
                return True, 'up'
        else:
            # Mostly vertical line - check x crossing
            prev_x = prev_pos[0]
            curr_x = current_pos[0]
            line_x = line_start[0]

            # Crossed from left to right
            if prev_x < line_x and curr_x >= line_x:
                return True, 'right'
            # Crossed from right to left
            elif prev_x > line_x and curr_x <= line_x:
                return True, 'left'

        return False, None

    def update_counts(self, tracked_objects: List[Dict]) -> bool:
        """
        Update vehicle counts based on tracked objects.

        Args:
            tracked_objects: List of tracked objects from tracker

        Returns:
            True if any counts were updated
        """
        updated = False

        logger.debug(f"Counter received {len(tracked_objects)} tracked objects")

        for obj in tracked_objects:
            track_id = obj['track_id']
            class_name = obj['class']
            position = obj['position']

            logger.debug(f"Processing track {track_id}: {class_name} at position {position}")

            # Skip if already counted
            if track_id in self.counted_tracks:
                logger.debug(f"Track {track_id} already counted, skipping")
                continue

            # Check if line was crossed
            crossed, direction = self._check_line_crossing(track_id, position)
            if crossed:
                # Count the vehicle
                if class_name in self.counts:
                    self.counts[class_name] += 1
                    self.counts['total'] += 1
                    self.counted_tracks.add(track_id)
                    updated = True
                    
                    # Update directional counts
                    if direction in self.direction_counts:
                        self.direction_counts[direction] += 1
                    
                    direction_label = f" ({direction})" if direction else ""
                    logger.info(f"✅ Counted {class_name} (ID: {track_id}){direction_label} - Total: {self.counts['total']}")
                else:
                    logger.warning(f"Unknown vehicle class: {class_name}")
            else:
                logger.debug(f"Track {track_id} has not crossed line yet")

            # Update previous position
            self.previous_positions[track_id] = position

        # Clean up old tracks (keep only recent ones)
        max_tracks = 1000
        if len(self.previous_positions) > max_tracks:
            # Remove oldest tracks
            oldest_tracks = list(self.previous_positions.keys())[:max_tracks//2]
            for track_id in oldest_tracks:
                self.previous_positions.pop(track_id, None)
                self.counted_tracks.discard(track_id)

        if updated:
            logger.info(f"Current counts: {self.counts}")

        return updated

    def get_counts(self) -> Dict[str, int]:
        """
        Get current vehicle counts.

        Returns:
            Dictionary with counts for each vehicle type and total
        """
        return self.counts.copy()

    def get_direction_counts(self) -> Dict[str, int]:
        """
        Get directional counts for two-way traffic.

        Returns:
            Dictionary with counts for each direction
        """
        return self.direction_counts.copy()

    def draw_counting_line(self, frame: np.ndarray, 
                          line_y_percent: float = None,
                          line_start_x_percent: float = None,
                          line_end_x_percent: float = None) -> np.ndarray:
        """
        Draw the counting line on the frame.

        Args:
            frame: Input frame
            line_y_percent: Vertical position as percentage (0-100), None for auto-center
            line_start_x_percent: Start X position as percentage (0-100), None for auto
            line_end_x_percent: End X position as percentage (0-100), None for auto

        Returns:
            Frame with counting line drawn
        """
        frame_copy = frame.copy()
        
        frame_height, frame_width = frame.shape[:2]
        
        # Use user-defined positions if provided, otherwise auto-center
        if line_y_percent is not None:
            center_y = int(frame_height * line_y_percent / 100)
        else:
            center_y = frame_height // 2
        
        if line_start_x_percent is not None:
            start_x = int(frame_width * line_start_x_percent / 100)
        else:
            start_x = 50  # Default margin
        
        if line_end_x_percent is not None:
            end_x = int(frame_width * line_end_x_percent / 100)
        else:
            end_x = frame_width - 50  # Default margin
        
        # Update counting line position
        self.counting_line = [(start_x, center_y), (end_x, center_y)]

        # Draw the counting line
        if len(self.counting_line) >= 2:
            points = np.array(self.counting_line, np.int32)
            cv2.polylines(frame_copy, [points], False, (0, 255, 255), 3)  # Yellow line

            # Add label
            mid_point = ((points[0][0] + points[-1][0]) // 2, (points[0][1] + points[-1][1]) // 2)
            cv2.putText(frame_copy, "COUNTING LINE (TWO-WAY)", (mid_point[0] - 120, mid_point[1] - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

            # Draw directional arrows to indicate two-way counting
            # Up arrow (above line)
            arrow_y_up = mid_point[1] - 30
            cv2.arrowedLine(frame_copy, (mid_point[0] - 40, arrow_y_up + 15), 
                          (mid_point[0] - 40, arrow_y_up - 15), (0, 255, 0), 2, tipLength=0.4)
            
            # Down arrow (below line)
            arrow_y_down = mid_point[1] + 30
            cv2.arrowedLine(frame_copy, (mid_point[0] + 40, arrow_y_down - 15), 
                          (mid_point[0] + 40, arrow_y_down + 15), (0, 255, 0), 2, tipLength=0.4)

        return frame_copy

    def reset_counts(self):
        """Reset all counts to zero."""
        self.counts = {
            'car': 0,
            'motorcycle': 0,
            'bus': 0,
            'truck': 0,
            'total': 0
        }
        self.direction_counts = {
            'up': 0,
            'down': 0
        }
        self.counted_tracks.clear()
        self.previous_positions.clear()
        logger.info("Counts reset")
