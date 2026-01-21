"""
Multi-Object Tracking Module using ByteTrack
Handles tracking of detected vehicles across frames.
"""

import numpy as np
import supervision as sv
from supervision import ByteTrack
from loguru import logger
from typing import List, Dict, Tuple


class VehicleTracker:
    """ByteTrack-based vehicle tracker for consistent ID assignment."""

    def __init__(self, track_activation_threshold: float = 0.25,
                 lost_track_buffer: int = 30,
                 minimum_matching_threshold: float = 0.8,
                 frame_rate: int = 30,
                 minimum_consecutive_frames: int = 1):
        """
        Initialize the vehicle tracker.

        Args:
            track_activation_threshold: Detection confidence threshold for tracking
            lost_track_buffer: Number of frames to keep lost tracks
            minimum_matching_threshold: Threshold for matching detections to tracks
            frame_rate: Video frame rate for time-based tracking
            minimum_consecutive_frames: Minimum consecutive frames for track activation
        """
        self.tracker = ByteTrack(
            track_activation_threshold=track_activation_threshold,
            lost_track_buffer=lost_track_buffer,
            minimum_matching_threshold=minimum_matching_threshold,
            frame_rate=frame_rate,
            minimum_consecutive_frames=minimum_consecutive_frames
        )

        self.track_history = {}  # track_id -> list of positions
        self.active_tracks = {}  # track_id -> current info
        logger.info("Vehicle tracker initialized")

    def update(self, detections: List[Dict], frame_shape: Tuple[int, int]) -> List[Dict]:
        """
        Update tracker with new detections.

        Args:
            detections: List of detections from detector
            frame_shape: (height, width) of the frame

        Returns:
            List of tracked objects with IDs
        """
        if not detections:
            # Update tracker with empty detections
            empty_detections = sv.Detections.empty()
            tracks = self.tracker.update_with_detections(empty_detections)
            return []

        # Convert detections to supervision Detections format
        bboxes = []
        confidences = []
        class_ids = []
        class_names = []

        for det in detections:
            bbox = det['bbox']  # [x1, y1, x2, y2]
            confidence = det['confidence']
            class_name = det['class']

            # Map class names to IDs (simplified mapping)
            class_id_map = {
                'car': 0,
                'motorcycle': 1,
                'bus': 2,
                'truck': 3
            }
            class_id = class_id_map.get(class_name, 0)

            bboxes.append(bbox)
            confidences.append(confidence)
            class_ids.append(class_id)
            class_names.append(class_name)

        # Create supervision Detections object
        sv_detections = sv.Detections(
            xyxy=np.array(bboxes),
            confidence=np.array(confidences),
            class_id=np.array(class_ids)
        )

        # Update tracker
        tracks = self.tracker.update_with_detections(sv_detections)

        # Process tracks - tracks is now a Detections object with tracker_id
        tracked_objects = []

        if tracks is not None and len(tracks) > 0:
            for i in range(len(tracks)):
                track_id = int(tracks.tracker_id[i]) if tracks.tracker_id is not None else i
                bbox = tracks.xyxy[i].tolist()
                confidence = float(tracks.confidence[i]) if tracks.confidence is not None else 0.0
                class_id = int(tracks.class_id[i]) if tracks.class_id is not None else 0

                # Map class ID back to class name
                class_name_map = {
                    0: 'car',
                    1: 'motorcycle',
                    2: 'bus',
                    3: 'truck'
                }
                class_name = class_names[i] if i < len(class_names) else class_name_map.get(class_id, 'unknown')

                # Store track history
                center_x = (bbox[0] + bbox[2]) / 2
                center_y = (bbox[1] + bbox[3]) / 2

                if track_id not in self.track_history:
                    self.track_history[track_id] = []

                self.track_history[track_id].append((center_x, center_y))

                # Keep only last N positions for memory efficiency
                if len(self.track_history[track_id]) > 100:
                    self.track_history[track_id] = self.track_history[track_id][-100:]

                # Update active tracks
                self.active_tracks[track_id] = {
                    'bbox': bbox,
                    'class': class_name,
                    'confidence': confidence,
                    'position': (center_x, center_y)
                }

                tracked_objects.append({
                    'track_id': track_id,
                    'bbox': bbox,
                    'class': class_name,
                    'confidence': confidence,
                    'position': (center_x, center_y)
                })

        logger.debug(f"Tracking {len(tracked_objects)} objects")
        if len(tracked_objects) > 0:
            logger.debug(f"  Sample: Track {tracked_objects[0]['track_id']}: {tracked_objects[0]['class']}")
        return tracked_objects

    def get_track_history(self, track_id: int) -> List[Tuple[float, float]]:
        """
        Get position history for a track.

        Args:
            track_id: Track ID to get history for

        Returns:
            List of (x, y) positions
        """
        return self.track_history.get(track_id, [])

    def get_active_tracks(self) -> Dict[int, Dict]:
        """
        Get all currently active tracks.

        Returns:
            Dictionary of track_id -> track_info
        """
        return self.active_tracks.copy()

    def reset(self):
        """Reset tracker state."""
        self.tracker = ByteTrack(
            track_activation_threshold=0.25,
            lost_track_buffer=30,
            minimum_matching_threshold=0.8,
            frame_rate=30,
            minimum_consecutive_frames=1
        )
        self.track_history.clear()
        self.active_tracks.clear()
        logger.info("Tracker reset")
