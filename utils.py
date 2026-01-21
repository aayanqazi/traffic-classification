"""
Utility functions for the vehicle counting application.
"""

import cv2
import numpy as np
import pandas as pd
from typing import Dict, List
from pathlib import Path
import json
from loguru import logger


def setup_logging():
    """Configure logging for the application."""
    logger.remove()  # Remove default handler
    logger.add(
        "vehicle_counter.log",
        rotation="10 MB",
        retention="1 week",
        level="INFO",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {name}:{function}:{line} | {message}"
    )
    logger.add(
        lambda msg: print(msg, end=""),
        level="INFO",
        format="{time:HH:mm:ss} | {level} | {message}"
    )


def save_results_to_json(counts: Dict[str, int], output_path: str):
    """
    Save counting results to JSON file.

    Args:
        counts: Dictionary with vehicle counts
        output_path: Path to save JSON file
    """
    try:
        with open(output_path, 'w') as f:
            json.dump({
                'vehicle_counts': counts,
                'timestamp': pd.Timestamp.now().isoformat()
            }, f, indent=2)
        logger.info(f"Results saved to {output_path}")
    except Exception as e:
        logger.error(f"Failed to save JSON results: {e}")


def save_results_to_csv(counts: Dict[str, int], output_path: str):
    """
    Save counting results to CSV file.

    Args:
        counts: Dictionary with vehicle counts
        output_path: Path to save CSV file
    """
    try:
        df = pd.DataFrame([counts])
        df.to_csv(output_path, index=False)
        logger.info(f"Results saved to {output_path}")
    except Exception as e:
        logger.error(f"Failed to save CSV results: {e}")


def get_video_info(video_path: str) -> Dict:
    """
    Get basic information about a video file.

    Args:
        video_path: Path to video file

    Returns:
        Dictionary with video properties
    """
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")

        info = {
            'fps': cap.get(cv2.CAP_PROP_FPS),
            'frame_count': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            'duration': cap.get(cv2.CAP_PROP_FRAME_COUNT) / cap.get(cv2.CAP_PROP_FPS)
        }

        cap.release()
        return info

    except Exception as e:
        logger.error(f"Failed to get video info: {e}")
        return {}


def validate_video_file(file_path: str) -> bool:
    """
    Validate if file is a valid video format.

    Args:
        file_path: Path to file

    Returns:
        True if valid video file
    """
    if not Path(file_path).exists():
        return False

    try:
        cap = cv2.VideoCapture(file_path)
        valid = cap.isOpened()
        cap.release()
        return valid
    except:
        return False


def resize_frame(frame: np.ndarray, max_width: int = 800, max_height: int = 600) -> np.ndarray:
    """
    Resize frame while maintaining aspect ratio.

    Args:
        frame: Input frame
        max_width: Maximum width
        max_height: Maximum height

    Returns:
        Resized frame
    """
    height, width = frame.shape[:2]

    # Calculate scaling factor
    scale_w = max_width / width
    scale_h = max_height / height
    scale = min(scale_w, scale_h)

    if scale < 1.0:
        new_width = int(width * scale)
        new_height = int(height * scale)
        frame = cv2.resize(frame, (new_width, new_height))

    return frame


def draw_tracking_info(frame: np.ndarray, tracked_objects: List[Dict]) -> np.ndarray:
    """
    Draw tracking information on frame.

    Args:
        frame: Input frame
        tracked_objects: List of tracked objects

    Returns:
        Frame with tracking info drawn
    """
    frame_copy = frame.copy()

    # Colors for different vehicle types
    colors = {
        'car': (0, 255, 0),        # Green
        'motorcycle': (255, 0, 0), # Blue
        'bus': (0, 0, 255),        # Red
        'truck': (255, 255, 0),    # Cyan
        'unknown': (255, 255, 255) # White
    }

    for obj in tracked_objects:
        track_id = obj['track_id']
        bbox = obj['bbox']
        class_name = obj.get('class', 'unknown')

        # Draw bounding box
        color = colors.get(class_name, (255, 255, 255))
        cv2.rectangle(frame_copy, (int(bbox[0]), int(bbox[1])),
                     (int(bbox[2]), int(bbox[3])), color, 2)

        # Draw track ID and class
        label = f"ID:{track_id} {class_name}"
        cv2.putText(frame_copy, label, (int(bbox[0]), int(bbox[1]) - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    return frame_copy


def draw_stats_panel(frame: np.ndarray, counts: Dict[str, int],
                    processing_fps: float = 0.0) -> np.ndarray:
    """
    Draw statistics panel on frame.

    Args:
        frame: Input frame
        counts: Vehicle counts dictionary
        processing_fps: Current processing FPS

    Returns:
        Frame with stats panel
    """
    frame_copy = frame.copy()
    height, width = frame.shape[:2]

    # Create semi-transparent overlay for stats
    overlay = frame_copy.copy()
    cv2.rectangle(overlay, (10, 10), (300, 200), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.7, frame_copy, 0.3, 0, frame_copy)

    # Draw stats text
    y_offset = 30
    cv2.putText(frame_copy, "VEHICLE COUNTS", (20, y_offset),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    y_offset += 30
    for vehicle_type, count in counts.items():
        if vehicle_type != 'total':
            cv2.putText(frame_copy, f"{vehicle_type.capitalize()}: {count}",
                       (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            y_offset += 25

    y_offset += 10
    cv2.putText(frame_copy, f"Total: {counts.get('total', 0)}",
               (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    y_offset += 30
    cv2.putText(frame_copy, f"FPS: {processing_fps:.1f}",
               (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    return frame_copy
