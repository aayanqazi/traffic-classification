"""
Vehicle Detection Module using YOLOv8
Handles object detection for vehicles in video frames.
"""

import cv2
import numpy as np
from ultralytics import YOLO
from loguru import logger
import torch


class VehicleDetector:
    """YOLOv8-based vehicle detector for cars, motorcycles, buses, and trucks."""

    # Vehicle classes we want to detect (COCO dataset class IDs)
    VEHICLE_CLASSES = {
        2: 'car',      # COCO class 2: car
        3: 'motorcycle',  # COCO class 3: motorcycle
        5: 'bus',      # COCO class 5: bus
        7: 'truck'     # COCO class 7: truck
    }

    def __init__(self, model_path: str = 'yolov8n.pt', confidence_threshold: float = 0.5):
        """
        Initialize the vehicle detector.

        Args:
            model_path: Path to YOLOv8 model (will download if not exists)
            confidence_threshold: Minimum confidence for detections
        """
        self.confidence_threshold = confidence_threshold
        self.model = None

        try:
            # Force CPU usage for offline compatibility
            torch.device('cpu')

            # Load YOLOv8 model
            logger.info(f"Loading YOLOv8 model: {model_path}")
            self.model = YOLO(model_path)
            
            # Optimize model for faster inference
            # Use smaller input size for faster processing
            self.model.overrides['imgsz'] = 640  # Standard size, can reduce to 416 for even faster
            logger.info("YOLOv8 model loaded successfully (optimized for speed)")

        except Exception as e:
            logger.error(f"Failed to load YOLOv8 model: {e}")
            raise

    def detect_vehicles(self, frame: np.ndarray) -> list:
        """
        Detect vehicles in a frame.

        Args:
            frame: Input frame as numpy array (BGR format)

        Returns:
            List of detections: [{'bbox': [x1, y1, x2, y2], 'class': str, 'confidence': float}, ...]
        """
        if self.model is None:
            logger.error("Model not loaded")
            return []

        try:
            # Run inference with optimizations
            # imgsz=640 for faster processing, half=False for CPU compatibility
            results = self.model(frame, verbose=False, imgsz=640, half=False)[0]

            detections = []
            for box in results.boxes:
                # Get bounding box coordinates
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()

                # Get class and confidence
                class_id = int(box.cls[0].cpu().numpy())
                confidence = float(box.conf[0].cpu().numpy())

                # Check if it's a vehicle class and meets confidence threshold
                if class_id in self.VEHICLE_CLASSES and confidence >= self.confidence_threshold:
                    detection = {
                        'bbox': [int(x1), int(y1), int(x2), int(y2)],
                        'class': self.VEHICLE_CLASSES[class_id],
                        'confidence': confidence
                    }
                    detections.append(detection)

            logger.debug(f"Detected {len(detections)} vehicles in frame")
            if len(detections) > 0:
                logger.debug(f"  Sample: {detections[0]['class']} (conf: {detections[0]['confidence']:.2f})")
            return detections

        except Exception as e:
            logger.error(f"Detection failed: {e}")
            return []

    def draw_detections(self, frame: np.ndarray, detections: list) -> np.ndarray:
        """
        Draw bounding boxes and labels on the frame.

        Args:
            frame: Input frame
            detections: List of detections

        Returns:
            Frame with drawings
        """
        frame_copy = frame.copy()

        # Colors for different vehicle types
        colors = {
            'car': (0, 255, 0),        # Green
            'motorcycle': (255, 0, 0), # Blue
            'bus': (0, 0, 255),        # Red
            'truck': (255, 255, 0)     # Cyan
        }

        for detection in detections:
            bbox = detection['bbox']
            class_name = detection['class']
            confidence = detection['confidence']

            # Draw bounding box
            color = colors.get(class_name, (255, 255, 255))
            cv2.rectangle(frame_copy, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)

            # Draw label
            label = f"{class_name}: {confidence:.2f}"
            cv2.putText(frame_copy, label, (bbox[0], bbox[1] - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        return frame_copy
