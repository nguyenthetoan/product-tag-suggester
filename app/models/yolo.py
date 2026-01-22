"""
YOLO-based object detection model.
Uses Ultralytics YOLO26 for efficient object detection (latest as of 2026).
Falls back to YOLOv8 if YOLO26 is not available.
"""

import os
import ssl
import torch
from PIL import Image
from functools import lru_cache
from dataclasses import dataclass
from ultralytics import YOLO

# Fix SSL certificate verification issues on macOS
# This is needed when downloading model weights
if hasattr(ssl, '_create_unverified_context'):
    ssl._create_default_https_context = ssl._create_unverified_context

# Alternative: Set environment variable to skip SSL verification for downloads
os.environ['CURL_CA_BUNDLE'] = ''


@dataclass
class Detection:
    """A detected object with bounding box and class info."""
    
    bbox: tuple[float, float, float, float]  # (x, y, width, height)
    class_id: int
    class_name: str
    confidence: float


class YOLOModel:
    """Wrapper for YOLO object detection model (YOLO26 or YOLOv8)."""

    def __init__(self, model_name: str = "yolo26m.pt"):
        """
        Initialize the YOLO model.

        Args:
            model_name: YOLO model variant. Options:
                YOLO26 (recommended - faster, more accurate):
                - "yolo26n.pt" (nano - fastest, least accurate)
                - "yolo26s.pt" (small - fast, good accuracy)
                - "yolo26m.pt" (medium - balanced) [default]
                - "yolo26l.pt" (large - slower, more accurate)
                - "yolo26x.pt" (extra large - slowest, most accurate)
                
                YOLOv8 (fallback):
                - "yolov8n.pt", "yolov8s.pt", "yolov8m.pt", etc.
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Try to load the model, with fallback to YOLOv8 if YOLO26 fails
        try:
            self.model = YOLO(model_name)
        except Exception as e:
            # Fallback to YOLOv8 if YOLO26 is not available
            fallback_name = model_name.replace("yolo26", "yolov8")
            print(f"Warning: Could not load {model_name}, falling back to {fallback_name}: {e}")
            self.model = YOLO(fallback_name)
        
        self.model.to(self.device)
        
    def detect(
        self,
        image: Image.Image,
        confidence_threshold: float = 0.25,
        classes: list[int] | None = None,
    ) -> list[Detection]:
        """
        Detect objects in an image.

        Args:
            image: PIL Image object
            confidence_threshold: Minimum confidence for detections
            classes: Optional list of class IDs to filter (None = all classes)

        Returns:
            List of Detection objects
        """
        # Run inference
        results = self.model(
            image,
            conf=confidence_threshold,
            classes=classes,
            verbose=False,
        )
        
        detections = []
        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue
                
            for i in range(len(boxes)):
                # Get xyxy format and convert to xywh
                xyxy = boxes.xyxy[i].cpu().numpy()
                x1, y1, x2, y2 = xyxy
                
                # Convert to (x, y, width, height) format
                bbox = (
                    float(x1),
                    float(y1),
                    float(x2 - x1),
                    float(y2 - y1),
                )
                
                class_id = int(boxes.cls[i].item())
                confidence = float(boxes.conf[i].item())
                class_name = self.model.names[class_id]
                
                detections.append(Detection(
                    bbox=bbox,
                    class_id=class_id,
                    class_name=class_name,
                    confidence=confidence,
                ))
        
        return detections

    def detect_with_embedding_regions(
        self,
        image: Image.Image,
        confidence_threshold: float = 0.25,
        padding_ratio: float = 0.1,
    ) -> list[Detection]:
        """
        Detect objects and expand bounding boxes slightly for better embedding extraction.

        Args:
            image: PIL Image object
            confidence_threshold: Minimum confidence for detections
            padding_ratio: Ratio to expand bbox on each side

        Returns:
            List of Detection objects with padded bboxes
        """
        detections = self.detect(image, confidence_threshold)
        img_w, img_h = image.size
        
        padded_detections = []
        for det in detections:
            x, y, w, h = det.bbox
            
            # Add padding
            pad_x = w * padding_ratio
            pad_y = h * padding_ratio
            
            new_x = max(0, x - pad_x)
            new_y = max(0, y - pad_y)
            new_w = min(img_w - new_x, w + 2 * pad_x)
            new_h = min(img_h - new_y, h + 2 * pad_y)
            
            padded_detections.append(Detection(
                bbox=(new_x, new_y, new_w, new_h),
                class_id=det.class_id,
                class_name=det.class_name,
                confidence=det.confidence,
            ))
        
        return padded_detections

    def get_class_names(self) -> dict[int, str]:
        """Get mapping of class IDs to class names."""
        return self.model.names

    def find_class_id(self, class_name: str) -> int | None:
        """Find class ID by name (case-insensitive partial match)."""
        class_name_lower = class_name.lower()
        for class_id, name in self.model.names.items():
            if class_name_lower in name.lower():
                return class_id
        return None


@lru_cache(maxsize=1)
def get_yolo_model() -> YOLOModel:
    """Get singleton instance of the YOLO model."""
    return YOLOModel()
