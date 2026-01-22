"""
YOLO-based product matching service.
Uses YOLO for object detection and class-based matching.
"""

import logging
from PIL import Image
from dataclasses import dataclass

from app.models.yolo import YOLOModel, Detection

logger = logging.getLogger(__name__)


@dataclass
class MatchResult:
    """Result of matching a product using YOLO detection."""

    product_id: str
    found: bool
    confidence: float
    suggested_bbox: tuple[float, float, float, float] | None = None
    detected_class: str | None = None


@dataclass
class TagInfo:
    """Information about a tagged product region."""

    product_id: str
    bbox: tuple[float, float, float, float]  # (x, y, width, height)


def bbox_iou(box1: tuple, box2: tuple) -> float:
    """Calculate Intersection over Union between two bboxes (x, y, w, h format)."""
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    
    # Convert to x1, y1, x2, y2 format
    box1_x2, box1_y2 = x1 + w1, y1 + h1
    box2_x2, box2_y2 = x2 + w2, y2 + h2
    
    # Calculate intersection
    inter_x1 = max(x1, x2)
    inter_y1 = max(y1, y2)
    inter_x2 = min(box1_x2, box2_x2)
    inter_y2 = min(box1_y2, box2_y2)
    
    if inter_x2 <= inter_x1 or inter_y2 <= inter_y1:
        return 0.0
    
    inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
    
    # Calculate union
    box1_area = w1 * h1
    box2_area = w2 * h2
    union_area = box1_area + box2_area - inter_area
    
    return inter_area / union_area if union_area > 0 else 0.0


class YOLOProductMatcher:
    """
    Product matcher that uses YOLO for detection and class-based matching.
    
    Strategy:
    1. Detect objects in source image to identify the class of each tagged product
    2. Detect objects in target image
    3. Match by object class - find the same class in target
    4. If multiple matches, use position/size heuristics
    """

    def __init__(
        self,
        yolo_model: YOLOModel,
        detection_confidence: float = 0.25,
        iou_threshold: float = 0.3,
    ):
        """
        Initialize the YOLO-based product matcher.

        Args:
            yolo_model: YOLO model for object detection
            detection_confidence: Minimum YOLO detection confidence
            iou_threshold: Minimum IoU to consider a detection as covering the tag
        """
        self.yolo = yolo_model
        self.detection_confidence = detection_confidence
        self.iou_threshold = iou_threshold

    def _find_class_for_tag(
        self,
        image: Image.Image,
        tag_bbox: tuple[float, float, float, float],
    ) -> Detection | None:
        """Find the YOLO detection that best covers the tagged region."""
        detections = self.yolo.detect(image, confidence_threshold=self.detection_confidence)
        
        best_detection = None
        best_iou = 0.0
        
        for det in detections:
            iou = bbox_iou(tag_bbox, det.bbox)
            if iou > best_iou and iou >= self.iou_threshold:
                best_iou = iou
                best_detection = det
        
        return best_detection

    def find_product_in_image(
        self,
        source_image: Image.Image,
        tag: TagInfo,
        target_image: Image.Image,
        target_detections: list[Detection] | None = None,
    ) -> MatchResult:
        """
        Find a tagged product from source image in target image using YOLO.

        Args:
            source_image: Image containing the tagged product
            tag: Tag information with product ID and bounding box
            target_image: Image to search for the product
            target_detections: Pre-computed detections (for batch efficiency)

        Returns:
            MatchResult with match details
        """
        # Step 1: Find what class the tagged product is
        source_detection = self._find_class_for_tag(source_image, tag.bbox)
        
        if source_detection is None:
            logger.info(
                f"Product {tag.product_id}: No YOLO detection covers the tagged region"
            )
            return MatchResult(
                product_id=tag.product_id,
                found=False,
                confidence=0.0,
            )
        
        logger.info(
            f"Product {tag.product_id}: Tagged region detected as '{source_detection.class_name}' "
            f"(conf={source_detection.confidence:.3f})"
        )
        
        # Step 2: Detect objects in target if not provided
        if target_detections is None:
            target_detections = self.yolo.detect(
                target_image, 
                confidence_threshold=self.detection_confidence,
            )
        
        # Step 3: Find matching class in target
        matching_detections = [
            det for det in target_detections 
            if det.class_name == source_detection.class_name
        ]
        
        if not matching_detections:
            logger.info(
                f"Product {tag.product_id}: No '{source_detection.class_name}' found in target"
            )
            return MatchResult(
                product_id=tag.product_id,
                found=False,
                confidence=0.0,
                detected_class=source_detection.class_name,
            )
        
        # Step 4: Pick the best match (highest confidence)
        best_match = max(matching_detections, key=lambda d: d.confidence)
        
        logger.info(
            f"Product {tag.product_id}: Found '{best_match.class_name}' in target "
            f"(conf={best_match.confidence:.3f}, bbox={best_match.bbox})"
        )
        
        return MatchResult(
            product_id=tag.product_id,
            found=True,
            confidence=round(best_match.confidence, 3),
            suggested_bbox=best_match.bbox,
            detected_class=best_match.class_name,
        )

    def find_all_products(
        self,
        source_image: Image.Image,
        tags: list[TagInfo],
        target_image: Image.Image,
    ) -> list[MatchResult]:
        """
        Find all tagged products from source image in target image.

        Args:
            source_image: Image containing tagged products
            tags: List of tag information
            target_image: Image to search for products

        Returns:
            List of MatchResult for each tag
        """
        # Pre-compute target detections once for efficiency
        target_detections = self.yolo.detect(
            target_image,
            confidence_threshold=self.detection_confidence,
        )
        
        logger.info(f"Detected {len(target_detections)} objects in target image")
        for det in target_detections:
            logger.debug(f"  - {det.class_name}: conf={det.confidence:.3f}, bbox={det.bbox}")

        results = []
        for tag in tags:
            result = self.find_product_in_image(
                source_image, tag, target_image, target_detections
            )
            results.append(result)
        
        return results

    def detect_objects(
        self,
        image: Image.Image,
    ) -> list[Detection]:
        """
        Detect all objects in an image.

        Args:
            image: PIL Image object

        Returns:
            List of Detection objects
        """
        return self.yolo.detect(image, confidence_threshold=self.detection_confidence)


class YOLODirectMatcher:
    """
    Simple YOLO detector for direct object detection.
    """

    def __init__(
        self,
        yolo_model: YOLOModel,
        detection_confidence: float = 0.3,
    ):
        self.yolo = yolo_model
        self.detection_confidence = detection_confidence

    def detect_and_classify(
        self,
        image: Image.Image,
        target_classes: list[str] | None = None,
    ) -> list[Detection]:
        """
        Detect objects, optionally filtering by class names.

        Args:
            image: PIL Image object
            target_classes: Optional list of class names to filter

        Returns:
            List of matching Detection objects
        """
        # Convert class names to IDs if provided
        class_ids = None
        if target_classes:
            class_ids = []
            for name in target_classes:
                class_id = self.yolo.find_class_id(name)
                if class_id is not None:
                    class_ids.append(class_id)
            
            if not class_ids:
                logger.warning(f"No matching class IDs found for: {target_classes}")
                return []

        detections = self.yolo.detect(
            image,
            confidence_threshold=self.detection_confidence,
            classes=class_ids,
        )
        
        return detections
