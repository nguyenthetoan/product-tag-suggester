"""
YOLO-based product matching service.
Uses YOLO for object detection combined with CLIP embeddings for product verification.
This approach is faster and more accurate than pure sliding-window search.
"""

import logging
import torch
from PIL import Image
from dataclasses import dataclass

from app.models.yolo import YOLOModel, Detection
from app.models.embeddings import EmbeddingModel

logger = logging.getLogger(__name__)


@dataclass
class YOLOMatchResult:
    """Result of matching a product using YOLO detection."""

    product_id: str
    found: bool
    confidence: float
    suggested_bbox: tuple[float, float, float, float] | None = None
    detected_class: str | None = None
    detection_confidence: float | None = None


@dataclass
class TagInfo:
    """Information about a tagged product region."""

    product_id: str
    bbox: tuple[float, float, float, float]  # (x, y, width, height)


class YOLOProductMatcher:
    """
    Product matcher that uses YOLO for detection and CLIP for verification.
    
    Strategy:
    1. Detect objects in target image using YOLO
    2. For each tagged product in source, get its CLIP embedding
    3. Compare source product embedding with detected objects in target
    4. Return the best matching detection that exceeds threshold
    """

    def __init__(
        self,
        yolo_model: YOLOModel,
        embedding_model: EmbeddingModel,
        similarity_threshold: float = 0.70,
        detection_confidence: float = 0.25,
    ):
        """
        Initialize the YOLO-based product matcher.

        Args:
            yolo_model: YOLO model for object detection
            embedding_model: CLIP model for embedding extraction
            similarity_threshold: Minimum CLIP similarity for a match
            detection_confidence: Minimum YOLO detection confidence
        """
        self.yolo = yolo_model
        self.clip = embedding_model
        self.similarity_threshold = similarity_threshold
        self.detection_confidence = detection_confidence

    def find_product_in_image(
        self,
        source_image: Image.Image,
        tag: TagInfo,
        target_image: Image.Image,
        target_detections: list[Detection] | None = None,
    ) -> YOLOMatchResult:
        """
        Find a tagged product from source image in target image using YOLO + CLIP.

        Args:
            source_image: Image containing the tagged product
            tag: Tag information with product ID and bounding box
            target_image: Image to search for the product
            target_detections: Pre-computed detections (for batch efficiency)

        Returns:
            YOLOMatchResult with match details
        """
        # Get embedding of the tagged product region
        source_embedding = self.clip.get_region_embedding(source_image, tag.bbox)
        
        # Detect objects in target if not provided
        if target_detections is None:
            target_detections = self.yolo.detect_with_embedding_regions(
                target_image, 
                confidence_threshold=self.detection_confidence,
            )
        
        logger.info(
            f"Searching for product {tag.product_id}: "
            f"source_bbox={tag.bbox}, detections_count={len(target_detections)}"
        )

        if not target_detections:
            logger.info(f"No objects detected in target image")
            return YOLOMatchResult(
                product_id=tag.product_id,
                found=False,
                confidence=0.0,
            )

        # Get embeddings for all detected regions in batch
        detection_bboxes = [det.bbox for det in target_detections]
        detection_embeddings = self.clip.get_region_embeddings_batch(
            target_image, detection_bboxes
        )

        # Compute similarities
        similarities = torch.matmul(detection_embeddings, source_embedding)
        
        # Find best match
        best_idx = torch.argmax(similarities).item()
        best_similarity = float(similarities[best_idx])
        best_detection = target_detections[best_idx]
        
        logger.info(
            f"Best match: class={best_detection.class_name}, "
            f"similarity={best_similarity:.3f}, "
            f"detection_conf={best_detection.confidence:.3f}, "
            f"bbox={best_detection.bbox}"
        )

        found = best_similarity >= self.similarity_threshold

        return YOLOMatchResult(
            product_id=tag.product_id,
            found=found,
            confidence=round(best_similarity, 3),
            suggested_bbox=best_detection.bbox if found else None,
            detected_class=best_detection.class_name if found else None,
            detection_confidence=round(best_detection.confidence, 3) if found else None,
        )

    def find_all_products(
        self,
        source_image: Image.Image,
        tags: list[TagInfo],
        target_image: Image.Image,
    ) -> list[YOLOMatchResult]:
        """
        Find all tagged products from source image in target image.

        Args:
            source_image: Image containing tagged products
            tags: List of tag information
            target_image: Image to search for products

        Returns:
            List of YOLOMatchResult for each tag
        """
        # Pre-compute detections once for efficiency
        target_detections = self.yolo.detect_with_embedding_regions(
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
    Simpler YOLO-only matcher that just uses object detection.
    
    For cases where you want to detect specific object types without 
    embedding comparison. Useful when you know the object category.
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
