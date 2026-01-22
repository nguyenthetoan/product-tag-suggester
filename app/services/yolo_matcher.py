"""
Hybrid YOLO + CLIP product matching service.
Uses YOLO for candidate detection + grid search fallback + CLIP for matching.
"""

import logging
import torch
from PIL import Image
from dataclasses import dataclass

from app.models.yolo import YOLOModel, Detection
from app.models.embeddings import EmbeddingModel

logger = logging.getLogger(__name__)


@dataclass
class MatchResult:
    """Result of matching a product."""

    product_id: str
    found: bool
    confidence: float  # CLIP similarity score
    suggested_bbox: tuple[float, float, float, float] | None = None
    detected_class: str | None = None  # YOLO class if available


@dataclass 
class Candidate:
    """A candidate region for matching."""
    
    bbox: tuple[float, float, float, float]
    source: str  # "yolo" or "grid"
    class_name: str | None = None


@dataclass
class TagInfo:
    """Information about a tagged product region."""

    product_id: str
    bbox: tuple[float, float, float, float]  # (x, y, width, height)


class ProductMatcher:
    """
    Product matcher using YOLO + coarse-to-fine grid search + CLIP.
    
    Strategy:
    1. Use YOLO to detect candidate regions (fast, but limited to known classes)
    2. Coarse grid scan at single scale with large stride
    3. Fine search around top candidates
    4. Return best match above similarity threshold
    """

    def __init__(
        self,
        yolo_model: YOLOModel,
        embedding_model: EmbeddingModel,
        similarity_threshold: float = 0.75,
        detection_confidence: float = 0.10,
    ):
        """
        Initialize the product matcher.

        Args:
            yolo_model: YOLO model for candidate detection
            embedding_model: CLIP model for embedding matching
            similarity_threshold: Minimum CLIP similarity for a match
            detection_confidence: Minimum YOLO detection confidence
        """
        self.yolo = yolo_model
        self.clip = embedding_model
        self.similarity_threshold = similarity_threshold
        self.detection_confidence = detection_confidence

    def _coarse_grid_search(
        self,
        target_image: Image.Image,
        source_embedding: torch.Tensor,
        source_bbox: tuple[float, float, float, float],
    ) -> list[tuple[float, Candidate]]:
        """
        Coarse grid search with large stride.
        Returns top candidates with their similarities.
        """
        img_w, img_h = target_image.size
        src_w, src_h = source_bbox[2], source_bbox[3]
        
        # Use source size as window, with 1.0 stride (no overlap)
        win_w, win_h = int(src_w), int(src_h)
        stride = max(win_w, win_h)  # Large stride for speed
        
        # Ensure minimum window size
        win_w = max(40, win_w)
        win_h = max(40, win_h)
        
        candidates = []
        for y in range(0, img_h - win_h + 1, stride):
            for x in range(0, img_w - win_w + 1, stride):
                candidates.append(Candidate(
                    bbox=(float(x), float(y), float(win_w), float(win_h)),
                    source="coarse",
                ))
        
        if not candidates:
            return []
        
        # Batch compute similarities
        bboxes = [c.bbox for c in candidates]
        embeddings = self.clip.get_region_embeddings_batch(target_image, bboxes)
        similarities = torch.matmul(embeddings, source_embedding).tolist()
        
        # Return candidates with similarities
        return list(zip(similarities, candidates))

    def _fine_search(
        self,
        target_image: Image.Image,
        source_embedding: torch.Tensor,
        source_bbox: tuple[float, float, float, float],
        center: tuple[float, float],
    ) -> list[tuple[float, Candidate]]:
        """
        Fine search around a center point at multiple scales.
        """
        img_w, img_h = target_image.size
        src_w, src_h = source_bbox[2], source_bbox[3]
        cx, cy = center
        
        candidates = []
        scales = [0.6, 0.8, 1.0, 1.2, 1.5]
        
        for scale in scales:
            win_w = int(src_w * scale)
            win_h = int(src_h * scale)
            
            if win_w < 20 or win_h < 20:
                continue
            
            # Search in a small region around center
            search_range = max(win_w, win_h)
            stride = max(1, int(min(win_w, win_h) * 0.25))  # Fine stride
            
            for dy in range(-search_range, search_range + 1, stride):
                for dx in range(-search_range, search_range + 1, stride):
                    x = int(cx + dx - win_w / 2)
                    y = int(cy + dy - win_h / 2)
                    
                    # Bounds check
                    if x < 0 or y < 0 or x + win_w > img_w or y + win_h > img_h:
                        continue
                    
                    candidates.append(Candidate(
                        bbox=(float(x), float(y), float(win_w), float(win_h)),
                        source="fine",
                    ))
        
        if not candidates:
            return []
        
        # Batch compute similarities
        bboxes = [c.bbox for c in candidates]
        embeddings = self.clip.get_region_embeddings_batch(target_image, bboxes)
        similarities = torch.matmul(embeddings, source_embedding).tolist()
        
        return list(zip(similarities, candidates))

    def find_product_in_image(
        self,
        source_image: Image.Image,
        tag: TagInfo,
        target_image: Image.Image,
        yolo_detections: list[Detection] | None = None,
    ) -> MatchResult:
        """
        Find a tagged product using coarse-to-fine search.
        """
        # Get CLIP embedding of the tagged product
        source_embedding = self.clip.get_region_embedding(source_image, tag.bbox)
        
        # Get YOLO detections
        if yolo_detections is None:
            yolo_detections = self.yolo.detect_with_embedding_regions(
                target_image,
                confidence_threshold=self.detection_confidence,
            )
        
        all_results: list[tuple[float, Candidate]] = []
        
        # Add YOLO detections
        if yolo_detections:
            yolo_bboxes = [det.bbox for det in yolo_detections]
            yolo_embeddings = self.clip.get_region_embeddings_batch(target_image, yolo_bboxes)
            yolo_sims = torch.matmul(yolo_embeddings, source_embedding).tolist()
            
            for det, sim in zip(yolo_detections, yolo_sims):
                all_results.append((sim, Candidate(
                    bbox=det.bbox,
                    source="yolo",
                    class_name=det.class_name,
                )))
        
        # Coarse grid search
        coarse_results = self._coarse_grid_search(
            target_image, source_embedding, tag.bbox
        )
        all_results.extend(coarse_results)
        
        logger.info(f"Coarse search: YOLO={len(yolo_detections)}, grid={len(coarse_results)}")
        
        if not all_results:
            return MatchResult(
                product_id=tag.product_id,
                found=False,
                confidence=0.0,
            )
        
        # Sort by similarity, get top 3 for fine search
        all_results.sort(key=lambda x: x[0], reverse=True)
        top_candidates = all_results[:3]
        
        # Fine search around top candidates
        fine_results = []
        for sim, candidate in top_candidates:
            cx = candidate.bbox[0] + candidate.bbox[2] / 2
            cy = candidate.bbox[1] + candidate.bbox[3] / 2
            fine = self._fine_search(target_image, source_embedding, tag.bbox, (cx, cy))
            fine_results.extend(fine)
        
        # Combine all results
        all_results.extend(fine_results)
        
        # Find best match
        best_sim, best_candidate = max(all_results, key=lambda x: x[0])
        
        logger.info(
            f"Best match: source={best_candidate.source}, "
            f"class={best_candidate.class_name}, "
            f"similarity={best_sim:.3f}, bbox={best_candidate.bbox}"
        )

        found = best_sim >= self.similarity_threshold

        return MatchResult(
            product_id=tag.product_id,
            found=found,
            confidence=round(best_sim, 3),
            suggested_bbox=best_candidate.bbox if found else None,
            detected_class=best_candidate.class_name if found else None,
        )

    def find_all_products(
        self,
        source_image: Image.Image,
        tags: list[TagInfo],
        target_image: Image.Image,
    ) -> list[MatchResult]:
        """
        Find all tagged products from source image in target image.
        """
        # Pre-compute YOLO detections once
        yolo_detections = self.yolo.detect_with_embedding_regions(
            target_image,
            confidence_threshold=self.detection_confidence,
        )
        
        results = []
        for tag in tags:
            result = self.find_product_in_image(
                source_image, tag, target_image, yolo_detections
            )
            results.append(result)
        
        return results

    def detect_objects(self, image: Image.Image) -> list[Detection]:
        """Detect all objects in an image using YOLO."""
        return self.yolo.detect(image, confidence_threshold=self.detection_confidence)


class DirectDetector:
    """Simple YOLO detector for direct object detection."""

    def __init__(
        self,
        yolo_model: YOLOModel,
        detection_confidence: float = 0.25,
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
        """
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

        return self.yolo.detect(
            image,
            confidence_threshold=self.detection_confidence,
            classes=class_ids,
        )
