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
        Coarse grid search with optimized stride to limit candidates.
        Returns top candidates with their similarities.
        """
        img_w, img_h = target_image.size
        src_w, src_h = source_bbox[2], source_bbox[3]
        
        # Use source size as window
        win_w, win_h = int(src_w), int(src_h)
        
        # Ensure minimum window size
        win_w = max(40, win_w)
        win_h = max(40, win_h)
        
        # Calculate stride to limit total candidates
        # Balanced limit for good performance vs accuracy
        max_candidates = 150  # Balanced for high-end GPU (was 300, too many)
        estimated_candidates_x = max(1, (img_w - win_w) // win_w + 1)
        estimated_candidates_y = max(1, (img_h - win_h) // win_h + 1)
        total_estimated = estimated_candidates_x * estimated_candidates_y
        
        if total_estimated > max_candidates:
            # Increase stride to reduce candidates
            stride_multiplier = (total_estimated / max_candidates) ** 0.5
            stride_x = max(win_w, int(win_w * stride_multiplier))
            stride_y = max(win_h, int(win_h * stride_multiplier))
        else:
            stride_x = win_w
            stride_y = win_h
        
        candidates = []
        for y in range(0, img_h - win_h + 1, stride_y):
            for x in range(0, img_w - win_w + 1, stride_x):
                candidates.append(Candidate(
                    bbox=(float(x), float(y), float(win_w), float(win_h)),
                    source="coarse",
                ))
        
        if not candidates:
            return []
        
        # Batch compute similarities (keep on GPU for speed)
        bboxes = [c.bbox for c in candidates]
        embeddings = self.clip.get_region_embeddings_batch(target_image, bboxes, return_cpu=False)
        
        # Ensure source_embedding is on same device and properly shaped
        if source_embedding.device != embeddings.device:
            source_embedding = source_embedding.to(embeddings.device)
        
        # Ensure source_embedding is 1D for proper matrix multiplication
        # embeddings: [N, embed_dim], source_embedding: [embed_dim] -> result: [N]
        if source_embedding.dim() > 1:
            source_embedding = source_embedding.squeeze()
        
        # Compute similarities on GPU (cosine similarity for normalized embeddings)
        similarities = torch.matmul(embeddings, source_embedding)
        
        # Ensure similarities is 1D
        if similarities.dim() > 1:
            similarities = similarities.squeeze()
        
        # Sort on GPU for better performance, then move top results to CPU
        # Get top-k indices on GPU (faster than sorting all on CPU)
        k = min(50, len(candidates))  # Keep top 50 for further processing
        top_k_values, top_k_indices = torch.topk(similarities, k, dim=0)
        
        # Move only top-k to CPU
        top_k_values_list = top_k_values.cpu().tolist()
        top_k_indices_list = top_k_indices.cpu().tolist()
        if not isinstance(top_k_values_list, list):
            top_k_values_list = [top_k_values_list]
        if not isinstance(top_k_indices_list, list):
            top_k_indices_list = [top_k_indices_list]
        
        # Return top candidates with similarities, sorted by similarity
        results = [(sim, candidates[idx]) for sim, idx in zip(top_k_values_list, top_k_indices_list)]
        results.sort(key=lambda x: x[0], reverse=True)
        return results

    def _fine_search(
        self,
        target_image: Image.Image,
        source_embedding: torch.Tensor,
        source_bbox: tuple[float, float, float, float],
        center: tuple[float, float],
    ) -> list[tuple[float, Candidate]]:
        """
        Fine search around a center point at multiple scales.
        Optimized to limit candidate count.
        """
        img_w, img_h = target_image.size
        src_w, src_h = source_bbox[2], source_bbox[3]
        cx, cy = center
        
        candidates = []
        # Balanced scales for good performance vs accuracy
        scales = [0.8, 1.0, 1.2]  # 3 scales for good balance
        
        for scale in scales:
            win_w = int(src_w * scale)
            win_h = int(src_h * scale)
            
            if win_w < 20 or win_h < 20:
                continue
            
            # Balanced search range for performance
            search_range = int(max(win_w, win_h) * 0.5)  # Balanced range
            # Balanced stride for performance
            stride = max(2, int(min(win_w, win_h) * 0.3))  # Balanced stride
            
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
        
        # Batch compute similarities (keep on GPU for speed)
        bboxes = [c.bbox for c in candidates]
        embeddings = self.clip.get_region_embeddings_batch(target_image, bboxes, return_cpu=False)
        
        # Ensure source_embedding is on same device and properly shaped
        if source_embedding.device != embeddings.device:
            source_embedding = source_embedding.to(embeddings.device)
        
        # Ensure source_embedding is 1D for proper matrix multiplication
        # embeddings: [N, embed_dim], source_embedding: [embed_dim] -> result: [N]
        if source_embedding.dim() > 1:
            source_embedding = source_embedding.squeeze()
        
        # Compute similarities on GPU (cosine similarity for normalized embeddings)
        similarities = torch.matmul(embeddings, source_embedding)
        
        # Ensure similarities is 1D
        if similarities.dim() > 1:
            similarities = similarities.squeeze()
        
        # Move to CPU only for final result
        similarities_list = similarities.cpu().tolist()
        if not isinstance(similarities_list, list):
            similarities_list = [similarities_list]
        
        return list(zip(similarities_list, candidates))

    def _validate_match(
        self,
        target_image: Image.Image,
        candidate_bbox: tuple[float, float, float, float],
        source_bbox: tuple[float, float, float, float],
        similarity: float,
        edge_margin: float = 10.0,
        size_ratio_tolerance: float = 0.5,
    ) -> bool:
        """
        Validate that a candidate match is reasonable.
        
        Args:
            target_image: Target image
            candidate_bbox: Candidate bounding box (x, y, width, height)
            source_bbox: Source bounding box (x, y, width, height)
            similarity: Similarity score
            edge_margin: Minimum distance from image edges (pixels)
            size_ratio_tolerance: Maximum allowed size difference ratio
            
        Returns:
            True if match is valid, False otherwise
        """
        img_w, img_h = target_image.size
        x, y, w, h = candidate_bbox
        src_w, src_h = source_bbox[2], source_bbox[3]
        
        # Check if bbox is at image edges (likely false positive)
        if x < edge_margin or y < edge_margin:
            return False
        if x + w > img_w - edge_margin or y + h > img_h - edge_margin:
            return False
        
        # Check if bbox size is reasonable compared to source
        # Allow some variation but not too much
        width_ratio = w / src_w if src_w > 0 else 1.0
        height_ratio = h / src_h if src_h > 0 else 1.0
        
        # Size should be within reasonable bounds (0.3x to 3x)
        if width_ratio < 0.3 or width_ratio > 3.0:
            return False
        if height_ratio < 0.3 or height_ratio > 3.0:
            return False
        
        # Check if bbox has reasonable dimensions (not too small)
        if w < 20 or h < 20:
            return False
        
        # For matches from grid search (not YOLO), require higher confidence
        # Grid search can find background regions that look similar
        # We'll check this in the caller by looking at the candidate source
        
        return True

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
        # Get CLIP embedding of the tagged product (keep on GPU)
        source_embedding = self.clip.get_region_embedding(source_image, tag.bbox, return_cpu=False)
        
        # Get YOLO detections
        if yolo_detections is None:
            yolo_detections = self.yolo.detect_with_embedding_regions(
                target_image,
                confidence_threshold=self.detection_confidence,
            )
        
        all_results: list[tuple[float, Candidate]] = []
        
        # Add YOLO detections (keep on GPU for speed)
        if yolo_detections:
            yolo_bboxes = [det.bbox for det in yolo_detections]
            yolo_embeddings = self.clip.get_region_embeddings_batch(target_image, yolo_bboxes, return_cpu=False)
            
            # Ensure source_embedding is on same device and properly shaped
            if source_embedding.device != yolo_embeddings.device:
                source_embedding = source_embedding.to(yolo_embeddings.device)
            
            # Ensure source_embedding is 1D for proper matrix multiplication
            # yolo_embeddings: [N, embed_dim], source_embedding: [embed_dim] -> result: [N]
            if source_embedding.dim() > 1:
                source_embedding = source_embedding.squeeze()
            
            # Compute similarities on GPU (cosine similarity for normalized embeddings)
            yolo_sims = torch.matmul(yolo_embeddings, source_embedding)
            
            # Ensure yolo_sims is 1D
            if yolo_sims.dim() > 1:
                yolo_sims = yolo_sims.squeeze()
            
            # Move to CPU only for final result
            yolo_sims_list = yolo_sims.cpu().tolist()
            if not isinstance(yolo_sims_list, list):
                yolo_sims_list = [yolo_sims_list]
            
            for det, sim in zip(yolo_detections, yolo_sims_list):
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
        
        # Sort by similarity, get top candidates for fine search
        all_results.sort(key=lambda x: x[0], reverse=True)
        best_coarse_sim = all_results[0][0] if all_results else 0.0
        
        # Early exit if we already have a very good match (above threshold + margin)
        # This avoids expensive fine search when not needed
        if best_coarse_sim >= self.similarity_threshold + 0.1:
            best_sim, best_candidate = all_results[0]
        else:
            # Do fine search on top 2 candidates (balanced for performance)
            top_candidates = all_results[:2]
            
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

        # Validate the match before accepting it
        is_valid_match = self._validate_match(
            target_image, best_candidate.bbox, tag.bbox, best_sim
        )
        
        # For grid search candidates (not YOLO detections), require higher confidence
        # Grid search can find background regions that look similar but aren't the object
        if best_candidate.source != "yolo" and best_sim < self.similarity_threshold + 0.15:
            # Grid search matches need higher confidence to avoid false positives
            is_valid_match = False
            logger.info(
                f"Grid search match requires higher confidence: "
                f"similarity={best_sim:.3f}, required={self.similarity_threshold + 0.15:.3f}"
            )
        
        found = best_sim >= self.similarity_threshold and is_valid_match
        
        if not is_valid_match and best_sim >= self.similarity_threshold:
            logger.warning(
                f"Match rejected by validation: similarity={best_sim:.3f}, "
                f"bbox={best_candidate.bbox}, source_bbox={tag.bbox}, "
                f"source={best_candidate.source}"
            )

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
    ) -> tuple[list[MatchResult], int]:
        """
        Find all tagged products from source image in target image.
        
        Returns:
            Tuple of (results, detections_count)
        """
        # Pre-compute YOLO detections once
        yolo_detections = self.yolo.detect_with_embedding_regions(
            target_image,
            confidence_threshold=self.detection_confidence,
        )
        detections_count = len(yolo_detections)
        
        results = []
        for tag in tags:
            result = self.find_product_in_image(
                source_image, tag, target_image, yolo_detections
            )
            results.append(result)
        
        return results, detections_count

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
