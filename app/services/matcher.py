"""
Product matching service using visual embeddings.
Implements coarse-to-fine detection for finding tagged products in new images.
"""

import logging
import torch
from PIL import Image
from dataclasses import dataclass

from app.models.embeddings import EmbeddingModel

logger = logging.getLogger(__name__)


@dataclass
class MatchResult:
    """Result of matching a product tag in a target image."""

    product_id: str
    found: bool
    confidence: float
    suggested_bbox: tuple[float, float, float, float] | None = None


@dataclass
class TagInfo:
    """Information about a tagged product region."""

    product_id: str
    bbox: tuple[float, float, float, float]  # (x, y, width, height)


class ProductMatcher:
    """Service for matching tagged products across images."""

    def __init__(
        self,
        embedding_model: EmbeddingModel,
        similarity_threshold: float = 0.75,
        window_scales: list[float] | None = None,
        coarse_stride_ratio: float = 0.5,
        fine_stride_ratio: float = 0.25,
    ):
        """
        Initialize the product matcher.

        Args:
            embedding_model: Model for extracting image embeddings
            similarity_threshold: Minimum similarity to consider a match (0-1)
            window_scales: Scales for sliding window relative to original tag size
            coarse_stride_ratio: Stride for coarse search (faster)
            fine_stride_ratio: Stride for fine search around candidates
        """
        self.model = embedding_model
        self.threshold = similarity_threshold
        # More scales to handle different product sizes in target image
        # With 80px source: 40, 56, 80, 112, 160, 224px windows
        self.window_scales = window_scales or [0.5, 0.7, 1.0, 1.4, 2.0, 2.8]
        self.coarse_stride_ratio = coarse_stride_ratio
        self.fine_stride_ratio = fine_stride_ratio

    def compute_similarity(
        self,
        embedding1: torch.Tensor,
        embedding2: torch.Tensor,
    ) -> float:
        """Compute cosine similarity between two embeddings."""
        return float(torch.dot(embedding1, embedding2))

    def find_product_in_image(
        self,
        source_image: Image.Image,
        tag: TagInfo,
        target_image: Image.Image,
    ) -> MatchResult:
        """
        Find a tagged product from source image in target image.

        Uses coarse-to-fine search: first quick scan, then refine around best candidates.

        Args:
            source_image: Image containing the tagged product
            tag: Tag information with product ID and bounding box
            target_image: Image to search for the product

        Returns:
            MatchResult with found status, confidence, and suggested bbox if found
        """
        # Get embedding of the tagged product region
        source_embedding = self.model.get_region_embedding(source_image, tag.bbox)

        target_w, target_h = target_image.size
        tag_w, tag_h = tag.bbox[2], tag.bbox[3]

        logger.info(
            f"Searching for product {tag.product_id}: "
            f"source_bbox={tag.bbox}, target_size={target_w}x{target_h}"
        )

        best_similarity = 0.0
        best_bbox = None
        windows_checked = 0
        candidates = []  # Top candidates for fine search

        # PHASE 1: Coarse search with large strides (batched for speed)
        BATCH_SIZE = 32  # Process 32 windows at a time
        
        for scale in self.window_scales:
            window_w = int(tag_w * scale)
            window_h = int(tag_h * scale)

            # Skip if window is larger than target image
            if window_w > target_w or window_h > target_h:
                continue

            # Skip very small windows (less than 40px)
            if window_w < 40 or window_h < 40:
                logger.debug(f"Skipping scale {scale}: window {window_w}x{window_h} too small")
                continue

            stride_x = max(1, int(window_w * self.coarse_stride_ratio))
            stride_y = max(1, int(window_h * self.coarse_stride_ratio))

            # Collect all bboxes for this scale
            bboxes = []
            for y in range(0, target_h - window_h + 1, stride_y):
                for x in range(0, target_w - window_w + 1, stride_x):
                    bboxes.append((x, y, window_w, window_h))

            # Track best for this scale
            scale_best_sim = 0.0
            scale_best_bbox = None

            # Process in batches
            for i in range(0, len(bboxes), BATCH_SIZE):
                batch_bboxes = bboxes[i:i + BATCH_SIZE]
                batch_embeddings = self.model.get_region_embeddings_batch(
                    target_image, batch_bboxes
                )
                windows_checked += len(batch_bboxes)

                # Compute similarities for batch
                similarities = torch.matmul(batch_embeddings, source_embedding)
                
                for j, (bbox, sim) in enumerate(zip(batch_bboxes, similarities)):
                    similarity = float(sim)
                    
                    # Track best for this scale
                    if similarity > scale_best_sim:
                        scale_best_sim = similarity
                        scale_best_bbox = bbox
                    
                    # Track top candidates for refinement
                    if similarity > self.threshold * 0.8:
                        candidates.append((similarity, bbox, scale))

                    if similarity > best_similarity:
                        best_similarity = similarity
                        best_bbox = bbox

                    # Early exit if very high confidence
                    if similarity > 0.92:
                        logger.info(f"Early exit: high confidence {similarity:.3f}")
                        return MatchResult(
                            product_id=tag.product_id,
                            found=True,
                            confidence=round(similarity, 3),
                            suggested_bbox=bbox,
                        )
            
            # Log best match for this scale
            if scale_best_bbox:
                center_x = (scale_best_bbox[0] + scale_best_bbox[2]/2) / target_w * 100
                center_y = (scale_best_bbox[1] + scale_best_bbox[3]/2) / target_h * 100
                logger.info(
                    f"Scale {scale}: window={window_w}x{window_h}, "
                    f"best_sim={scale_best_sim:.3f}, pos=({center_x:.1f}%, {center_y:.1f}%)"
                )

        # PHASE 2: Fine search around top candidates (batched)
        candidates.sort(reverse=True, key=lambda x: x[0])
        top_candidates = candidates[:5]  # Refine top 5
        
        # Log top candidates
        logger.info(f"Top {len(top_candidates)} candidates:")
        for i, (sim, (cx, cy, cw, ch), scale) in enumerate(top_candidates):
            center_x = (cx + cw/2) / target_w * 100
            center_y = (cy + ch/2) / target_h * 100
            logger.info(f"  {i+1}. sim={sim:.3f}, pos=({center_x:.1f}%, {center_y:.1f}%), scale={scale}")

        for _, (cx, cy, cw, ch), scale in top_candidates:
            window_w = int(tag_w * scale)
            window_h = int(tag_h * scale)
            
            stride_x = max(1, int(window_w * self.fine_stride_ratio))
            stride_y = max(1, int(window_h * self.fine_stride_ratio))

            # Search in neighborhood of candidate
            search_range = int(window_w * 0.5)
            
            # Collect bboxes for fine search
            fine_bboxes = []
            for dy in range(-search_range, search_range + 1, stride_y):
                for dx in range(-search_range, search_range + 1, stride_x):
                    x = cx + dx
                    y = cy + dy
                    
                    # Bounds check
                    if x < 0 or y < 0 or x + window_w > target_w or y + window_h > target_h:
                        continue

                    fine_bboxes.append((x, y, window_w, window_h))

            if fine_bboxes:
                # Process fine search in batch
                fine_embeddings = self.model.get_region_embeddings_batch(
                    target_image, fine_bboxes
                )
                windows_checked += len(fine_bboxes)
                
                similarities = torch.matmul(fine_embeddings, source_embedding)
                
                for bbox, sim in zip(fine_bboxes, similarities):
                    similarity = float(sim)
                    if similarity > best_similarity:
                        best_similarity = similarity
                        best_bbox = bbox

        found = best_similarity >= self.threshold

        logger.info(
            f"Result: found={found}, confidence={best_similarity:.3f}, "
            f"bbox={best_bbox}, windows_checked={windows_checked}"
        )

        return MatchResult(
            product_id=tag.product_id,
            found=found,
            confidence=round(best_similarity, 3),
            suggested_bbox=best_bbox if found else None,
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
        results = []
        for tag in tags:
            result = self.find_product_in_image(source_image, tag, target_image)
            results.append(result)
        return results
