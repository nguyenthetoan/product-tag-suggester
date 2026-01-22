"""
Product matching service using visual embeddings.
Implements sliding window detection for finding tagged products in new images.
"""

import torch
from PIL import Image
from dataclasses import dataclass

from app.models.embeddings import EmbeddingModel


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
        stride_ratio: float = 0.25,
    ):
        """
        Initialize the product matcher.

        Args:
            embedding_model: Model for extracting image embeddings
            similarity_threshold: Minimum similarity to consider a match (0-1)
            window_scales: Scales for sliding window relative to original tag size
            stride_ratio: Stride as ratio of window size (smaller = more thorough but slower)
        """
        self.model = embedding_model
        self.threshold = similarity_threshold
        self.window_scales = window_scales or [0.75, 1.0, 1.25, 1.5]
        self.stride_ratio = stride_ratio

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

        Uses sliding window approach with multiple scales.

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

        best_similarity = 0.0
        best_bbox = None

        # Try multiple window scales
        for scale in self.window_scales:
            window_w = int(tag_w * scale)
            window_h = int(tag_h * scale)

            # Skip if window is larger than target image
            if window_w > target_w or window_h > target_h:
                continue

            stride_x = max(1, int(window_w * self.stride_ratio))
            stride_y = max(1, int(window_h * self.stride_ratio))

            # Slide window across target image
            for y in range(0, target_h - window_h + 1, stride_y):
                for x in range(0, target_w - window_w + 1, stride_x):
                    bbox = (x, y, window_w, window_h)
                    target_embedding = self.model.get_region_embedding(
                        target_image, bbox
                    )

                    similarity = self.compute_similarity(
                        source_embedding, target_embedding
                    )

                    if similarity > best_similarity:
                        best_similarity = similarity
                        best_bbox = bbox

        found = best_similarity >= self.threshold

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
