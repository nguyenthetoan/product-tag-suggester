"""
CLIP-based image embedding extractor.
Uses OpenAI's CLIP model to generate visual embeddings for product matching.
"""

import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from functools import lru_cache


class EmbeddingModel:
    """Wrapper for CLIP model to extract image embeddings."""

    def __init__(self, model_name: str = "openai/clip-vit-base-patch32"):
        """
        Initialize the CLIP model.

        Args:
            model_name: HuggingFace model identifier. Options:
                - "openai/clip-vit-base-patch32" (faster, less accurate)
                - "openai/clip-vit-large-patch14" (slower, more accurate)
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = CLIPModel.from_pretrained(model_name).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.model.eval()

    @torch.no_grad()
    def get_embedding(self, image: Image.Image) -> torch.Tensor:
        """
        Extract embedding vector from an image.

        Args:
            image: PIL Image object

        Returns:
            Normalized embedding tensor of shape (512,) or (768,) depending on model
        """
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        embedding = self.model.get_image_features(**inputs)
        # Normalize for cosine similarity
        embedding = embedding / embedding.norm(dim=-1, keepdim=True)
        return embedding.squeeze(0).cpu()

    @torch.no_grad()
    def get_embeddings_batch(self, images: list[Image.Image]) -> torch.Tensor:
        """
        Extract embeddings from multiple images in a batch (faster than one-by-one).

        Args:
            images: List of PIL Image objects

        Returns:
            Normalized embedding tensor of shape (N, embed_dim)
        """
        if not images:
            return torch.tensor([])
        
        inputs = self.processor(images=images, return_tensors="pt", padding=True).to(self.device)
        embeddings = self.model.get_image_features(**inputs)
        # Normalize for cosine similarity
        embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)
        return embeddings.cpu()

    def get_region_embedding(
        self,
        image: Image.Image,
        bbox: tuple[float, float, float, float],
    ) -> torch.Tensor:
        """
        Extract embedding from a specific region of an image.

        Args:
            image: PIL Image object
            bbox: Bounding box as (x, y, width, height) in pixels

        Returns:
            Normalized embedding tensor
        """
        x, y, w, h = bbox
        cropped = image.crop((x, y, x + w, y + h))
        return self.get_embedding(cropped)

    def get_region_embeddings_batch(
        self,
        image: Image.Image,
        bboxes: list[tuple[float, float, float, float]],
    ) -> torch.Tensor:
        """
        Extract embeddings from multiple regions in a batch.

        Args:
            image: PIL Image object
            bboxes: List of bounding boxes as (x, y, width, height) in pixels

        Returns:
            Normalized embedding tensor of shape (N, embed_dim)
        """
        crops = []
        for x, y, w, h in bboxes:
            cropped = image.crop((x, y, x + w, y + h))
            crops.append(cropped)
        return self.get_embeddings_batch(crops)


@lru_cache(maxsize=1)
def get_embedding_model() -> EmbeddingModel:
    """Get singleton instance of the embedding model."""
    return EmbeddingModel()
