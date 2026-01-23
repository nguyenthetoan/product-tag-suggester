"""
CLIP-based image embedding extractor.
Uses OpenAI's CLIP model to generate visual embeddings for product matching.
Optimized for high-performance GPU systems.
"""

import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from functools import lru_cache

try:
    from app.models.embeddings_onnx import ONNXEmbeddingModel, get_onnx_embedding_model
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False
    print("Warning: ONNX embedding model not available. Falling back to PyTorch CLIP.")


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
        # Use fast processor for better performance
        self.processor = CLIPProcessor.from_pretrained(model_name, use_fast=True)
        self.model.eval()
        
        # Enable mixed precision if CUDA is available and supports it
        self.use_amp = torch.cuda.is_available() and hasattr(torch.cuda, 'amp')
        
        # Note: torch.compile can add overhead on first run and may not always be faster
        # Disabled by default - enable if needed after profiling
        # if hasattr(torch, 'compile') and self.device == "cuda":
        #     try:
        #         self.model = torch.compile(self.model, mode="reduce-overhead")
        #         print("Model compiled with torch.compile for optimal performance")
        #     except Exception as e:
        #         print(f"Could not compile model: {e}")

    @torch.no_grad()
    def get_embedding(self, image: Image.Image, return_cpu: bool = False) -> torch.Tensor:
        """
        Extract embedding vector from an image.

        Args:
            image: PIL Image object
            return_cpu: If True, return CPU tensor. If False, keep on GPU.

        Returns:
            Normalized embedding tensor of shape (512,) or (768,) depending on model
        """
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        
        if self.use_amp:
            with torch.cuda.amp.autocast():
                embedding = self.model.get_image_features(**inputs)
        else:
            embedding = self.model.get_image_features(**inputs)
        
        # Normalize for cosine similarity
        embedding = embedding / embedding.norm(dim=-1, keepdim=True)
        result = embedding.squeeze(0)
        return result.cpu() if return_cpu else result

    @torch.no_grad()
    def get_embeddings_batch(self, images: list[Image.Image], return_cpu: bool = False) -> torch.Tensor:
        """
        Extract embeddings from multiple images in a batch (faster than one-by-one).

        Args:
            images: List of PIL Image objects
            return_cpu: If True, return CPU tensor. If False, keep on GPU.

        Returns:
            Normalized embedding tensor of shape (N, embed_dim)
        """
        if not images:
            empty_tensor = torch.tensor([], device=self.device)
            return empty_tensor.cpu() if return_cpu else empty_tensor
        
        inputs = self.processor(images=images, return_tensors="pt", padding=True).to(self.device)
        
        if self.use_amp:
            with torch.cuda.amp.autocast():
                embeddings = self.model.get_image_features(**inputs)
        else:
            embeddings = self.model.get_image_features(**inputs)
        
        # Normalize for cosine similarity
        embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)
        return embeddings.cpu() if return_cpu else embeddings

    def get_region_embedding(
        self,
        image: Image.Image,
        bbox: tuple[float, float, float, float],
        return_cpu: bool = False,
    ) -> torch.Tensor:
        """
        Extract embedding from a specific region of an image.

        Args:
            image: PIL Image object
            bbox: Bounding box as (x, y, width, height) in pixels
            return_cpu: If True, return CPU tensor. If False, keep on GPU.

        Returns:
            Normalized embedding tensor
        """
        x, y, w, h = bbox
        cropped = image.crop((x, y, x + w, y + h))
        return self.get_embedding(cropped, return_cpu=return_cpu)

    def get_region_embeddings_batch(
        self,
        image: Image.Image,
        bboxes: list[tuple[float, float, float, float]],
        return_cpu: bool = False,
    ) -> torch.Tensor:
        """
        Extract embeddings from multiple regions in a batch.

        Args:
            image: PIL Image object
            bboxes: List of bounding boxes as (x, y, width, height) in pixels
            return_cpu: If True, return CPU tensor. If False, keep on GPU.

        Returns:
            Normalized embedding tensor of shape (N, embed_dim)
        """
        crops = []
        for x, y, w, h in bboxes:
            cropped = image.crop((x, y, x + w, y + h))
            crops.append(cropped)
        return self.get_embeddings_batch(crops, return_cpu=return_cpu)


@lru_cache(maxsize=1)
def get_embedding_model():
    """
    Get singleton instance of the embedding model.
    
    Uses ONNX Runtime model for faster inference (4-12x speedup).
    Falls back to PyTorch CLIP if ONNX is not available.
    """
    if ONNX_AVAILABLE:
        try:
            # Use ONNX model (faster - 4-12x speedup)
            return get_onnx_embedding_model()
        except Exception as e:
            print(f"Warning: Failed to load ONNX model: {e}")
            print("Falling back to PyTorch CLIP model.")
            return EmbeddingModel()
    else:
        # Fallback to PyTorch CLIP
        return EmbeddingModel()
