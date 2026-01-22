"""
Product Tag Suggester API

FastAPI service for suggesting product tags across images using visual similarity.
"""

from contextlib import asynccontextmanager
from io import BytesIO

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from pydantic import BaseModel, HttpUrl

from app.models.embeddings import get_embedding_model
from app.services.matcher import ProductMatcher, TagInfo


# Request/Response models
class BoundingBox(BaseModel):
    """Bounding box in pixels (x, y, width, height)."""

    x: float
    y: float
    width: float
    height: float

    def to_tuple(self) -> tuple[float, float, float, float]:
        return (self.x, self.y, self.width, self.height)


class SourceTag(BaseModel):
    """A tagged product in the source image."""

    product_id: str
    bbox: BoundingBox


class SuggestTagsRequest(BaseModel):
    """Request to find tagged products in a target image."""

    source_image_url: HttpUrl
    source_tags: list[SourceTag]
    target_image_url: HttpUrl
    similarity_threshold: float = 0.75


class TagSuggestion(BaseModel):
    """Suggestion for a product tag in target image."""

    product_id: str
    found: bool
    confidence: float
    suggested_bbox: BoundingBox | None = None


class SuggestTagsResponse(BaseModel):
    """Response with tag suggestions."""

    suggestions: list[TagSuggestion]


# App setup
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize model on startup."""
    # Warm up the model
    get_embedding_model()
    yield


app = FastAPI(
    title="Product Tag Suggester",
    description="Suggest product tags across images using visual similarity",
    version="0.1.0",
    lifespan=lifespan,
)

# CORS middleware - allow requests from frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",  # Vite dev server
        "http://localhost:3000",  # Alternative dev port
        "http://127.0.0.1:5173",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# HTTP client for fetching images
http_client = httpx.AsyncClient(timeout=30.0)


async def fetch_image(url: str) -> Image.Image:
    """Fetch image from URL and return as PIL Image."""
    try:
        response = await http_client.get(str(url))
        response.raise_for_status()
        return Image.open(BytesIO(response.content)).convert("RGB")
    except httpx.HTTPError as e:
        raise HTTPException(status_code=400, detail=f"Failed to fetch image: {e}")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image: {e}")


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}


@app.post("/api/suggest-tags", response_model=SuggestTagsResponse)
async def suggest_tags(request: SuggestTagsRequest):
    """
    Find tagged products from source image in target image.

    Given an image with tagged products (source), checks if any of those
    products appear in another image (target) and suggests tag positions.

    Example:
    - Source image: Living room with tagged sofa (product_123)
    - Target image: Another angle of the same room
    - Result: {"suggestions": [{"product_id": "product_123", "found": true, ...}]}
    """
    # Fetch images
    source_image, target_image = await fetch_image(request.source_image_url), await fetch_image(request.target_image_url)

    # Convert tags to internal format
    tags = [
        TagInfo(product_id=tag.product_id, bbox=tag.bbox.to_tuple())
        for tag in request.source_tags
    ]

    # Run matcher
    model = get_embedding_model()
    matcher = ProductMatcher(
        embedding_model=model,
        similarity_threshold=request.similarity_threshold,
    )

    results = matcher.find_all_products(source_image, tags, target_image)

    # Convert to response format
    suggestions = []
    for result in results:
        suggestion = TagSuggestion(
            product_id=result.product_id,
            found=result.found,
            confidence=result.confidence,
            suggested_bbox=(
                BoundingBox(
                    x=result.suggested_bbox[0],
                    y=result.suggested_bbox[1],
                    width=result.suggested_bbox[2],
                    height=result.suggested_bbox[3],
                )
                if result.suggested_bbox
                else None
            ),
        )
        suggestions.append(suggestion)

    return SuggestTagsResponse(suggestions=suggestions)


@app.post("/api/compare-regions")
async def compare_regions(
    image1_url: HttpUrl,
    bbox1: BoundingBox,
    image2_url: HttpUrl,
    bbox2: BoundingBox,
):
    """
    Compare two specific regions and return similarity score.

    Useful for debugging or manual verification.
    """
    image1, image2 = await fetch_image(str(image1_url)), await fetch_image(str(image2_url))

    model = get_embedding_model()
    emb1 = model.get_region_embedding(image1, bbox1.to_tuple())
    emb2 = model.get_region_embedding(image2, bbox2.to_tuple())

    similarity = float((emb1 @ emb2).item())

    return {"similarity": round(similarity, 4)}
