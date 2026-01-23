"""
Product Tag Suggester API

FastAPI service for detecting tagged products across images.
Uses YOLO for fast candidate detection + CLIP for accurate matching.
"""

import asyncio
from contextlib import asynccontextmanager
from io import BytesIO

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from pydantic import BaseModel, HttpUrl

from app.models.yolo import get_yolo_model
from app.models.embeddings import get_embedding_model
from app.services.yolo_matcher import ProductMatcher, DirectDetector, TagInfo


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
    """Request to find tagged products in target image."""

    source_image_url: HttpUrl
    source_tags: list[SourceTag]
    target_image_url: HttpUrl
    similarity_threshold: float = 0.75


class TagSuggestion(BaseModel):
    """Suggestion for a product tag."""

    product_id: str
    found: bool
    confidence: float  # CLIP similarity score (0-1)
    suggested_bbox: BoundingBox | None = None
    detected_class: str | None = None  # YOLO class if available


class SuggestTagsResponse(BaseModel):
    """Response with tag suggestions."""

    suggestions: list[TagSuggestion]
    detections_count: int


class DetectedObject(BaseModel):
    """A detected object in an image."""

    bbox: BoundingBox
    class_id: int
    class_name: str
    confidence: float


class DetectObjectsRequest(BaseModel):
    """Request to detect objects in an image."""

    image_url: HttpUrl
    confidence_threshold: float = 0.25
    target_classes: list[str] | None = None


class DetectObjectsResponse(BaseModel):
    """Response with detected objects."""

    objects: list[DetectedObject]
    total_count: int


# App setup
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize models on startup."""
    import torch
    
    # Optimize PyTorch for high-performance GPU systems
    if torch.cuda.is_available():
        # Enable cuDNN benchmarking for consistent input sizes (faster)
        torch.backends.cudnn.benchmark = True
        # Enable cuDNN deterministic mode (optional, can disable for speed)
        torch.backends.cudnn.deterministic = False
        # Enable TensorFloat-32 for faster computation on Ampere+ GPUs (RTX 5070 Ti)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        print(f"GPU optimizations enabled: CUDA device {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    # Initialize models (this will trigger compilation and optimization)
    get_yolo_model()
    get_embedding_model()
    yield


app = FastAPI(
    title="Product Tag Suggester",
    description="Detect tagged products across images using YOLO + CLIP",
    version="0.3.0",
    lifespan=lifespan,
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://localhost:3000",
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

    How it works:
    1. YOLO detects candidate regions in target image (fast)
    2. CLIP compares the source tagged product with each candidate
    3. Returns matches above the similarity threshold

    This can match ANY product (kettles, toasters, furniture, etc.) 
    regardless of whether YOLO knows the object class.

    Example:
    - Source image: Kitchen with tagged SMEG kettle
    - Target image: Another angle of the same kitchen
    - Result: Finds the same kettle and returns its position
    """
    # Fetch images in parallel
    source_image, target_image = await asyncio.gather(
        fetch_image(request.source_image_url),
        fetch_image(request.target_image_url),
    )

    # Convert tags to internal format
    tags = [
        TagInfo(product_id=tag.product_id, bbox=tag.bbox.to_tuple())
        for tag in request.source_tags
    ]

    # Run hybrid YOLO + CLIP matcher
    yolo_model = get_yolo_model()
    clip_model = get_embedding_model()
    matcher = ProductMatcher(
        yolo_model=yolo_model,
        embedding_model=clip_model,
        similarity_threshold=request.similarity_threshold,
    )

    results, detections_count = matcher.find_all_products(source_image, tags, target_image)

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
            detected_class=result.detected_class,
        )
        suggestions.append(suggestion)

    return SuggestTagsResponse(
        suggestions=suggestions,
        detections_count=detections_count,
    )


@app.post("/api/detect", response_model=DetectObjectsResponse)
async def detect_objects(request: DetectObjectsRequest):
    """
    Detect objects in an image using YOLO.

    Returns all detected objects with bounding boxes and class names.
    """
    image = await fetch_image(request.image_url)

    yolo_model = get_yolo_model()
    detector = DirectDetector(
        yolo_model=yolo_model,
        detection_confidence=request.confidence_threshold,
    )

    detections = detector.detect_and_classify(
        image,
        target_classes=request.target_classes,
    )

    objects = [
        DetectedObject(
            bbox=BoundingBox(
                x=det.bbox[0],
                y=det.bbox[1],
                width=det.bbox[2],
                height=det.bbox[3],
            ),
            class_id=det.class_id,
            class_name=det.class_name,
            confidence=round(det.confidence, 3),
        )
        for det in detections
    ]

    return DetectObjectsResponse(
        objects=objects,
        total_count=len(objects),
    )


@app.get("/api/classes")
async def get_yolo_classes():
    """Get all available YOLO class names (80 COCO classes)."""
    yolo_model = get_yolo_model()
    return {
        "classes": yolo_model.get_class_names(),
        "total_count": len(yolo_model.get_class_names()),
    }
