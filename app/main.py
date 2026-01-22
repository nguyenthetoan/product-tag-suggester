"""
Product Tag Suggester API

FastAPI service for suggesting product tags across images using YOLO + CLIP.
"""

from contextlib import asynccontextmanager
from io import BytesIO

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from pydantic import BaseModel, HttpUrl

from app.models.yolo import get_yolo_model
from app.services.yolo_matcher import YOLOProductMatcher, YOLODirectMatcher, TagInfo


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
    """Request to find tagged products using YOLO detection."""

    source_image_url: HttpUrl
    source_tags: list[SourceTag]
    target_image_url: HttpUrl
    detection_confidence: float = 0.25


class TagSuggestion(BaseModel):
    """Suggestion for a product tag using YOLO detection."""

    product_id: str
    found: bool
    confidence: float  # YOLO detection confidence
    suggested_bbox: BoundingBox | None = None
    detected_class: str | None = None  # COCO class name (e.g., "couch", "chair")


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
    """Initialize YOLO model on startup."""
    get_yolo_model()
    yield


app = FastAPI(
    title="Product Tag Suggester",
    description="Suggest product tags across images using YOLO + CLIP",
    version="0.2.0",
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
    Find tagged products using YOLO object detection.

    Uses YOLO26 to detect the object class of tagged products in the source image,
    then finds matching object classes in the target image.
    This approach is fast (~50-100ms).

    Example:
    - Source image: Living room with tagged sofa (product_123)
    - Target image: Another angle of the same room
    - Result: YOLO detects "couch" class in both images, suggests tag position
    """
    # Fetch images
    source_image, target_image = (
        await fetch_image(request.source_image_url),
        await fetch_image(request.target_image_url),
    )

    # Convert tags to internal format
    tags = [
        TagInfo(product_id=tag.product_id, bbox=tag.bbox.to_tuple())
        for tag in request.source_tags
    ]

    # Run YOLO matcher
    yolo_model = get_yolo_model()
    matcher = YOLOProductMatcher(
        yolo_model=yolo_model,
        detection_confidence=request.detection_confidence,
    )

    # Get detections count for response
    detections = matcher.detect_objects(target_image)
    detections_count = len(detections)

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

    Returns all detected objects with their bounding boxes, class names, and confidence.
    Optionally filter by specific class names.

    Example:
    - Request: {"image_url": "...", "target_classes": ["chair", "sofa"]}
    - Response: List of detected chairs and sofas with bounding boxes
    """
    image = await fetch_image(request.image_url)

    yolo_model = get_yolo_model()
    direct_matcher = YOLODirectMatcher(
        yolo_model=yolo_model,
        detection_confidence=request.confidence_threshold,
    )

    detections = direct_matcher.detect_and_classify(
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
    """
    Get all available YOLO class names.

    Returns the 80 COCO classes that YOLO can detect.
    Useful for filtering detections or understanding what objects can be detected.
    """
    yolo_model = get_yolo_model()
    class_names = yolo_model.get_class_names()

    return {
        "classes": class_names,
        "total_count": len(class_names),
    }
