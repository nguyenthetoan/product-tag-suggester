# Product Tag Suggester

A microservice for suggesting product tags across images using YOLO26 + CLIP.

## Overview

Given an image with tagged products, this service can detect if those products appear in another image and suggest tag positions.

**How it works:**
1. YOLO26 detects objects in the target image (fast, ~50-100ms)
2. CLIP embeddings verify which detected objects match the source products
3. Returns matched objects with bounding boxes and confidence scores

**Example Use Case:**
1. User uploads Image 1 (living room) and tags a sofa (product_123)
2. User uploads Image 2 (another angle of the room)
3. Service detects the same sofa in Image 2 and suggests tagging it

## Quick Start

### Local Development

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Option 1: Gunicorn (recommended - more stable, better signal handling)
gunicorn app.main:app -c gunicorn.conf.py

# Option 2: Uvicorn with timeout (for development with hot reload)
uvicorn app.main:app --reload --port 8000 --timeout-keep-alive 120

# Option 3: Uvicorn without reload (more stable than --reload)
uvicorn app.main:app --port 8000 --timeout-keep-alive 120
```

**Tip:** If uvicorn won't stop with Ctrl+C, use `kill -9 $(lsof -t -i:8000)` to force kill.

### Docker

```bash
# Build
docker build -t product-tag-suggester .

# Run
docker run -p 8000:8000 product-tag-suggester
```

## API Usage

### Suggest Tags

Find tagged products from a source image in a target image.

```bash
POST /api/suggest-tags
Content-Type: application/json

{
  "source_image_url": "https://example.com/image1.jpg",
  "source_tags": [
    {
      "product_id": "prod_123",
      "bbox": { "x": 100, "y": 150, "width": 200, "height": 180 }
    }
  ],
  "target_image_url": "https://example.com/image2.jpg",
  "similarity_threshold": 0.70,
  "detection_confidence": 0.25
}
```

**Response:**

```json
{
  "suggestions": [
    {
      "product_id": "prod_123",
      "found": true,
      "confidence": 0.85,
      "suggested_bbox": { "x": 250, "y": 180, "width": 210, "height": 195 },
      "detected_class": "couch",
      "detection_confidence": 0.92
    }
  ],
  "detections_count": 5
}
```

### Detect Objects

Detect all objects in an image. Useful for understanding what objects are present.

```bash
POST /api/detect
Content-Type: application/json

{
  "image_url": "https://example.com/image.jpg",
  "confidence_threshold": 0.25,
  "target_classes": ["chair", "couch", "bed"]  // optional filter
}
```

**Response:**

```json
{
  "objects": [
    {
      "bbox": { "x": 100, "y": 150, "width": 200, "height": 180 },
      "class_id": 57,
      "class_name": "couch",
      "confidence": 0.92
    }
  ],
  "total_count": 1
}
```

### Get Available Classes

List all 80 COCO classes that YOLO can detect.

```bash
GET /api/classes
```

### Health Check

```bash
GET /health
```

## Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `similarity_threshold` | 0.70 | Minimum CLIP similarity to consider a match (0-1) |
| `detection_confidence` | 0.25 | Minimum YOLO detection confidence (0-1) |
| Model | `yolo26m.pt` | YOLO26 medium (balanced speed/accuracy) |

### Using a Different YOLO Model

Edit `app/models/yolo.py`:

```python
# Change from:
model_name: str = "yolo26m.pt"

# To (faster, less accurate):
model_name: str = "yolo26s.pt"

# Or (slower, more accurate):
model_name: str = "yolo26l.pt"
```

## Performance

- **Speed:** ~50-100ms per image (YOLO26 is 43% faster than YOLOv8)
- **GPU:** Automatically uses CUDA if available (~10x faster)
- **Detectable objects:** 80 COCO classes (furniture, electronics, vehicles, etc.)

## Integration with communa-web

Call this service from your React Router action/loader:

```typescript
// In your post form handler
const response = await fetch('http://localhost:8000/api/suggest-tags', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    source_image_url: images[0].url,
    source_tags: images[0].tags.map(tag => ({
      product_id: tag.productId,
      bbox: tag.bbox,
    })),
    target_image_url: images[1].url,
    similarity_threshold: 0.70,
    detection_confidence: 0.25,
  }),
});

const { suggestions, detections_count } = await response.json();

// Show suggestions to user
suggestions
  .filter(s => s.found)
  .forEach(s => {
    console.log(`Found ${s.detected_class} with ${s.confidence} confidence`);
    showTagSuggestion(s.product_id, s.suggested_bbox);
  });
```

## Troubleshooting

### SSL Certificate Error on macOS

If you see `SSL: CERTIFICATE_VERIFY_FAILED` when downloading YOLO weights:

```bash
# Option 1: Install certificates for Python
/Applications/Python\ 3.x/Install\ Certificates.command

# Option 2: Manual download (if above doesn't work)
# Download model from: https://github.com/ultralytics/assets/releases
# Place in the project root or ~/.cache/ultralytics/
```

The code includes an automatic SSL bypass, but if it still fails, use the manual download option.

### Model Download Issues

YOLO model weights (~50MB) are downloaded on first run. If the download fails:

1. Check your internet connection
2. Try a smaller model: edit `app/models/yolo.py` to use `yolo26n.pt` instead of `yolo26m.pt`
3. Pre-download manually from [Ultralytics releases](https://github.com/ultralytics/assets/releases)

## License

MIT
