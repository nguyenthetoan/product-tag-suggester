# Product Tag Suggester

A microservice for detecting tagged products across images using YOLO + CLIP (with optional ONNX Runtime optimization).

## Overview

Given an image with tagged products, this service detects if those same products appear in another image and returns their positions.

**How it works:**
1. **YOLO** detects candidate regions in target image (fast, ~50ms)
2. **CLIP** compares the source tagged product with each candidate region
3. Returns matches above the similarity threshold with bounding boxes

This approach can match **any product** (kettles, toasters, furniture, etc.) regardless of whether YOLO knows the object class.

**Performance Optimization:**
- CLIP can be converted to ONNX format for 2-4x faster inference
- ONNX Runtime with CUDA support provides significant speedup
- See `scripts/convert_clip_to_onnx.py` for conversion instructions

**Example Use Case:**
1. User uploads Image 1 (kitchen) and tags a SMEG kettle
2. User uploads Image 2 (another angle of the kitchen)
3. Service finds the same kettle in Image 2 and returns its position

## Quick Start

### Local Development

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run with Gunicorn (recommended)
gunicorn app.main:app -c gunicorn.conf.py

# Or with Uvicorn (development)
uvicorn app.main:app -b 0.0.0.0:8000
```

### Docker

```bash
docker build -t product-tag-suggester .
docker run -p 8000:8000 product-tag-suggester
```

## API Usage

### Suggest Tags

Find tagged products from source image in target image.

```bash
POST /api/suggest-tags
Content-Type: application/json

{
  "source_image_url": "https://example.com/kitchen1.jpg",
  "source_tags": [
    {
      "product_id": "smeg_kettle_123",
      "bbox": { "x": 100, "y": 150, "width": 80, "height": 120 }
    }
  ],
  "target_image_url": "https://example.com/kitchen2.jpg",
  "similarity_threshold": 0.75
}
```

**Response:**

```json
{
  "suggestions": [
    {
      "product_id": "smeg_kettle_123",
      "found": true,
      "confidence": 0.87,
      "suggested_bbox": { "x": 250, "y": 180, "width": 85, "height": 125 },
      "detected_class": "bottle"
    }
  ],
  "detections_count": 12
}
```

### Detect Objects

Detect all objects in an image using YOLO.

```bash
POST /api/detect
Content-Type: application/json

{
  "image_url": "https://example.com/image.jpg",
  "confidence_threshold": 0.25,
  "target_classes": ["chair", "couch", "bed"]
}
```

### Get Available Classes

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
| `similarity_threshold` | 0.75 | Minimum CLIP similarity for a match (0-1) |
| `detection_confidence` | 0.15 | Internal YOLO confidence (lower = more candidates) |

### System Requirements

- **Python**: 3.10-3.13 (for ONNX Runtime support, if using)
- **CUDA**: 12.x or 13.x (tested with CUDA 13.1)
- **GPU**: NVIDIA GPU with CUDA support (recommended)
- **PyTorch**: 2.9.1 with CUDA 13.0 wheels (for RTX 5070 Ti and newer GPUs)

## Performance

- **Speed (PyTorch CLIP):** 3-5 seconds per request (YOLO ~50ms + CLIP ~50-150ms per batch)
- **Speed (ONNX Runtime):** 0.8-1.5 seconds per request (2-4x faster with ONNX)
- **GPU:** Automatically uses CUDA if available (~5-10x faster)
- **Accuracy:** Can match any product, not limited to 80 COCO classes

### ONNX Optimization (Optional)

For faster performance, you can convert CLIP to ONNX format:

```bash
# Install ONNX dependencies
pip install onnxruntime-gpu onnxruntime-tools onnx onnxscript

# Convert CLIP to ONNX
python scripts/convert_clip_to_onnx.py

# The ONNX model will be saved to models/clip_vision.onnx
# Update app/models/embeddings.py to use ONNXEmbeddingModel
```

**Note:** ONNX Runtime requires Python 3.10-3.13 (Python 3.14 not yet supported).

## Integration with communa-web

```typescript
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
    similarity_threshold: 0.75,
  }),
});

const { suggestions } = await response.json();

suggestions
  .filter(s => s.found)
  .forEach(s => showTagSuggestion(s.product_id, s.suggested_bbox));
```

## Troubleshooting

### SSL Certificate Error on macOS

```bash
/Applications/Python\ 3.x/Install\ Certificates.command
```

### Model Download Issues

Models are downloaded on first run (~200MB total). If download fails:
1. Check internet connection
2. Pre-download from [Ultralytics releases](https://github.com/ultralytics/assets/releases)

### ONNX Runtime Issues

If using ONNX Runtime:
- **Python 3.14:** Not supported yet, use Python 3.13 or 3.12
- **CUDA:** Requires CUDA 12.x or 13.x with compatible drivers
- **Shape errors:** The ONNX model may need re-exporting with fixed batch sizes
- See `scripts/convert_clip_to_onnx.py` for conversion options

## License

MIT
