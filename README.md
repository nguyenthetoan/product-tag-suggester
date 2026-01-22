# Product Tag Suggester

A microservice for suggesting product tags across images using visual similarity (CLIP embeddings).

## Overview

Given an image with tagged products, this service can detect if those products appear in another image and suggest tag positions.

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
  "similarity_threshold": 0.75
}
```

**Response:**

```json
{
  "suggestions": [
    {
      "product_id": "prod_123",
      "found": true,
      "confidence": 0.87,
      "suggested_bbox": { "x": 250, "y": 180, "width": 200, "height": 180 }
    }
  ]
}
```

### Compare Regions (Debug)

```bash
POST /api/compare-regions?image1_url=...&bbox1=...&image2_url=...&bbox2=...
```

Returns similarity score between two specific regions.

### Health Check

```bash
GET /health
```

## Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `similarity_threshold` | 0.75 | Minimum similarity to consider a match (0-1) |
| Model | `clip-vit-base-patch32` | CLIP model variant |

### Using a More Accurate Model

Edit `app/models/embeddings.py`:

```python
# Change from:
model_name: str = "openai/clip-vit-base-patch32"

# To:
model_name: str = "openai/clip-vit-large-patch14"
```

Note: Larger model requires more memory (~2GB) and is slower.

## Performance Considerations

### Sliding Window Approach

The current implementation uses sliding windows to search for products. This is:
- **Pros:** Simple, works without training
- **Cons:** Can be slow for large images or many tags

### Optimization Ideas

1. **Add object detection pre-filter**: Use YOLO/DETR to find candidate regions first
2. **Downscale images**: Reduce resolution before processing
3. **GPU acceleration**: Run on CUDA for ~10x speedup
4. **Batch processing**: Process multiple windows in parallel
5. **Caching**: Cache embeddings for frequently used product images

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
  }),
});

const { suggestions } = await response.json();

// Show suggestions to user
suggestions
  .filter(s => s.found)
  .forEach(s => showTagSuggestion(s.product_id, s.suggested_bbox));
```

## License

MIT
