# Product Tag Suggester - Agent Guide

## Project Overview

A FastAPI microservice that detects tagged products across images using a hybrid YOLO + CLIP approach. Given a source image with tagged products, the service finds those same products in a target image and returns their positions with bounding boxes.

**Key Capability**: Can match ANY product (kettles, toasters, furniture, etc.) regardless of whether YOLO knows the object class, using CLIP's semantic understanding.

## Architecture

### Core Components

1. **YOLO Model** (`app/models/yolo.py`)
   - Object detection using YOLO26 (falls back to YOLOv8)
   - Fast candidate region detection (~50ms)
   - Uses FP16 precision on GPU for optimal performance
   - Singleton pattern via `get_yolo_model()`

2. **CLIP Embedding Model** (`app/models/embeddings.py`)
   - OpenAI CLIP for semantic image embeddings
   - Batch processing for multiple regions
   - GPU-optimized with mixed precision (AMP)
   - Keeps tensors on GPU for performance
   - Singleton pattern via `get_embedding_model()`

3. **Product Matcher** (`app/services/yolo_matcher.py`)
   - Hybrid matching strategy:
     - YOLO detections (fast, limited to known classes)
     - Coarse grid search (covers entire image)
     - Fine search (refines around top candidates)
     - Ultra-fine search (pixel-level precision)
   - Multi-scale search for size variations
   - GPU-accelerated similarity computations

4. **API Layer** (`app/main.py`)
   - FastAPI REST endpoints
   - Async image fetching with parallel downloads
   - Pydantic models for request/response validation

## File Structure

```
app/
├── __init__.py
├── main.py                 # FastAPI app, endpoints, request/response models
├── models/
│   ├── __init__.py
│   ├── yolo.py            # YOLO model wrapper (YOLO26/YOLOv8)
│   └── embeddings.py       # CLIP embedding model wrapper
└── services/
    ├── __init__.py
    └── yolo_matcher.py     # Product matching logic (coarse/fine/ultra-fine search)
```

## Key Technologies

- **FastAPI**: Async web framework
- **PyTorch 2.9.1**: Deep learning framework
- **Ultralytics YOLO**: Object detection (YOLO26/YOLOv8)
- **Transformers**: CLIP model (HuggingFace)
- **PIL/Pillow**: Image processing
- **httpx**: Async HTTP client for image fetching

## Performance Optimizations

### GPU Optimizations
- **FP16 Mixed Precision**: Enabled for both YOLO and CLIP (~2x speedup)
- **GPU Tensor Operations**: All computations stay on GPU, minimal CPU transfers
- **Batch Processing**: Multiple regions processed in single batch
- **cuDNN Benchmarking**: Enabled for consistent input sizes
- **TensorFloat-32**: Enabled for Ampere+ GPUs (RTX 5070 Ti)

### Search Strategy Optimizations
- **Coarse Grid Search**: Limited to ~150 candidates (adaptive stride)
- **Fine Search**: 5 scales (0.75, 0.875, 1.0, 1.125, 1.25) with 0.75x search range
- **Ultra-Fine Search**: Pixel-level refinement (2px stride) around best candidate
- **GPU-Based Sorting**: Uses `torch.topk()` for efficient top-k selection
- **Parallel Image Fetching**: Both images downloaded simultaneously

### Performance Characteristics
- **Typical Response Time**: 3-5 seconds (down from ~30s before optimizations)
- **YOLO Detection**: ~50-100ms
- **CLIP Embedding**: ~50-150ms per batch
- **GPU Utilization**: High (FP16, batched operations, minimal transfers)

## API Endpoints

### POST `/api/suggest-tags`
Find tagged products from source image in target image.

**Request:**
```json
{
  "source_image_url": "https://...",
  "source_tags": [
    {
      "product_id": "product_123",
      "bbox": { "x": 100, "y": 150, "width": 80, "height": 120 }
    }
  ],
  "target_image_url": "https://...",
  "similarity_threshold": 0.75
}
```

**Response:**
```json
{
  "suggestions": [
    {
      "product_id": "product_123",
      "found": true,
      "confidence": 0.87,
      "suggested_bbox": { "x": 250, "y": 180, "width": 85, "height": 125 },
      "detected_class": "bottle"
    }
  ],
  "detections_count": 12
}
```

### POST `/api/detect`
Detect all objects in an image using YOLO.

### GET `/api/classes`
Get all available YOLO class names.

### GET `/health`
Health check endpoint.

## Important Conventions

### Model Initialization
- Models are initialized on app startup (lifespan context manager)
- Singleton pattern: `get_yolo_model()` and `get_embedding_model()`
- Models are loaded once and reused across requests

### Tensor Device Management
- **Always keep tensors on GPU** during computation
- Only move to CPU at final step (when returning results)
- Use `return_cpu=False` for intermediate embeddings
- Ensure device consistency before operations

### Bounding Box Format
- Format: `(x, y, width, height)` in pixels
- `x, y`: Top-left corner coordinates
- All bboxes use this format consistently

### Search Strategy
1. **YOLO Detections**: Fast, but limited to known COCO classes
2. **Coarse Grid**: Covers entire image with adaptive stride
3. **Fine Search**: Multi-scale search around top 2 candidates
4. **Ultra-Fine Search**: Pixel-level refinement if match found

### Error Handling
- Image fetching errors → HTTP 400 with descriptive message
- Model errors → Logged, graceful degradation
- Invalid requests → Pydantic validation errors

## Code Patterns

### Embedding Extraction
```python
# Single region (keeps on GPU)
source_embedding = clip_model.get_region_embedding(image, bbox, return_cpu=False)

# Batch regions (keeps on GPU)
embeddings = clip_model.get_region_embeddings_batch(image, bboxes, return_cpu=False)
```

### Similarity Computation
```python
# Compute on GPU
similarities = torch.matmul(embeddings, source_embedding)

# Only move to CPU at final step
similarities_list = similarities.cpu().tolist()
```

### YOLO Detection
```python
# Detections with padding for better embeddings
detections = yolo_model.detect_with_embedding_regions(
    image,
    confidence_threshold=0.10,  # Low threshold for more candidates
    padding_ratio=0.1
)
```

## Configuration

### Default Parameters
- `similarity_threshold`: 0.75 (minimum CLIP similarity for match)
- `detection_confidence`: 0.10 (YOLO confidence, lower = more candidates)
- `max_candidates`: 150 (coarse grid search limit)
- `fine_search_scales`: [0.75, 0.875, 1.0, 1.125, 1.25]
- `ultra_fine_stride`: 2 pixels

### GPU Settings
- Automatically detects CUDA availability
- FP16 enabled for both models
- cuDNN benchmarking enabled
- TensorFloat-32 enabled for Ampere+ GPUs

## Development Notes

### Adding New Features
- Keep GPU operations in mind - minimize CPU transfers
- Use batch processing when possible
- Profile performance before/after changes
- Maintain singleton pattern for models

### Debugging
- Check GPU memory usage if OOM errors
- Monitor tensor device placement
- Use logging for search strategy insights
- Profile with `torch.profiler` if needed

### Testing
- Test with various image sizes
- Test with different similarity thresholds
- Verify GPU vs CPU behavior
- Test edge cases (no matches, multiple matches)

## Integration

### With communa-web
The service is designed to integrate with the communa-web frontend:
- TypeScript/vanilla-extract frontend
- OpenAPI for API requests
- Motion for animations

See README.md for integration example code.

## Performance Targets

- **Response Time**: 3-5 seconds (typical)
- **GPU Memory**: Efficient batching to avoid OOM
- **Throughput**: Handles concurrent requests via async FastAPI
- **Accuracy**: High precision via multi-stage search

## Future Optimizations

Potential areas for further improvement:
- Model quantization (INT8)
- TensorRT optimization
- Caching frequently accessed embeddings
- Request batching for multiple tags
- Model serving optimization (TorchServe, etc.)
