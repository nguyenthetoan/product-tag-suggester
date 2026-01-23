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
   - **Optional ONNX Runtime** (`app/models/embeddings_onnx.py`): 2-4x faster inference

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
│   ├── embeddings.py       # CLIP embedding model wrapper (PyTorch)
│   └── embeddings_onnx.py  # ONNX Runtime wrapper (optional, faster)
└── services/
    ├── __init__.py
    └── yolo_matcher.py     # Product matching logic (coarse/fine/ultra-fine search)
scripts/
└── convert_clip_to_onnx.py # Script to convert CLIP to ONNX format
models/
├── clip_vision.onnx        # ONNX model (if converted)
└── clip_quantized.onnx     # Quantized ONNX model (if converted)
```

## Key Technologies

- **FastAPI**: Async web framework
- **PyTorch 2.9.1**: Deep learning framework (with CUDA 13.0 support)
- **Ultralytics YOLO**: Object detection (YOLO26/YOLOv8)
- **Transformers**: CLIP model (HuggingFace)
- **ONNX Runtime** (optional): Faster inference engine for CLIP (2-4x speedup)
- **PIL/Pillow**: Image processing
- **httpx**: Async HTTP client for image fetching

### Python & CUDA Requirements

- **Python**: 3.10-3.13 (for ONNX Runtime support)
- **CUDA**: 12.x or 13.x (tested with CUDA 13.1)
- **PyTorch**: 2.9.1 with CUDA 13.0 wheels (compatible with CUDA 13.1 drivers)

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
- **Typical Response Time (PyTorch CLIP)**: 3-5 seconds (down from ~30s before optimizations)
- **Typical Response Time (ONNX Runtime)**: 0.8-1.5 seconds (2-4x faster)
- **YOLO Detection**: ~50-100ms
- **CLIP Embedding (PyTorch)**: ~50-150ms per batch
- **CLIP Embedding (ONNX)**: ~10-30ms per batch (2-4x faster)
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

**PyTorch CLIP:**
```python
# Single region (keeps on GPU)
source_embedding = clip_model.get_region_embedding(image, bbox, return_cpu=False)

# Batch regions (keeps on GPU)
embeddings = clip_model.get_region_embeddings_batch(image, bboxes, return_cpu=False)
```

**ONNX Runtime (same interface):**
```python
# ONNX model has same interface as PyTorch CLIP
onnx_model = get_onnx_embedding_model()
source_embedding = onnx_model.get_region_embedding(image, bbox, return_cpu=False)
embeddings = onnx_model.get_region_embeddings_batch(image, bboxes, return_cpu=False)
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

- **Response Time (PyTorch CLIP)**: 3-5 seconds (typical)
- **Response Time (ONNX Runtime)**: 0.8-1.5 seconds (target: 1-2s)
- **GPU Memory**: Efficient batching to avoid OOM
- **Throughput**: Handles concurrent requests via async FastAPI
- **Accuracy**: High precision via multi-stage search

## ONNX Runtime Integration

### Converting CLIP to ONNX

The project includes a script to convert CLIP to ONNX format for faster inference:

```bash
# Install ONNX dependencies
pip install onnxruntime-gpu onnxruntime-tools onnx onnxscript

# Convert CLIP to ONNX
python scripts/convert_clip_to_onnx.py
```

This creates:
- `models/clip_vision.onnx` - Non-quantized ONNX model
- `models/clip_quantized.onnx` - INT8 quantized model (faster, smaller)

### Using ONNX Model

To use the ONNX model instead of PyTorch CLIP:

1. Convert the model (see above)
2. Update `app/models/embeddings.py`:
   ```python
   from app.models.embeddings_onnx import get_onnx_embedding_model
   
   @lru_cache(maxsize=1)
   def get_embedding_model():
       return get_onnx_embedding_model()
   ```

### ONNX Performance

- **2-4x faster** than PyTorch CLIP
- **Lower memory usage** (especially with quantization)
- **Better GPU utilization** with ONNX Runtime
- **Target response time**: 0.8-1.5 seconds (vs 3-5 seconds with PyTorch)

### ONNX Limitations

- Requires Python 3.10-3.13 (Python 3.14 not yet supported)
- May have shape compatibility issues with dynamic batch sizes
- Currently processes batches of size 1 to avoid shape mismatches
- Quantization may fail for some model architectures

## Future Optimizations

Potential areas for further improvement:
- ✅ **ONNX Runtime integration** (implemented, optional)
- Model quantization (INT8) - available via ONNX
- TensorRT optimization
- Caching frequently accessed embeddings
- Request batching for multiple tags
- Model serving optimization (TorchServe, etc.)
- Fix ONNX dynamic batch size support for true batching
