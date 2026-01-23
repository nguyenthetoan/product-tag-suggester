"""
Script to convert CLIP model to ONNX format for faster inference.

Usage:
    python scripts/convert_clip_to_onnx.py

This will:
1. Load the CLIP model
2. Export to ONNX format
3. Optionally apply INT8 quantization for 2-4x additional speedup

Requirements:
    pip install onnxruntime-gpu onnxruntime-tools onnx onnxscript
"""

import torch
import numpy as np
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import os

# Optional: For quantization
try:
    from onnxruntime.quantization import quantize_dynamic, QuantType
    QUANTIZATION_AVAILABLE = True
except ImportError:
    QUANTIZATION_AVAILABLE = False
    print("Warning: onnxruntime-tools not installed. Quantization will be skipped.")
    print("Install with: pip install onnxruntime-tools")


def export_clip_to_onnx(
    model_name: str = "openai/clip-vit-base-patch32",
    output_dir: str = "models",
    quantize: bool = True,
):
    """
    Export CLIP model to ONNX format.

    Args:
        model_name: HuggingFace model identifier
        output_dir: Directory to save ONNX model
        quantize: Whether to apply INT8 quantization
    """
    os.makedirs(output_dir, exist_ok=True)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Load CLIP model
    print(f"Loading CLIP model: {model_name}")
    model = CLIPModel.from_pretrained(model_name).to(device)
    processor = CLIPProcessor.from_pretrained(model_name)
    model.eval()
    
    # Create dummy input (batch_size=1, 3 channels, 224x224)
    dummy_image = Image.new("RGB", (224, 224), color="white")
    inputs = processor(images=dummy_image, return_tensors="pt").to(device)
    
    # Export the full CLIP model's image encoder
    # We'll export the get_image_features method which is more stable
    onnx_path = os.path.join(output_dir, "clip_vision.onnx")
    print(f"Exporting CLIP image encoder to ONNX: {onnx_path}")
    
    # Create dummy input
    pixel_values = inputs["pixel_values"]
    
    # Create a wrapper function for easier export
    # We need to export the full pipeline: vision_model -> pooler -> projection
    class CLIPImageEncoder(torch.nn.Module):
        def __init__(self, clip_model):
            super().__init__()
            self.vision_model = clip_model.vision_model
            self.visual_projection = clip_model.visual_projection
            
        def forward(self, pixel_values):
            # Get vision model outputs
            vision_outputs = self.vision_model(pixel_values=pixel_values)
            # vision_outputs is a tuple: (last_hidden_state, pooler_output)
            # We want pooler_output (index 1) which is already pooled
            pooler_output = vision_outputs[1]  # Shape: (batch, hidden_size)
            # Apply projection to get final embedding
            image_embeds = self.visual_projection(pooler_output)  # Shape: (batch, embed_dim)
            return image_embeds
    
    # Create wrapper model
    encoder_model = CLIPImageEncoder(model).eval()
    
    # Export with opset 17 for better transformer support
    # Opset 14 is too old for modern CLIP models (LayerNormalization issues)
    try:
        torch.onnx.export(
            encoder_model,
            pixel_values,
            onnx_path,
            input_names=["pixel_values"],
            output_names=["image_embeds"],
            dynamic_axes={
                "pixel_values": {0: "batch_size"},
                "image_embeds": {0: "batch_size"},
            },
            opset_version=17,  # Updated from 14 to 17 for better transformer support
            do_constant_folding=True,
        )
    except Exception as e:
        print(f"Error with opset 17, trying opset 18: {e}")
        # Try with opset 18 if 17 fails
        torch.onnx.export(
            encoder_model,
            pixel_values,
            onnx_path,
            input_names=["pixel_values"],
            output_names=["image_embeds"],
            dynamic_axes={
                "pixel_values": {0: "batch_size"},
                "image_embeds": {0: "batch_size"},
            },
            opset_version=18,
            do_constant_folding=True,
        )
    
    print(f"✓ ONNX model exported to {onnx_path}")
    print(f"  Model size: {os.path.getsize(onnx_path) / 1024 / 1024:.2f} MB")
    
    # Apply quantization if requested
    if quantize and QUANTIZATION_AVAILABLE:
        quantized_path = os.path.join(output_dir, "clip_quantized.onnx")
        print(f"\nApplying INT8 quantization...")
        
        try:
            quantize_dynamic(
                model_input=onnx_path,
                model_output=quantized_path,
                weight_type=QuantType.QUInt8,
            )
            
            print(f"✓ Quantized model saved to {quantized_path}")
            print(f"  Original size: {os.path.getsize(onnx_path) / 1024 / 1024:.2f} MB")
            print(f"  Quantized size: {os.path.getsize(quantized_path) / 1024 / 1024:.2f} MB")
        except Exception as e:
            print(f"⚠ Quantization failed: {e}")
            print(f"  Continuing with non-quantized model: {onnx_path}")
            print(f"  Note: Non-quantized model is still faster than PyTorch CLIP")
            quantized_path = None
    else:
        if quantize:
            print("⚠ Quantization skipped (onnxruntime-tools not available)")
        quantized_path = None
    
    print("\n✓ Conversion complete!")
    print(f"\nTo use the ONNX model:")
    print(f"  1. Update app/models/embeddings.py to use ONNXEmbeddingModel")
    print(f"  2. Or modify get_embedding_model() to return ONNXEmbeddingModel")
    final_model_path = quantized_path if quantized_path else onnx_path
    print(f"  3. Model path: {final_model_path}")
    if not quantized_path:
        print(f"     (Using non-quantized model - still faster than PyTorch CLIP)")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Convert CLIP model to ONNX")
    parser.add_argument(
        "--model",
        type=str,
        default="openai/clip-vit-base-patch32",
        help="CLIP model name (default: openai/clip-vit-base-patch32)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="models",
        help="Output directory for ONNX model (default: models)",
    )
    parser.add_argument(
        "--no-quantize",
        action="store_true",
        help="Skip INT8 quantization",
    )
    
    args = parser.parse_args()
    
    export_clip_to_onnx(
        model_name=args.model,
        output_dir=args.output_dir,
        quantize=not args.no_quantize,
    )
