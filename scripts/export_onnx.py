#!/usr/bin/env python3
"""
Export MODEL-W to ONNX format for deployment.

Useful for:
- Running inference without PyTorch
- Integration with DAW plugins (via ONNX runtime)
- Edge deployment
"""

import argparse
from pathlib import Path

import torch
import torch.onnx

from modelw.api import ModelW


def export_to_onnx(
    checkpoint_path: str,
    output_path: str,
    max_seq_len: int = 512,
    opset_version: int = 14,
):
    """Export model to ONNX format."""
    
    print(f"Loading model from {checkpoint_path}...")
    model_api = ModelW.load(checkpoint_path, device="cpu")
    model = model_api.model
    model.eval()
    
    print(f"Model loaded: {model_api}")
    
    # Create dummy input
    batch_size = 1
    seq_len = max_seq_len
    dummy_input = torch.randint(0, model_api.tokenizer.vocab_size, (batch_size, seq_len))
    
    # Export
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"Exporting to {output_path}...")
    
    torch.onnx.export(
        model,
        (dummy_input,),
        str(output_path),
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=["input_ids"],
        output_names=["logits"],
        dynamic_axes={
            "input_ids": {0: "batch_size", 1: "sequence"},
            "logits": {0: "batch_size", 1: "sequence"},
        },
    )
    
    print(f"✓ Exported to {output_path}")
    
    # Verify
    try:
        import onnx
        import onnxruntime
        
        print("\nVerifying ONNX model...")
        onnx_model = onnx.load(str(output_path))
        onnx.checker.check_model(onnx_model)
        print("✓ ONNX model is valid")
        
        # Test inference
        session = onnxruntime.InferenceSession(str(output_path))
        test_input = dummy_input.numpy()
        outputs = session.run(None, {"input_ids": test_input})
        print(f"✓ ONNX inference works, output shape: {outputs[0].shape}")
        
    except ImportError:
        print("\n(Install onnx and onnxruntime to verify the export)")


def main():
    parser = argparse.ArgumentParser(description="Export MODEL-W to ONNX")
    parser.add_argument(
        "checkpoint",
        type=str,
        help="Path to checkpoint directory or .pt file",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./exports/model_w.onnx",
        help="Output ONNX file path",
    )
    parser.add_argument(
        "--max-seq-len",
        type=int,
        default=512,
        help="Maximum sequence length for export",
    )
    args = parser.parse_args()
    
    export_to_onnx(args.checkpoint, args.output, args.max_seq_len)


if __name__ == "__main__":
    main()

