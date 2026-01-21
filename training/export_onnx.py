#!/usr/bin/env python3
"""
Export NLLB model to ONNX format using Optimum.
"""

import argparse
from pathlib import Path
from optimum.onnxruntime import ORTModelForSeq2SeqLM
from transformers import AutoTokenizer

def export_model(model_name: str, output_dir: str):
    print(f"Exporting model: {model_name}")
    print(f"Output directory: {output_dir}")
    
    # Load and export
    model = ORTModelForSeq2SeqLM.from_pretrained(model_name, export=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Save model and tokenizer
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    print(f"\n✅ Model exported successfully to: {output_dir}")

def main():
    parser = argparse.ArgumentParser(description="Export NLLB to ONNX")
    parser.add_argument('--model', type=str, default='facebook/nllb-200-distilled-600M',
                       help='Model name or path')
    parser.add_argument('--output', type=str, default='models/nllb_onnx',
                       help='Output directory for ONNX model')
    
    args = parser.parse_args()
    
    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)
    
    export_model(args.model, str(output_path))

if __name__ == "__main__":
    main()
