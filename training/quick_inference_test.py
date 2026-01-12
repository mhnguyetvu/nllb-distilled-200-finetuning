#!/usr/bin/env python3
"""
Quick inference test for NLLB models
Test both quality and performance before/after finetuning

Usage:
    # Test pretrained model
    python quick_inference_test.py --model facebook/nllb-200-distilled-600M
    
    # Test finetuned model
    python quick_inference_test.py --model ../outputs/nllb-ko-vi-finetuned
"""

import torch
import json
import time
import argparse
from pathlib import Path
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import evaluate
from typing import List, Dict
import psutil
import os

def get_gpu_memory():
    """Get current GPU memory usage in GB"""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1024**3
    return 0

def load_test_samples(test_file: str, num_samples: int = 100) -> Dict[str, List[str]]:
    """Load first N test samples"""
    sources = []
    references = []
    
    with open(test_file, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= num_samples:
                break
            item = json.loads(line)
            translation = item['translation']
            sources.append(translation['kor_Hang'])
            references.append(translation['vie_Latn'])
    
    return {'sources': sources, 'references': references}

def run_inference_test(
    model_name: str,
    test_file: str,
    num_samples: int = 100,
    batch_size: int = 8,
    num_beams: int = 5,
):
    """Run inference test and collect metrics"""
    
    print(f"\n{'='*70}")
    print(f"INFERENCE TEST: {model_name}")
    print(f"{'='*70}\n")
    
    # Load test data
    print(f"Loading {num_samples} test samples from {test_file}...")
    test_data = load_test_samples(test_file, num_samples)
    sources = test_data['sources']
    references = test_data['references']
    print(f"✓ Loaded {len(sources)} sentences\n")
    
    # Load model
    print(f"Loading model: {model_name}")
    start_load = time.time()
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()
    
    load_time = time.time() - start_load
    
    if device == "cuda":
        gpu_name = torch.cuda.get_device_name(0)
        print(f"✓ Device: {gpu_name}")
    else:
        print(f"⚠ Device: CPU (slower)")
    
    print(f"✓ Model loaded in {load_time:.1f}s")
    print(f"✓ Model parameters: {model.num_parameters() / 1e6:.1f}M\n")
    
    # Warm-up
    print("Warming up GPU...")
    tokenizer.src_lang = "kor_Hang"
    tgt_token_id = tokenizer.convert_tokens_to_ids("vie_Latn")
    warmup_inputs = tokenizer([sources[0]], return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        model.generate(**warmup_inputs, forced_bos_token_id=tgt_token_id, max_length=50)
    torch.cuda.synchronize() if device == "cuda" else None
    print("✓ Warm-up done\n")
    
    # Translation with performance measurement
    print(f"Translating {len(sources)} sentences...")
    print(f"Batch size: {batch_size}, Num beams: {num_beams}\n")
    
    predictions = []
    total_tokens_generated = 0
    
    # Memory before
    mem_before = get_gpu_memory()
    
    start_time = time.time()
    
    for i in range(0, len(sources), batch_size):
        batch_sources = sources[i:i+batch_size]
        
        # Tokenize
        inputs = tokenizer(
            batch_sources,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=256,
        ).to(device)
        
        # Generate
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                forced_bos_token_id=tgt_token_id,
                max_length=256,
                num_beams=num_beams,
                early_stopping=True,
            )
        
        # Decode
        batch_predictions = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        predictions.extend(batch_predictions)
        total_tokens_generated += outputs.numel()
        
        # Progress
        if (i + batch_size) % 40 == 0:
            progress = min(i + batch_size, len(sources))
            print(f"  Progress: {progress}/{len(sources)} sentences")
    
    torch.cuda.synchronize() if device == "cuda" else None
    elapsed_time = time.time() - start_time
    
    # Memory after
    mem_after = get_gpu_memory()
    mem_peak = torch.cuda.max_memory_allocated() / 1024**3 if device == "cuda" else 0
    
    # Calculate metrics
    sentences_per_sec = len(sources) / elapsed_time
    tokens_per_sec = total_tokens_generated / elapsed_time
    
    print(f"\n✓ Translation completed!")
    print(f"  Time: {elapsed_time:.2f}s")
    print(f"  Speed: {sentences_per_sec:.2f} sent/s")
    print(f"  Tokens/s: {tokens_per_sec:.1f}")
    
    if device == "cuda":
        print(f"  GPU memory: {mem_before:.2f}GB → {mem_after:.2f}GB (peak: {mem_peak:.2f}GB)")
    
    # Compute BLEU
    print(f"\nComputing BLEU score...")
    bleu_metric = evaluate.load("sacrebleu")
    bleu_result = bleu_metric.compute(
        predictions=predictions,
        references=[[r] for r in references]
    )
    
    print(f"✓ BLEU: {bleu_result['score']:.2f}")
    
    # Sample translations
    print(f"\n{'='*70}")
    print("SAMPLE TRANSLATIONS:")
    print(f"{'='*70}\n")
    
    for i in range(min(5, len(sources))):
        print(f"Example {i+1}:")
        print(f"  Source:     {sources[i]}")
        print(f"  Reference:  {references[i]}")
        print(f"  Prediction: {predictions[i]}")
        print()
    
    # Summary
    results = {
        'model': model_name,
        'device': device,
        'test_size': len(sources),
        'batch_size': batch_size,
        'num_beams': num_beams,
        'performance': {
            'load_time_s': load_time,
            'translation_time_s': elapsed_time,
            'sentences_per_sec': sentences_per_sec,
            'tokens_per_sec': tokens_per_sec,
            'gpu_memory_gb': {
                'before': mem_before,
                'after': mem_after,
                'peak': mem_peak,
            } if device == "cuda" else None,
        },
        'quality': {
            'bleu': bleu_result['score'],
            'precisions': bleu_result['precisions'],
        },
        'samples': [
            {
                'source': src,
                'reference': ref,
                'prediction': pred,
            }
            for src, ref, pred in list(zip(sources, references, predictions))[:10]
        ]
    }
    
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"Model: {model_name}")
    print(f"BLEU Score: {bleu_result['score']:.2f}")
    print(f"Translation Speed: {sentences_per_sec:.2f} sent/s")
    print(f"GPU Memory Peak: {mem_peak:.2f} GB" if device == "cuda" else "CPU mode")
    print(f"{'='*70}\n")
    
    return results

def save_results(results: dict, output_file: str):
    """Save results to JSON file"""
    os.makedirs(os.path.dirname(output_file) or '.', exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"Results saved to: {output_file}\n")

def main():
    parser = argparse.ArgumentParser(description="Quick inference test for NLLB models")
    parser.add_argument('--model', type=str, default='facebook/nllb-200-distilled-600M',
                       help='Model name or path')
    parser.add_argument('--test', type=str, default='../data/final/nllb_test.jsonl',
                       help='Test data file')
    parser.add_argument('--num-samples', type=int, default=100,
                       help='Number of test samples (default: 100)')
    parser.add_argument('--batch-size', type=int, default=8,
                       help='Batch size for inference')
    parser.add_argument('--num-beams', type=int, default=5,
                       help='Number of beams for generation')
    parser.add_argument('--output', type=str, default=None,
                       help='Output JSON file for results')
    
    args = parser.parse_args()
    
    # Run test
    results = run_inference_test(
        model_name=args.model,
        test_file=args.test,
        num_samples=args.num_samples,
        batch_size=args.batch_size,
        num_beams=args.num_beams,
    )
    
    # Save results if output specified
    if args.output:
        save_results(results, args.output)
    else:
        # Auto-generate filename
        model_short = args.model.split('/')[-1]
        output_file = f"../results/inference_test_{model_short}.json"
        save_results(results, output_file)

if __name__ == "__main__":
    main()
