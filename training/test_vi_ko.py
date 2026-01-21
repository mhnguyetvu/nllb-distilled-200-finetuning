#!/usr/bin/env python3
"""
Test script for Vietnamese-to-Korean (vi-ko) translation 
to check if the model can handle reverse translation.
"""

import torch
import json
import argparse
import time
from pathlib import Path
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

def run_vi_ko_test(
    model_name: str,
    test_file: str = None,
    num_samples: int = 10,
    batch_size: int = 1,
):
    print(f"\n{'='*70}")
    print(f"VIETNAMESE -> KOREAN TEST: {model_name}")
    print(f"{'='*70}\n")
    
    # Load model and tokenizer
    print(f"Loading model: {model_name}")
    start_load = time.time()
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()
    
    load_time = time.time() - start_load
    print(f"✓ Model loaded in {load_time:.1f}s on {device}\n")
    
    # Set languages for NLLB
    # For VI -> KO
    tokenizer.src_lang = "vie_Latn"
    tgt_lang_token = "kor_Hang"
    tgt_token_id = tokenizer.convert_tokens_to_ids(tgt_lang_token)
    
    # Prepare test data
    test_samples = []
    if test_file and Path(test_file).exists():
        print(f"Loading test samples from {test_file}...")
        with open(test_file, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if i >= num_samples:
                    break
                item = json.loads(line)
                test_samples.append({
                    'vi': item['translation']['vie_Latn'],
                    'ko_ref': item['translation']['kor_Hang']
                })
    else:
        # Default manual samples if no file provided
        print("Using manual test samples...")
        manual_data = [
            {"vi": "Chào buổi sáng, bạn khỏe không?", "ko_ref": "좋은 아침입니다, 어떻게 지내세요?"},
            {"vi": "Tôi yêu tiếng Việt và tiếng Hàn.", "ko_ref": "나는 베트남어와 한국어를 사랑합니다."},
            {"vi": "Hôm nay thời tiết rất đẹp.", "ko_ref": "오늘 날씨가 매우 좋습니다."},
            {"vi": "Bạn có thể giúp tôi dịch câu này không?", "ko_ref": "이 문장을 번역하는 것을 도와주실 수 있나요?"},
            {"vi": "Mô hình này hoạt động rất tốt.", "ko_ref": "이 모델은 매우 잘 작동합니다."}
        ]
        test_samples = []
        for item in manual_data[:num_samples]:
            test_samples.append({
                'vi': item['vi'],
                'ko_ref': item['ko_ref']
            })
            
    # Run translation
    print(f"Translating {len(test_samples)} sentences (vi -> ko)...\n")
    
    for i, sample in enumerate(test_samples):
        vi_text = sample['vi']
        ko_ref = sample['ko_ref']
        
        # Tokenize
        inputs = tokenizer(vi_text, return_tensors="pt").to(device)
        
        # Generate
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                forced_bos_token_id=tgt_token_id,
                max_length=256,
                num_beams=5,
            )
        
        # Decode
        prediction = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
        
        print(f"Example {i+1}:")
        print(f"  VI (Source):  {vi_text}")
        print(f"  KO (Ref):     {ko_ref}")
        print(f"  KO (Pred):    {prediction}")
        print("-" * 30)

def main():
    parser = argparse.ArgumentParser(description="Test vi-ko reverse translation")
    parser.add_argument('--model', type=str, default='facebook/nllb-200-distilled-600M',
                       help='Model name or path')
    parser.add_argument('--test', type=str, default=None,
                       help='Optional JSONL test file with vie_Latn/kor_Hang pairs')
    parser.add_argument('--num-samples', type=int, default=5,
                       help='Number of samples to test')
    
    args = parser.parse_args()
    
    # Try to auto-detect test file if not provided
    if args.test is None:
        possible_paths = [
            'data/final_semantic/nllb_test.jsonl',
            '../data/final_semantic/nllb_test.jsonl',
            'data/sweep/semantic_80/nllb_test.jsonl'
        ]
        for p in possible_paths:
            if Path(p).exists():
                args.test = p
                break
                
    run_vi_ko_test(
        model_name=args.model,
        test_file=args.test,
        num_samples=args.num_samples
    )

if __name__ == "__main__":
    main()
