#!/usr/bin/env python3
"""
Evaluation Script for NLLB Translation Models
Supports both pretrained and finetuned models

Usage:
    # Evaluate pretrained model
    python evaluate_model.py \
        --model facebook/nllb-200-distilled-600M \
        --test data/final/nllb_test.jsonl \
        --output results/pretrained_eval.json

    # Evaluate finetuned model
    python evaluate_model.py \
        --model outputs/nllb-ko-vi-finetuned \
        --test data/final/nllb_test.jsonl \
        --output results/finetuned_eval.json
"""

import os
import json
import logging
from typing import List, Dict, Any
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import evaluate
from tqdm import tqdm
import time

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_test_data(test_file: str, src_key: str = 'kor_Hang', tgt_key: str = 'vie_Latn') -> Dict[str, List[str]]:
    """Load test data from JSONL file"""
    sources = []
    references = []
    
    with open(test_file, 'r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line)
            translation = item['translation']
            sources.append(translation[src_key])
            references.append(translation[tgt_key])
    
    logger.info(f"Loaded {len(sources)} test examples from {test_file}")
    return {
        'sources': sources,
        'references': references
    }


def translate_batch(
    model,
    tokenizer,
    sources: List[str],
    src_lang: str,
    tgt_lang: str,
    batch_size: int = 8,
    num_beams: int = 5,
    max_length: int = 256,
) -> List[str]:
    """Translate a batch of source sentences"""
    predictions = []
    
    # Set source language
    tokenizer.src_lang = src_lang
    
    # Get target language token ID
    tgt_token_id = tokenizer.convert_tokens_to_ids(tgt_lang)
    
    for i in tqdm(range(0, len(sources), batch_size), desc="Translating"):
        batch_sources = sources[i:i+batch_size]
        
        # Tokenize
        inputs = tokenizer(
            batch_sources,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        ).to(model.device)
        
        # Generate
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                forced_bos_token_id=tgt_token_id,
                max_length=max_length,
                num_beams=num_beams,
                early_stopping=True,
            )
        
        # Decode
        batch_predictions = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        predictions.extend(batch_predictions)
    
    return predictions


def compute_metrics(predictions: List[str], references: List[str], sources: List[str]) -> Dict[str, Any]:
    """Compute evaluation metrics"""
    metrics_results = {}
    
    # BLEU
    try:
        bleu = evaluate.load("sacrebleu")
        bleu_result = bleu.compute(predictions=predictions, references=[[r] for r in references])
        metrics_results['bleu'] = {
            'score': bleu_result['score'],
            'precisions': bleu_result['precisions'],
        }
        logger.info(f"✓ BLEU: {bleu_result['score']:.2f}")
    except Exception as e:
        logger.error(f"✗ BLEU computation failed: {e}")
    
    # chrF
    try:
        chrf = evaluate.load("chrf")
        chrf_result = chrf.compute(predictions=predictions, references=references)
        metrics_results['chrf'] = chrf_result['score']
        logger.info(f"✓ chrF: {chrf_result['score']:.2f}")
    except Exception as e:
        logger.error(f"✗ chrF computation failed: {e}")
    
    # TER (Translation Error Rate)
    try:
        ter = evaluate.load("ter")
        ter_result = ter.compute(predictions=predictions, references=references)
        metrics_results['ter'] = ter_result['score']
        logger.info(f"✓ TER: {ter_result['score']:.2f}")
    except Exception as e:
        logger.error(f"✗ TER computation failed: {e}")
    
    # COMET (Neural metric - requires more resources)
    try:
        comet = evaluate.load("comet")
        comet_result = comet.compute(
            predictions=predictions,
            references=references,
            sources=sources
        )
        metrics_results['comet'] = comet_result['mean_score']
        logger.info(f"✓ COMET: {comet_result['mean_score']:.4f}")
    except Exception as e:
        logger.warning(f"⚠ COMET computation skipped: {e}")
    
    return metrics_results


def evaluate_translation(
    model_path: str,
    test_file: str,
    output_file: str,
    src_lang: str = "kor_Hang",
    tgt_lang: str = "vie_Latn",
    batch_size: int = 8,
    num_beams: int = 5,
    num_samples: int = 10,
) -> Dict[str, Any]:
    """Main evaluation function"""
    
    # Load test data
    test_data = load_test_data(test_file, src_lang, tgt_lang)
    sources = test_data['sources']
    references = test_data['references']
    
    # Load model and tokenizer
    logger.info(f"Loading model: {model_path}")
    
    # Check if it's a local path (starts with . or / or is a valid directory)
    import os
    is_local = model_path.startswith('.') or model_path.startswith('/') or os.path.isdir(model_path)
    
    if is_local:
        logger.info(f"Loading from local path: {model_path}")
        tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_path, local_files_only=True)
    else:
        logger.info(f"Loading from HuggingFace Hub: {model_path}")
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
    
    # Move to GPU if available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()
    
    if device == "cuda":
        gpu_name = torch.cuda.get_device_name(0)
        logger.info(f"✓ Using GPU: {gpu_name}")
    else:
        logger.warning("⚠ Running on CPU (slower)")
    
    # Translate
    logger.info(f"Translating {len(sources)} sentences...")
    start_time = time.time()
    
    predictions = translate_batch(
        model=model,
        tokenizer=tokenizer,
        sources=sources,
        src_lang=src_lang,
        tgt_lang=tgt_lang,
        batch_size=batch_size,
        num_beams=num_beams,
    )
    
    elapsed_time = time.time() - start_time
    logger.info(f"Translation completed in {elapsed_time:.1f}s ({len(sources)/elapsed_time:.1f} sent/s)")
    
    # Compute metrics
    logger.info("Computing evaluation metrics...")
    metrics = compute_metrics(predictions, references, sources)
    
    # Prepare results
    results = {
        'model': model_path,
        'test_file': test_file,
        'test_size': len(sources),
        'translation_time': elapsed_time,
        'sentences_per_second': len(sources) / elapsed_time,
        'metrics': metrics,
        'samples': [
            {
                'source': src,
                'reference': ref,
                'prediction': pred,
            }
            for src, ref, pred in list(zip(sources, references, predictions))[:num_samples]
        ]
    }
    
    # Save results
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    logger.info(f"Results saved to: {output_file}")
    
    # Print summary
    logger.info("\n" + "="*60)
    logger.info("EVALUATION SUMMARY")
    logger.info("="*60)
    logger.info(f"Model: {model_path}")
    logger.info(f"Test size: {len(sources)} sentences")
    logger.info(f"Translation speed: {len(sources)/elapsed_time:.1f} sent/s")
    logger.info("-"*60)
    logger.info("Metrics:")
    for metric_name, metric_value in metrics.items():
        if isinstance(metric_value, dict):
            logger.info(f"  {metric_name.upper()}: {metric_value['score']:.2f}")
        else:
            logger.info(f"  {metric_name.upper()}: {metric_value:.2f}")
    logger.info("="*60)
    
    # Print sample translations
    logger.info("\nSample translations:")
    for i, sample in enumerate(results['samples'][:3], 1):
        logger.info(f"\nExample {i}:")
        logger.info(f"  Source:     {sample['source']}")
        logger.info(f"  Reference:  {sample['reference']}")
        logger.info(f"  Prediction: {sample['prediction']}")
    
    return results


def compare_models(eval_file1: str, eval_file2: str, output_file: str = None):
    """Compare evaluation results from two models"""
    
    # Load results
    with open(eval_file1, 'r', encoding='utf-8') as f:
        results1 = json.load(f)
    
    with open(eval_file2, 'r', encoding='utf-8') as f:
        results2 = json.load(f)
    
    # Compare metrics
    comparison = {
        'model1': {
            'name': results1['model'],
            'metrics': results1['metrics']
        },
        'model2': {
            'name': results2['model'],
            'metrics': results2['metrics']
        },
        'improvements': {}
    }
    
    # Calculate improvements
    for metric_name in results1['metrics']:
        if metric_name in results2['metrics']:
            val1 = results1['metrics'][metric_name]
            val2 = results2['metrics'][metric_name]
            
            if isinstance(val1, dict):
                val1 = val1['score']
                val2 = val2['score']
            
            # For TER, lower is better
            if metric_name == 'ter':
                improvement = val1 - val2
            else:
                improvement = val2 - val1
            
            comparison['improvements'][metric_name] = {
                'model1': val1,
                'model2': val2,
                'delta': improvement,
                'percent': (improvement / val1 * 100) if val1 != 0 else 0
            }
    
    # Print comparison
    logger.info("\n" + "="*60)
    logger.info("MODEL COMPARISON")
    logger.info("="*60)
    logger.info(f"Model 1: {results1['model']}")
    logger.info(f"Model 2: {results2['model']}")
    logger.info("-"*60)
    
    for metric_name, values in comparison['improvements'].items():
        logger.info(f"\n{metric_name.upper()}:")
        logger.info(f"  Model 1: {values['model1']:.2f}")
        logger.info(f"  Model 2: {values['model2']:.2f}")
        logger.info(f"  Delta:   {values['delta']:+.2f} ({values['percent']:+.1f}%)")
    
    logger.info("="*60)
    
    # Save comparison
    if output_file:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(comparison, f, ensure_ascii=False, indent=2)
        logger.info(f"Comparison saved to: {output_file}")
    
    return comparison


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate NLLB translation model")
    parser.add_argument('--model', type=str, required=True, help='Model path or HuggingFace model name')
    parser.add_argument('--test', type=str, default='data/final/nllb_test.jsonl', help='Test data file')
    parser.add_argument('--output', type=str, required=True, help='Output JSON file for results')
    parser.add_argument('--src-lang', type=str, default='kor_Hang', help='Source language code')
    parser.add_argument('--tgt-lang', type=str, default='vie_Latn', help='Target language code')
    parser.add_argument('--batch-size', type=int, default=8, help='Batch size for translation')
    parser.add_argument('--num-beams', type=int, default=5, help='Number of beams for generation')
    parser.add_argument('--num-samples', type=int, default=10, help='Number of sample translations to save')
    
    # Comparison mode
    parser.add_argument('--compare', type=str, nargs=2, metavar=('FILE1', 'FILE2'),
                       help='Compare two evaluation result files')
    parser.add_argument('--compare-output', type=str, help='Output file for comparison results')
    
    args = parser.parse_args()
    
    if args.compare:
        # Comparison mode
        compare_models(args.compare[0], args.compare[1], args.compare_output)
    else:
        # Evaluation mode
        evaluate_translation(
            model_path=args.model,
            test_file=args.test,
            output_file=args.output,
            src_lang=args.src_lang,
            tgt_lang=args.tgt_lang,
            batch_size=args.batch_size,
            num_beams=args.num_beams,
            num_samples=args.num_samples,
        )


if __name__ == "__main__":
    main()
