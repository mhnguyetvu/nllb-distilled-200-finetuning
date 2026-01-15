#!/usr/bin/env python3
"""
Analyze evaluation results to understand BLEU discrepancy
"""

import json
import argparse
from collections import defaultdict
from typing import List, Tuple
import difflib

def load_eval_results(file_path: str) -> dict:
    """Load evaluation results JSON"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def analyze_translation_differences(results: dict, num_examples: int = 50):
    """Analyze differences between predictions and references"""
    
    samples = results['samples']
    
    print("\n" + "="*80)
    print("TRANSLATION QUALITY ANALYSIS")
    print("="*80)
    
    # Categorize translations by similarity
    perfect_matches = []
    close_matches = []
    different_style = []
    wrong_translations = []
    
    for i, sample in enumerate(samples[:num_examples]):
        src = sample['source']
        ref = sample['reference']
        pred = sample['prediction']
        
        # Normalize for comparison
        ref_norm = ref.lower().strip()
        pred_norm = pred.lower().strip()
        
        # Calculate similarity ratio
        similarity = difflib.SequenceMatcher(None, ref_norm, pred_norm).ratio()
        
        if similarity >= 0.95:
            perfect_matches.append((i, src, ref, pred, similarity))
        elif similarity >= 0.7:
            close_matches.append((i, src, ref, pred, similarity))
        elif similarity >= 0.4:
            different_style.append((i, src, ref, pred, similarity))
        else:
            wrong_translations.append((i, src, ref, pred, similarity))
    
    # Print statistics
    print(f"\n📊 SIMILARITY DISTRIBUTION (first {num_examples} examples):")
    print(f"  ✅ Perfect matches (≥95% similar): {len(perfect_matches)} ({len(perfect_matches)/num_examples*100:.1f}%)")
    print(f"  🟢 Close matches (70-95% similar): {len(close_matches)} ({len(close_matches)/num_examples*100:.1f}%)")
    print(f"  🟡 Different style (40-70% similar): {len(different_style)} ({len(different_style)/num_examples*100:.1f}%)")
    print(f"  🔴 Wrong translations (<40% similar): {len(wrong_translations)} ({len(wrong_translations)/num_examples*100:.1f}%)")
    
    # Show examples from each category
    print("\n" + "-"*80)
    print("🔴 PROBLEMATIC TRANSLATIONS (Low Similarity):")
    print("-"*80)
    for i, src, ref, pred, sim in wrong_translations[:10]:
        print(f"\nExample {i+1} (Similarity: {sim:.2%}):")
        print(f"  Source:     {src}")
        print(f"  Reference:  {ref}")
        print(f"  Prediction: {pred}")
    
    print("\n" + "-"*80)
    print("🟡 DIFFERENT STYLE (Medium Similarity):")
    print("-"*80)
    for i, src, ref, pred, sim in different_style[:10]:
        print(f"\nExample {i+1} (Similarity: {sim:.2%}):")
        print(f"  Source:     {src}")
        print(f"  Reference:  {ref}")
        print(f"  Prediction: {pred}")
    
    print("\n" + "-"*80)
    print("🟢 CLOSE MATCHES (High Similarity):")
    print("-"*80)
    for i, src, ref, pred, sim in close_matches[:5]:
        print(f"\nExample {i+1} (Similarity: {sim:.2%}):")
        print(f"  Source:     {src}")
        print(f"  Reference:  {ref}")
        print(f"  Prediction: {pred}")
    
    return {
        'perfect_matches': len(perfect_matches),
        'close_matches': len(close_matches),
        'different_style': len(different_style),
        'wrong_translations': len(wrong_translations)
    }

def analyze_reference_quality(results: dict, num_check: int = 100):
    """Analyze if references are consistent and high quality"""
    
    print("\n" + "="*80)
    print("REFERENCE QUALITY ANALYSIS")
    print("="*80)
    
    samples = results['samples'][:num_check]
    
    # Check for common issues
    issues = defaultdict(list)
    
    for i, sample in enumerate(samples):
        ref = sample['reference']
        
        # Check for very short references (might be incomplete)
        if len(ref.split()) < 3:
            issues['too_short'].append((i, sample['source'], ref))
        
        # Check for references with mixed languages
        if any(ord(char) > 0x1100 and ord(char) < 0x11FF for char in ref):  # Korean characters
            issues['mixed_language'].append((i, sample['source'], ref))
        
        # Check for references that look like literal translations
        literal_markers = ['các', 'những', 'một', 'cái', 'này', 'đó']
        if sum(1 for marker in literal_markers if marker in ref.lower()) >= 3:
            issues['potentially_literal'].append((i, sample['source'], ref))
    
    print(f"\n🔍 Potential Reference Issues (checked {num_check} samples):")
    print(f"  • Too short (<3 words): {len(issues['too_short'])}")
    print(f"  • Mixed language: {len(issues['mixed_language'])}")
    print(f"  • Potentially literal: {len(issues['potentially_literal'])}")
    
    if issues['too_short']:
        print("\n⚠️  Short references (might be incomplete):")
        for i, src, ref in issues['too_short'][:5]:
            print(f"  [{i+1}] '{src}' → '{ref}'")
    
    if issues['mixed_language']:
        print("\n⚠️  Mixed language references:")
        for i, src, ref in issues['mixed_language'][:5]:
            print(f"  [{i+1}] '{src}' → '{ref}'")
    
    return issues

def compare_with_baseline(finetuned_file: str, baseline_file: str = None):
    """Compare finetuned model with baseline"""
    
    finetuned = load_eval_results(finetuned_file)
    
    print("\n" + "="*80)
    print("MODEL COMPARISON")
    print("="*80)
    
    if baseline_file:
        baseline = load_eval_results(baseline_file)
        
        print(f"\n📊 Metric Comparison:")
        print(f"{'Metric':<15} {'Baseline':<15} {'Finetuned':<15} {'Delta':<15}")
        print("-"*60)
        
        for metric in ['bleu', 'chrf', 'ter', 'comet']:
            if metric in finetuned['metrics'] and metric in baseline['metrics']:
                fine_val = finetuned['metrics'][metric]
                base_val = baseline['metrics'][metric]
                
                if isinstance(fine_val, dict):
                    fine_val = fine_val['score']
                    base_val = base_val['score']
                
                delta = fine_val - base_val
                delta_str = f"{delta:+.2f}"
                
                print(f"{metric.upper():<15} {base_val:<15.2f} {fine_val:<15.2f} {delta_str:<15}")
    else:
        print("\n📊 Finetuned Model Metrics:")
        for metric, value in finetuned['metrics'].items():
            if isinstance(value, dict):
                print(f"  {metric.upper()}: {value['score']:.2f}")
            else:
                print(f"  {metric.upper()}: {value:.2f}")
        
        print("\n⚠️  No baseline file provided. Run this to compare:")
        print("  python evaluate_model.py --model facebook/nllb-200-distilled-600M \\")
        print("      --test ../data/final_semantic/nllb_test.jsonl \\")
        print("      --output ../results/baseline_eval.json")

def main():
    parser = argparse.ArgumentParser(description="Analyze evaluation results")
    parser.add_argument('--finetuned', type=str, required=True, help='Finetuned model evaluation JSON')
    parser.add_argument('--baseline', type=str, help='Baseline model evaluation JSON (optional)')
    parser.add_argument('--num-examples', type=int, default=100, help='Number of examples to analyze')
    
    args = parser.parse_args()
    
    # Load results
    results = load_eval_results(args.finetuned)
    
    print(f"\n📁 Analyzing: {args.finetuned}")
    print(f"   Model: {results['model']}")
    print(f"   Test size: {results['test_size']}")
    print(f"   BLEU: {results['metrics']['bleu']['score'] if isinstance(results['metrics']['bleu'], dict) else results['metrics']['bleu']:.2f}")
    print(f"   COMET: {results['metrics'].get('comet', 'N/A')}")
    
    # Analyze translations
    stats = analyze_translation_differences(results, args.num_examples)
    
    # Analyze reference quality
    ref_issues = analyze_reference_quality(results, args.num_examples)
    
    # Compare with baseline if provided
    compare_with_baseline(args.finetuned, args.baseline)
    
    # Summary and recommendations
    print("\n" + "="*80)
    print("💡 RECOMMENDATIONS")
    print("="*80)
    
    wrong_pct = stats['wrong_translations'] / args.num_examples * 100
    style_pct = stats['different_style'] / args.num_examples * 100
    
    if wrong_pct > 20:
        print("\n🔴 High number of wrong translations (>20%):")
        print("   → Need more training data or better quality filtering")
        print("   → Consider increasing semantic similarity threshold to 0.75")
    
    if style_pct > 30:
        print("\n🟡 Many style mismatches (>30%):")
        print("   → References might have idiomatic translations")
        print("   → Model is translating more literally")
        print("   → COMET score is more reliable than BLEU for this case")
    
    if results['metrics'].get('comet', 0) > 0.8:
        print("\n✅ High COMET score (>0.8):")
        print("   → Translations are semantically correct")
        print("   → Low BLEU might be due to reference quality, not model quality")
    
    print("\n📋 Next Steps:")
    print("   1. Run baseline evaluation for comparison")
    print("   2. Manually review problematic translations")
    print("   3. Consider retraining with stricter threshold (0.75)")
    print("   4. Check if references are professionally translated or auto-aligned")

if __name__ == "__main__":
    main()
