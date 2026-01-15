#!/usr/bin/env python3
"""
Enhanced OPUS Quality Filtering with Multiple Improvements
- Stricter length ratio checks
- Language ID verification (fasttext or langdetect)
- Punctuation/number consistency
- Boilerplate/subtitle noise removal
- Near-duplicate detection
- Multiple semantic similarity thresholds (sweep mode)
"""

import json
import re
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Set
from collections import defaultdict
import torch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import numpy as np
import unicodedata
import gc

# Try to import langdetect (lightweight) or fasttext
try:
    from langdetect import detect_langs, LangDetectException
    LANGDETECT_AVAILABLE = True
except ImportError:
    LANGDETECT_AVAILABLE = False
    print("⚠️  langdetect not available. Install: pip install langdetect")

# For near-duplicate detection
try:
    from datasketch import MinHash, MinHashLSH
    MINHASH_AVAILABLE = True
except ImportError:
    MINHASH_AVAILABLE = False
    print("⚠️  datasketch not available. Install: pip install datasketch")


class EnhancedFilter:
    """Enhanced filtering with all improvements"""
    
    # Boilerplate patterns to remove
    BOILERPLATE_PATTERNS = [
        r'\[.*?\]',  # [music], [laughs], etc.
        r'\(.*?\)',  # (cười), (applause)
        r'♪+',       # Music notes
        r'www\.',    # URLs
        r'http[s]?://',  # URLs
        r'subscribe',  # YouTube noise
        r'episode \d+',  # Episode markers
        r'^\s*[A-Z]:\s*',  # Speaker labels (A:, B:)
        r'^\d{1,2}:\d{2}',  # Timestamps
        r'^-+\s*$',  # Dashes only
    ]
    
    # Very short/useless phrases
    USELESS_PHRASES_KO = ['ㅋㅋ', 'ㅎㅎ', '응', '아', '예', '네']
    USELESS_PHRASES_VI = ['ờ', 'ừ', 'à', 'ô', 'à']
    
    def __init__(self, 
                 use_langid: bool = True,
                 use_near_dedup: bool = True,
                 langid_threshold: float = 0.8,
                 length_ratio_min: float = 0.4,
                 length_ratio_max: float = 2.5,
                 min_length: int = 4,
                 max_number_diff: float = 0.3):
        
        self.use_langid = use_langid and LANGDETECT_AVAILABLE
        self.use_near_dedup = use_near_dedup and MINHASH_AVAILABLE
        self.langid_threshold = langid_threshold
        self.length_ratio_min = length_ratio_min
        self.length_ratio_max = length_ratio_max
        self.min_length = min_length
        self.max_number_diff = max_number_diff
        
        # For near-duplicate detection
        if self.use_near_dedup:
            self.lsh = MinHashLSH(threshold=0.85, num_perm=128)
            self.seen_hashes = {}
        
        # Compile patterns
        self.boilerplate_regex = re.compile('|'.join(self.BOILERPLATE_PATTERNS), re.IGNORECASE)
        
        self.stats = defaultdict(int)
    
    def normalize_text(self, text: str) -> str:
        """Normalize text for deduplication"""
        # Lowercase
        text = text.lower()
        # Remove extra spaces
        text = re.sub(r'\s+', ' ', text)
        # Normalize quotes
        text = text.replace('"', '"').replace('"', '"').replace("'", "'").replace("'", "'")
        # Strip
        text = text.strip()
        return text
    
    def check_language_id(self, text: str, expected_lang: str) -> bool:
        """Check if text is in expected language using langdetect"""
        if not self.use_langid:
            return True
        
        try:
            langs = detect_langs(text)
            if not langs:
                return False
            
            # Map expected language codes
            lang_map = {
                'ko': 'ko',
                'vi': 'vi',
                'kor_Hang': 'ko',
                'vie_Latn': 'vi'
            }
            
            expected = lang_map.get(expected_lang, expected_lang)
            
            # Check if expected language is detected with high confidence
            for lang in langs:
                if lang.lang == expected and lang.prob >= self.langid_threshold:
                    return True
            
            return False
            
        except LangDetectException:
            # Text too short or unclear - be lenient
            return True
    
    def has_boilerplate(self, text: str) -> bool:
        """Check if text contains boilerplate/noise"""
        # Check patterns
        if self.boilerplate_regex.search(text):
            return True
        
        # Check useless phrases
        text_lower = text.lower()
        if text_lower in self.USELESS_PHRASES_KO or text_lower in self.USELESS_PHRASES_VI:
            return True
        
        return False
    
    def check_length_ratio(self, text1: str, text2: str) -> bool:
        """Check if length ratio is within acceptable range"""
        len1 = len(text1.split())
        len2 = len(text2.split())
        
        if len1 == 0 or len2 == 0:
            return False
        
        ratio = len2 / len1
        return self.length_ratio_min <= ratio <= self.length_ratio_max
    
    def check_number_consistency(self, text1: str, text2: str) -> bool:
        """Check if numbers are consistent between texts"""
        # Extract numbers
        nums1 = re.findall(r'\d+', text1)
        nums2 = re.findall(r'\d+', text2)
        
        # If no numbers, pass
        if not nums1 and not nums2:
            return True
        
        # If one has numbers but other doesn't, check difference
        diff = abs(len(nums1) - len(nums2))
        total = max(len(nums1), len(nums2))
        
        if total == 0:
            return True
        
        return (diff / total) <= self.max_number_diff
    
    def check_punctuation_density(self, text: str) -> bool:
        """Check if text has abnormal punctuation density"""
        if len(text) == 0:
            return False
        
        # Count punctuation
        punct_count = sum(1 for c in text if unicodedata.category(c).startswith('P'))
        density = punct_count / len(text)
        
        # Reject if > 30% punctuation
        return density < 0.3
    
    def get_minhash(self, text: str) -> MinHash:
        """Create MinHash for near-duplicate detection"""
        m = MinHash(num_perm=128)
        # Tokenize by words
        words = self.normalize_text(text).split()
        for word in words:
            m.update(word.encode('utf-8'))
        return m
    
    def is_near_duplicate(self, text: str, text_id: str) -> bool:
        """Check if text is near-duplicate of previously seen text"""
        if not self.use_near_dedup:
            return False
        
        minhash = self.get_minhash(text)
        
        # Check against LSH
        result = self.lsh.query(minhash)
        if result:
            return True
        
        # Add to LSH
        self.lsh.insert(text_id, minhash)
        self.seen_hashes[text_id] = minhash
        
        return False
    
    def filter_pair(self, korean: str, vietnamese: str, pair_id: str) -> Tuple[bool, str]:
        """
        Filter a single pair through all checks
        Returns (keep, reason)
        """
        self.stats['total'] += 1
        
        # 1. Length check
        if len(korean) < self.min_length or len(vietnamese) < self.min_length:
            self.stats['too_short'] += 1
            return False, "too_short"
        
        # 2. Boilerplate check
        if self.has_boilerplate(korean) or self.has_boilerplate(vietnamese):
            self.stats['boilerplate'] += 1
            return False, "boilerplate"
        
        # 3. Language ID check
        if not self.check_language_id(korean, 'ko'):
            self.stats['wrong_lang_ko'] += 1
            return False, "wrong_lang_ko"
        
        if not self.check_language_id(vietnamese, 'vi'):
            self.stats['wrong_lang_vi'] += 1
            return False, "wrong_lang_vi"
        
        # 4. Length ratio check
        if not self.check_length_ratio(korean, vietnamese):
            self.stats['length_ratio'] += 1
            return False, "length_ratio"
        
        # 5. Number consistency check
        if not self.check_number_consistency(korean, vietnamese):
            self.stats['number_inconsistent'] += 1
            return False, "number_inconsistent"
        
        # 6. Punctuation density check
        if not self.check_punctuation_density(korean) or not self.check_punctuation_density(vietnamese):
            self.stats['punct_density'] += 1
            return False, "punct_density"
        
        # 7. Near-duplicate check
        if self.is_near_duplicate(korean, pair_id):
            self.stats['near_duplicate'] += 1
            return False, "near_duplicate"
        
        self.stats['passed'] += 1
        return True, "passed"


def semantic_filter_with_threshold_preloaded(
    data: List[dict],
    model: SentenceTransformer,
    output_file: str,
    threshold: float,
    batch_size: int = 32,
    device: str = 'cuda'
) -> Dict[str, int]:
    """Apply semantic similarity filter with preloaded model and data"""
    
    print(f"\n{'='*70}")
    print(f"SEMANTIC FILTER: Threshold {threshold:.2f}")
    print(f"{'='*70}")
    
    print(f"Total pairs: {len(data):,}")
    
    # Process in batches
    kept_pairs = []
    similarities = []
    
    for i in tqdm(range(0, len(data), batch_size), desc="Computing similarities"):
        batch = data[i:i+batch_size]
        
        korean_texts = [item['translation']['kor_Hang'] for item in batch]
        vietnamese_texts = [item['translation']['vie_Latn'] for item in batch]
        
        # Encode
        with torch.no_grad():
            korean_embeddings = model.encode(korean_texts, convert_to_tensor=True, show_progress_bar=False)
            vietnamese_embeddings = model.encode(vietnamese_texts, convert_to_tensor=True, show_progress_bar=False)
        
        # Compute cosine similarity
        batch_similarities = torch.nn.functional.cosine_similarity(
            korean_embeddings, vietnamese_embeddings
        ).cpu().numpy()
        
        # Filter by threshold
        for j, sim in enumerate(batch_similarities):
            similarities.append(sim)
            if sim >= threshold:
                kept_pairs.append(batch[j])
    
    # Save
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for pair in kept_pairs:
            f.write(json.dumps(pair, ensure_ascii=False) + '\n')
    
    # Statistics
    stats = {
        'input': len(data),
        'output': len(kept_pairs),
        'threshold': threshold,
        'retention': len(kept_pairs) / len(data) * 100,
        'mean_similarity': float(np.mean(similarities)),
        'median_similarity': float(np.median(similarities)),
        'std_similarity': float(np.std(similarities))
    }
    
    print(f"\n{'='*70}")
    print(f"RESULTS (Threshold {threshold:.2f}):")
    print(f"  Input:  {stats['input']:,} pairs")
    print(f"  Output: {stats['output']:,} pairs")
    print(f"  Retention: {stats['retention']:.1f}%")
    print(f"  Mean similarity: {stats['mean_similarity']:.3f}")
    print(f"  Median similarity: {stats['median_similarity']:.3f}")
    print(f"  Std similarity: {stats['std_similarity']:.3f}")
    print(f"{'='*70}\n")
    
    return stats


def semantic_filter_with_threshold(
    input_file: str,
    output_file: str,
    threshold: float,
    batch_size: int = 32,
    device: str = None
) -> Dict[str, int]:
    """Apply semantic similarity filter with specific threshold"""
    
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f"\n{'='*70}")
    print(f"SEMANTIC FILTER: Threshold {threshold:.2f}")
    print(f"{'='*70}")
    
    # Load LaBSE model
    print("Loading LaBSE model...")
    model = SentenceTransformer('sentence-transformers/LaBSE')
    model = model.to(device)
    
    # Load data
    print(f"Loading data from: {input_file}")
    with open(input_file, 'r', encoding='utf-8') as f:
        data = [json.loads(line) for line in f]
    
    print(f"Total pairs: {len(data):,}")
    
    # Process in batches
    kept_pairs = []
    similarities = []
    
    for i in tqdm(range(0, len(data), batch_size), desc="Computing similarities"):
        batch = data[i:i+batch_size]
        
        korean_texts = [item['translation']['kor_Hang'] for item in batch]
        vietnamese_texts = [item['translation']['vie_Latn'] for item in batch]
        
        # Encode
        with torch.no_grad():
            korean_embeddings = model.encode(korean_texts, convert_to_tensor=True, show_progress_bar=False)
            vietnamese_embeddings = model.encode(vietnamese_texts, convert_to_tensor=True, show_progress_bar=False)
        
        # Compute cosine similarity
        batch_similarities = torch.nn.functional.cosine_similarity(
            korean_embeddings, vietnamese_embeddings
        ).cpu().numpy()
        
        # Filter by threshold
        for j, sim in enumerate(batch_similarities):
            similarities.append(sim)
            if sim >= threshold:
                kept_pairs.append(batch[j])
    
    # Save
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for pair in kept_pairs:
            f.write(json.dumps(pair, ensure_ascii=False) + '\n')
    
    # Statistics
    stats = {
        'input': len(data),
        'output': len(kept_pairs),
        'threshold': threshold,
        'retention': len(kept_pairs) / len(data) * 100,
        'mean_similarity': float(np.mean(similarities)),
        'median_similarity': float(np.median(similarities)),
        'std_similarity': float(np.std(similarities))
    }
    
    print(f"\n{'='*70}")
    print(f"RESULTS (Threshold {threshold:.2f}):")
    print(f"  Input:  {stats['input']:,} pairs")
    print(f"  Output: {stats['output']:,} pairs")
    print(f"  Retention: {stats['retention']:.1f}%")
    print(f"  Mean similarity: {stats['mean_similarity']:.3f}")
    print(f"  Median similarity: {stats['median_similarity']:.3f}")
    print(f"  Std similarity: {stats['std_similarity']:.3f}")
    print(f"{'='*70}\n")
    
    return stats


def sweep_semantic_thresholds(
    input_file: str,
    output_dir: str,
    thresholds: List[float] = [0.65, 0.70, 0.75, 0.80],
    batch_size: int = 32
):
    """Sweep multiple semantic thresholds and create datasets"""
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Load model ONCE and reuse for all thresholds
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\n{'='*70}")
    print("Loading LaBSE model (will be reused for all thresholds)...")
    print(f"Device: {device}")
    print(f"{'='*70}\n")
    
    model = SentenceTransformer('sentence-transformers/LaBSE')
    model = model.to(device)
    
    # Load data ONCE
    print(f"Loading data from: {input_file}")
    with open(input_file, 'r', encoding='utf-8') as f:
        data = [json.loads(line) for line in f]
    print(f"Total pairs: {len(data):,}\n")
    
    all_stats = {}
    
    for threshold in thresholds:
        output_file = output_path / f"semantic_{int(threshold*100)}.jsonl"
        stats = semantic_filter_with_threshold_preloaded(
            data=data,
            model=model,
            output_file=str(output_file),
            threshold=threshold,
            batch_size=batch_size,
            device=device
        )
        all_stats[threshold] = stats
        
        # Free memory between thresholds
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
    
    # Save summary
    summary_file = output_path / "sweep_summary.json"
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(all_stats, f, indent=2)
    
    # Print comparison
    print("\n" + "="*70)
    print("THRESHOLD SWEEP SUMMARY")
    print("="*70)
    print(f"{'Threshold':<12} {'Pairs':<10} {'Retention':<12} {'Mean Sim':<12}")
    print("-"*70)
    
    for threshold in thresholds:
        stats = all_stats[threshold]
        print(f"{threshold:<12.2f} {stats['output']:<10,} {stats['retention']:<12.1f}% {stats['mean_similarity']:<12.3f}")
    
    print("="*70)
    print(f"\n✅ Sweep complete! Files saved to: {output_dir}")
    print(f"📊 Summary: {summary_file}")


def main():
    parser = argparse.ArgumentParser(description="Enhanced OPUS filtering with multiple improvements")
    
    # Input/output
    parser.add_argument('--input', type=str, required=True, help='Input JSONL file')
    parser.add_argument('--output', type=str, help='Output JSONL file (for single filter)')
    parser.add_argument('--output-dir', type=str, help='Output directory (for sweep mode)')
    
    # Filter modes
    parser.add_argument('--mode', type=str, choices=['basic', 'semantic', 'sweep'], default='basic',
                       help='Filter mode: basic (enhanced filters), semantic (single threshold), sweep (multiple thresholds)')
    
    # Basic filter options
    parser.add_argument('--use-langid', action='store_true', default=True, help='Use language ID verification')
    parser.add_argument('--no-langid', action='store_false', dest='use_langid', help='Disable language ID')
    parser.add_argument('--use-near-dedup', action='store_true', default=True, help='Use near-duplicate detection')
    parser.add_argument('--no-near-dedup', action='store_false', dest='use_near_dedup', help='Disable near-dedup')
    parser.add_argument('--langid-threshold', type=float, default=0.8, help='Language ID confidence threshold')
    parser.add_argument('--length-ratio-min', type=float, default=0.4, help='Minimum length ratio')
    parser.add_argument('--length-ratio-max', type=float, default=2.5, help='Maximum length ratio')
    parser.add_argument('--min-length', type=int, default=4, help='Minimum text length (characters)')
    
    # Semantic filter options
    parser.add_argument('--threshold', type=float, default=0.70, help='Semantic similarity threshold (single mode)')
    parser.add_argument('--thresholds', type=float, nargs='+', default=[0.65, 0.70, 0.75, 0.80],
                       help='Thresholds for sweep mode')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size for semantic encoding')
    parser.add_argument('--device', type=str, default=None, help='Device (cuda/cpu)')
    
    args = parser.parse_args()
    
    if args.mode == 'basic':
        # Enhanced basic filtering
        if not args.output:
            parser.error("--output required for basic mode")
        
        print("\n" + "="*70)
        print("ENHANCED BASIC FILTERING")
        print("="*70)
        print(f"Input:  {args.input}")
        print(f"Output: {args.output}")
        print(f"Options:")
        print(f"  Language ID: {args.use_langid} (threshold: {args.langid_threshold})")
        print(f"  Near-dedup: {args.use_near_dedup}")
        print(f"  Length ratio: {args.length_ratio_min:.2f} - {args.length_ratio_max:.2f}")
        print(f"  Min length: {args.min_length} chars")
        print("="*70 + "\n")
        
        # Create filter
        filter_obj = EnhancedFilter(
            use_langid=args.use_langid,
            use_near_dedup=args.use_near_dedup,
            langid_threshold=args.langid_threshold,
            length_ratio_min=args.length_ratio_min,
            length_ratio_max=args.length_ratio_max,
            min_length=args.min_length
        )
        
        # Load data
        print("Loading data...")
        with open(args.input, 'r', encoding='utf-8') as f:
            data = [json.loads(line) for line in f]
        
        print(f"Total pairs: {len(data):,}")
        
        # Filter
        kept_pairs = []
        rejection_reasons = defaultdict(int)
        
        for i, item in enumerate(tqdm(data, desc="Filtering")):
            korean = item['translation']['kor_Hang']
            vietnamese = item['translation']['vie_Latn']
            
            keep, reason = filter_obj.filter_pair(korean, vietnamese, f"pair_{i}")
            
            if keep:
                kept_pairs.append(item)
            else:
                rejection_reasons[reason] += 1
        
        # Save
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(args.output, 'w', encoding='utf-8') as f:
            for pair in kept_pairs:
                f.write(json.dumps(pair, ensure_ascii=False) + '\n')
        
        # Print stats
        print("\n" + "="*70)
        print("FILTERING RESULTS")
        print("="*70)
        print(f"Input:  {len(data):,} pairs")
        print(f"Output: {len(kept_pairs):,} pairs")
        print(f"Retention: {len(kept_pairs)/len(data)*100:.1f}%")
        print("\nRejection reasons:")
        for reason, count in sorted(rejection_reasons.items(), key=lambda x: -x[1]):
            print(f"  {reason:<20}: {count:>8,} ({count/len(data)*100:>5.1f}%)")
        print("="*70)
        
    elif args.mode == 'semantic':
        # Single threshold semantic filter
        if not args.output:
            parser.error("--output required for semantic mode")
        
        semantic_filter_with_threshold(
            input_file=args.input,
            output_file=args.output,
            threshold=args.threshold,
            batch_size=args.batch_size,
            device=args.device
        )
        
    elif args.mode == 'sweep':
        # Sweep multiple thresholds
        if not args.output_dir:
            parser.error("--output-dir required for sweep mode")
        
        sweep_semantic_thresholds(
            input_file=args.input,
            output_dir=args.output_dir,
            thresholds=args.thresholds,
            batch_size=args.batch_size
        )


if __name__ == "__main__":
    main()
