#!/usr/bin/env python3
"""
Filter OPUS Dataset with Quality Criteria
- Length ratio check (0.5-2.0x)
- Korean Hangul content validation (>30%)
- Vietnamese Latin script validation
- Minimum length threshold
- Semantic similarity filtering (LaBSE)
- Deduplication
"""

import json
import re
import logging
from typing import List, Dict, Tuple
from collections import defaultdict
import numpy as np
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def check_korean_content(text: str, min_ratio: float = 0.3) -> bool:
    """Check if text has sufficient Korean Hangul characters"""
    if not text:
        return False
    
    hangul_chars = len(re.findall(r'[가-힣]', text))
    total_chars = len(text.replace(' ', ''))
    
    if total_chars == 0:
        return False
    
    ratio = hangul_chars / total_chars
    return ratio >= min_ratio


def check_vietnamese_content(text: str) -> bool:
    """Check if text contains Vietnamese characters"""
    if not text:
        return False
    
    # Vietnamese special characters
    viet_pattern = r'[àáảãạăằắẳẵặâầấẩẫậèéẻẽẹêềếểễệìíỉĩịòóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵđ]'
    
    # Check for Vietnamese chars OR Latin script with proper structure
    has_viet_chars = bool(re.search(viet_pattern, text.lower()))
    has_latin = bool(re.search(r'[a-z]', text.lower()))
    
    return has_viet_chars or has_latin


def check_length_ratio(text1: str, text2: str, min_ratio: float = 0.5, max_ratio: float = 2.0) -> bool:
    """Check if length ratio between two texts is within acceptable range"""
    len1, len2 = len(text1), len(text2)
    
    if len1 == 0 or len2 == 0:
        return False
    
    ratio = len1 / len2
    return min_ratio <= ratio <= max_ratio


def check_min_length(text: str, min_chars: int = 5) -> bool:
    """Check if text meets minimum length requirement"""
    return len(text.strip()) >= min_chars


def is_valid_pair(ko_text: str, vi_text: str) -> bool:
    """Check if a Korean-Vietnamese pair meets basic quality criteria"""
    
    # Check minimum length
    if not (check_min_length(ko_text) and check_min_length(vi_text)):
        return False
    
    # Check Korean content
    if not check_korean_content(ko_text):
        return False
    
    # Check Vietnamese content
    if not check_vietnamese_content(vi_text):
        return False
    
    # Check length ratio
    if not check_length_ratio(ko_text, vi_text):
        return False
    
    return True


def compute_semantic_similarity_batch(ko_texts: List[str], vi_texts: List[str], 
                                     model, batch_size: int = 32) -> np.ndarray:
    """Compute semantic similarity scores using LaBSE in batches"""
    similarities = []
    
    try:
        for i in range(0, len(ko_texts), batch_size):
            batch_ko = ko_texts[i:i+batch_size]
            batch_vi = vi_texts[i:i+batch_size]
            
            # Encode
            logger.debug(f"Encoding batch {i//batch_size + 1}/{(len(ko_texts) + batch_size - 1)//batch_size}")
            ko_emb = model.encode(batch_ko, convert_to_numpy=True, normalize_embeddings=True)
            vi_emb = model.encode(batch_vi, convert_to_numpy=True, normalize_embeddings=True)
            
            # Compute cosine similarity
            batch_sim = np.sum(ko_emb * vi_emb, axis=1)
            similarities.extend(batch_sim.tolist())
            
            # Progress update every 100 batches
            if (i // batch_size + 1) % 100 == 0:
                logger.info(f"Progress: {i + batch_size}/{len(ko_texts)} pairs processed")
        
        return np.array(similarities)
    except Exception as e:
        logger.error(f"Error in compute_semantic_similarity_batch: {e}", exc_info=True)
        raise


def deduplicate_pairs(pairs: List[Dict]) -> List[Dict]:
    """Remove duplicate pairs based on source text"""
    seen = set()
    unique_pairs = []
    
    for pair in pairs:
        ko_text = pair['translation']['kor_Hang']
        
        if ko_text not in seen:
            seen.add(ko_text)
            unique_pairs.append(pair)
    
    return unique_pairs


def filter_opus_dataset(
    input_file: str,
    output_file: str,
    use_semantic_filter: bool = True,
    similarity_threshold: float = 0.6,
    batch_size: int = 32,
    max_pairs: int = None
):
    """Filter OPUS dataset with quality criteria"""
    
    logger.info(f"Loading data from {input_file}...")
    
    # Load data
    pairs = []
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            pairs.append(json.loads(line))
    
    total_input = len(pairs)
    logger.info(f"Loaded {total_input:,} pairs")
    
    # Step 1: Basic quality filters
    logger.info("Step 1: Applying basic quality filters...")
    filtered_pairs = []
    
    for pair in tqdm(pairs, desc="Basic filtering"):
        ko_text = pair['translation']['kor_Hang']
        vi_text = pair['translation']['vie_Latn']
        
        if is_valid_pair(ko_text, vi_text):
            filtered_pairs.append(pair)
    
    logger.info(f"After basic filters: {len(filtered_pairs):,} pairs ({len(filtered_pairs)/total_input*100:.1f}%)")
    
    # Step 2: Deduplication
    logger.info("Step 2: Deduplicating...")
    filtered_pairs = deduplicate_pairs(filtered_pairs)
    logger.info(f"After deduplication: {len(filtered_pairs):,} pairs ({len(filtered_pairs)/total_input*100:.1f}%)")
    
    # Step 3: Semantic similarity filtering
    if use_semantic_filter:
        logger.info("Step 3: Computing semantic similarity with LaBSE...")
        logger.info("Loading LaBSE model...")
        
        try:
            from sentence_transformers import SentenceTransformer
            import torch
            
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            logger.info(f"Using device: {device}")
            
            if device == 'cuda':
                logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
                logger.info(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
            
            logger.info("Downloading/loading LaBSE model (this may take a few minutes on first run)...")
            model = SentenceTransformer('sentence-transformers/LaBSE')
            logger.info("Model loaded successfully!")
            
            model = model.to(device)
            logger.info(f"Model moved to {device}")
            
            # Extract texts
            ko_texts = [p['translation']['kor_Hang'] for p in filtered_pairs]
            vi_texts = [p['translation']['vie_Latn'] for p in filtered_pairs]
            
            # Compute similarities
            logger.info(f"Computing similarity for {len(filtered_pairs):,} pairs (batch_size={batch_size})...")
            logger.info(f"Estimated batches: {(len(filtered_pairs) + batch_size - 1) // batch_size}")
            
            try:
                similarities = compute_semantic_similarity_batch(ko_texts, vi_texts, model, batch_size)
                logger.info("Similarity computation completed!")
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    logger.error(f"GPU Out of Memory! Try reducing batch_size (current: {batch_size})")
                    logger.error("Suggested: --batch-size 8 or --batch-size 4")
                raise
            except Exception as e:
                logger.error(f"Error during similarity computation: {e}", exc_info=True)
                raise
            
            # Filter by similarity threshold
            high_quality_pairs = []
            for pair, sim in zip(filtered_pairs, similarities):
                if sim >= similarity_threshold:
                    pair['similarity'] = float(sim)
                    high_quality_pairs.append(pair)
            
            filtered_pairs = high_quality_pairs
            logger.info(f"After similarity filter (≥{similarity_threshold}): {len(filtered_pairs):,} pairs ({len(filtered_pairs)/total_input*100:.1f}%)")
            
            # Log similarity stats
            logger.info(f"Similarity stats: min={similarities.min():.3f}, max={similarities.max():.3f}, mean={similarities.mean():.3f}")
            
        except ImportError:
            logger.warning("sentence-transformers not installed. Skipping semantic filtering.")
            logger.warning("Install with: pip install sentence-transformers")
    
    # Step 4: Limit to max_pairs if specified
    if max_pairs and len(filtered_pairs) > max_pairs:
        logger.info(f"Step 4: Sampling {max_pairs:,} pairs...")
        
        if use_semantic_filter and 'similarity' in filtered_pairs[0]:
            # Sort by similarity and take top N
            filtered_pairs.sort(key=lambda x: x['similarity'], reverse=True)
            filtered_pairs = filtered_pairs[:max_pairs]
            logger.info(f"Selected top {max_pairs:,} pairs by similarity")
        else:
            # Random sample
            import random
            random.shuffle(filtered_pairs)
            filtered_pairs = filtered_pairs[:max_pairs]
            logger.info(f"Random sampled {max_pairs:,} pairs")
    
    # Save filtered data
    logger.info(f"Saving filtered data to {output_file}...")
    with open(output_file, 'w', encoding='utf-8') as f:
        for pair in filtered_pairs:
            # Remove similarity score before saving
            if 'similarity' in pair:
                del pair['similarity']
            f.write(json.dumps(pair, ensure_ascii=False) + '\n')
    
    # Summary
    logger.info("\n" + "="*60)
    logger.info("FILTERING SUMMARY")
    logger.info("="*60)
    logger.info(f"Input pairs:      {total_input:,}")
    logger.info(f"Output pairs:     {len(filtered_pairs):,}")
    logger.info(f"Retention rate:   {len(filtered_pairs)/total_input*100:.1f}%")
    logger.info(f"Filtered out:     {total_input - len(filtered_pairs):,} ({(total_input - len(filtered_pairs))/total_input*100:.1f}%)")
    logger.info("="*60 + "\n")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Filter OPUS dataset with quality criteria")
    parser.add_argument('--input', required=True, help='Input JSONL file')
    parser.add_argument('--output', required=True, help='Output JSONL file')
    parser.add_argument('--no-semantic', action='store_true', help='Skip semantic similarity filtering')
    parser.add_argument('--similarity-threshold', type=float, default=0.6, help='Minimum similarity score (default: 0.6)')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size for encoding (default: 32)')
    parser.add_argument('--max-pairs', type=int, default=None, help='Maximum number of pairs to keep')
    
    args = parser.parse_args()
    
    filter_opus_dataset(
        input_file=args.input,
        output_file=args.output,
        use_semantic_filter=not args.no_semantic,
        similarity_threshold=args.similarity_threshold,
        batch_size=args.batch_size,
        max_pairs=args.max_pairs
    )


if __name__ == "__main__":
    main()