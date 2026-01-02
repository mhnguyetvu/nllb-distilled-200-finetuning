"""
Main Pipeline Runner for Korean-Vietnamese Parallel Corpus
Orchestrates the entire pipeline from crawling to final dataset.

Usage:
    python run_pipeline.py --config config.yaml --stage all
    python run_pipeline.py --config config.yaml --stage crawl
    python run_pipeline.py --config config.yaml --stage extract
    python run_pipeline.py --config config.yaml --stage align
    python run_pipeline.py --config config.yaml --stage filter
    python run_pipeline.py --config config.yaml --stage select
"""

import argparse
import yaml
import logging
from pathlib import Path
import sys
import json
from typing import Dict, List
from collections import defaultdict
import numpy as np

# Import pipeline modules
# from sentence_align import process_all_documents
from filter_pipeline import FilterPipeline

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DatasetSelector:
    """
    Final dataset selection from filtered corpus.
    Implements quality-based and source-balanced strategies.
    """
    
    def __init__(self, config: Dict):
        self.config = config
        
    def compute_quality_score(self, sent_pair: Dict) -> float:
        """
        Compute quality score for ranking.
        Combines alignment score and LaBSE similarity.
        """
        weights = self.config['weights']
        
        alignment_score = sent_pair.get('alignment_score', 0.5)
        
        # Extract LaBSE similarity from filter results
        filter_results = sent_pair.get('filter_results', {})
        labse_result = filter_results.get('labse_similarity', 'sim=0.5')
        
        # Parse similarity score
        try:
            labse_sim = float(labse_result.split('=')[1])
        except:
            labse_sim = 0.5
        
        # Weighted combination
        score = (weights['alignment_score'] * alignment_score + 
                weights['labse_similarity'] * labse_sim)
        
        return score
    
    def select_top_quality(self, sent_pairs: List[Dict], target_size: int) -> List[Dict]:
        """
        Select top-quality sentences by score.
        """
        # Compute scores
        for pair in sent_pairs:
            pair['quality_score'] = self.compute_quality_score(pair)
        
        # Sort by quality
        sorted_pairs = sorted(sent_pairs, key=lambda x: x['quality_score'], reverse=True)
        
        # Select top N
        selected = sorted_pairs[:target_size]
        
        logger.info(f"Selected {len(selected)} top-quality sentences")
        logger.info(f"Quality score range: {selected[-1]['quality_score']:.3f} - {selected[0]['quality_score']:.3f}")
        
        return selected
    
    def select_source_balanced(self, sent_pairs: List[Dict], target_size: int) -> List[Dict]:
        """
        Select sentences with source diversity.
        Ensures representation from all sources.
        """
        # Group by source
        by_source = defaultdict(list)
        for pair in sent_pairs:
            source = pair['source_site']
            pair['quality_score'] = self.compute_quality_score(pair)
            by_source[source].append(pair)
        
        # Sort each source by quality
        for source in by_source:
            by_source[source].sort(key=lambda x: x['quality_score'], reverse=True)
        
        # Ensure minimum per source
        min_per_source = self.config.get('min_per_source', 1000)
        selected = []
        
        for source, pairs in by_source.items():
            n_select = min(min_per_source, len(pairs))
            selected.extend(pairs[:n_select])
            logger.info(f"Selected {n_select} from {source}")
        
        # If we haven't reached target, add more by quality
        if len(selected) < target_size:
            remaining_target = target_size - len(selected)
            
            # Collect remaining candidates
            remaining = []
            for source, pairs in by_source.items():
                remaining.extend(pairs[min_per_source:])
            
            # Sort by quality
            remaining.sort(key=lambda x: x['quality_score'], reverse=True)
            
            # Add top remaining
            selected.extend(remaining[:remaining_target])
        
        # If we have too many, trim to target
        if len(selected) > target_size:
            selected.sort(key=lambda x: x['quality_score'], reverse=True)
            selected = selected[:target_size]
        
        logger.info(f"Total selected: {len(selected)} sentences")
        
        # Distribution by source
        source_dist = defaultdict(int)
        for pair in selected:
            source_dist[pair['source_site']] += 1
        
        logger.info("Source distribution:")
        for source, count in sorted(source_dist.items()):
            logger.info(f"  {source}: {count} ({count/len(selected)*100:.1f}%)")
        
        return selected
    
    def split_dataset(self, selected: List[Dict]) -> Dict[str, List[Dict]]:
        """
        Split into train/dev/test sets.
        """
        train_ratio = self.config.get('train_ratio', 0.95)
        dev_ratio = self.config.get('dev_ratio', 0.025)
        test_ratio = self.config.get('test_ratio', 0.025)
        
        # Shuffle
        np.random.seed(42)
        indices = np.random.permutation(len(selected))
        
        n_train = int(len(selected) * train_ratio)
        n_dev = int(len(selected) * dev_ratio)
        
        train_idx = indices[:n_train]
        dev_idx = indices[n_train:n_train+n_dev]
        test_idx = indices[n_train+n_dev:]
        
        splits = {
            'train': [selected[i] for i in train_idx],
            'dev': [selected[i] for i in dev_idx],
            'test': [selected[i] for i in test_idx],
        }
        
        logger.info(f"\nDataset splits:")
        logger.info(f"  Train: {len(splits['train'])} ({len(splits['train'])/len(selected)*100:.1f}%)")
        logger.info(f"  Dev: {len(splits['dev'])} ({len(splits['dev'])/len(selected)*100:.1f}%)")
        logger.info(f"  Test: {len(splits['test'])} ({len(splits['test'])/len(selected)*100:.1f}%)")
        
        return splits
    
    def save_splits(self, splits: Dict[str, List[Dict]], output_dir: str):
        """Save train/dev/test splits"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        for split_name, pairs in splits.items():
            output_file = output_path / f"kovi_{split_name}.jsonl"
            
            with open(output_file, 'w', encoding='utf-8') as f:
                for pair in pairs:
                    # Simplified format for MT training
                    simplified = {
                        'ko': pair['ko_sent'],
                        'vi': pair['vi_sent'],
                        'score': pair.get('quality_score', 0.0),
                        'source': pair['source_site'],
                    }
                    f.write(json.dumps(simplified, ensure_ascii=False) + '\n')
            
            logger.info(f"Saved {len(pairs)} pairs to {output_file}")
    
    def process_selection(self, input_path: str, output_dir: str):
        """Run selection pipeline"""
        logger.info(f"Loading filtered sentences from {input_path}")
        
        # Load all filtered sentences
        sent_pairs = []
        with open(input_path, 'r', encoding='utf-8') as f:
            for line in f:
                sent_pairs.append(json.loads(line))
        
        logger.info(f"Loaded {len(sent_pairs)} filtered sentences")
        
        # Select based on strategy
        target_size = self.config.get('target_size', 200000)
        strategy = self.config.get('strategy', 'source_balanced')
        
        logger.info(f"\nSelection strategy: {strategy}")
        logger.info(f"Target size: {target_size:,}")
        
        if strategy == 'top_quality':
            selected = self.select_top_quality(sent_pairs, target_size)
        elif strategy == 'source_balanced':
            selected = self.select_source_balanced(sent_pairs, target_size)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
        
        # Split into train/dev/test
        splits = self.split_dataset(selected)
        
        # Save splits
        self.save_splits(splits, output_dir)
        
        logger.info(f"\n{'='*70}")
        logger.info(f"Final dataset ready!")
        logger.info(f"Output directory: {output_dir}")
        logger.info(f"{'='*70}")


def run_crawling(config: Dict):
    """Stage 1: Crawl + Extract (optimized)"""
    from crawl_and_extract import OptimizedCrawler
    
    crawler_config = config['crawling']
    crawler = OptimizedCrawler(crawler_config)
    
    for seed_url in crawler_config['seed_urls']:
        # respect configured max pages per site (fall back to 500)
        max_pages = crawler_config.get('max_pages_per_site', 500)
        crawler.crawl_site(seed_url, max_pages=max_pages)
    
    # Save direct to processed (skip raw)
    crawler.save_results('data/processed/clean_documents.jsonl')

def run_extraction(config: Dict):
    """Stage 2: Extract clean text from HTML"""
    logger.info("\n" + "="*70)
    logger.info("STAGE 2: TEXT EXTRACTION")
    logger.info("="*70 + "\n")
    
    extract_config = config['extraction']
    extractor = TextExtractor(extract_config)
    
    extractor.process_document_pairs(
        input_path=extract_config['input_file'],
        output_path=extract_config['output_file']
    )
    
    logger.info("Stage 2 complete.\n")


def run_alignment(config: Dict):
    """Stage 3: Sentence segmentation and alignment"""
    logger.info("\n" + "="*70)
    logger.info("STAGE 3: SENTENCE ALIGNMENT")
    logger.info("="*70 + "\n")

    # sentence_align.py của bạn có hàm này
    from sentence_align import run_alignment_stage
    run_alignment_stage(config)

    logger.info("Stage 3 complete.\n")



def run_filtering(config: Dict):
    """Stage 4: Filter pipeline"""
    logger.info("\n" + "="*70)
    logger.info("STAGE 4: FILTERING PIPELINE")
    logger.info("="*70 + "\n")
    
    filter_config = config['filtering']
    pipeline = FilterPipeline(filter_config)
    
    pipeline.process_aligned_sentences(
        input_path=filter_config['input_file'],
        output_path=filter_config['output_file']
    )
    
    logger.info("Stage 4 complete.\n")


def run_selection(config: Dict):
    """Stage 5: Final selection and splitting"""
    logger.info("\n" + "="*70)
    logger.info("STAGE 5: FINAL SELECTION")
    logger.info("="*70 + "\n")
    
    select_config = config['selection']
    selector = DatasetSelector(select_config)
    
    selector.process_selection(
        input_path=select_config['input_file'],
        output_dir='data/final'
    )
    
    logger.info("Stage 5 complete.\n")


def main():
    parser = argparse.ArgumentParser(description='Korean-Vietnamese Parallel Corpus Pipeline')
    parser.add_argument('--config', type=str, default='config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--stage', type=str, default='all',
                       choices=['all', 'crawl', 'extract', 'align', 'filter', 'select'],
                       help='Pipeline stage to run')
    
    args = parser.parse_args()
    
    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Create output directories
    for directory in ['data/raw', 'data/processed', 'data/final', 'logs', 'models']:
        Path(directory).mkdir(parents=True, exist_ok=True)
    
    # Run requested stages
    stages = {
        'crawl': run_crawling,
        'extract': run_extraction,
        'align': run_alignment,
        'filter': run_filtering,
        'select': run_selection,
    }
    
    if args.stage == 'all':
        for stage_name, stage_func in stages.items():
            try:
                stage_func(config)
            except Exception as e:
                logger.error(f"Error in stage '{stage_name}': {e}", exc_info=True)
                sys.exit(1)
    else:
        try:
            stages[args.stage](config)
        except Exception as e:
            logger.error(f"Error in stage '{args.stage}': {e}", exc_info=True)
            sys.exit(1)
    
    logger.info("\n" + "="*70)
    logger.info("PIPELINE COMPLETE!")
    logger.info("="*70 + "\n")


if __name__ == '__main__':
    main()