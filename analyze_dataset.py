"""
Dataset Analysis and Quality Assessment
Provides statistics, quality metrics, and visualizations for the corpus.

Usage:
    python analyze_dataset.py --input data/final/kovi_train.jsonl
    python analyze_dataset.py --input data/final/kovi_train.jsonl --output analysis_report.txt
"""

import json
import argparse
from pathlib import Path
from collections import Counter, defaultdict
from typing import List, Dict
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DatasetAnalyzer:
    """Analyze Korean-Vietnamese parallel corpus"""
    
    def __init__(self, dataset_path: str):
        self.dataset_path = dataset_path
        self.data = []
        self.load_data()
        
    def load_data(self):
        """Load dataset from JSONL"""
        logger.info(f"Loading dataset from {self.dataset_path}")
        with open(self.dataset_path, 'r', encoding='utf-8') as f:
            for line in f:
                self.data.append(json.loads(line))
        logger.info(f"Loaded {len(self.data)} sentence pairs")
    
    def basic_statistics(self) -> Dict:
        """Compute basic statistics"""
        stats = {
            'total_pairs': len(self.data),
            'avg_ko_length': np.mean([len(item['ko']) for item in self.data]),
            'avg_vi_length': np.mean([len(item['vi']) for item in self.data]),
            'avg_ko_words': 0,  # Would need tokenization
            'avg_vi_words': np.mean([len(item['vi'].split()) for item in self.data]),
            'avg_score': np.mean([item.get('score', 0) for item in self.data]),
        }
        
        # Length distributions
        ko_lengths = [len(item['ko']) for item in self.data]
        vi_lengths = [len(item['vi']) for item in self.data]
        
        stats['ko_length_min'] = min(ko_lengths)
        stats['ko_length_max'] = max(ko_lengths)
        stats['ko_length_median'] = np.median(ko_lengths)
        
        stats['vi_length_min'] = min(vi_lengths)
        stats['vi_length_max'] = max(vi_lengths)
        stats['vi_length_median'] = np.median(vi_lengths)
        
        # Length ratios
        ratios = [max(ko_len, vi_len) / min(ko_len, vi_len) 
                 for ko_len, vi_len in zip(ko_lengths, vi_lengths)
                 if min(ko_len, vi_len) > 0]
        stats['avg_length_ratio'] = np.mean(ratios)
        stats['max_length_ratio'] = max(ratios)
        
        return stats
    
    def source_distribution(self) -> Dict[str, int]:
        """Distribution by source site"""
        source_counts = Counter([item.get('source', 'unknown') for item in self.data])
        return dict(source_counts)
    
    def score_distribution(self) -> Dict:
        """Quality score distribution"""
        scores = [item.get('score', 0) for item in self.data]
        
        return {
            'mean': np.mean(scores),
            'median': np.median(scores),
            'std': np.std(scores),
            'min': min(scores),
            'max': max(scores),
            'q25': np.percentile(scores, 25),
            'q75': np.percentile(scores, 75),
        }
    
    def sample_pairs(self, n: int = 10, by_score: bool = True) -> List[Dict]:
        """Sample pairs for manual inspection"""
        if by_score:
            # Sort by score and sample across quality range
            sorted_data = sorted(self.data, key=lambda x: x.get('score', 0), reverse=True)
            # Sample from top, middle, and bottom
            samples = []
            samples.extend(sorted_data[:n//3])  # Top quality
            samples.extend(sorted_data[len(sorted_data)//2:len(sorted_data)//2 + n//3])  # Medium
            samples.extend(sorted_data[-n//3:])  # Lower quality
            return samples
        else:
            # Random sample
            indices = np.random.choice(len(self.data), min(n, len(self.data)), replace=False)
            return [self.data[i] for i in indices]
    
    def detect_potential_issues(self) -> Dict[str, List[Dict]]:
        """Detect potential quality issues"""
        issues = {
            'very_short': [],
            'very_long': [],
            'extreme_ratio': [],
            'low_score': [],
        }
        
        for item in self.data:
            ko_len = len(item['ko'])
            vi_len = len(item['vi'])
            score = item.get('score', 0)
            
            # Very short
            if ko_len < 20 or vi_len < 20:
                issues['very_short'].append(item)
            
            # Very long
            if ko_len > 300 or vi_len > 300:
                issues['very_long'].append(item)
            
            # Extreme length ratio
            if min(ko_len, vi_len) > 0:
                ratio = max(ko_len, vi_len) / min(ko_len, vi_len)
                if ratio > 2.5:
                    issues['extreme_ratio'].append({**item, 'ratio': ratio})
            
            # Low score
            if score < 0.5:
                issues['low_score'].append(item)
        
        return issues
    
    def character_distribution(self) -> Dict:
        """Analyze character usage"""
        ko_chars = set()
        vi_chars = set()
        
        for item in self.data:
            ko_chars.update(item['ko'])
            vi_chars.update(item['vi'])
        
        # Categorize Korean characters
        hangul_count = sum(1 for c in ko_chars if 0xAC00 <= ord(c) <= 0xD7A3)
        
        # Categorize Vietnamese characters
        latin_count = sum(1 for c in vi_chars if ord(c) < 0x0250)
        
        return {
            'ko_total_chars': len(ko_chars),
            'ko_hangul_chars': hangul_count,
            'vi_total_chars': len(vi_chars),
            'vi_latin_chars': latin_count,
        }
    
    def generate_report(self, output_path: str = None):
        """Generate comprehensive analysis report"""
        report = []
        report.append("="*70)
        report.append("KOREAN-VIETNAMESE PARALLEL CORPUS ANALYSIS")
        report.append("="*70)
        report.append("")
        
        # Basic statistics
        report.append("BASIC STATISTICS")
        report.append("-"*70)
        stats = self.basic_statistics()
        report.append(f"Total sentence pairs: {stats['total_pairs']:,}")
        report.append(f"Average Korean length: {stats['avg_ko_length']:.1f} characters")
        report.append(f"Average Vietnamese length: {stats['avg_vi_length']:.1f} characters")
        report.append(f"Average Vietnamese words: {stats['avg_vi_words']:.1f}")
        report.append(f"Average quality score: {stats['avg_score']:.3f}")
        report.append("")
        report.append(f"Korean length range: {stats['ko_length_min']} - {stats['ko_length_max']} (median: {stats['ko_length_median']:.0f})")
        report.append(f"Vietnamese length range: {stats['vi_length_min']} - {stats['vi_length_max']} (median: {stats['vi_length_median']:.0f})")
        report.append(f"Average length ratio: {stats['avg_length_ratio']:.2f}")
        report.append(f"Max length ratio: {stats['max_length_ratio']:.2f}")
        report.append("")
        
        # Source distribution
        report.append("SOURCE DISTRIBUTION")
        report.append("-"*70)
        sources = self.source_distribution()
        for source, count in sorted(sources.items(), key=lambda x: x[1], reverse=True):
            percentage = count / stats['total_pairs'] * 100
            report.append(f"{source:30s}: {count:7,} ({percentage:5.1f}%)")
        report.append("")
        
        # Score distribution
        report.append("QUALITY SCORE DISTRIBUTION")
        report.append("-"*70)
        score_stats = self.score_distribution()
        report.append(f"Mean:   {score_stats['mean']:.3f}")
        report.append(f"Median: {score_stats['median']:.3f}")
        report.append(f"Std:    {score_stats['std']:.3f}")
        report.append(f"Range:  {score_stats['min']:.3f} - {score_stats['max']:.3f}")
        report.append(f"Q1-Q3:  {score_stats['q25']:.3f} - {score_stats['q75']:.3f}")
        report.append("")
        
        # Character distribution
        report.append("CHARACTER DISTRIBUTION")
        report.append("-"*70)
        char_stats = self.character_distribution()
        report.append(f"Korean unique characters: {char_stats['ko_total_chars']} (Hangul: {char_stats['ko_hangul_chars']})")
        report.append(f"Vietnamese unique characters: {char_stats['vi_total_chars']} (Latin: {char_stats['vi_latin_chars']})")
        report.append("")
        
        # Potential issues
        report.append("POTENTIAL QUALITY ISSUES")
        report.append("-"*70)
        issues = self.detect_potential_issues()
        report.append(f"Very short pairs (<20 chars): {len(issues['very_short']):,}")
        report.append(f"Very long pairs (>300 chars): {len(issues['very_long']):,}")
        report.append(f"Extreme length ratios (>2.5): {len(issues['extreme_ratio']):,}")
        report.append(f"Low quality scores (<0.5): {len(issues['low_score']):,}")
        report.append("")
        
        # Sample pairs
        report.append("SAMPLE SENTENCE PAIRS")
        report.append("-"*70)
        samples = self.sample_pairs(n=10)
        for i, sample in enumerate(samples[:5], 1):
            report.append(f"\nExample {i} (score: {sample.get('score', 0):.3f}, source: {sample.get('source', 'unknown')})")
            report.append(f"KO: {sample['ko']}")
            report.append(f"VI: {sample['vi']}")
        report.append("")
        
        report.append("="*70)
        report.append("END OF REPORT")
        report.append("="*70)
        
        # Print report
        report_text = "\n".join(report)
        print(report_text)
        
        # Save to file if specified
        if output_path:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(report_text)
            logger.info(f"Report saved to {output_path}")
        
        return report_text
    
    def export_issues_for_review(self, output_dir: str):
        """Export problematic pairs for manual review"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        issues = self.detect_potential_issues()
        
        for issue_type, pairs in issues.items():
            if pairs:
                output_file = output_path / f"review_{issue_type}.jsonl"
                with open(output_file, 'w', encoding='utf-8') as f:
                    for pair in pairs[:100]:  # Limit to 100 per category
                        f.write(json.dumps(pair, ensure_ascii=False) + '\n')
                logger.info(f"Exported {len(pairs[:100])} pairs with issue '{issue_type}' to {output_file}")


def main():
    parser = argparse.ArgumentParser(description='Analyze Korean-Vietnamese parallel corpus')
    parser.add_argument('--input', type=str, required=True,
                       help='Path to dataset JSONL file')
    parser.add_argument('--output', type=str, default=None,
                       help='Path to save analysis report')
    parser.add_argument('--export-issues', action='store_true',
                       help='Export problematic pairs for manual review')
    
    args = parser.parse_args()
    
    # Create analyzer
    analyzer = DatasetAnalyzer(args.input)
    
    # Generate report
    analyzer.generate_report(output_path=args.output)
    
    # Export issues if requested
    if args.export_issues:
        analyzer.export_issues_for_review('data/review')
        logger.info("Issues exported to data/review/ for manual inspection")


if __name__ == '__main__':
    main()