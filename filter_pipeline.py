"""
Production-grade Filtering Pipeline for Ko-Vi Parallel Corpus
Implements V2 + V3 filtering strategies adapted for Korean-Vietnamese.

Design philosophy:
- Aggressive filtering: better 100k high-quality than 500k noisy
- Each filter is independent and configurable
- Filters are ordered from cheap to expensive
- Preserve alignment scores for final ranking
"""

import json
import re
import unicodedata
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path
import logging
from collections import Counter
import hashlib

import fasttext
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# For tokenization
from kiwipiepy import Kiwi  # Korean tokenizer
import underthesea  # Vietnamese NLP

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class FilterStats:
    """Statistics for each filter"""
    total_input: int = 0
    passed: int = 0
    failed: int = 0
    
    def pass_rate(self) -> float:
        return self.passed / max(self.total_input, 1) * 100


class FilterPipeline:
    """
    Complete filtering pipeline combining V2 and V3 strategies.
    """
    
    def __init__(self, config: Dict):
        self.config = config
        self.stats = {}  # filter_name -> FilterStats
        
        # Initialize components
        self._init_language_detector()
        self._init_tokenizers()
        self._init_embeddings()
        
        # Deduplication cache
        self.seen_hashes = set()
        
        # Spam/commercial keywords
        self.spam_keywords_ko = [
            '할인', '쿠폰', '이벤트', '특가', '최저가', '무료배송',
            '회원가입', '로그인', '장바구니', 'B2B', '도매', '공급',
        ]
        self.spam_keywords_vi = [
            'giảm giá', 'khuyến mãi', 'miễn phí', 'đăng ký', 
            'đăng nhập', 'mua ngay', 'liên hệ', 'hotline',
        ]
        
    def _init_language_detector(self):
        """Initialize fastText language detector"""
        logger.info("Loading language detector...")
        try:
            # Download: https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin
            self.lang_detector = fasttext.load_model('models/lid.176.bin')
        except Exception as e:
            logger.warning(f"fastText model not found: {e}")
            self.lang_detector = None
    
    def _init_tokenizers(self):
        """Initialize Korean and Vietnamese tokenizers"""
        logger.info("Loading tokenizers...")
        self.ko_tokenizer = Kiwi()
        # underthesea used for Vietnamese word segmentation
        
    def _init_embeddings(self):
        """Initialize LaBSE for semantic similarity"""
        logger.info("Loading LaBSE embeddings...")
        self.embedder = SentenceTransformer('sentence-transformers/LaBSE')
        
    def compute_hash(self, text: str) -> str:
        """Compute hash for deduplication"""
        normalized = text.lower().strip()
        return hashlib.md5(normalized.encode()).hexdigest()
    
    # ========== V2 FILTERS ==========
    
    def filter_language_detection(self, ko_sent: str, vi_sent: str) -> Tuple[bool, str]:
        """
        1. Language Detection
        Verify Korean sentence is Korean and Vietnamese is Vietnamese.
        """
        if not self.lang_detector:
            return True, "no_detector"
        
        try:
            # Predict languages
            ko_pred = self.lang_detector.predict(ko_sent, k=1)
            vi_pred = self.lang_detector.predict(vi_sent, k=1)
            
            ko_lang = ko_pred[0][0].replace('__label__', '')
            vi_lang = vi_pred[0][0].replace('__label__', '')
            
            ko_conf = ko_pred[1][0]
            vi_conf = vi_pred[1][0]
            
            # Check if detected languages match expected
            min_conf = self.config.get('lang_detect_confidence', 0.8)
            
            if ko_lang == 'ko' and ko_conf >= min_conf and \
               vi_lang == 'vi' and vi_conf >= min_conf:
                return True, "pass"
            else:
                return False, f"ko={ko_lang}({ko_conf:.2f}),vi={vi_lang}({vi_conf:.2f})"
                
        except Exception as e:
            logger.warning(f"Language detection failed: {e}")
            return True, "error"
    
    def filter_deduplication(self, ko_sent: str, vi_sent: str) -> Tuple[bool, str]:
        """
        2. Exact Deduplication
        Remove exact duplicates based on Korean sentence.
        """
        ko_hash = self.compute_hash(ko_sent)
        
        if ko_hash in self.seen_hashes:
            return False, "duplicate"
        
        self.seen_hashes.add(ko_hash)
        return True, "pass"
    
    def filter_labse_similarity(self, ko_sent: str, vi_sent: str) -> Tuple[bool, str]:
        """
        3. Cross-lingual Semantic Coherence (LaBSE)
        Verify sentences are semantically similar.
        """
        try:
            ko_emb = self.embedder.encode([ko_sent], convert_to_numpy=True)
            vi_emb = self.embedder.encode([vi_sent], convert_to_numpy=True)
            
            similarity = cosine_similarity(ko_emb, vi_emb)[0][0]
            
            min_sim = self.config.get('min_labse_similarity', 0.5)
            
            if similarity >= min_sim:
                return True, f"sim={similarity:.3f}"
            else:
                return False, f"sim={similarity:.3f}_low"
                
        except Exception as e:
            logger.warning(f"LaBSE similarity failed: {e}")
            return True, "error"
    
    def filter_length_ratio(self, ko_sent: str, vi_sent: str) -> Tuple[bool, str]:
        """
        4. Length Ratio Filter
        Korean and Vietnamese should have similar lengths (characters).
        Korean is more compact, so allow wider ratio.
        """
        ko_len = len(ko_sent)
        vi_len = len(vi_sent)
        
        if ko_len == 0 or vi_len == 0:
            return False, "empty"
        
        ratio = max(ko_len, vi_len) / min(ko_len, vi_len)
        
        max_ratio = self.config.get('max_length_ratio', 3.0)
        
        if ratio <= max_ratio:
            return True, f"ratio={ratio:.2f}"
        else:
            return False, f"ratio={ratio:.2f}_high"
    
    def filter_content_quality(self, ko_sent: str, vi_sent: str) -> Tuple[bool, str]:
        """
        5. Content Quality Filters
        - Minimum/maximum length
        - Not too many URLs, numbers, special chars
        - Not all uppercase
        """
        min_len = self.config.get('min_sent_length', 10)
        max_len = self.config.get('max_sent_length', 500)
        
        # Length check
        if len(ko_sent) < min_len or len(vi_sent) < min_len:
            return False, "too_short"
        if len(ko_sent) > max_len or len(vi_sent) > max_len:
            return False, "too_long"
        
        # URL check (shouldn't have URLs after cleaning, but double check)
        url_pattern = r'https?://|www\.'
        if re.search(url_pattern, ko_sent) or re.search(url_pattern, vi_sent):
            return False, "has_url"
        
        # Digit ratio (not more than 40% digits)
        max_digit_ratio = 0.4
        ko_digit_ratio = sum(c.isdigit() for c in ko_sent) / len(ko_sent)
        vi_digit_ratio = sum(c.isdigit() for c in vi_sent) / len(vi_sent)
        
        if ko_digit_ratio > max_digit_ratio or vi_digit_ratio > max_digit_ratio:
            return False, "too_many_digits"
        
        # Special character ratio
        max_special_ratio = 0.3
        ko_special = sum(not c.isalnum() and not c.isspace() for c in ko_sent) / len(ko_sent)
        vi_special = sum(not c.isalnum() and not c.isspace() for c in vi_sent) / len(vi_sent)
        
        if ko_special > max_special_ratio or vi_special > max_special_ratio:
            return False, "too_many_special"
        
        # Uppercase check (all caps is usually spam)
        if ko_sent.isupper() or vi_sent.isupper():
            return False, "all_caps"
        
        return True, "pass"
    
    def filter_script_validation(self, ko_sent: str, vi_sent: str) -> Tuple[bool, str]:
        """
        8. Script Ratio Validation
        Korean should be mostly Hangul, Vietnamese mostly Latin.
        """
        # Count Hangul characters in Korean
        hangul_count = sum(0xAC00 <= ord(c) <= 0xD7A3 for c in ko_sent)
        hangul_ratio = hangul_count / len(ko_sent)
        
        min_hangul = self.config.get('min_hangul_ratio', 0.3)
        
        if hangul_ratio < min_hangul:
            return False, f"hangul={hangul_ratio:.2f}_low"
        
        # Count Latin characters in Vietnamese
        latin_count = sum(c.isalpha() and ord(c) < 0x0250 for c in vi_sent)
        latin_ratio = latin_count / len(vi_sent)
        
        min_latin = self.config.get('min_latin_ratio', 0.5)
        
        if latin_ratio < min_latin:
            return False, f"latin={latin_ratio:.2f}_low"
        
        return True, "pass"
    
    def filter_repetition(self, ko_sent: str, vi_sent: str) -> Tuple[bool, str]:
        """
        6. Repetition Detection
        Detect sentences with excessive repetition.
        """
        def has_repetition(text: str) -> bool:
            # Character n-gram repetition
            for n in [3, 4, 5]:
                ngrams = [text[i:i+n] for i in range(len(text)-n+1)]
                if ngrams:
                    most_common = Counter(ngrams).most_common(1)[0][1]
                    if most_common > len(ngrams) * 0.3:  # >30% repetition
                        return True
            
            # Word repetition
            words = text.split()
            if len(words) > 3:
                most_common_word = Counter(words).most_common(1)[0][1]
                if most_common_word > len(words) * 0.4:  # >40% same word
                    return True
            
            return False
        
        if has_repetition(ko_sent) or has_repetition(vi_sent):
            return False, "repetitive"
        
        return True, "pass"
    
    # ========== V3 FILTERS ==========
    
    def filter_web_artifacts(self, ko_sent: str, vi_sent: str) -> Tuple[bool, str]:
        """
        3. Web Artifact Removal
        Detect navigation, UI elements, cookies, etc.
        """
        web_patterns = [
            r'cookie', r'javascript', r'©', r'copyright',
            r'privacy policy', r'terms of service',
            r'click here', r'read more', r'sign up', r'log in',
            # Korean
            r'쿠키', r'개인정보', r'이용약관', r'로그인', r'회원가입',
            # Vietnamese  
            r'cookie', r'chính sách', r'điều khoản', r'đăng nhập', r'đăng ký',
        ]
        
        combined = (ko_sent + ' ' + vi_sent).lower()
        
        for pattern in web_patterns:
            if re.search(pattern, combined, re.IGNORECASE):
                return False, f"web_artifact:{pattern}"
        
        return True, "pass"
    
    def filter_commercial_spam(self, ko_sent: str, vi_sent: str) -> Tuple[bool, str]:
        """
        4. B2B / Commercial Spam Filtering
        Detect promotional, sales, B2B content.
        """
        # Check Korean keywords
        ko_lower = ko_sent.lower()
        for keyword in self.spam_keywords_ko:
            if keyword in ko_lower:
                return False, f"spam_ko:{keyword}"
        
        # Check Vietnamese keywords
        vi_lower = vi_sent.lower()
        for keyword in self.spam_keywords_vi:
            if keyword in vi_lower:
                return False, f"spam_vi:{keyword}"
        
        # Excessive punctuation (!!!, ???)
        if re.search(r'[!?]{3,}', ko_sent) or re.search(r'[!?]{3,}', vi_sent):
            return False, "excessive_punct"
        
        return True, "pass"
    
    def filter_fragments(self, ko_sent: str, vi_sent: str) -> Tuple[bool, str]:
        """
        6. Fragment Detection
        Detect incomplete sentences (no verb, too short, etc.)
        """
        # Check if sentences end with proper punctuation
        if not ko_sent[-1] in '.!?。' or not vi_sent[-1] in '.!?':
            # Allow if very short (might be title)
            if len(ko_sent) > 20 or len(vi_sent) > 20:
                return False, "no_ending_punct"
        
        # Check minimum word count
        ko_words = len(self.ko_tokenizer.tokenize(ko_sent))
        vi_words = len(underthesea.word_tokenize(vi_sent))
        
        min_words = self.config.get('min_word_count', 3)
        
        if ko_words < min_words or vi_words < min_words:
            return False, f"too_few_words(ko={ko_words},vi={vi_words})"
        
        return True, "pass"
    
    def filter_mt_artifacts(self, ko_sent: str, vi_sent: str) -> Tuple[bool, str]:
        """
        7. MT Artifact Detection
        Detect machine translation artifacts.
        """
        mt_patterns = [
            # Untranslated source language
            # (too much Latin in Korean, or too much Hangul in Vietnamese)
            
            # Bracket artifacts
            r'\[\[.*?\]\]',
            r'\{\{.*?\}\}',
            
            # Translation notes
            r'\(translated\)', r'\(translation\)',
            r'\(dịch\)', r'\(번역\)',
        ]
        
        combined = ko_sent + ' ' + vi_sent
        
        for pattern in mt_patterns:
            if re.search(pattern, combined, re.IGNORECASE):
                return False, f"mt_artifact:{pattern}"
        
        # Check if Vietnamese has too much Hangul (MT leak)
        hangul_in_vi = sum(0xAC00 <= ord(c) <= 0xD7A3 for c in vi_sent)
        if hangul_in_vi > 0:
            return False, "hangul_in_vietnamese"
        
        return True, "pass"
    
    def filter_token_ratio(self, ko_sent: str, vi_sent: str) -> Tuple[bool, str]:
        """
        9. Token Ratio (more refined than char ratio)
        Korean and Vietnamese token counts should be similar.
        """
        try:
            ko_tokens = len(self.ko_tokenizer.tokenize(ko_sent))
            vi_tokens = len(underthesea.word_tokenize(vi_sent))
            
            if ko_tokens == 0 or vi_tokens == 0:
                return False, "no_tokens"
            
            token_ratio = max(ko_tokens, vi_tokens) / min(ko_tokens, vi_tokens)
            
            max_token_ratio = self.config.get('max_token_ratio', 2.5)
            
            if token_ratio <= max_token_ratio:
                return True, f"token_ratio={token_ratio:.2f}"
            else:
                return False, f"token_ratio={token_ratio:.2f}_high"
                
        except Exception as e:
            logger.warning(f"Token ratio failed: {e}")
            return True, "error"
    
    # ========== PIPELINE EXECUTION ==========
    
    def apply_filters(self, ko_sent: str, vi_sent: str) -> Tuple[bool, Dict[str, str]]:
        """
        Apply all filters in sequence.
        Returns (passed, filter_results)
        """
        filters = [
            ('lang_detection', self.filter_language_detection),
            ('deduplication', self.filter_deduplication),
            ('content_quality', self.filter_content_quality),
            ('length_ratio', self.filter_length_ratio),
            ('script_validation', self.filter_script_validation),
            ('repetition', self.filter_repetition),
            ('web_artifacts', self.filter_web_artifacts),
            ('commercial_spam', self.filter_commercial_spam),
            ('fragments', self.filter_fragments),
            ('mt_artifacts', self.filter_mt_artifacts),
            ('token_ratio', self.filter_token_ratio),
            ('labse_similarity', self.filter_labse_similarity),  # Expensive, do last
        ]
        
        results = {}
        
        for filter_name, filter_func in filters:
            # Initialize stats if needed
            if filter_name not in self.stats:
                self.stats[filter_name] = FilterStats()
            
            self.stats[filter_name].total_input += 1
            
            # Apply filter
            passed, reason = filter_func(ko_sent, vi_sent)
            results[filter_name] = reason
            
            if passed:
                self.stats[filter_name].passed += 1
            else:
                self.stats[filter_name].failed += 1
                # Stop at first failure (early exit)
                return False, results
        
        return True, results
    
    def process_aligned_sentences(self, input_path: str, output_path: str):
        """
        Process aligned sentences through filter pipeline.
        """
        input_file = Path(input_path)
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        total_input = 0
        total_passed = 0
        
        with open(input_file, 'r', encoding='utf-8') as f_in, \
             open(output_file, 'w', encoding='utf-8') as f_out:
            
            for line in f_in:
                total_input += 1
                
                sent_pair = json.loads(line)
                ko_sent = sent_pair['ko_sent']
                vi_sent = sent_pair['vi_sent']
                
                # Apply filters
                passed, filter_results = self.apply_filters(ko_sent, vi_sent)
                
                if passed:
                    # Add filter results to output
                    sent_pair['filter_results'] = filter_results
                    f_out.write(json.dumps(sent_pair, ensure_ascii=False) + '\n')
                    total_passed += 1
                
                if total_input % 1000 == 0:
                    logger.info(f"Processed {total_input} sentences, "
                              f"passed {total_passed} ({total_passed/total_input*100:.1f}%)")
        
        # Print statistics
        self.print_statistics(total_input, total_passed)
        
        logger.info(f"\nFiltered dataset saved to: {output_path}")
    
    def print_statistics(self, total_input: int, total_passed: int):
        """Print detailed filter statistics"""
        logger.info(f"\n{'='*70}")
        logger.info(f"FILTER PIPELINE STATISTICS")
        logger.info(f"{'='*70}")
        logger.info(f"Total input sentences: {total_input:,}")
        logger.info(f"Total passed: {total_passed:,} ({total_passed/total_input*100:.1f}%)")
        logger.info(f"Total filtered: {total_input - total_passed:,}")
        logger.info(f"\nPer-filter statistics:")
        logger.info(f"{'-'*70}")
        logger.info(f"{'Filter':<25} {'Input':>10} {'Passed':>10} {'Failed':>10} {'Pass%':>8}")
        logger.info(f"{'-'*70}")
        
        for filter_name, stats in self.stats.items():
            logger.info(f"{filter_name:<25} {stats.total_input:>10,} "
                       f"{stats.passed:>10,} {stats.failed:>10,} "
                       f"{stats.pass_rate():>7.1f}%")
        
        logger.info(f"{'='*70}\n")


def main():
    """Example usage"""
    config = {
        # Language detection
        'lang_detect_confidence': 0.8,
        
        # Semantic similarity
        'min_labse_similarity': 0.5,
        
        # Length constraints
        'min_sent_length': 10,
        'max_sent_length': 500,
        'max_length_ratio': 3.0,
        'max_token_ratio': 2.5,
        
        # Script validation
        'min_hangul_ratio': 0.3,
        'min_latin_ratio': 0.5,
        
        # Quality
        'min_word_count': 3,
    }
    
    pipeline = FilterPipeline(config)
    
    pipeline.process_aligned_sentences(
        input_path='data/processed/aligned_sentences.jsonl',
        output_path='data/final/filtered_kovi_parallel.jsonl'
    )


if __name__ == '__main__':
    main()