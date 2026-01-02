"""
Optimized Crawler: Extract text immediately, don't store HTML
Tiết kiệm storage bằng cách extract text ngay khi crawl

Output trực tiếp: clean document pairs (không lưu HTML)
Storage giảm: 500MB → 50MB

Fixes included:
- Avoid BeautifulSoup mutation during iteration (fix NoneType.get crashes)
- Robust boilerplate removal (collect then decompose)
- Canonicalize URL query correctly (doseq=True)
- Skip non-text pages (video/search/login/media)
- Deduplicate document pairs
- Better logging (exc_info=True)
"""

import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse, parse_qs, urlencode
import json
import time
import logging
from collections import deque
from dataclasses import dataclass, asdict
from typing import List, Dict, Set, Optional, Tuple
import re
from pathlib import Path
from trafilatura import extract

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class CleanDocumentPair:
    """Clean text pair (no HTML stored)"""
    ko_url: str
    vi_url: str
    ko_title: str
    vi_title: str
    ko_paragraphs: List[str]  # Clean paragraphs
    vi_paragraphs: List[str]  # Clean paragraphs
    ko_char_count: int
    vi_char_count: int
    source_site: str
    crawl_timestamp: str
    confidence: float


class OptimizedCrawler:
    """Crawl + extract in one pass - không lưu HTML"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': config.get(
                'user_agent',
                'Mozilla/5.0 (compatible; KoViDatasetBot/1.0)'
            )
        })

        # runtime state
        self.visited_urls: Set[str] = set()
        self.seen_pairs: Set[Tuple[str, str]] = set()
        self.document_pairs: List[CleanDocumentPair] = []

        # persistent checkpoint file for visited URLs
        self.visited_file: str = config.get('visited_file', 'data/state/visited_urls.txt')
        # how often (new URLs) to flush visited set to disk
        self.save_visited_every: int = int(config.get('save_visited_every', 100))
        self._new_visited_counter: int = 0

        # load existing visited URLs and seen pairs if present (resume support)
        try:
            self._load_visited()
        except Exception:
            logger.debug('No visited file to load (starting fresh)')

        try:
            # config may provide output_file where previous pairs live
            output_file = config.get('output_file', 'data/processed/clean_documents.jsonl')
            self._load_seen_pairs(output_file)
        except Exception:
            logger.debug('No existing pairs file to load (starting fresh)')
        
        # Language patterns
        self.ko_patterns = [
            r'/ko/', r'/kr/', r'/korean/',
            r'\?lang=ko', r'\?language=korean', r'/ko-KR/',
        ]
        self.vi_patterns = [
            r'/vi/', r'/vn/', r'/vietnamese/',
            r'\?lang=vi', r'\?language=vietnamese', r'/vi-VN/',
        ]
        
        # Boilerplate patterns (class/id)
        self.boilerplate_patterns = [
            r'nav', r'menu', r'header', r'footer', r'sidebar',
            r'advertisement', r'ad-', r'social', r'comment',
            r'cookie', r'banner', r'popup', r'modal'
        ]

        # Skip patterns (non-text / low-value pages)
        self.skip_substrings = [
            "/search", "/검색", "/login", "/signin", "/account",
            "/라이브러리/동영상", "/video", "/media", "/watch", "/player",
            "/cart", "/checkout"
        ]
    
    # ----------------------------
    # URL Helpers
    # ----------------------------
    def canonicalize_url(self, url: str) -> str:
        """Normalize URL"""
        parsed = urlparse(url)
        query = parse_qs(parsed.query)
        # IMPORTANT: doseq=True to keep query list values correctly
        sorted_query = urlencode(sorted(query.items()), doseq=True)
        canonical = f"{parsed.scheme}://{parsed.netloc}{parsed.path}"
        if sorted_query:
            canonical += f"?{sorted_query}"
        return canonical.lower()

    # ----------------------------
    # Persistence helpers
    # ----------------------------
    def _load_visited(self):
        p = Path(self.visited_file)
        if not p.exists():
            return
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open('r', encoding='utf-8') as f:
            for line in f:
                u = line.strip()
                if u:
                    self.visited_urls.add(u)

    def _save_visited(self):
        p = Path(self.visited_file)
        p.parent.mkdir(parents=True, exist_ok=True)
        try:
            with p.open('w', encoding='utf-8') as f:
                for u in sorted(self.visited_urls):
                    f.write(u + '\n')
        except Exception:
            logger.exception('Failed to save visited file')

    def _load_seen_pairs(self, path: str):
        p = Path(path)
        if not p.exists():
            return
        with p.open('r', encoding='utf-8') as f:
            for line in f:
                try:
                    d = json.loads(line)
                    ko = self.canonicalize_url(d.get('ko_url', ''))
                    vi = self.canonicalize_url(d.get('vi_url', ''))
                    if ko and vi:
                        self.seen_pairs.add((ko, vi))
                except Exception:
                    continue

    def should_skip_url(self, url: str) -> bool:
        u = (url or "").lower()
        if any(s in u for s in self.skip_substrings):
            return True
        if u.startswith(("mailto:", "javascript:")):
            return True
        if any(u.endswith(ext) for ext in [".pdf", ".jpg", ".png", ".zip", ".mp4", ".avi", ".mov"]):
            return True
        return False
    
    def detect_language_from_url(self, url: str) -> Optional[str]:
        """Detect language from URL"""
        url_lower = url.lower()
        for pattern in self.ko_patterns:
            if re.search(pattern, url_lower):
                return 'ko'
        for pattern in self.vi_patterns:
            if re.search(pattern, url_lower):
                return 'vi'
        return None
    
    # ----------------------------
    # Parallel URL discovery
    # ----------------------------
    def find_parallel_url(self, url: str, soup: BeautifulSoup, source_lang: str) -> Optional[str]:
        """Find parallel URL (hreflang > switcher > heuristic transform)"""
        if soup is None:
            return None

        target_lang = 'vi' if source_lang == 'ko' else 'ko'
        
        # Method 1: hreflang
        hreflang_tags = soup.find_all('link', rel='alternate', hreflang=True)
        for tag in hreflang_tags:
            if not tag:
                continue
            hreflang = (tag.get('hreflang') or '').lower()
            href = tag.get('href')
            if not href:
                continue

            if target_lang == 'vi':
                if 'vi' in hreflang or 'vn' in hreflang:
                    return urljoin(url, href)
            else:
                if 'ko' in hreflang or 'kr' in hreflang:
                    return urljoin(url, href)
        
        # Method 2: Language switcher links
        all_links = soup.find_all('a', href=True)
        target_patterns = self.vi_patterns if target_lang == 'vi' else self.ko_patterns
        
        for link in all_links:
            href = link.get('href')
            if not href:
                continue

            link_text = link.get_text().strip().lower()
            
            # Check text hint
            if target_lang in link_text or ('vietnamese' if target_lang == 'vi' else 'korean') in link_text:
                full_url = urljoin(url, href)
                if self.detect_language_from_url(full_url) == target_lang:
                    return full_url
            
            # Check URL pattern directly
            for pattern in target_patterns:
                if re.search(pattern, href.lower()):
                    return urljoin(url, href)
        
        # Method 3: URL transformation heuristics
        parsed = urlparse(url)
        path = parsed.path
        query = parsed.query
        
        # Replace in query
        if 'lang=' in query:
            new_query = re.sub(r'lang=\w+', f'lang={target_lang}', query)
            if new_query != query:
                return f"{parsed.scheme}://{parsed.netloc}{path}?{new_query}"

        # Replace in path (simple and safe heuristic)
        if source_lang == "ko":
            candidates = [path.replace("/ko/", "/vi/"), path.replace("/ko-KR/", "/vi-VN/")]
        else:
            candidates = [path.replace("/vi/", "/ko/"), path.replace("/vi-VN/", "/ko-KR/")]

        for new_path in candidates:
            if new_path != path:
                return f"{parsed.scheme}://{parsed.netloc}{new_path}"

        return None
    
    # ----------------------------
    # Extraction helpers
    # ----------------------------
    def extract_title(self, soup: BeautifulSoup) -> str:
        """Extract page title"""
        if soup is None:
            return ""

        title_tag = soup.find('title')
        if title_tag:
            title = title_tag.get_text().strip()
            title = re.sub(r'\s*[\|\-]\s*.*$', '', title)
            return title
        
        h1_tag = soup.find('h1')
        if h1_tag:
            return h1_tag.get_text().strip()
        
        return ''
    
    def clean_text(self, text: str, language: str) -> str:
        """Clean text"""
        if not text:
            return ""

        text = re.sub(r'https?://\S+', '', text)
        text = re.sub(r'www\.\S+', '', text)
        text = re.sub(r'\S+@\S+', '', text)
        text = re.sub(r'\s+', ' ', text)

        # Generic junk
        text = re.sub(r'(click here|read more|learn more)', '', text, flags=re.IGNORECASE)
        
        if language == 'ko':
            text = re.sub(r'(더보기|자세히|클릭|바로가기)', '', text)
        elif language == 'vi':
            text = re.sub(r'(xem thêm|đọc thêm|chi tiết|nhấn vào)', '', text, flags=re.IGNORECASE)
        
        # Remove leading bullets/numbers
        text = re.sub(r'^[\d\.\)]+\s*', '', text, flags=re.MULTILINE)
        text = re.sub(r'^[•\-\*]\s*', '', text, flags=re.MULTILINE)
        
        return text.strip()

    def _remove_boilerplate_elements(self, soup: BeautifulSoup):
        """
        SAFE boilerplate removal:
        - remove obvious tags first
        - then collect nodes with class/id matches and decompose AFTER iteration
        """
        if soup is None:
            return

        # remove obvious structural boilerplate
        for tag in ['nav', 'header', 'footer', 'aside', 'script', 'style', 'noscript']:
            for element in soup.find_all(tag):
                try:
                    element.decompose()
                except Exception:
                    pass

        # collect nodes to remove (DO NOT decompose during iteration)
        to_remove = []

        for element in soup.find_all(True):
            try:
                for attr in ['class', 'id']:
                    values = element.get(attr)
                    if not values:
                        continue

                    if isinstance(values, str):
                        values = [values]

                    for value in values:
                        if not value:
                            continue
                        value_lower = str(value).lower()
                        # simple contains check is enough here
                        for pattern in self.boilerplate_patterns:
                            if pattern in value_lower:
                                to_remove.append(element)
                                raise StopIteration
            except StopIteration:
                continue
            except Exception:
                # never crash extraction due to a broken node
                continue

        # now decompose
        for el in set(to_remove):
            try:
                el.decompose()
            except Exception:
                pass
    
    def extract_paragraphs(self, html: str, url: str, language: str) -> List[str]:
        """Extract clean paragraphs from HTML"""
        paragraphs = []
        
        # Try trafilatura first
        try:
            text = extract(
                html,
                include_comments=False,
                include_tables=True,
                no_fallback=False,
                output_format='txt'
            )
            if text:
                raw_paragraphs = text.split('\n')
                for para in raw_paragraphs:
                    para = para.strip()
                    if len(para) > 30:
                        cleaned = self.clean_text(para, language)
                        if cleaned:
                            paragraphs.append(cleaned)
        except Exception as e:
            logger.warning(f"Trafilatura failed for {url}: {e}", exc_info=True)
        
        # Fallback to BeautifulSoup if not enough content
        if len(paragraphs) < 3:
            soup = BeautifulSoup(html, 'lxml')

            self._remove_boilerplate_elements(soup)

            # Extract text from common content tags
            for tag in ['p', 'article', 'li', 'div']:
                for element in soup.find_all(tag):
                    try:
                        text = element.get_text(separator=' ', strip=True)
                    except Exception:
                        continue

                    if text and len(text) > 30:
                        cleaned = self.clean_text(text, language)
                        if cleaned:
                            paragraphs.append(cleaned)
        
        # Deduplicate (preserve order)
        seen = set()
        unique = []
        for para in paragraphs:
            if para not in seen:
                seen.add(para)
                unique.append(para)
        
        return unique
    
    def fetch_and_extract(self, url: str, language: str) -> Optional[Tuple[str, List[str], int]]:
        """
        Fetch page and extract clean text immediately
        Returns: (title, paragraphs, char_count) or None
        """
        try:
            if self.should_skip_url(url):
                return None

            response = self.session.get(
                url,
                timeout=self.config.get('timeout', 10),
                allow_redirects=True
            )
            response.raise_for_status()
            
            content_type = response.headers.get('Content-Type', '')
            if 'text/html' not in content_type:
                return None
            
            html = response.text
            soup = BeautifulSoup(html, 'lxml')
            
            title = self.extract_title(soup)
            paragraphs = self.extract_paragraphs(html, url, language)
            
            if not paragraphs or len(paragraphs) < 2:
                return None
            
            full_text = '\n'.join(paragraphs)
            char_count = len(full_text)
            
            # NOTE: min_chars should be set in crawling config
            min_chars = self.config.get('min_chars', 100)
            if char_count < min_chars:
                return None
            
            return title, paragraphs, char_count
            
        except Exception as e:
            logger.warning(f"Failed to fetch/extract {url}: {e}", exc_info=True)
            return None
    
    # ----------------------------
    # Crawl
    # ----------------------------
    def crawl_site(self, seed_url: str, max_pages: int = 1000):
        """
        Crawl site and extract clean text immediately
        Không lưu HTML, chỉ lưu clean text
        """
        domain = urlparse(seed_url).netloc
        queue = deque([seed_url])
        pages_crawled = 0
        
        logger.info(f"Starting crawl of {domain} from {seed_url}")
        
        while queue and pages_crawled < max_pages:
            url = queue.popleft()

            if not url or self.should_skip_url(url):
                continue

            canonical_url = self.canonicalize_url(url)
            if canonical_url in self.visited_urls:
                continue

            # new visit
            self.visited_urls.add(canonical_url)
            self._new_visited_counter += 1
            if self._new_visited_counter >= self.save_visited_every:
                try:
                    self._save_visited()
                    self._new_visited_counter = 0
                except Exception:
                    logger.debug('Failed to flush visited file')
            pages_crawled += 1
            
            # Fetch page
            try:
                response = self.session.get(
                    url,
                    timeout=self.config.get('timeout', 10),
                    allow_redirects=True
                )
                response.raise_for_status()
                
                content_type = response.headers.get('Content-Type', '')
                if 'text/html' not in content_type:
                    continue
                
                html = response.text
                soup = BeautifulSoup(html, 'lxml')
                
            except Exception as e:
                logger.warning(f"Failed to fetch {url}: {e}", exc_info=True)
                continue
            
            logger.info(f"[{pages_crawled}/{max_pages}] Crawled: {url}")
            
            # Detect language
            source_lang = self.detect_language_from_url(url)
            if not source_lang:
                html_tag = soup.find('html')
                if html_tag and html_tag.get('lang'):
                    lang_attr = html_tag.get('lang').lower()
                    if 'ko' in lang_attr:
                        source_lang = 'ko'
                    elif 'vi' in lang_attr or 'vn' in lang_attr:
                        source_lang = 'vi'
            
            # If ko or vi page, find parallel and extract both
            if source_lang in ['ko', 'vi']:
                parallel_url = self.find_parallel_url(url, soup, source_lang)
                
                if parallel_url and not self.should_skip_url(parallel_url):
                    source_result = self.fetch_and_extract(url, source_lang)
                    if source_result:
                        source_title, source_paragraphs, source_chars = source_result
                        
                        target_lang = 'vi' if source_lang == 'ko' else 'ko'
                        target_result = self.fetch_and_extract(parallel_url, target_lang)
                        
                        if target_result:
                            target_title, target_paragraphs, target_chars = target_result
                            
                            # Assign ko/vi correctly
                            if source_lang == 'ko':
                                ko_url, ko_title, ko_paragraphs, ko_chars = url, source_title, source_paragraphs, source_chars
                                vi_url, vi_title, vi_paragraphs, vi_chars = parallel_url, target_title, target_paragraphs, target_chars
                            else:
                                vi_url, vi_title, vi_paragraphs, vi_chars = url, source_title, source_paragraphs, source_chars
                                ko_url, ko_title, ko_paragraphs, ko_chars = parallel_url, target_title, target_paragraphs, target_chars
                            
                            # Deduplicate pair
                            ko_can = self.canonicalize_url(ko_url)
                            vi_can = self.canonicalize_url(vi_url)
                            key = (ko_can, vi_can)

                            if key not in self.seen_pairs:
                                self.seen_pairs.add(key)

                                pair = CleanDocumentPair(
                                    ko_url=ko_url,
                                    vi_url=vi_url,
                                    ko_title=ko_title,
                                    vi_title=vi_title,
                                    ko_paragraphs=ko_paragraphs,
                                    vi_paragraphs=vi_paragraphs,
                                    ko_char_count=ko_chars,
                                    vi_char_count=vi_chars,
                                    source_site=domain,
                                    crawl_timestamp=time.strftime('%Y-%m-%d %H:%M:%S'),
                                    confidence=0.9
                                )
                                
                                self.document_pairs.append(pair)
                                logger.info(
                                    f"✓ Extracted pair: {ko_url} ↔ {vi_url} "
                                    f"(ko:{ko_chars} vi:{vi_chars} chars)"
                                )
                            
                            # Mark parallel as visited too (and possibly flush)
                            par_can = self.canonicalize_url(parallel_url)
                            if par_can not in self.visited_urls:
                                self.visited_urls.add(par_can)
                                self._new_visited_counter += 1
                                if self._new_visited_counter >= self.save_visited_every:
                                    try:
                                        self._save_visited()
                                        self._new_visited_counter = 0
                                    except Exception:
                                        logger.debug('Failed to flush visited file')
            
            # Extract links for crawling
            if pages_crawled < max_pages:
                links = soup.find_all('a', href=True)
                for link in links:
                    href = link.get('href')
                    if not href:
                        continue

                    full_url = urljoin(url, href)
                    if self.should_skip_url(full_url):
                        continue
                    
                    if urlparse(full_url).netloc == domain:
                        canonical = self.canonicalize_url(full_url)
                        if canonical not in self.visited_urls:
                            queue.append(full_url)
            if url.rstrip("/") in ["https://www.jw.org/ko", "https://www.jw.org/ko/"]:
                continue

            # Rate limiting
            time.sleep(self.config.get('delay', 0.5))
    
    # ----------------------------
    # Save
    # ----------------------------
    def save_results(self, output_path: str):
        """Save clean document pairs (NO HTML)"""
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        # If the file already exists, load its seen pairs to avoid duplicates
        existing_pairs: Set[Tuple[str, str]] = set()
        if output_file.exists():
            try:
                with output_file.open('r', encoding='utf-8') as f:
                    for line in f:
                        try:
                            d = json.loads(line)
                            ko = self.canonicalize_url(d.get('ko_url', ''))
                            vi = self.canonicalize_url(d.get('vi_url', ''))
                            if ko and vi:
                                existing_pairs.add((ko, vi))
                        except Exception:
                            continue
            except Exception:
                logger.debug('Failed to read existing output file for dedupe')

        written = 0
        with open(output_file, 'a', encoding='utf-8') as f:
            for pair in self.document_pairs:
                ko_can = self.canonicalize_url(pair.ko_url)
                vi_can = self.canonicalize_url(pair.vi_url)
                key = (ko_can, vi_can)
                if key in existing_pairs:
                    continue
                f.write(json.dumps(asdict(pair), ensure_ascii=False) + '\n')
                existing_pairs.add(key)
                written += 1

        # Ensure visited file flushed at the end
        try:
            self._save_visited()
        except Exception:
            pass

        logger.info(f"Saved {written} new clean document pairs to {output_path} (total in memory: {len(self.document_pairs)})")


def main():
    """Example usage"""
    config = {
        'user_agent': 'Mozilla/5.0 (compatible; KoViDatasetBot/1.0)',
        'timeout': 15,
        'delay': 0.5,
        'min_chars': 100,  # NOTE: should be in crawling config
    }
    
    seed_urls = [
        'https://www.jw.org/ko/',
    ]
    
    crawler = OptimizedCrawler(config)
    
    for seed_url in seed_urls:
        logger.info(f"\n{'='*60}")
        logger.info(f"Crawling: {seed_url}")
        logger.info(f"{'='*60}\n")
        
        try:
            crawler.crawl_site(seed_url, max_pages=50)
        except Exception as e:
            logger.error(f"Error crawling {seed_url}: {e}", exc_info=True)
            continue
    
    crawler.save_results('data/processed/clean_documents.jsonl')
    
    logger.info(f"\n{'='*60}")
    logger.info(f"Crawling complete!")
    logger.info(f"Total document pairs: {len(crawler.document_pairs)}")
    logger.info(f"{'='*60}")


if __name__ == '__main__':
    main()
