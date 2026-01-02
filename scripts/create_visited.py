"""Create visited_urls.txt from existing clean_documents.jsonl

Usage:
    python scripts/create_visited.py
    python scripts/create_visited.py --input data/processed/clean_documents.jsonl --output data/state/visited_urls.txt

This script canonicalizes URLs similarly to the crawler and writes one normalized URL per line.
"""
from pathlib import Path
import json
import argparse
from urllib.parse import urlparse, parse_qs, urlencode


def canonicalize_url(url: str) -> str:
    if not url:
        return ''
    parsed = urlparse(url)
    query = parse_qs(parsed.query)
    sorted_query = urlencode(sorted(query.items()), doseq=True)
    canonical = f"{parsed.scheme}://{parsed.netloc}{parsed.path}"
    if sorted_query:
        canonical += f"?{sorted_query}"
    return canonical.lower()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--input', type=str, default='data/processed/clean_documents.jsonl')
    p.add_argument('--output', type=str, default='data/state/visited_urls.txt')
    args = p.parse_args()

    inp = Path(args.input)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)

    seen = set()
    if not inp.exists():
        print(f"Input file not found: {inp}")
        return

    with inp.open('r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except Exception:
                continue
            ko = d.get('ko_url') or d.get('doc_ko_url') or ''
            vi = d.get('vi_url') or d.get('doc_vi_url') or ''
            if ko:
                k = canonicalize_url(ko)
                if k:
                    seen.add(k)
            if vi:
                v = canonicalize_url(vi)
                if v:
                    seen.add(v)

    with out.open('w', encoding='utf-8') as f:
        for u in sorted(seen):
            f.write(u + '\n')

    print(f"Wrote {len(seen)} visited URLs to {out}")


if __name__ == '__main__':
    main()
