"""
sentence_align.py

Stage 3: Sentence Segmentation & Alignment for Korean-Vietnamese Parallel Corpus

Input:
  data/processed/clean_documents.jsonl
  Each line contains:
    {
      "ko_url": ...,
      "vi_url": ...,
      "ko_title": ...,
      "vi_title": ...,
      "ko_paragraphs": [...],
      "vi_paragraphs": [...],
      "source_site": ...,
      ...
    }

Output:
  data/processed/aligned_sentences.jsonl
  Each line contains aligned sentence pairs:
    {
      "ko_sent": "...",
      "vi_sent": "...",
      "alignment_score": 0.81,
      "doc_ko_url": "...",
      "doc_vi_url": "...",
      "source_site": "...",
      "meta": {...}
    }
"""

import json
import logging
import re
from typing import Any, Dict, List, Tuple

import numpy as np

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


# ----------------------------
# Sentence splitters
# ----------------------------
def split_korean(paragraphs: List[str]) -> List[str]:
    """Split Korean paragraphs into sentences using kss."""
    try:
        import kss
        sents = []
        for p in paragraphs:
            p = p.strip()
            if not p:
                continue
            sents.extend([s.strip() for s in kss.split_sentences(p) if s.strip()])
        return sents
    except Exception as e:
        logger.warning(f"[KSS] Failed, fallback regex: {e}")
        # regex fallback (not perfect)
        text = "\n".join(paragraphs)
        text = re.sub(r"\s+", " ", text)
        parts = re.split(r"(?<=[\.\?\!])\s+", text)
        return [p.strip() for p in parts if p.strip()]


def split_vietnamese(paragraphs: List[str]) -> List[str]:
    """Split Vietnamese paragraphs into sentences using underthesea (fallback regex)."""
    try:
        from underthesea import sent_tokenize
        sents = []
        for p in paragraphs:
            p = p.strip()
            if not p:
                continue
            sents.extend([s.strip() for s in sent_tokenize(p) if s.strip()])
        return sents
    except Exception as e:
        logger.warning(f"[underthesea] Failed, fallback regex: {e}")
        text = "\n".join(paragraphs)
        text = re.sub(r"\s+", " ", text)
        parts = re.split(r"(?<=[\.\?\!])\s+", text)
        return [p.strip() for p in parts if p.strip()]


# ----------------------------
# Alignment helpers
# ----------------------------
def position_bias(i: int, j: int, window: int) -> float:
    """
    Position bias to encourage alignments near diagonal.
    i, j are sentence indices.
    window controls tolerance.
    """
    dist = abs(i - j)
    if dist <= window:
        return 1.0
    return np.exp(- (dist - window) / max(window, 1))


def cosine_sim_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Compute cosine similarity matrix between embeddings a and b."""
    # normalize
    a_norm = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-9)
    b_norm = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-9)
    return np.matmul(a_norm, b_norm.T)


def greedy_align(
    ko_sents: List[str],
    vi_sents: List[str],
    sim: np.ndarray,
    window: int,
    min_score: float
) -> List[Tuple[int, int, float]]:
    """
    Greedy alignment:
      - apply position bias
      - repeatedly pick best remaining pair
      - ensure 1-to-1 mapping
    """
    n, m = sim.shape
    used_ko = set()
    used_vi = set()
    aligned = []

    # apply position bias
    biased = sim.copy()
    for i in range(n):
        for j in range(m):
            biased[i, j] *= position_bias(i, j, window)

    # flatten candidates
    candidates = []
    for i in range(n):
        for j in range(m):
            candidates.append((biased[i, j], i, j))

    # sort descending
    candidates.sort(reverse=True, key=lambda x: x[0])

    for score, i, j in candidates:
        if score < min_score:
            break
        if i in used_ko or j in used_vi:
            continue
        used_ko.add(i)
        used_vi.add(j)
        aligned.append((i, j, float(score)))

    return aligned


# ----------------------------
# Main processing
# ----------------------------
def align_document_pair(
    doc: Dict[str, Any],
    model,
    config: Dict[str, Any]
) -> List[Dict[str, Any]]:
    """
    Align all sentences for a single document pair.
    Returns list of aligned sentence dicts.
    """
    ko_paras = doc.get("ko_paragraphs", [])
    vi_paras = doc.get("vi_paragraphs", [])
    if not ko_paras or not vi_paras:
        return []

    ko_sents = split_korean(ko_paras)
    vi_sents = split_vietnamese(vi_paras)

    # basic filtering
    ko_sents = [s for s in ko_sents if len(s) >= 5]
    vi_sents = [s for s in vi_sents if len(s) >= 5]

    if len(ko_sents) == 0 or len(vi_sents) == 0:
        return []

    # embed
    batch_size = config.get("batch_size", 32)
    ko_emb = model.encode(ko_sents, batch_size=batch_size, convert_to_numpy=True, show_progress_bar=False)
    vi_emb = model.encode(vi_sents, batch_size=batch_size, convert_to_numpy=True, show_progress_bar=False)

    sim = cosine_sim_matrix(ko_emb, vi_emb)

    window = config.get("position_window", 5)
    min_score = config.get("min_alignment_score", 0.5)

    aligned_idx = greedy_align(ko_sents, vi_sents, sim, window, min_score)

    results = []
    for i, j, score in aligned_idx:
        results.append({
            "ko_sent": ko_sents[i],
            "vi_sent": vi_sents[j],
            "alignment_score": score,
            "doc_ko_url": doc.get("ko_url", ""),
            "doc_vi_url": doc.get("vi_url", ""),
            "source_site": doc.get("source_site", "unknown"),
            "meta": {
                "ko_title": doc.get("ko_title", ""),
                "vi_title": doc.get("vi_title", ""),
                "ko_index": i,
                "vi_index": j
            }
        })

    return results


def run_alignment_stage(config: Dict[str, Any]):
    """
    Entry point used by run_pipeline.py:
      from sentence_align import run_alignment_stage
      run_alignment_stage(config)
    """
    align_cfg = config["alignment"]
    input_path = align_cfg["input_file"]
    output_path = align_cfg["output_file"]

    logger.info(f"Loading clean document pairs from: {input_path}")
    docs = []
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            docs.append(json.loads(line))

    logger.info(f"Loaded {len(docs)} document pairs")

    # load LaBSE
    model_name = align_cfg.get("model_name", "sentence-transformers/LaBSE")
    logger.info(f"Loading sentence-transformer model: {model_name}")

    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(model_name)

    total_aligned = 0
    with open(output_path, "w", encoding="utf-8") as out:
        for idx, doc in enumerate(docs, start=1):
            try:
                aligned = align_document_pair(doc, model, align_cfg)
                for pair in aligned:
                    out.write(json.dumps(pair, ensure_ascii=False) + "\n")
                total_aligned += len(aligned)

                if idx % 10 == 0:
                    logger.info(f"[{idx}/{len(docs)}] aligned_sentences={total_aligned}")

            except Exception as e:
                logger.warning(f"Failed aligning doc pair {idx}: {e}", exc_info=True)
                continue

    logger.info(f"Saved {total_aligned} aligned sentence pairs to {output_path}")


# Optional CLI
if __name__ == "__main__":
    import argparse
    import yaml

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    args = parser.parse_args()

    config = yaml.safe_load(open(args.config))
    run_alignment_stage(config)
