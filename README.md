# Korean–Vietnamese Parallel Corpus Pipeline

Production-ready pipeline for building a high-quality Korean↔Vietnamese parallel corpus for fine-tuning NLLB-200.

**Target**: 200,000 high-quality sentence pairs from ~1–2M crawled candidates.

---

## Features

✅ **Bilingual website crawler** with hreflang detection  
✅ **Clean text extraction** removing boilerplate  
✅ **LaBSE-based sentence alignment** with position bias  
✅ **Comprehensive filtering** (V2 + V3 strategies)  
✅ **Source-balanced selection** for dataset diversity  
✅ **Fully configurable** via YAML  

---

## Pipeline Overview

```
Stage 1: Crawl Bilingual Websites
         ↓
Stage 2: Extract Clean Text (remove HTML boilerplate)
         ↓
Stage 3: Sentence Segmentation & Alignment (kss + LaBSE)
         ↓
Stage 4: Filtering Pipeline (13 filters)
         ↓
Stage 5: Final Selection & Train/Dev/Test Split
         ↓
      200k Ko-Vi Dataset
```

---

## Installation

### 1. Install Python dependencies

```bash
pip install -r requirements.txt
```

### 2. Download fastText language detection model

```bash
mkdir -p models
cd models
wget https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin
cd ..
```

### 3. Verify installation

```bash
python -c "import kss; import underthesea; from sentence_transformers import SentenceTransformer; print('✓ All dependencies installed')"
```

---

## Usage

### Quick Start (Run Full Pipeline)

```bash
python run_pipeline.py --config config.yaml --stage all
```

### Run Individual Stages

```bash
# Stage 1: Crawl bilingual websites
python run_pipeline.py --config config.yaml --stage crawl

# Stage 2: Extract clean text
python run_pipeline.py --config config.yaml --stage extract

# Stage 3: Align sentences
python run_pipeline.py --config config.yaml --stage align

# Stage 4: Filter corpus
python run_pipeline.py --config config.yaml --stage filter

# Stage 5: Select final dataset
python run_pipeline.py --config config.yaml --stage select
```

---

## Configuration

Edit `config.yaml` to customize:

### Crawling
- **seed_urls**: List of bilingual websites to crawl
- **max_pages_per_site**: Maximum pages to crawl per site
- **delay**: Seconds between requests (politeness)

### Alignment
- **min_alignment_score**: Minimum LaBSE similarity (0.5 = moderate, 0.7 = high)
- **position_window**: Context window for position bias

### Filtering
- **min_labse_similarity**: Semantic similarity threshold
- **min_hangul_ratio**: Korean must be ≥30% Hangul
- **min_latin_ratio**: Vietnamese must be ≥50% Latin
- **spam_keywords**: Add domain-specific spam terms

### Selection
- **target_size**: Final dataset size (default: 200k)
- **strategy**: `top_quality` or `source_balanced`
- **train_ratio**: Train/dev/test split ratios

---

## Output Format

Final dataset (`data/final/kovi_train.jsonl`):

```json
{"ko": "한국어 문장입니다.", "vi": "Đây là câu tiếng Việt.", "score": 0.85, "source": "jw.org"}
{"ko": "또 다른 예시입니다.", "vi": "Đây là ví dụ khác.", "score": 0.82, "source": "liveinkorea.kr"}
```

---

## Filtering Pipeline

The pipeline implements **13 filters** combining V2 and V3 strategies:

### V2 Filters
1. **Language Detection** (fastText): Verify ko/vi languages
2. **Exact Deduplication**: Remove duplicate Korean sentences
3. **LaBSE Semantic Similarity**: Cross-lingual coherence check
4. **Length Ratio**: Character-level length balance
5. **Content Quality**: Length, digits, special chars, URLs
6. **Repetition Detection**: Remove repetitive content

### V3 Filters
7. **Web Artifacts**: Navigation, cookies, UI elements
8. **Commercial Spam**: B2B keywords, promotions
9. **Fragment Detection**: Incomplete sentences
10. **MT Artifacts**: Translation notes, untranslated text
11. **Script Validation**: Hangul ratio (ko), Latin ratio (vi)
12. **Token Ratio**: Word-level length balance

### Expected Filter Rates
- Input: ~1–2M sentence pairs (from crawling)
- After filtering: ~300–500k pairs
- Final selection: 200k pairs

---

## Recommended Bilingual Sites for Ko-Vi

### Religious/Educational
- **jw.org** (Jehovah's Witnesses) - High quality, extensive ko/vi
- **biblegateway.com** - Religious texts

### Government/Immigration
- **liveinkorea.kr** - Korea immigration portal
- **hikorea.go.kr** - Immigration services

### News/Media
- **arirang.com** - Korean international broadcasting
- Look for Vietnamese news sites with Korean versions

### E-Learning
- **korean.go.kr** - National Institute of Korean Language
- TOPIK preparation sites with Vietnamese

### Tourism
- **visitkorea.or.kr** - Korea tourism
- Korean embassy sites in Vietnam

### How to Find More
1. Search: `site:*.kr "?lang=vi"` or `site:*.kr "/vi/"`
2. Look for language switchers on Korean government sites
3. Check Korean companies with Vietnam operations

---

## Project Structure

```
ko-vi-corpus/
├── config.yaml                    # Main configuration
├── requirements.txt               # Python dependencies
├── README.md                      # This file
├── run_pipeline.py                # Main pipeline orchestrator
├── crawl_bilingual_pages.py      # Stage 1: Crawler
├── extract_clean_text.py         # Stage 2: Text extraction
├── sentence_align.py              # Stage 3: Alignment
├── filter_pipeline.py             # Stage 4: Filtering
├── data/
│   ├── raw/                       # Crawled HTML
│   ├── processed/                 # Clean text + alignments
│   └── final/                     # Final dataset
│       ├── kovi_train.jsonl       # Training set
│       ├── kovi_dev.jsonl         # Dev set
│       └── kovi_test.jsonl        # Test set
├── models/
│   └── lid.176.bin                # fastText LID model
└── logs/                          # Pipeline logs
```

---

## Performance Notes

### Hardware Requirements
- **Minimum**: 16GB RAM, CPU
- **Recommended**: 32GB RAM, GPU (for LaBSE alignment)
- **Storage**: ~10GB for models + data

### Speed Estimates (on GPU)
- Crawling: ~500 pages/hour (limited by politeness delay)
- Text extraction: ~1000 docs/minute
- Sentence alignment: ~50 doc pairs/minute (LaBSE on GPU)
- Filtering: ~5000 sentences/minute

### Full Pipeline Runtime
- **Crawling**: 10–20 hours (for 5–10k pages across multiple sites)
- **Processing**: 4–8 hours (extraction + alignment + filtering)
- **Total**: ~1–2 days for complete 200k dataset

---

## Quality Assurance

### Automatic Checks
- Language detection (fastText confidence ≥0.8)
- Semantic similarity (LaBSE ≥0.5)
- Script validation (Hangul/Latin ratios)
- Length ratios (character + token)

### Manual Inspection (Recommended)
1. Sample 100 random pairs from final dataset
2. Check for:
   - Translation accuracy
   - Fluency in both languages
   - Alignment correctness
   - Domain appropriateness

### Debugging Failed Filters
Check `logs/filtering.log` for detailed per-filter statistics.

---

## Troubleshooting

### Issue: "ModuleNotFoundError: No module named 'kss'"
**Solution**: Install Korean dependencies:
```bash
pip install kss kiwipiepy
```

### Issue: "fastText model not found"
**Solution**: Download LID model:
```bash
wget https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin -P models/
```

### Issue: "CUDA out of memory"
**Solution**: Reduce batch size in `config.yaml`:
```yaml
alignment:
  batch_size: 16  # reduce from 32
```

### Issue: "Too few sentence pairs after filtering"
**Solutions**:
1. Lower filtering thresholds in `config.yaml`:
   ```yaml
   filtering:
     min_labse_similarity: 0.4  # from 0.5
     max_length_ratio: 4.0      # from 3.0
   ```
2. Add more seed URLs
3. Increase max_pages_per_site

---

## Fine-tuning NLLB-200

After building your corpus, fine-tune NLLB-200:

```python
from transformers import NllbTokenizer, AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments, Seq2SeqTrainer

# Load model
model = AutoModelForSeq2SeqLM.from_pretrained("facebook/nllb-200-distilled-600M")
tokenizer = NllbTokenizer.from_pretrained("facebook/nllb-200-distilled-600M")

# Load your dataset
train_data = load_jsonl("data/final/kovi_train.jsonl")

# Set language codes
tokenizer.src_lang = "kor_Hang"  # Korean
tokenizer.tgt_lang = "vie_Latn"  # Vietnamese

# Fine-tune (see HuggingFace documentation for full training loop)
```

---

## Citation

If you use this pipeline, please cite:

```bibtex
@misc{kovi_corpus_pipeline,
  title={Korean-Vietnamese Parallel Corpus Pipeline},
  author={Your Name},
  year={2025},
  url={https://github.com/yourusername/ko-vi-corpus}
}
```

---

## License

MIT License - See LICENSE file

---

## Contributing

Contributions welcome! Please:
1. Add new bilingual sites to `config.yaml`
2. Improve filters in `filter_pipeline.py`
3. Report issues with specific examples

---

## Contact

For questions or issues, open a GitHub issue or contact: your.email@example.com