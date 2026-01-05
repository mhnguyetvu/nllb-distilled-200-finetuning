

# Korean–Vietnamese Parallel Corpus Pipeline

Pipeline xây dựng song ngữ Hàn–Việt chất lượng cao để fine-tune NLLB-200.

**Mục tiêu:** 200.000 cặp câu chất lượng từ ~1–2 triệu cặp crawl ban đầu.

---

## Tính năng chính

- Crawl website song ngữ, trích xuất văn bản sạch
- Căn chỉnh câu (kss, LaBSE)
- Lọc mạnh mẽ, cấu hình linh hoạt (V2 + V3)
- Chọn mẫu cân bằng nguồn hoặc chất lượng cao nhất
- Các bước pipeline tách biệt, dễ mở rộng

---

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
├── crawl_bilingual_pages.py       # Stage 1: Crawler
├── extract_clean_text.py          # Stage 2: Text extraction
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

