# NLLB-200 Fine-tuning for Korean-Vietnamese Translation

Comprehensive pipeline for fine-tuning NLLB-200-distilled-600M on Korean-Vietnamese parallel data with advanced quality filtering and semantic alignment.

##  Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Requirements](#requirements)
- [Quick Start](#quick-start)
- [Data Preparation Pipeline](#data-preparation-pipeline)
- [Training Workflow](#training-workflow)
- [Results](#results)
- [Project Structure](#project-structure)
- [Troubleshooting](#troubleshooting)

---

##  Overview

This project implements a complete pipeline for fine-tuning Facebook''s NLLB-200 model on Korean-Vietnamese translation, achieving significant improvements through:

- **Multi-stage quality filtering**: Basic + Enhanced + Semantic alignment
- **Threshold optimization**: Automated sweep to find optimal data quality
- **Production-ready training**: Optimized configs for A100 GPUs

**Key Results:**
- BLEU: **24.31** on high-quality test, **24.08** on diverse test
- COMET: **0.87** (both test sets)
- Improvement: **+2.8% to +9.9%** over pretrained baseline
- Dataset: 27K high-quality pairs (semantic threshold 0.80)

---

##  Features

### 1. Advanced Data Filtering
- **Basic Filtering**: Length, ratio, special characters
- **Enhanced Filtering**: 
  - Language ID verification (langdetect, 80% confidence)
  - Near-duplicate detection (MinHash LSH, 85% similarity)
  - Boilerplate/subtitle noise removal
  - Number consistency checking (30% tolerance)
  - Punctuation density control (<30%)
  - Useless phrase filtering

### 2. Semantic Alignment
- **LaBSE embeddings** for cross-lingual similarity
- **Threshold sweep** (0.65, 0.70, 0.75, 0.80) to optimize quality vs quantity
- **Automatic comparison** to select best threshold

### 3. Automated Training Pipeline
- Train/dev/test splitting (95/2.5/2.5)
- Multi-threshold training and evaluation
- Comprehensive metrics (BLEU, chrF, TER, COMET)
- Best model selection

---

##  Requirements

### Environment Setup

```bash
# Create conda environment
conda create -n nllb-finetuning python=3.10
conda activate nllb-finetuning

# Install dependencies
pip install -r requirements.txt
```

### Key Dependencies

```
transformers>=4.30.0
datasets>=2.14.0
torch>=2.0.0
sentence-transformers>=2.2.0
sacrebleu>=2.3.0
unbabel-comet>=2.0.0
langdetect>=1.0.9
datasketch>=1.6.0
pyyaml>=6.0
tqdm>=4.65.0
```

---

##  Quick Start

### For New Users (Clone & Run)

```bash
# 1. Clone repository
git clone <your-repo-url>
cd nllb-distilled-200-finetuning

# 2. Setup environment
conda create -n nllb-finetuning python=3.10
conda activate nllb-finetuning
pip install -r requirements.txt

# 3. Prepare your data
# Option A: Download OPUS data (if you don't have data yet)
pip install opustools-pkg
opus_read -d OpenSubtitles -s ko -t vi -wm moses -w data/raw/opus_ko_vi.txt
# Convert to JSONL format (ko/vi fields) - see OPUS documentation

# Option B: Use your own parallel data
# Place your data in OPUS format: data/raw/your_data.jsonl
# Required format: {"translation": {"kor_Hang": "한국어", "vie_Latn": "Tiếng Việt"}}

# 4. Run complete pipeline (automated)
# If you have raw data, run filtering + sweep
python filter_opus_enhanced.py \
    --input data/raw/your_data.jsonl \
    --output-dir data/sweep \
    --mode sweep \
    --thresholds 0.75 0.80 \
    --batch-size 32

python quick_sweep_train.py \
    --base-dir . \
    --filtered-dir data/sweep \
    --thresholds 0.75 0.80 \
    --max-steps 5000

# Option B: If you already have filtered data at threshold 0.80
python training/finetune_nllb.py --config final_config_80.yaml

# 5. View results
python analyze_eval_results.py results/comparison.json
```

### Step-by-Step Workflow

1. **Environment Setup** (5 minutes)
   - Create conda environment with Python 3.10
   - Install all dependencies from requirements.txt

2. **Data Preparation** (1-2 hours depending on dataset size)
   - Place raw parallel data in `data/raw/`
   - Run semantic filtering with multiple thresholds
   - Creates train/dev/test splits automatically

3. **Threshold Optimization** (1-2 hours on GPU)
   - Quick training (5K steps) on each threshold
   - Automatic evaluation and comparison
   - Identifies best quality/quantity trade-off

4. **Production Training** (1.5-2 hours on A100)
   - Full 10-epoch training on optimal threshold
   - Final BLEU: **24.31** (high-quality test), **24.08** (diverse test)
   - Model checkpoint: `outputs/final_semantic_80/checkpoint-1000`
   - Proven generalization: +9.9% improvement on challenging data

---

##  Data Format Requirements

### Input Data Format

All scripts expect JSONL (JSON Lines) format with the following structure:

```json
{"translation": {"kor_Hang": "한국어 문장", "vie_Latn": "Câu tiếng Việt"}}
{"translation": {"kor_Hang": "또 다른 문장", "vie_Latn": "Câu khác"}}
```

**Field Requirements:**
- `translation`: Object containing source and target language
- `kor_Hang`: Korean text in Hangul script (source)
- `vie_Latn`: Vietnamese text in Latin script (target)
- Each line must be valid JSON
- One sentence pair per line

### Getting Data

#### Option 1: OPUS Corpus (Recommended for beginners)

```bash
# Install opustools
pip install opustools-pkg

# Download Korean-Vietnamese parallel data from OpenSubtitles
opus_read -d OpenSubtitles -s ko -t vi -wm moses -w data/raw/opus_ko_vi.txt

# Convert to JSONL format (you'll need to write a simple converter)
# Expected output: data/raw/opus_ko_vi.jsonl
```

**Available OPUS datasets for ko-vi:**
- OpenSubtitles: Movie/TV subtitles (~200K pairs)
- Tatoeba: Short sentences (~10K pairs)
- WikiMatrix: Wikipedia sentences (~50K pairs)

#### Option 2: Custom Data

If you have your own parallel data:

```python
# Example converter script
import json

# Assuming you have parallel files: ko.txt and vi.txt
with open('ko.txt', 'r', encoding='utf-8') as f_ko, \
     open('vi.txt', 'r', encoding='utf-8') as f_vi, \
     open('data/raw/custom.jsonl', 'w', encoding='utf-8') as f_out:
    
    for ko_line, vi_line in zip(f_ko, f_vi):
        obj = {
            "translation": {
                "kor_Hang": ko_line.strip(),
                "vie_Latn": vi_line.strip()
            }
        }
        f_out.write(json.dumps(obj, ensure_ascii=False) + '\n')
```

### Data Validation

Before running the pipeline, validate your data format:

```bash
# Quick check
python -c "
import json
with open('data/raw/your_data.jsonl', 'r', encoding='utf-8') as f:
    for i, line in enumerate(f, 1):
        try:
            obj = json.loads(line)
            assert 'translation' in obj
            assert 'kor_Hang' in obj['translation']
            assert 'vie_Latn' in obj['translation']
            if i == 1:
                print('✓ Format valid!')
                print(f'Sample: {obj}')
            if i >= 5:
                break
        except Exception as e:
            print(f'✗ Line {i} error: {e}')
            break
"
```

---

##  Configuration & Paths

### Important: Update Absolute Paths

⚠️ **Before running training scripts, you MUST update absolute paths in config files:**

#### In `final_config_80.yaml`:

```yaml
# CHANGE THESE PATHS to match your system:
model_name_or_path: facebook/nllb-200-distilled-600M  # OK as-is (HuggingFace hub)

# UPDATE these paths:
output_dir: /path/to/your/project/outputs/final_semantic_80
train_file: /path/to/your/project/data/sweep/semantic_80/nllb_train.jsonl
validation_file: /path/to/your/project/data/sweep/semantic_80/nllb_dev.jsonl
test_file: /path/to/your/project/data/sweep/semantic_80/nllb_test.jsonl
logging_dir: /path/to/your/project/logs
```

**Quick find & replace:**
```bash
# On Windows (PowerShell)
(Get-Content final_config_80.yaml) -replace '/data/AITeam/nguyetnvm/nllb', 'C:/Users/YourName/Documents/nllb-distilled-200-finetuning' | Set-Content final_config_80.yaml

# On Linux/Mac
sed -i 's|/data/AITeam/nguyetnvm/nllb|/home/yourname/nllb-distilled-200-finetuning|g' final_config_80.yaml
```

### Dependencies Installation Order

Install dependencies in this order to avoid conflicts:

```bash
# 1. Base dependencies (data processing, embeddings)
pip install -r requirements.txt

# 2. Filtering dependencies (language detection, deduplication)
pip install -r requirements_filter.txt

# 3. Training dependencies (transformers, evaluation metrics)
pip install -r training/requirements_train.txt

# 4. Optional: Install flash-attention for 2-3x speedup (requires CUDA)
# pip install flash-attn --no-build-isolation
```

**Note:** If you only need training (already have filtered data), you can skip steps 1-2.

---

##  Data Preparation Pipeline

### Overview

```
Raw Data (OPUS/Custom)
    
Basic Filtering (length, ratio, special chars)
    
Enhanced Filtering (LangID, dedup, boilerplate)
    
Semantic Filtering (LaBSE similarity)
    
Threshold Sweep (0.75, 0.80, 0.85...)
    
Dataset Splitting (95/2.5/2.5)
    
Training & Evaluation
```

### Complete Pipeline Example

```bash
# 1. Basic filtering
python filter_pipeline.py \
    --input data/raw/your_data.jsonl \
    --output data/processed/filtered.jsonl \
    --min-length 5 \
    --max-length 256

# 2. Enhanced + Semantic filtering with threshold sweep
python filter_opus_enhanced.py \
    --input data/processed/filtered.jsonl \
    --output-dir data/sweep_filtered \
    --mode sweep \
    --thresholds 0.75 0.80 \
    --batch-size 32

# 3. Quick training sweep to find best threshold
python quick_sweep_train.py \
    --base-dir . \
    --filtered-dir data/sweep_filtered \
    --thresholds 0.75 0.80 \
    --max-steps 5000

# 4. Full training on best threshold
python training/finetune_nllb.py --config final_config_80.yaml
```

---

##  Results

### Final Production Results (10 Epochs on Threshold 0.80)

**Test on High-Quality Data (semantic_80, 908 samples):**

| Model | BLEU | chrF | TER | COMET | Improvement |
|-------|------|------|-----|-------|-------------|
| Baseline (pretrained) | 23.65 | 44.91 | 64.97 | 0.87 | - |
| **Fine-tuned (ours)** | **24.31** | **45.23** | **64.07** | **0.87** | **+2.8% BLEU** |

**Test on Diverse Data (semantic_75, 1,399 samples):**

| Model | BLEU | chrF | TER | COMET | Improvement |
|-------|------|------|-----|-------|-------------|
| Baseline (pretrained) | 21.92 | 42.73 | 68.53 | 0.86 | - |
| **Fine-tuned (ours)** | **24.08** | **44.40** | **65.72** | **0.86** | **+9.9% BLEU** |

### Key Findings

✅ **Model Generalizes Well - No Overfitting!**
- Fine-tuned on high-quality data (threshold 0.80, 27K pairs)
- **Better improvement on challenging data** (+9.9%) than on easy data (+2.8%)
- Baseline struggles with lower quality: BLEU drops 23.65→21.92 (-7.3%)
- Fine-tuned model robust: BLEU drops only 24.31→24.08 (-0.9%)

✅ **Quality > Quantity Validated:**
- 27K high-quality pairs (threshold 0.80) outperform larger datasets
- Semantic alignment critical for translation quality
- Production-ready: Consistent performance across different data distributions

### Threshold Comparison (Quick Sweep - 5K steps)

| Threshold | Data Size | BLEU | chrF | TER | COMET |
|-----------|-----------|------|------|-----|-------|
| 0.65 (baseline) | 59K | 19.89 | 40.46 | 71.63 | 0.830 |
| 0.75 | 36K | 22.91 | 43.53 | 66.14 | 0.860 |
| **0.80 (winner)** | **27K** | **23.85** | **44.91** | **64.67** | **0.869** |

*Quick sweep results used to select optimal threshold for full production training.*

---

##  Testing & Inference

### Quick Inference Test

Test model quality and performance before/after fine-tuning with `quick_inference_test.py`:

```bash
# Test pretrained baseline model
python training/quick_inference_test.py \
    --model facebook/nllb-200-distilled-600M \
    --test data/final/nllb_test.jsonl \
    --num-samples 100

# Test your fine-tuned model
python training/quick_inference_test.py \
    --model outputs/final_semantic_80 \
    --test data/final/nllb_test.jsonl \
    --num-samples 100 \
    --output results/inference_finetuned.json
```

**What it measures:**
- **Quality Metrics**: BLEU score with sample translations
- **Performance Metrics**: 
  - Translation speed (sentences/sec)
  - GPU memory usage (before/after/peak)
  - Model loading time
  - Token generation speed
- **Sample Outputs**: First 5 translations for visual inspection

**Typical Results:**
- Baseline model: ~15-20 BLEU, ~25-30 sent/s on A100
- Fine-tuned model: ~23-27 BLEU, ~25-30 sent/s on A100
- GPU memory: ~2-3GB peak for 600M model

Use this to quickly validate model improvements without running full evaluation suite.

---

##  Project Structure

```
nllb-distilled-200-finetuning/
├── README.md                      # This file
├── requirements.txt               # Python dependencies
├── requirements_filter.txt        # Additional filtering dependencies
├── .gitignore                     # Git ignore rules
│
├── filter_opus_enhanced.py        # Main filtering script (basic + enhanced + semantic)
├── quick_sweep_train.py          # Automated threshold comparison training
├── review_samples.py             # Generate HTML report for manual quality review
├── analyze_eval_results.py       # Compare evaluation results across thresholds
├── final_config_80.yaml          # Production training config (threshold 0.80)
│
├── training/
│   ├── finetune_nllb.py          # Main training script
│   ├── evaluate_model.py         # Comprehensive evaluation (BLEU/chrF/TER/COMET)
│   ├── quick_inference_test.py   # Quick quality & performance test
│   └── requirements_train.txt     # Training-specific dependencies
│
├── data/
│   ├── raw/                      # Raw parallel data (user-provided)
│   ├── sweep/                    # Filtered data at multiple thresholds
│   │   ├── semantic_75/         # Threshold 0.75 data
│   │   └── semantic_80/         # Threshold 0.80 data (production)
│   └── final/                    # Final production dataset
│
├── outputs/                      # Model checkpoints (gitignored)
├── logs/                         # Training logs (gitignored)
└── results/                      # Evaluation results (gitignored)
```

### Key Scripts

- **filter_opus_enhanced.py**: Complete filtering pipeline with semantic alignment
- **quick_sweep_train.py**: Automates training on multiple thresholds for comparison
- **quick_inference_test.py**: Fast quality & performance validation (before/after training)
- **review_samples.py**: Generates HTML report with 100 random samples for manual review
- **analyze_eval_results.py**: Compares metrics across different thresholds

---

##  Troubleshooting

### Common Issues

**1. CUDA Out of Memory**
```bash
# Reduce batch size in config
per_device_train_batch_size: 4  # Instead of 8
gradient_accumulation_steps: 4  # To maintain effective batch size
```

**2. Windows Virtual Memory Error**
```bash
# Increase pagefile size or use WSL2
# See FULL_GUIDE.md for detailed fix
```

**3. Slow Filtering**
```bash
# Use smaller batch size for semantic filtering
python filter_opus_enhanced.py --batch-size 16  # Instead of 32
```

**4. Import Errors**
```bash
# Ensure all dependencies installed
pip install -r requirements.txt
pip install -r requirements_filter.txt
pip install -r training/requirements_train.txt
```

---

For complete documentation, see the full README.md
