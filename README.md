# NLLB-200 Korean-Vietnamese Finetuning

Fine-tuning [NLLB-200-distilled-600M](https://huggingface.co/facebook/nllb-200-distilled-600M) for **Korean ↔ Vietnamese** translation using a diverse 300K parallel corpus from OPUS.

## 📊 Project Overview

**Objective**: Improve translation quality on diverse domains (news, media, tech, conversation) beyond religious-only texts.

**Approach**: 
- Downloaded 5 high-quality parallel corpora from OPUS
- Created balanced 300K dataset with quality filtering
- Fine-tuned on NVIDIA A100 with BF16 mixed precision
- Evaluated with SacreBlEU metric

**Current Status**:
- ✅ OPUS corpus download (TED2020, OpenSubtitles, WikiMatrix, CCMatrix, QED)
- ✅ Balanced dataset creation (300K pairs → 285K train / 7.5K dev / 7.5K test)
- ✅ Training pipeline with A100 optimizations
- 🔄 Model training in progress (on A100 server)

## 📁 Repository Structure

```
nllb-distilled-200-finetuning/
├── training/                              # Training scripts (transfer to A100)
│   ├── finetune_nllb.py                  # Main training script
│   ├── finetune_config_balanced.yaml     # Training configuration
│   ├── evaluate_model.py                 # Full evaluation script
│   ├── quick_inference_test.py           # Quick baseline testing
│   ├── requirements_train.txt            # Training dependencies
│   └── README.md                          # A100 training guide
├── data/
│   ├── final_balanced/                   # Balanced 300K dataset (CURRENT)
│   │   ├── nllb_train.jsonl             # 285,000 pairs
│   │   ├── nllb_dev.jsonl               # 7,500 pairs
│   │   └── nllb_test.jsonl              # 7,500 pairs
│   └── opus/                             # OPUS corpus files
├── requirements.txt                       # Base dependencies
├── A100_TRANSFER_GUIDE.md                # A100 server setup instructions
└── README.md                              # This file
```

## 🚀 Quick Start

### Option 1: Use Pre-trained Baseline

```bash
pip install transformers torch

python training/quick_inference_test.py \
    --model facebook/nllb-200-distilled-600M \
    --test data/final_balanced/nllb_test.jsonl \
    --num-samples 100
```

**Expected**: BLEU ~14-15 on balanced test set

### Option 2: Fine-tune on A100

**Prerequisites**: NVIDIA A100 GPU (40GB or 80GB), CUDA 11.8+

```bash
# 1. Transfer files to A100 server
# See A100_TRANSFER_GUIDE.md for complete instructions

# 2. Setup environment on A100
conda create -n nllb-train python=3.10 -y
conda activate nllb-train
cd training
pip install -r requirements_train.txt

# 3. Start training
python finetune_nllb.py --config finetune_config_balanced.yaml

# 4. Monitor progress
# Training takes ~4 hours on A100 80GB
# Checkpoints saved every 1000 steps
# Evaluation every 500 steps

# 5. Evaluate final model
python evaluate_model.py \
    --model ../outputs/nllb-balanced-300k \
    --test ../data/final_balanced/nllb_test.jsonl
```

**Expected**: BLEU ~30-35 on balanced test set (+115-140% improvement over baseline)

## 📦 Dataset Details

### Data Filtering Pipeline

OPUS corpus goes through multiple filtering stages to ensure quality:

| Stage | File | Pairs | Size | Description |
|-------|------|-------|------|-------------|
| **1. Raw OPUS** | `opus_combined.jsonl` | 1,721,638 | 339 MB | All downloaded corpora combined |
| **2. Balanced Sample** | `opus_balanced_300k.jsonl` | 180,000 | 41 MB | Balanced across 4 domains |
| **3. Basic Filter** | `opus_filtered_basic.jsonl` | 93,429 | 21 MB | Length, script, dedup filters |
| **4. Semantic Filter** | `opus_filtered_semantic.jsonl` | 58,889 | 13 MB | LaBSE similarity ≥0.65 |
| **5. Train/Dev/Test** | `data/final_balanced/` | 285,000 | 76 MB | Split from balanced sample |

### Quality Filtering Criteria

**Basic Filters** ([filter_opus_quality.py](filter_opus_quality.py)):
1. ✅ **Length ratio check**: 0.5-2.0x between Korean-Vietnamese
2. ✅ **Korean Hangul validation**: ≥30% Hangul characters
3. ✅ **Vietnamese script check**: Latin script presence
4. ✅ **Minimum length**: ≥5 characters
5. ✅ **Deduplication**: Remove duplicate source texts
   - Retention: 93,429 / 180,000 = **51.9%**

**Semantic Similarity Filter** (LaBSE embeddings):
6. ✅ **Semantic alignment**: Cosine similarity ≥0.65
   - Uses [LaBSE model](https://huggingface.co/sentence-transformers/LaBSE) (Language-agnostic BERT)
   - Removes misaligned pairs (especially from OpenSubtitles)
   - Retention: 58,889 / 93,429 = **63.0%**
   - Overall: 58,889 / 180,000 = **32.7%** (high quality pairs)

### Current Training Dataset (data/final_balanced/)

| Split | Pairs | File Size |
|-------|-------|-----------|
| Train | 285,000 | 76 MB |
| Dev | 7,500 | 2 MB |
| Test | 7,500 | 2 MB |

**Source Breakdown** (by alignment quality):
- 🎤 **TED2020** (80K pairs): Conference talks, educational content - ⭐⭐⭐ High quality (manual translation)
- 🌐 **CCMatrix** (70K pairs): Web-crawled parallel sentences - ⭐⭐⭐ High quality (margin-based mining)
- 📰 **WikiMatrix** (50K pairs): Wikipedia articles, encyclopedic text - ⭐⭐ Good quality
- 🎬 **OpenSubtitles** (100K pairs): Movie/TV subtitles, conversational language - ⭐ Variable quality (time-based alignment)

**⚠️ Note**: Current training uses balanced sample (285K) without semantic filtering. For higher quality, use semantic filtered dataset (58K pairs).

### Data Format

NLLB-compatible JSONL format:

```json
{"translation": {"kor_Hang": "안녕하세요", "vie_Latn": "Xin chào"}}
{"translation": {"kor_Hang": "좋은 아침입니다", "vie_Latn": "Chào buổi sáng"}}
```

## 🧹 Data Quality Improvement

To create a higher quality dataset with semantic filtering:

```bash
# Step 1: Basic filtering (fast, 1-2 min)
python filter_opus_quality.py \
    --input data/opus/opus_balanced_300k.jsonl \
    --output data/opus/opus_filtered_basic.jsonl \
    --no-semantic

# Step 2: Semantic filtering (requires GPU, 15-25 min on RTX 3050 Ti)
python filter_opus_quality.py \
    --input data/opus/opus_filtered_basic.jsonl \
    --output data/opus/opus_filtered_semantic.jsonl \
    --similarity-threshold 0.65 \
    --batch-size 16

# Step 3: Split into train/dev/test (similar to split_opus_dataset.py)
# Creates new data/final_semantic/ directory with 95/2.5/2.5 split
```

**Expected Results**:
- Input: 180K balanced pairs
- After basic: 93K pairs (52% retention)
- After semantic: 59K pairs (33% overall retention)
- **Quality**: High - misaligned pairs removed via LaBSE similarity

## 🔧 Training Configuration

**Model**: `facebook/nllb-200-distilled-600M` (615M parameters)

**Key Hyperparameters**:
- Batch size: 32 (per device)
- Learning rate: 5e-5
- Epochs: 5
- Optimizer: AdamW (fused)
- Mixed precision: BF16 (A100 native)
- Gradient checkpointing: Enabled

**A100 Optimizations**:
- BF16 mixed precision (better than FP16 on A100)
- TF32 tensor cores enabled
- Fused AdamW optimizer
- Gradient checkpointing for memory efficiency

**Evaluation**:
- Metric: SacreBlEU
- Beam search: 5 beams
- Frequency: Every 500 steps
- Best model selection: Highest BLEU score

## 🧪 Testing & Evaluation

### Quick Baseline Test (100 samples)

```bash
python training/quick_inference_test.py \
    --model facebook/nllb-200-distilled-600M \
    --test data/final_balanced/nllb_test.jsonl \
    --num-samples 100 \
    --batch-size 8 \
    --num-beams 5
```

### Full Evaluation (7500 samples)

```bash
python training/evaluate_model.py \
    --model outputs/nllb-balanced-300k \
    --test data/final_balanced/nllb_test.jsonl \
    --batch-size 16 \
    --num-beams 5 \
    --output results/evaluation.json
```

### Compare Models

```bash
# Baseline
python training/evaluate_model.py \
    --model facebook/nllb-200-distilled-600M \
    --test data/final_balanced/nllb_test.jsonl \
    --output results/baseline.json

# Finetuned
python training/evaluate_model.py \
    --model outputs/nllb-balanced-300k \
    --test data/final_balanced/nllb_test.jsonl \
    --output results/finetuned.json
```

## 📈 Expected Results

| Model | BLEU | Improvement |
|-------|------|-------------|
| Pretrained baseline | ~14.56 | - |
| Fine-tuned (balanced 300K) | ~30-35 | +115-140% |

*Note: Results measured on balanced test set with diverse domains*

## 🛠️ Technical Details

### Training Script Features

**[finetune_nllb.py](training/finetune_nllb.py)**:
- HuggingFace Seq2SeqTrainer integration
- YAML-based configuration
- Automatic checkpoint resumption
- SacreBlEU metric with beam search
- TensorBoard logging
- Early stopping support
- Token overflow protection (vocab clipping)

**Configuration**:
- Single YAML file for all hyperparameters
- Easy A100 optimization toggles
- Flexible batch size and memory settings

### Evaluation Scripts

**[quick_inference_test.py](training/quick_inference_test.py)**:
- Fast baseline testing (100 samples)
- Performance metrics (speed, memory)
- Sample translation display

**[evaluate_model.py](training/evaluate_model.py)**:
- Full test set evaluation
- Detailed metrics and error analysis
- JSON output for comparison

## 🐛 Troubleshooting

### Out of Memory Error

```yaml
# Reduce batch size in config
per_device_train_batch_size: 16  # Try 8 or 4
gradient_accumulation_steps: 2   # Increase to maintain effective batch size
```

### Training Too Slow

```bash
# Install Flash Attention 2 for 2-3x speedup
pip install flash-attn --no-build-isolation

# Enable in config (already default)
advanced:
  use_flash_attention: true
```

### Resume from Checkpoint

```bash
python finetune_nllb.py \
    --config finetune_config_balanced.yaml \
    --resume ../outputs/nllb-balanced-300k/checkpoint-1000
```

## 📝 Dependencies

**Base requirements** ([requirements.txt](requirements.txt)):
- Data processing: requests, beautifulsoup4, trafilatura
- Korean NLP: kss, kiwipiepy
- Vietnamese NLP: underthesea
- ML utilities: torch, sentence-transformers

**Training requirements** ([training/requirements_train.txt](training/requirements_train.txt)):
- transformers>=4.36.0
- datasets>=2.16.0
- evaluate>=0.4.1
- sacrebleu>=2.3.1
- torch>=2.1.0
- tensorboard>=2.15.0

## 🔗 Resources

- **NLLB Model**: https://huggingface.co/facebook/nllb-200-distilled-600M
- **OPUS Corpus**: https://opus.nlpl.eu/
- **FLORES Benchmark**: https://github.com/facebookresearch/flores

## 📄 License

MIT License - See [A100_TRANSFER_GUIDE.md](A100_TRANSFER_GUIDE.md) for complete setup instructions.

---

**Last Updated**: January 2026  
