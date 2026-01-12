# NLLB-200 Korean-Vietnamese Finetuning on A100

Complete training package for finetuning NLLB-200-distilled-600M on Korean-Vietnamese parallel corpus.

## 📦 Contents

```
training/
├── finetune_nllb.py         # Main training script
├── evaluate_model.py         # Evaluation and comparison script
├── finetune_config.yaml      # Training hyperparameters
├── requirements_train.txt    # Python dependencies
└── README.md                 # This file
```

## 🚀 Quick Start on A100 Server

### 1. Setup Environment

```bash
# Create conda environment
conda create -n nllb-train python=3.10 -y
conda activate nllb-train

# Install dependencies
pip install -r requirements_train.txt

# Optional: Install Flash Attention 2 for 2-3x speedup
pip install flash-attn --no-build-isolation
```

### 2. Prepare Data

Transfer data files to server:

```bash
# On local machine
scp data/final/*.jsonl user@a100-server:~/nllb-finetuning/data/final/

# On A100 server
mkdir -p ~/nllb-finetuning/data/final
mkdir -p ~/nllb-finetuning/results
mkdir -p ~/nllb-finetuning/outputs
```

### 3. Run Training

```bash
# Start training
python finetune_nllb.py --config finetune_config.yaml

# Monitor with TensorBoard
tensorboard --logdir logs/tensorboard --port 6006
```

### 4. Evaluate Models

```bash
# Evaluate pretrained baseline
python evaluate_model.py \
    --model facebook/nllb-200-distilled-600M \
    --test data/final/nllb_test.jsonl \
    --output results/pretrained_eval.json

# Evaluate finetuned model
python evaluate_model.py \
    --model outputs/nllb-ko-vi-finetuned \
    --test data/final/nllb_test.jsonl \
    --output results/finetuned_eval.json

# Compare results
python evaluate_model.py \
    --compare results/pretrained_eval.json results/finetuned_eval.json \
    --compare-output results/comparison.json
```

## ⚙️ Configuration

### Key Hyperparameters (finetune_config.yaml)

**For A100 40GB:**
- `per_device_train_batch_size: 16`
- `gradient_accumulation_steps: 2`
- Effective batch size: 32

**For A100 80GB:**
- `per_device_train_batch_size: 32`
- `gradient_accumulation_steps: 2`
- Effective batch size: 64

**Optimizations:**
- ✅ BF16 mixed precision (native A100 support)
- ✅ TF32 enabled for faster computation
- ✅ Fused AdamW optimizer
- ✅ Gradient checkpointing (saves memory)
- ✅ Flash Attention 2 (if installed)

### Adjust for Your Setup

Edit `finetune_config.yaml`:

```yaml
# Increase batch size if you have more memory
per_device_train_batch_size: 32  # Default: 16

# Adjust epochs based on dataset size
num_train_epochs: 3  # Default: 5

# Learning rate (try 3e-5 to 5e-5)
learning_rate: 5.0e-5
```

## 📊 Expected Results

### Training Time

| Setup | Batch Size | Time/Epoch | Total (5 epochs) |
|-------|------------|------------|------------------|
| **A100 40GB** | 16 | ~8 min | **~40 min** |
| **A100 80GB** | 32 | ~5 min | **~25 min** |

### Expected Performance Gains

With 15K training sentences:

| Metric | Pretrained | After Finetuning | Improvement |
|--------|------------|------------------|-------------|
| **BLEU** | 15-25 | 30-40 | +15-20 pts |
| **COMET** | 0.40-0.55 | 0.65-0.75 | +0.15-0.25 |
| **chrF** | 40-50 | 55-65 | +10-15 pts |

## 🔧 Troubleshooting

### OOM (Out of Memory)

```yaml
# In finetune_config.yaml, reduce batch size:
per_device_train_batch_size: 8
gradient_accumulation_steps: 4
```

### Slow Training

```bash
# Check if Flash Attention is installed
python -c "import flash_attn; print('Flash Attention OK')"

# If not, install it:
pip install flash-attn --no-build-isolation
```

### Low BLEU Scores

- Train for more epochs (increase `num_train_epochs`)
- Collect more training data (aim for 50K+ sentences)
- Adjust learning rate (try 3e-5 or 7e-5)

## 📈 Monitoring Training

### TensorBoard

```bash
tensorboard --logdir logs/tensorboard --port 6006
```

Then access: `http://localhost:6006`

### Watch Training Logs

```bash
tail -f outputs/nllb-ko-vi-finetuned/trainer_state.json
```

## 💾 Model Checkpoints

Checkpoints saved to: `outputs/nllb-ko-vi-finetuned/`

```
outputs/nllb-ko-vi-finetuned/
├── checkpoint-500/      # Every 500 steps
├── checkpoint-1000/
├── ...
├── config.json          # Model config
├── pytorch_model.bin    # Final model weights
└── tokenizer files
```

### Resume from Checkpoint

```bash
python finetune_nllb.py \
    --config finetune_config.yaml \
    --resume outputs/nllb-ko-vi-finetuned/checkpoint-1000
```

## 🎯 Next Steps After Training

1. **Evaluate on test set** - Compare pretrained vs finetuned
2. **Error analysis** - Inspect failed translations
3. **Collect more data** - If results unsatisfactory, aim for 50K+ sentences
4. **Deploy model** - Use for production Korean→Vietnamese translation

## 📚 Resources

- NLLB Paper: https://arxiv.org/abs/2207.04672
- HuggingFace NLLB: https://huggingface.co/facebook/nllb-200-distilled-600M
- Flash Attention: https://github.com/Dao-AILab/flash-attention

## 🐛 Issues

For issues with training scripts, check:
- CUDA version compatibility: `nvidia-smi`
- PyTorch CUDA: `python -c "import torch; print(torch.cuda.is_available())"`
- GPU memory: `nvidia-smi --query-gpu=memory.used --format=csv`
