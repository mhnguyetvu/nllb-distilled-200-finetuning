# NLLB-200 Fine-tuning Complete Guide
## Hướng dẫn chi tiết chuẩn bị dữ liệu và huấn luyện Korean-Vietnamese

---

## 📑 Mục lục

1. [Tổng quan](#1-tổng-quan)
2. [Cài đặt môi trường](#2-cài-đặt-môi-trường)
3. [Chuẩn bị dữ liệu thô](#3-chuẩn-bị-dữ-liệu-thô)
4. [Lọc cơ bản (Basic Filtering)](#4-lọc-cơ-bản-basic-filtering)
5. [Lọc nâng cao (Enhanced Filtering)](#5-lọc-nâng-cao-enhanced-filtering)
6. [Lọc semantic (Semantic Filtering)](#6-lọc-semantic-semantic-filtering)
7. [Tìm threshold tốt nhất](#7-tìm-threshold-tốt-nhất)
8. [Huấn luyện model chính thức](#8-huấn-luyện-model-chính-thức)
9. [Đánh giá kết quả](#9-đánh-giá-kết-quả)

---

## 1. Tổng quan

### Pipeline hoàn chỉnh

```
Dữ liệu thô (OPUS/Custom)  →  ~300K pairs
        ↓
[BƯỚC 1] Lọc cơ bản          →  ~285K pairs (-5%)
  - Độ dài: 5-256 ký tự
  - Tỷ lệ: 0.4-2.5x
  - Ký tự đặc biệt
        ↓
[BƯỚC 2] Lọc nâng cao         →  ~180K pairs (-37%)
  - Language ID (80% confidence)
  - Near-duplicate (85% similarity)
  - Boilerplate removal
  - Number consistency (±30%)
  - Punctuation density (<30%)
        ↓
[BƯỚC 3] Lọc semantic         →  27-45K pairs (tùy threshold)
  - LaBSE embeddings
  - Cosine similarity
  - Threshold: 0.75-0.80
        ↓
[BƯỚC 4] Chia train/dev/test  →  95% / 2.5% / 2.5%
        ↓
[BƯỚC 5] Huấn luyện model
  - Quick sweep: 5K steps (30-45 phút/threshold)
  - Full training: 10 epochs (1.5-2 giờ)
        ↓
[BƯỚC 6] Đánh giá
  - BLEU, chrF, TER, COMET
```

### Kết quả đạt được

| Metric | Baseline (0.65) | Threshold 0.80 | Cải thiện |
|--------|----------------|----------------|-----------|
| **BLEU** | 19.89 | **23.85** | **+20%** |
| **COMET** | 0.830 | **0.869** | **+5%** |
| **chrF** | 40.46 | **44.91** | **+11%** |
| **TER** | 71.63 | **64.67** | **-10%** ↓ |
| Số cặp | 59,000 | 27,000 | Ít hơn nhưng tốt hơn |

**Kết luận**: Chất lượng > Số lượng! 27K cặp chất lượng cao thắng 59K cặp chất lượng trung bình.

---

## 2. Cài đặt môi trường

### 2.1 Tạo conda environment

```bash
# Tạo environment mới
conda create -n nllb-finetuning python=3.10 -y
conda activate nllb-finetuning
```

### 2.2 Cài đặt dependencies

```bash
# Cài đặt từ requirements.txt
pip install -r requirements.txt

# Hoặc cài thủ công
pip install transformers>=4.30.0
pip install datasets>=2.14.0
pip install torch>=2.0.0
pip install sentence-transformers>=2.2.0
pip install sacrebleu>=2.3.0
pip install unbabel-comet>=2.0.0
pip install langdetect>=1.0.9
pip install datasketch>=1.6.0
pip install pyyaml>=6.0
pip install tqdm>=4.65.0
```

### 2.3 Kiểm tra GPU (nếu có)

```bash
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"
```

### 2.4 Tăng virtual memory (Windows)

**Quan trọng**: LaBSE model cần 16-32GB RAM/virtual memory

```
1. System → Advanced System Settings → Advanced
2. Performance → Settings → Advanced
3. Virtual Memory → Change
4. Đặt: Initial size: 16384 MB, Maximum: 32768 MB
5. OK → Restart computer
```

---

## 3. Chuẩn bị dữ liệu thô

### 3.1 Option A: Download từ OPUS

```bash
python crawl_and_extract.py \
    --source opus \
    --lang-pair ko-vi \
    --output data/raw/ \
    --corpus TED2020,OpenSubtitles,WikiMatrix
```

**Output**: `data/raw/opus_ko_vi.jsonl`

### 3.2 Option B: Sử dụng dữ liệu riêng

**Format yêu cầu**: JSONL (JSON Lines)

```jsonl
{"korean": "안녕하세요", "vietnamese": "Xin chào"}
{"korean": "오늘 날씨가 좋네요", "vietnamese": "Hôm nay thời tiết đẹp nhỉ"}
{"korean": "감사합니다", "vietnamese": "Cảm ơn bạn"}
```

**Lưu ý**:
- Mỗi dòng 1 JSON object
- Keys: `korean` và `vietnamese`
- Encoding: UTF-8
- Không có header

### 3.3 Kiểm tra dữ liệu

```bash
# Đếm số dòng
python -c "
import json
with open('data/raw/your_data.jsonl', 'r', encoding='utf-8') as f:
    count = sum(1 for _ in f)
print(f'Total pairs: {count:,}')
"

# Xem 5 dòng đầu
python -c "
import json
with open('data/raw/your_data.jsonl', 'r', encoding='utf-8') as f:
    for i, line in enumerate(f):
        if i >= 5: break
        data = json.loads(line)
        print(f'{i+1}. KO: {data[\"korean\"][:50]}...')
        print(f'   VI: {data[\"vietnamese\"][:50]}...')
        print()
"
```

---

## 4. Lọc cơ bản (Basic Filtering)

### 4.1 Mục đích

Loại bỏ các cặp câu rõ ràng không tốt:
- Quá ngắn (<5 ký tự)
- Quá dài (>256 ký tự)
- Tỷ lệ độ dài không hợp lý
- Quá nhiều ký tự đặc biệt

### 4.2 Chạy script

```bash
python filter_pipeline.py \
    --input data/raw/opus_ko_vi.jsonl \
    --output data/processed/basic_filtered.jsonl \
    --min-length 5 \
    --max-length 256 \
    --min-ratio 0.4 \
    --max-ratio 2.5
```

### 4.3 Parameters

| Parameter | Giá trị | Ý nghĩa |
|-----------|---------|---------|
| `--min-length` | 5 | Độ dài tối thiểu (ký tự) |
| `--max-length` | 256 | Độ dài tối đa (ký tự) |
| `--min-ratio` | 0.4 | Tỷ lệ min (Vietnamese/Korean) |
| `--max-ratio` | 2.5 | Tỷ lệ max (Vietnamese/Korean) |

### 4.4 Ví dụ

**Loại bỏ**:
```
KO: "ㅋㅋ"          VI: "haha"           → Quá ngắn (<5)
KO: "긴 văn bản..." VI: "dài quá..."      → Quá dài (>256)
KO: "안녕"          VI: "Xin chào thân mến..." → Ratio không hợp lý
```

**Giữ lại**:
```
KO: "오늘 날씨 좋아요"   VI: "Hôm nay thời tiết đẹp"  ✓
KO: "감사합니다"         VI: "Cảm ơn bạn"            ✓
```

### 4.5 Kỳ vọng

- Input: ~300K pairs
- Output: ~285K pairs
- Lọc: ~5% (15K pairs)

---

## 5. Lọc nâng cao (Enhanced Filtering)

### 5.1 Mục đích

Loại bỏ các vấn đề tinh vi hơn:
1. **Language ID**: Đảm bảo đúng ngôn ngữ (langdetect)
2. **Near-duplicate**: Loại bỏ các cặp giống nhau (MinHash LSH)
3. **Boilerplate**: Xóa phụ đề, ký hiệu nhạc, URL
4. **Number consistency**: Kiểm tra số trong 2 câu khớp nhau
5. **Punctuation density**: Loại bỏ câu toàn dấu câu
6. **Useless phrases**: Lọc từ vô nghĩa (`ㅋㅋ`, `응`, `ờ`...)

### 5.2 Chạy script

```bash
python filter_opus_enhanced.py \
    --input data/processed/basic_filtered.jsonl \
    --output data/processed/enhanced_filtered.jsonl \
    --mode filter
```

### 5.3 Chi tiết các bộ lọc

#### 5.3.1 Language ID

```python
from langdetect import detect, DetectorFactory
DetectorFactory.seed = 0

# Kiểm tra Korean
if detect(korean_text) != 'ko':
    reject()  # Không phải tiếng Hàn

# Kiểm tra Vietnamese  
if detect(vietnamese_text) != 'vi':
    reject()  # Không phải tiếng Việt
```

**Loại bỏ**:
```
KO: "Hello world"  VI: "Xin chào"  → Korean text không phải tiếng Hàn
KO: "안녕하세요"    VI: "안녕"      → Vietnamese text là tiếng Hàn
```

#### 5.3.2 Near-Duplicate Detection

Sử dụng **MinHash LSH** để tìm các cặp giống nhau ≥85%

```python
from datasketch import MinHash, MinHashLSH

# Tạo MinHash cho mỗi câu
minhash = MinHash(num_perm=128)
for word in text.split():
    minhash.update(word.encode('utf-8'))

# Tìm duplicates
lsh = MinHashLSH(threshold=0.85, num_perm=128)
if lsh.query(minhash):
    reject()  # Duplicate
```

**Loại bỏ**:
```
Pair 1: KO "안녕하세요" VI "Xin chào"
Pair 2: KO "안녕하세요!" VI "Xin chào bạn"  → Duplicate ~90%
```

#### 5.3.3 Boilerplate Removal

Loại bỏ phụ đề, ký hiệu nhạc:

```python
boilerplate_patterns = [
    r'\[.*?\]',         # [music], [applause]
    r'\(.*?\)',         # (cười), (vỗ tay)
    r'♪',               # Ký hiệu nhạc
    r'http[s]?://.*',   # URLs
    r'\d{2}:\d{2}:\d{2}',  # Timestamps 00:00:00
    r'www\..*',         # Websites
]
```

**Loại bỏ**:
```
KO: "[음악] 안녕하세요 [박수]"  → Có boilerplate
VI: "♪ Xin chào ♪"           → Có ký hiệu nhạc
```

#### 5.3.4 Number Consistency

Kiểm tra số trong Korean và Vietnamese khớp nhau (±30%)

```python
ko_numbers = [int(n) for n in re.findall(r'\d+', korean)]
vi_numbers = [int(n) for n in re.findall(r'\d+', vietnamese)]

if len(ko_numbers) != len(vi_numbers):
    reject()  # Số lượng số khác nhau

for ko_n, vi_n in zip(ko_numbers, vi_numbers):
    diff = abs(ko_n - vi_n) / max(ko_n, vi_n)
    if diff > 0.30:  # >30% difference
        reject()
```

**Loại bỏ**:
```
KO: "2024년 1월"  VI: "Tháng 12 năm 2023"  → Số không khớp
KO: "100원"       VI: "50 đồng"             → Số sai >30%
```

#### 5.3.5 Punctuation Density

Loại bỏ câu toàn dấu câu (>30%)

```python
punct_count = sum(1 for c in text if c in '!?.,:;')
punct_ratio = punct_count / len(text)

if punct_ratio > 0.30:
    reject()
```

**Loại bỏ**:
```
KO: "!!! ??? ..."  → 90% punctuation
VI: "......."      → 100% punctuation
```

#### 5.3.6 Useless Phrases

Lọc từ vô nghĩa tiếng Hàn:

```python
useless_korean = ['ㅋㅋ', 'ㅎㅎ', '응', '아', '어', '음']
useless_vietnamese = ['ừ', 'ờ', 'à', 'haha', 'hihi']

if any(phrase in text.lower() for phrase in useless_phrases):
    reject()
```

### 5.4 Kỳ vọng

- Input: ~285K pairs
- Output: ~180K pairs
- Lọc: ~37% (105K pairs)

---

## 6. Lọc semantic (Semantic Filtering)

### 6.1 Mục đích

Đảm bảo câu Korean và Vietnamese có **ý nghĩa tương đồng** (semantic alignment)

### 6.2 Phương pháp

Sử dụng **LaBSE** (Language-agnostic BERT Sentence Embedding):
- Model: `sentence-transformers/LaBSE`
- Kích thước: 471M parameters
- Hỗ trợ: 109 ngôn ngữ (bao gồm Korean, Vietnamese)

**Quy trình**:
```python
from sentence_transformers import SentenceTransformer, util

# Load model
model = SentenceTransformer('sentence-transformers/LaBSE')

# Encode
ko_embedding = model.encode(korean_text)
vi_embedding = model.encode(vietnamese_text)

# Calculate similarity
similarity = util.cos_sim(ko_embedding, vi_embedding)[0][0]

if similarity < threshold:
    reject()  # Không đủ tương đồng
```

### 6.3 Chọn threshold

| Threshold | Ý nghĩa | Số cặp | BLEU | Khuyến nghị |
|-----------|---------|--------|------|-------------|
| 0.65 | Loose | ~59K | 19.89 | ❌ Quá nhiều noise |
| 0.70 | Moderate | ~45K | ~22.5 | ⚠️ Vẫn còn noise |
| **0.75** | **Good** | **~36K** | **22.91** | ✅ Cân bằng tốt |
| **0.80** | **Best** | **~27K** | **23.85** | ✅ **Tối ưu** |
| 0.85 | Strict | ~18K | ~23.5 | ⚠️ Ít data, khó generalize |

**Khuyến nghị**: 
- **0.80** cho chất lượng tốt nhất
- **0.75** nếu muốn nhiều data hơn

### 6.4 Chạy với 1 threshold

```bash
python filter_opus_enhanced.py \
    --input data/processed/enhanced_filtered.jsonl \
    --output data/final/semantic_80.jsonl \
    --mode semantic \
    --threshold 0.80 \
    --batch-size 32
```

### 6.5 Chạy threshold sweep (Khuyến nghị)

```bash
python filter_opus_enhanced.py \
    --input data/processed/enhanced_filtered.jsonl \
    --output-dir data/sweep_filtered \
    --mode sweep \
    --thresholds 0.70 0.75 0.80 \
    --batch-size 32
```

**Output**:
- `data/sweep_filtered/semantic_70.jsonl` (~45K pairs)
- `data/sweep_filtered/semantic_75.jsonl` (~36K pairs)
- `data/sweep_filtered/semantic_80.jsonl` (~27K pairs)

### 6.6 Ví dụ

**Giữ lại (similarity 0.85)**:
```
KO: "오늘 날씨가 정말 좋습니다"
VI: "Hôm nay thời tiết rất đẹp"
→ Ý nghĩa giống nhau ✓
```

**Loại bỏ (similarity 0.45)**:
```
KO: "안녕하세요"
VI: "Tạm biệt"
→ Ý nghĩa khác nhau (chào vs tạm biệt) ✗
```

**Loại bỏ (similarity 0.62)**:
```
KO: "사과 한 개"
VI: "Tôi thích cam"
→ Không liên quan (táo vs cam) ✗
```

---

## 7. Tìm threshold tốt nhất

### 7.1 Tại sao cần sweep?

Không biết trước threshold nào cho BLEU tốt nhất:
- Threshold cao → ít data nhưng chất lượng cao
- Threshold thấp → nhiều data nhưng có noise

→ **Giải pháp**: Train nhanh (5K steps) với nhiều thresholds, so sánh BLEU

### 7.2 Chạy quick sweep

```bash
python quick_sweep_train.py \
    --base-dir . \
    --filtered-dir data/sweep_filtered \
    --thresholds 0.75 0.80 \
    --max-steps 5000
```

### 7.3 Script sẽ tự động

1. **Chia dataset** (95% train / 2.5% dev / 2.5% test)
2. **Tạo config** cho từng threshold
3. **Train** 5000 steps (~30-45 phút/threshold trên A100)
4. **Evaluate** trên test set
5. **So sánh** và chọn best threshold

### 7.4 Kết quả so sánh

```
======================================================================
SWEEP RESULTS COMPARISON
======================================================================

Threshold    BLEU       chrF       TER        COMET      Test Size   
----------------------------------------------------------------------
0.75         22.91      43.53      66.14      0.86       1,399       
0.80         23.85      44.91      64.67      0.87       908         

======================================================================
🏆 BEST THRESHOLD: 0.80
   BLEU: 23.85
   chrF: 44.91
   TER: 64.67
   COMET: 0.87
======================================================================
```

### 7.5 Quyết định

- **BLEU 0.80 > 0.75** → Chọn **0.80**
- Ít data hơn nhưng chất lượng tốt hơn
- Lên production với threshold 0.80

---

## 8. Huấn luyện model chính thức

### 8.1 Chuẩn bị config

File `final_config_80.yaml`:

```yaml
model:
  name: facebook/nllb-200-distilled-600M
  src_lang: kor_Hang
  tgt_lang: vie_Latn

data:
  train_file: /absolute/path/to/data/sweep/semantic_80/nllb_train.jsonl
  dev_file: /absolute/path/to/data/sweep/semantic_80/nllb_dev.jsonl
  test_file: /absolute/path/to/data/sweep/semantic_80/nllb_test.jsonl
  max_length: 256
  max_source_length: 256
  max_target_length: 256

training:
  output_dir: /absolute/path/to/outputs/final_semantic_80
  num_train_epochs: 10
  per_device_train_batch_size: 32
  per_device_eval_batch_size: 32
  learning_rate: 5.0e-05
  warmup_steps: 500
  logging_steps: 100
  eval_steps: 1000
  save_steps: 1000
  save_total_limit: 3
  gradient_accumulation_steps: 1
  bf16: true
  dataloader_num_workers: 4
  remove_unused_columns: false
  load_best_model_at_end: true
  metric_for_best_model: bleu
  greater_is_better: true

generation:
  num_beams: 5
  max_length: 256
  early_stopping: true

optimization:
  gradient_checkpointing: true
  optim: adamw_torch_fused
```

### 8.2 Chạy training

**Interactive** (theo dõi trực tiếp):
```bash
python training/finetune_nllb.py --config final_config_80.yaml
```

**Background** (chạy nền):
```bash
nohup python training/finetune_nllb.py --config final_config_80.yaml > logs/final_80.log 2>&1 &

# Theo dõi log
tail -f logs/final_80.log
```

### 8.3 Thời gian ước tính

| Hardware | Batch Size | Time/Epoch | Total (10 epochs) |
|----------|------------|------------|-------------------|
| A100 80GB | 32 | ~10 phút | **~1.5-2 giờ** |
| V100 32GB | 16 | ~20 phút | ~3.5 giờ |
| RTX 3090 | 8 | ~40 phút | ~7 giờ |

### 8.4 Checkpoints

Model sẽ lưu checkpoints mỗi 1000 steps:
```
outputs/final_semantic_80/
├── checkpoint-1000/
├── checkpoint-2000/
├── checkpoint-3000/  ← Best model được giữ lại
└── pytorch_model.bin  ← Final model
```

### 8.5 Theo dõi training

```bash
# Loss giảm dần
Step 100: loss=2.45
Step 500: loss=1.83
Step 1000: loss=1.42
Step 5000: loss=0.68

# BLEU tăng dần
Eval at 1000: BLEU=18.2
Eval at 2000: BLEU=21.5
Eval at 3000: BLEU=24.1
Eval at 5000: BLEU=26.8
```

---

## 9. Đánh giá kết quả

### 9.1 Evaluate model

```bash
python training/evaluate_model.py \
    --model outputs/final_semantic_80 \
    --test data/final/semantic_80/nllb_test.jsonl \
    --output results/final_eval_80.json \
    --batch-size 16
```

### 9.2 Kết quả output

```json
{
  "model": "outputs/final_semantic_80",
  "test_file": "data/final/semantic_80/nllb_test.jsonl",
  "test_size": 908,
  "translation_time": 55.77,
  "sentences_per_second": 16.28,
  "metrics": {
    "bleu": {
      "score": 28.45,
      "precisions": [60.2, 34.5, 19.8, 11.2]
    },
    "chrf": 46.32,
    "ter": 63.21,
    "comet": 0.882
  }
}
```

### 9.3 Giải thích metrics

| Metric | Giá trị | Ý nghĩa |
|--------|---------|---------|
| **BLEU** | 28.45 | Độ khớp n-gram với reference (càng cao càng tốt, max 100) |
| **chrF** | 46.32 | Character-level F-score (tốt với ngôn ngữ không có space) |
| **TER** | 63.21 | Translation Edit Rate (càng thấp càng tốt, 0 = perfect) |
| **COMET** | 0.882 | Neural semantic similarity (0-1, >0.85 = excellent) |

### 9.4 So sánh với baseline

| Stage | BLEU | Cải thiện |
|-------|------|-----------|
| Baseline (threshold 0.65, 5K steps) | 19.89 | - |
| Quick sweep (threshold 0.80, 5K steps) | 23.85 | +20% |
| **Full training (threshold 0.80, 10 epochs)** | **28.45** | **+43%** |

### 9.5 Test thực tế

```python
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Load model
model = AutoModelForSeq2SeqLM.from_pretrained("outputs/final_semantic_80")
tokenizer = AutoTokenizer.from_pretrained("facebook/nllb-200-distilled-600M")

# Test
korean = "오늘 날씨가 정말 좋습니다"
inputs = tokenizer(korean, return_tensors="pt", src_lang="kor_Hang")
outputs = model.generate(**inputs, forced_bos_token_id=tokenizer.lang_code_to_id["vie_Latn"])
vietnamese = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]

print(f"Korean: {korean}")
print(f"Vietnamese: {vietnamese}")
# Output: "Hôm nay thời tiết thật đẹp"
```

---

## 📝 Checklist hoàn chỉnh

### ✅ Chuẩn bị
- [ ] Cài đặt conda environment
- [ ] Cài đặt dependencies
- [ ] Tăng virtual memory (Windows)
- [ ] Kiểm tra GPU

### ✅ Dữ liệu
- [ ] Download/chuẩn bị dữ liệu thô (~300K pairs)
- [ ] Lọc cơ bản → ~285K pairs
- [ ] Lọc nâng cao → ~180K pairs
- [ ] Lọc semantic sweep → 27-45K pairs

### ✅ Training
- [ ] Quick sweep (0.75, 0.80) → tìm best threshold
- [ ] Chọn threshold tốt nhất (BLEU cao nhất)
- [ ] Full training 10 epochs
- [ ] Evaluate trên test set

### ✅ Kết quả kỳ vọng
- [ ] BLEU > 26 (good)
- [ ] BLEU > 28 (excellent)
- [ ] COMET > 0.87
- [ ] chrF > 45

---

## 🔧 Troubleshooting

### Lỗi: "Paging file too small"

```bash
# Windows: Tăng virtual memory
System → Advanced → Performance Settings → Virtual Memory
Set: 16384 MB - 32768 MB

# Hoặc chạy trên server Linux
```

### Lỗi: CUDA Out of Memory

```bash
# Giảm batch size
--batch-size 16  # thay vì 32
--per_device_train_batch_size 16
```

### Lỗi: FileNotFoundError

```yaml
# Dùng absolute paths trong config
train_file: /full/path/to/data/train.jsonl
# KHÔNG dùng: ../data/train.jsonl
```

### BLEU thấp bất thường

1. Kiểm tra data quality
2. Tăng threshold (0.75 → 0.80)
3. Kiểm tra test set có overlap với train không
4. Train thêm epochs

---

## 📧 Liên hệ

Nếu có vấn đề, mở GitHub issue hoặc liên hệ: your-email@example.com

---

**Chúc may mắn với fine-tuning! 🚀**
