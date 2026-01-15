#!/usr/bin/env python3
"""
Quick sweep: Split datasets + Train + Evaluate for thresholds 0.75 and 0.80
All-in-one script for fast threshold comparison
"""

import json
import random
import subprocess
import sys
from pathlib import Path
from datetime import datetime

def split_dataset(input_file: str, output_dir: str, seed: int = 42):
    """Split dataset into train/dev/test (95/2.5/2.5)"""
    
    print(f"\n{'='*70}")
    print(f"Splitting: {input_file}")
    print(f"Output: {output_dir}")
    print(f"{'='*70}")
    
    # Load data
    with open(input_file, 'r', encoding='utf-8') as f:
        data = [json.loads(line) for line in f]
    
    total = len(data)
    print(f"Total pairs: {total:,}")
    
    # Shuffle
    random.seed(seed)
    random.shuffle(data)
    
    # Split (95/2.5/2.5)
    train_size = int(0.95 * total)
    dev_size = int(0.025 * total)
    
    train = data[:train_size]
    dev = data[train_size:train_size+dev_size]
    test = data[train_size+dev_size:]
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save splits
    splits = {
        'train': train,
        'dev': dev,
        'test': test
    }
    
    for split_name, split_data in splits.items():
        output_file = output_path / f"nllb_{split_name}.jsonl"
        with open(output_file, 'w', encoding='utf-8') as f:
            for item in split_data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        print(f"  ✓ {split_name}: {len(split_data):,} pairs → {output_file}")
    
    return len(train), len(dev), len(test)


def create_training_config(threshold: float, base_dir: str = '.', max_steps: int = 5000):
    """Create training config for specific threshold"""
    
    threshold_int = int(threshold * 100)
    
    # Use absolute paths
    base_path = Path(base_dir).resolve()
    
    config_content = f"""model:
  name: facebook/nllb-200-distilled-600M
  src_lang: kor_Hang
  tgt_lang: vie_Latn

data:
  train_file: {base_path}/data/sweep/semantic_{threshold_int}/nllb_train.jsonl
  dev_file: {base_path}/data/sweep/semantic_{threshold_int}/nllb_dev.jsonl
  test_file: {base_path}/data/sweep/semantic_{threshold_int}/nllb_test.jsonl
  max_length: 256
  max_source_length: 256
  max_target_length: 256

training:
  output_dir: {base_path}/outputs/sweep/semantic_{threshold_int}
  num_train_epochs: 1
  max_steps: {max_steps}
  per_device_train_batch_size: 32
  per_device_eval_batch_size: 32
  learning_rate: 5.0e-05
  warmup_steps: 500
  logging_steps: 100
  eval_steps: 500
  save_steps: {max_steps}
  save_total_limit: 1
  gradient_accumulation_steps: 1
  bf16: true
  dataloader_num_workers: 4
  remove_unused_columns: false
  load_best_model_at_end: true
  metric_for_best_model: bleu
  greater_is_better: true
  log_dir: {base_path}/logs/sweep

generation:
  num_beams: 5
  max_length: 256
  early_stopping: true

optimization:
  gradient_checkpointing: true
  optim: adamw_torch_fused
"""
    
    # Save config
    config_file = Path(f"{base_dir}/training/sweep_config_{threshold_int}.yaml")
    config_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(config_file, 'w', encoding='utf-8') as f:
        f.write(config_content)
    
    print(f"  ✓ Config created: {config_file}")
    return str(config_file)


def run_training(config_file: str, threshold: float, base_dir: str = '.'):
    """Run training with logging"""
    
    threshold_int = int(threshold * 100)
    
    print(f"\n{'#'*70}")
    print(f"# TRAINING: Threshold {threshold:.2f}")
    print(f"# Config: {config_file}")
    print(f"{'#'*70}\n")
    
    # Check if finetune script exists (try multiple locations)
    train_script_locations = [
        Path(base_dir) / "training" / "finetune_nllb.py",
        Path("training/finetune_nllb.py"),
        Path("finetune_nllb.py")
    ]
    
    train_script = None
    for script_path in train_script_locations:
        if script_path.exists():
            train_script = script_path
            break
    
    if train_script is None:
        print(f"❌ Training script not found in:")
        for loc in train_script_locations:
            print(f"   - {loc}")
        return False
    
    print(f"✓ Found training script: {train_script}")
    
    # Create log directory
    log_dir = Path(base_dir) / "logs" / "sweep"
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Log file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"sweep_{threshold_int}_{timestamp}.log"
    
    # Run training
    cmd = [
        sys.executable,  # Use same Python interpreter
        str(train_script),
        '--config', config_file
    ]
    
    print(f"Command: {' '.join(cmd)}")
    print(f"Log file: {log_file}\n")
    
    try:
        with open(log_file, 'w', encoding='utf-8') as f:
            # Run and capture output
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1
            )
            
            # Stream output to both console and file
            for line in process.stdout:
                print(line, end='')
                f.write(line)
                f.flush()
            
            process.wait()
            
            if process.returncode == 0:
                print(f"\n✅ Training complete for threshold {threshold:.2f}")
                print(f"   Log saved: {log_file}")
                return True
            else:
                print(f"\n❌ Training failed for threshold {threshold:.2f}")
                print(f"   Check log: {log_file}")
                return False
                
    except Exception as e:
        print(f"\n❌ Error during training: {e}")
        return False


def run_evaluation(threshold: float, base_dir: str = '.'):
    """Run evaluation on trained model"""
    
    threshold_int = int(threshold * 100)
    
    print(f"\n{'#'*70}")
    print(f"# EVALUATION: Threshold {threshold:.2f}")
    print(f"{'#'*70}\n")
    
    model_dir = f"{base_dir}/outputs/sweep/semantic_{threshold_int}"
    test_file = f"{base_dir}/data/sweep/semantic_{threshold_int}/nllb_test.jsonl"
    output_file = f"{base_dir}/results/sweep/eval_{threshold_int}.json"
    
    # Check if model exists
    if not Path(model_dir).exists():
        print(f"❌ Model not found: {model_dir}")
        return False
    
    # Create results directory
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    
    # Run evaluation - try multiple locations
    eval_script_locations = [
        Path(base_dir) / "training" / "evaluate_model.py",
        Path("training/evaluate_model.py"),
        Path("evaluate_model.py")
    ]
    
    eval_script = None
    for script_path in eval_script_locations:
        if script_path.exists():
            eval_script = script_path
            break
    
    if eval_script is None:
        print(f"❌ Evaluation script not found in:")
        for loc in eval_script_locations:
            print(f"   - {loc}")
        return False
    
    print(f"✓ Found evaluation script: {eval_script}")
    cmd = [
        sys.executable,
        str(eval_script),
        '--model', model_dir,
        '--test', test_file,
        '--output', output_file,
        '--batch-size', '16'
    ]
    
    print(f"Command: {' '.join(cmd)}\n")
    
    try:
        subprocess.run(cmd, check=True)
        print(f"\n✅ Evaluation complete")
        print(f"   Results: {output_file}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Evaluation failed: {e}")
        return False


def compare_results(thresholds: list, base_dir: str = '.'):
    """Compare results across thresholds"""
    
    print(f"\n{'='*70}")
    print("SWEEP RESULTS COMPARISON")
    print(f"{'='*70}\n")
    
    results = {}
    
    for threshold in thresholds:
        threshold_int = int(threshold * 100)
        eval_file = f"{base_dir}/results/sweep/eval_{threshold_int}.json"
        
        if not Path(eval_file).exists():
            print(f"⚠️  No results for threshold {threshold:.2f}")
            continue
        
        with open(eval_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        results[threshold] = {
            'bleu': data['metrics']['bleu']['score'] if isinstance(data['metrics']['bleu'], dict) else data['metrics']['bleu'],
            'chrf': data['metrics']['chrf'],
            'ter': data['metrics']['ter'],
            'comet': data['metrics'].get('comet', None),
            'test_size': data['test_size']
        }
    
    if not results:
        print("❌ No results found")
        return
    
    # Print table
    print(f"{'Threshold':<12} {'BLEU':<10} {'chrF':<10} {'TER':<10} {'COMET':<10} {'Test Size':<12}")
    print("-"*70)
    
    for threshold in sorted(results.keys()):
        r = results[threshold]
        comet_str = f"{r['comet']:.2f}" if r['comet'] else "N/A"
        print(f"{threshold:<12.2f} {r['bleu']:<10.2f} {r['chrf']:<10.2f} {r['ter']:<10.2f} {comet_str:<10} {r['test_size']:<12,}")
    
    # Find best
    best_threshold = max(results.keys(), key=lambda t: results[t]['bleu'])
    
    print("\n" + "="*70)
    print(f"🏆 BEST THRESHOLD: {best_threshold:.2f}")
    print(f"   BLEU: {results[best_threshold]['bleu']:.2f}")
    print(f"   chrF: {results[best_threshold]['chrf']:.2f}")
    print(f"   TER: {results[best_threshold]['ter']:.2f}")
    if results[best_threshold]['comet']:
        print(f"   COMET: {results[best_threshold]['comet']:.2f}")
    print("="*70)
    
    # Save summary
    summary_file = f"{base_dir}/results/sweep/comparison_summary.json"
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump({
            'results': results,
            'best_threshold': best_threshold,
            'best_bleu': results[best_threshold]['bleu'],
            'timestamp': datetime.now().isoformat()
        }, f, indent=2)
    
    print(f"\n📊 Summary saved: {summary_file}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Quick sweep for thresholds 0.75 and 0.80")
    parser.add_argument('--base-dir', type=str, default='.',
                       help='Base directory (default: current directory)')
    parser.add_argument('--filtered-dir', type=str, default='data/sweep_filtered',
                       help='Directory with semantic_XX.jsonl files (default: data/sweep_filtered)')
    parser.add_argument('--thresholds', type=float, nargs='+', default=[0.75, 0.80],
                       help='Thresholds to train (default: 0.75 0.80)')
    parser.add_argument('--max-steps', type=int, default=5000,
                       help='Max training steps (default: 5000)')
    parser.add_argument('--skip-split', action='store_true',
                       help='Skip dataset splitting (already done)')
    parser.add_argument('--skip-train', action='store_true',
                       help='Skip training (only evaluate)')
    
    args = parser.parse_args()
    
    print("\n" + "="*70)
    print(f"QUICK SWEEP: Thresholds {args.thresholds}")
    print("="*70)
    print("This script will:")
    if not args.skip_split:
        print("  1. Split datasets into train/dev/test")
    if not args.skip_train:
        print("  2. Train models (5000 steps each)")
    print("  3. Evaluate on test sets")
    print("  4. Compare results")
    print(f"\nBase directory: {args.base_dir}")
    print(f"Filtered data: {args.filtered_dir}")
    print("="*70 + "\n")
    
    thresholds = args.thresholds
    base_dir = args.base_dir
    filtered_dir = args.filtered_dir
    
    # Step 1: Split datasets
    if not args.skip_split:
        print("\n" + "#"*70)
        print("# STEP 1: SPLITTING DATASETS")
        print("#"*70)
        
        for threshold in thresholds:
            threshold_int = int(threshold * 100)
            input_file = f"{filtered_dir}/semantic_{threshold_int}.jsonl"
            output_dir = f"{base_dir}/data/sweep/semantic_{threshold_int}"
            
            if not Path(input_file).exists():
                print(f"\n❌ Input file not found: {input_file}")
                print("   Please run filter_opus_enhanced.py first!")
                continue
            
            split_dataset(input_file, output_dir)
    else:
        print("\n⏭️  Skipping dataset splitting (--skip-split)")
    
    # Step 2: Create configs and train
    if not args.skip_train:
        print("\n" + "#"*70)
        print("# STEP 2: TRAINING MODELS")
        print("#"*70)
        
        for threshold in thresholds:
            # Create config
            config_file = create_training_config(threshold, base_dir, args.max_steps)
            
            # Train
            success = run_training(config_file, threshold, base_dir)
            
            if not success:
                print(f"\n⚠️  Skipping evaluation for threshold {threshold:.2f}")
                continue
    else:
        print("\n⏭️  Skipping training (--skip-train)")
    
    # Step 3: Evaluate
    print("\n" + "#"*70)
    print("# STEP 3: EVALUATING MODELS")
    print("#"*70)
    
    for threshold in thresholds:
        run_evaluation(threshold, base_dir)
    
    # Step 4: Compare
    print("\n" + "#"*70)
    print("# STEP 4: COMPARING RESULTS")
    print("#"*70)
    
    compare_results(thresholds, base_dir)
    
    print("\n" + "="*70)
    print("🎉 SWEEP COMPLETE!")
    print("="*70)
    print("\nNext steps:")
    print("  1. Check logs in logs/sweep/")
    print("  2. Review results in results/sweep/")
    print("  3. Train full model on best threshold (10 epochs)")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
