#!/usr/bin/env python3
"""
NLLB-200 Finetuning Script for Korean-Vietnamese Translation
Optimized for NVIDIA A100 GPU

Usage:
    python finetune_nllb.py --config finetune_config.yaml
"""

import os
import json
import yaml
import logging
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Dict, Any
import torch
from datasets import Dataset
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback,
)
import evaluate
import numpy as np


def setup_logging(log_dir: str = "../logs", log_name: str = None) -> logging.Logger:
    """Setup logging to both console and file with detailed formatting"""
    
    # Create logs directory
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    
    # Generate log filename with timestamp
    if log_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_name = f"training_{timestamp}.log"
    
    log_file = Path(log_dir) / log_name
    
    # Create logger
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    
    # Remove existing handlers
    logger.handlers.clear()
    
    # Create formatters
    detailed_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    simple_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%H:%M:%S'
    )
    
    # Console handler (simple format)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(simple_formatter)
    
    # File handler (detailed format)
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(detailed_formatter)
    
    # Add handlers
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    
    # Log setup info
    logger.info("="*70)
    logger.info("LOGGING SETUP")
    logger.info("="*70)
    logger.info(f"Log file: {log_file}")
    logger.info(f"Log level: INFO")
    logger.info("="*70 + "\n")
    
    return logger


# Setup logging (will be called in main)
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def load_jsonl_dataset(file_path: str) -> Dataset:
    """Load JSONL dataset"""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line)
            translation = item['translation']
            data.append({
                'source': list(translation.values())[0],  # kor_Hang
                'target': list(translation.values())[1],  # vie_Latn
            })
    
    logger.info(f"Loaded {len(data)} examples from {file_path}")
    return Dataset.from_list(data)


def preprocess_function(examples, tokenizer, src_lang, tgt_lang, max_source_length, max_target_length):
    """Preprocess examples for training"""
    # Set source language
    tokenizer.src_lang = src_lang
    
    # Tokenize sources
    model_inputs = tokenizer(
        examples['source'],
        max_length=max_source_length,
        truncation=True,
        padding=False,  # Will be padded by data collator
    )
    
    # Tokenize targets
    tokenizer.src_lang = tgt_lang  # Set target language for labels
    labels = tokenizer(
        examples['target'],
        max_length=max_target_length,
        truncation=True,
        padding=False,
    )
    
    model_inputs['labels'] = labels['input_ids']
    return model_inputs


def compute_metrics(eval_preds, tokenizer, metric_bleu):
    """Compute evaluation metrics"""
    preds, labels = eval_preds
    
    # Decode predictions
    if isinstance(preds, tuple):
        preds = preds[0]
    
    # Replace -100 in predictions and labels (used for padding)
    preds = np.where(preds != -100, preds, tokenizer.pad_token_id)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    
    # Clip to valid vocab range (prevent overflow)
    vocab_size = tokenizer.vocab_size
    preds = np.clip(preds, 0, vocab_size - 1)
    labels = np.clip(labels, 0, vocab_size - 1)
    
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    
    # Post-process
    decoded_preds = [pred.strip() for pred in decoded_preds]
    decoded_labels = [[label.strip()] for label in decoded_labels]
    
    # Compute BLEU
    result = metric_bleu.compute(predictions=decoded_preds, references=decoded_labels)
    
    # Log sample translations
    if len(decoded_preds) > 0:
        logger.info(f"\nSample translation:")
        logger.info(f"  Prediction: {decoded_preds[0]}")
        logger.info(f"  Reference:  {decoded_labels[0][0]}")
    
    # Return only scalar metrics (precisions is list, will cause logging error)
    return {
        'bleu': result['score'],
    }


def setup_flash_attention(model):
    """Enable Flash Attention 2 if available"""
    try:
        from flash_attn import flash_attn_func
        logger.info("✓ Flash Attention 2 detected and enabled")
        return True
    except ImportError:
        logger.warning("✗ Flash Attention 2 not available. Install with: pip install flash-attn")
        return False


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Finetune NLLB-200 for Korean-Vietnamese translation")
    parser.add_argument('--config', type=str, default='finetune_config.yaml', help='Path to config file')
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume from')
    parser.add_argument('--log-dir', type=str, default=None, help='Directory for log files (overrides config)')
    parser.add_argument('--log-name', type=str, default=None, help='Custom log file name')
    args = parser.parse_args()
    
    # Load configuration first to get log_dir from config if specified
    config = load_config(args.config)
    
    # Determine log directory: CLI arg > config > default
    log_dir = args.log_dir or config.get('training', {}).get('log_dir', '../logs')
    
    # Setup logging
    global logger
    logger = setup_logging(log_dir=log_dir, log_name=args.log_name)
    
    logger.info("="*70)
    logger.info("CONFIGURATION LOADED")
    logger.info("="*70)
    logger.info(f"Config file: {args.config}")
    logger.info(f"Resume from: {args.resume if args.resume else 'None (training from scratch)'}")
    logger.info("="*70 + "\n")
    
    # Check GPU
    if not torch.cuda.is_available():
        logger.error("❌ CUDA not available! This script requires GPU.")
        raise RuntimeError("❌ CUDA not available! This script requires GPU.")
    
    gpu_name = torch.cuda.get_device_name(0)
    gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
    logger.info("="*70)
    logger.info("GPU INFORMATION")
    logger.info("="*70)
    logger.info(f"GPU: {gpu_name}")
    logger.info(f"Memory: {gpu_memory:.1f} GB")
    logger.info(f"CUDA Version: {torch.version.cuda}")
    logger.info("="*70 + "\n")
    
    # Load tokenizer and model
    model_name = config['model']['name']
    logger.info("="*70)
    logger.info("MODEL LOADING")
    logger.info("="*70)
    logger.info(f"Model: {model_name}")
    logger.info(f"Source language: {config['model']['src_lang']}")
    logger.info(f"Target language: {config['model']['tgt_lang']}")
    
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        cache_dir=config['model'].get('cache_dir'),
        src_lang=config['model']['src_lang'],
        tgt_lang=config['model']['tgt_lang'],
    )
    logger.info("✓ Tokenizer loaded")
    
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_name,
        cache_dir=config['model'].get('cache_dir'),
    )
    logger.info("✓ Model loaded")
    
    # Enable Flash Attention if available
    if config.get('advanced', {}).get('use_flash_attention', True):
        setup_flash_attention(model)
    
    logger.info(f"Model parameters: {model.num_parameters() / 1e6:.1f}M")
    logger.info("="*70 + "\n")
    
    # Load datasets
    logger.info("="*70)
    logger.info("DATASET LOADING")
    logger.info(f"Train file: {config['data']['train_file']}")
    logger.info(f"Dev file: {config['data']['dev_file']}")
    train_dataset = load_jsonl_dataset(config['data']['train_file'])
    eval_dataset = load_jsonl_dataset(config['data']['dev_file'])
    logger.info("="*70 + "\n")
    
    # Preprocess datasets
    logger.info("="*70)
    logger.info("DATA PREPROCESSING")
    logger.info("="*70)
    logger.info(f"Max source length: {config['data']['max_source_length']}")
    logger.info(f"Max target length: {config['data']['max_target_length']}")
    
    src_lang = config['model']['src_lang']
    tgt_lang = config['model']['tgt_lang']
    max_source_length = config['data']['max_source_length']
    max_target_length = config['data']['max_target_length']
    
    train_dataset = train_dataset.map(
        lambda x: preprocess_function(x, tokenizer, src_lang, tgt_lang, max_source_length, max_target_length),
        batched=True,
        remove_columns=train_dataset.column_names,
        desc="Preprocessing training data",
    )
    logger.info("✓ Training data preprocessed")
    
    eval_dataset = eval_dataset.map(
        lambda x: preprocess_function(x, tokenizer, src_lang, tgt_lang, max_source_length, max_target_length),
        batched=True,
        remove_columns=eval_dataset.column_names,
        desc="Preprocessing evaluation data",
    )
    logger.info("✓ Evaluation data preprocessed")
    logger.info("="*70 + "\n")
    
    # Data collator
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True,
    )
    
    # Load evaluation metric
    metric_bleu = evaluate.load("sacrebleu")
    
    # Training arguments
    training_args = Seq2SeqTrainingArguments(
        output_dir=config['training']['output_dir'],
        num_train_epochs=config['training']['num_train_epochs'],
        per_device_train_batch_size=config['training']['per_device_train_batch_size'],
        per_device_eval_batch_size=config['training']['per_device_eval_batch_size'],
        gradient_accumulation_steps=config['training']['gradient_accumulation_steps'],
        learning_rate=config['training']['learning_rate'],
        warmup_steps=config['training']['warmup_steps'],
        weight_decay=config['training']['weight_decay'],
        max_grad_norm=config['training']['max_grad_norm'],
        
        # Mixed precision
        bf16=config['training']['bf16'],
        fp16=config['training']['fp16'],
        tf32=config['training']['tf32'],
        
        # Memory optimization
        gradient_checkpointing=config['training']['gradient_checkpointing'],
        optim=config['training']['optim'],
        
        # Logging
        logging_dir=config['training']['logging_dir'],
        logging_steps=config['training']['logging_steps'],
        save_steps=config['training']['save_steps'],
        eval_steps=config['training']['eval_steps'],
        save_total_limit=config['training']['save_total_limit'],
        eval_strategy=config['training']['evaluation_strategy'],  # Fixed: evaluation_strategy deprecated
        load_best_model_at_end=config['training']['load_best_model_at_end'],
        metric_for_best_model=config['training']['metric_for_best_model'],
        greater_is_better=config['training']['greater_is_better'],
        
        # Generation
        predict_with_generate=config['training']['predict_with_generate'],
        generation_max_length=config['training']['generation_max_length'],
        generation_num_beams=config['training']['generation_num_beams'],
        
        # Performance
        dataloader_num_workers=config['training']['dataloader_num_workers'],
        dataloader_pin_memory=config['training']['dataloader_pin_memory'],
        remove_unused_columns=config['training']['remove_unused_columns'],
        
        # Reproducibility
        seed=config['training']['seed'],
        
        # Reporting
        report_to=["tensorboard"],
        run_name=f"nllb-ko-vi-{config['training']['num_train_epochs']}ep",
    )
    
    # Callbacks
    callbacks = []
    if config.get('early_stopping'):
        callbacks.append(
            EarlyStoppingCallback(
                early_stopping_patience=config['early_stopping']['patience'],
                early_stopping_threshold=config['early_stopping']['threshold'],
            )
        )
    
    # Initialize Trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=lambda x: compute_metrics(x, tokenizer, metric_bleu),
        callbacks=callbacks,
    )
    
    # Print training info
    logger.info("\n" + "="*60)
    logger.info("Training Configuration:")
    logger.info(f"  Train examples: {len(train_dataset)}")
    logger.info(f"  Eval examples: {len(eval_dataset)}")
    logger.info(f"  Batch size: {config['training']['per_device_train_batch_size']}")
    logger.info(f"  Gradient accumulation: {config['training']['gradient_accumulation_steps']}")
    logger.info(f"  Effective batch size: {config['training']['per_device_train_batch_size'] * config['training']['gradient_accumulation_steps']}")
    logger.info(f"  Epochs: {config['training']['num_train_epochs']}")
    logger.info(f"  Learning rate: {config['training']['learning_rate']}")
    logger.info(f"  Mixed precision: {'BF16' if config['training']['bf16'] else 'FP32'}")
    logger.info(f"  Gradient checkpointing: {config['training']['gradient_checkpointing']}")
    logger.info("="*60 + "\n")
    
    # Train
    logger.info("="*70)
    logger.info("TRAINING START")
    logger.info("="*70)
    if args.resume:
        logger.info(f"Resuming from checkpoint: {args.resume}")
    else:
        logger.info("Training from scratch")
    logger.info("="*70 + "\n")
    
    try:
        if args.resume:
            train_result = trainer.train(resume_from_checkpoint=args.resume)
        else:
            train_result = trainer.train()
    except Exception as e:
        logger.error("="*70)
        logger.error("TRAINING ERROR")
        logger.error("="*70)
        logger.error(f"Error: {str(e)}", exc_info=True)
        logger.error("="*70)
        raise
    
    # Save final model
    logger.info("\n" + "="*70)
    logger.info("SAVING MODEL")
    logger.info("="*70)
    trainer.save_model()
    tokenizer.save_pretrained(training_args.output_dir)
    logger.info(f"✓ Model saved to: {training_args.output_dir}")
    
    # Save training metrics
    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()
    logger.info("✓ Training metrics saved")
    logger.info("="*70 + "\n")
    
    # Final evaluation
    logger.info("="*70)
    logger.info("FINAL EVALUATION")
    logger.info("="*70)
    metrics = trainer.evaluate()
    trainer.log_metrics("eval", metrics)
    trainer.save_metrics("eval", metrics)
    logger.info(f"Eval loss: {metrics['eval_loss']:.4f}")
    if 'eval_bleu' in metrics:
        logger.info(f"Eval BLEU: {metrics['eval_bleu']:.2f}")
    logger.info("="*70 + "\n")
    
    # Success summary
    logger.info("="*70)
    logger.info("TRAINING COMPLETED SUCCESSFULLY!")
    logger.info("="*70)
    logger.info(f"Model: {training_args.output_dir}")
    logger.info(f"Final loss: {metrics['eval_loss']:.4f}")
    if 'eval_bleu' in metrics:
        logger.info(f"Final BLEU: {metrics['eval_bleu']:.2f}")
    logger.info(f"Total epochs: {config['training']['num_train_epochs']}")
    logger.info(f"Training examples: {len(train_dataset):,}")
    logger.info("="*70 + "\n")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger = logging.getLogger(__name__)
        logger.error("="*70)
        logger.error("FATAL ERROR")
        logger.error("="*70)
        logger.error(f"Error: {str(e)}", exc_info=True)
        logger.error("="*70)
        raise
