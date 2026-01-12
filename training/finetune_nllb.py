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

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
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
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    logger.info(f"Loaded configuration from {args.config}")
    
    # Check GPU
    if not torch.cuda.is_available():
        raise RuntimeError("❌ CUDA not available! This script requires GPU.")
    
    gpu_name = torch.cuda.get_device_name(0)
    gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
    logger.info(f"✓ GPU: {gpu_name}, Memory: {gpu_memory:.1f} GB")
    
    # Load tokenizer and model
    model_name = config['model']['name']
    logger.info(f"Loading model: {model_name}")
    
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        cache_dir=config['model'].get('cache_dir'),
        src_lang=config['model']['src_lang'],
        tgt_lang=config['model']['tgt_lang'],
    )
    
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_name,
        cache_dir=config['model'].get('cache_dir'),
    )
    
    # Enable Flash Attention if available
    if config.get('advanced', {}).get('use_flash_attention', True):
        setup_flash_attention(model)
    
    logger.info(f"Model parameters: {model.num_parameters() / 1e6:.1f}M")
    
    # Load datasets
    logger.info("Loading datasets...")
    train_dataset = load_jsonl_dataset(config['data']['train_file'])
    eval_dataset = load_jsonl_dataset(config['data']['dev_file'])
    
    # Preprocess datasets
    logger.info("Preprocessing datasets...")
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
    
    eval_dataset = eval_dataset.map(
        lambda x: preprocess_function(x, tokenizer, src_lang, tgt_lang, max_source_length, max_target_length),
        batched=True,
        remove_columns=eval_dataset.column_names,
        desc="Preprocessing evaluation data",
    )
    
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
    logger.info("Starting training...")
    
    if args.resume:
        logger.info(f"Resuming from checkpoint: {args.resume}")
        train_result = trainer.train(resume_from_checkpoint=args.resume)
    else:
        train_result = trainer.train()
    
    # Save final model
    logger.info("Saving final model...")
    trainer.save_model()
    tokenizer.save_pretrained(training_args.output_dir)
    
    # Save training metrics
    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()
    
    # Final evaluation
    logger.info("Running final evaluation...")
    metrics = trainer.evaluate()
    trainer.log_metrics("eval", metrics)
    trainer.save_metrics("eval", metrics)
    
    logger.info("\n" + "="*60)
    logger.info("Training completed successfully!")
    logger.info(f"Model saved to: {training_args.output_dir}")
    logger.info(f"Final eval loss: {metrics['eval_loss']:.4f}")
    if 'eval_bleu' in metrics:
        logger.info(f"Final eval BLEU: {metrics['eval_bleu']:.2f}")
    logger.info("="*60 + "\n")


if __name__ == "__main__":
    main()
