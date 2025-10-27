#!/usr/bin/env python3
"""
Quick Start: CIFAR-10 with TensorBoard & WandB Logging
======================================================
"""

import sys
import os

# Import the enhanced logging version
from main_multiexp_with_logging import *

# Override configuration for quick testing
config.DATASET = "CIFAR10"
config.NUM_CLASSES = 10
config.DATA_PATH = "./data"

# Quick test settings
config.NUM_EPOCHS = 10  # Quick test (use 200 for full training)
config.BATCH_SIZE = 128 if torch.cuda.is_available() else 32

# Test fewer ratios for speed
config.RATIOS_TO_TEST = [0.3, 0.5, 0.7]

# Single model multi-ratio
config.SINGLE_MODEL_MULTI_RATIO = True
config.ENABLE_MULTI_MODEL = False

# Model settings
config.MODEL_NAME = "resnet18"
config.REFERENCE_MODEL_NAME = "resnet18"

# Logging settings
config.USE_TENSORBOARD = True
config.USE_WANDB = True  # Set to False if you don't want wandb
config.WANDB_PROJECT = "cifar10-selective-coreset"
config.WANDB_ENTITY = None  # Set to your wandb username
config.LOG_INTERVAL = 10
config.LOG_IMAGES = False  # Disable for faster training
config.LOG_HISTOGRAMS = True
config.LOG_GRADIENTS = True

# Evaluation and checkpointing
config.EVAL_FREQUENCY = 2
config.SELECTION_FREQUENCY = 3
config.SAVE_FREQUENCY = 5

# Experiment naming
config.EXPERIMENT_NAME = f"cifar10_logged_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
config.TENSORBOARD_DIR = "./runs"
config.RESULTS_DIR = "./experiment_results"
config.CHECKPOINT_DIR = "./checkpoints"

# Device
config.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print("="*70)
print("CIFAR-10 WITH TENSORBOARD & WANDB LOGGING")
print("="*70)
print(f"Dataset: {config.DATASET}")
print(f"Epochs: {config.NUM_EPOCHS}")
print(f"Batch Size: {config.BATCH_SIZE}")
print(f"Device: {config.DEVICE}")
print(f"Ratios: {config.RATIOS_TO_TEST}")
print(f"Model: {config.MODEL_NAME}")
print()
print("Logging:")
print(f"  TensorBoard: {config.USE_TENSORBOARD}")
print(f"  WandB: {config.USE_WANDB and WANDB_AVAILABLE}")
print()
print("To view TensorBoard:")
print(f"  tensorboard --logdir={config.TENSORBOARD_DIR}")
print()
if config.USE_WANDB and not WANDB_AVAILABLE:
    print("WARNING: WandB not installed! Install with: pip install wandb")
    print()
print("="*70)
print()

if __name__ == "__main__":
    main()
