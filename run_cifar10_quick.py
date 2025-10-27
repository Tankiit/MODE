#!/usr/bin/env python3
"""
Quick test run for CIFAR-10 with reduced epochs for faster experimentation
"""

import sys
import os

# Import the main experiment code
from main_multiexp import *

# Override configuration for quick testing
config.DATASET = "CIFAR10"
config.NUM_CLASSES = 10
config.DATA_PATH = "./data"

# Quick test settings (change these for full experiments)
config.NUM_EPOCHS = 10  # Use 10 for quick test, 200 for full training
config.BATCH_SIZE = 128 if torch.cuda.is_available() else 32

# Test fewer ratios for speed
config.RATIOS_TO_TEST = [0.3, 0.5, 0.7]  # Quick test
# config.RATIOS_TO_TEST = [0.1, 0.2, 0.3, 0.5, 0.7, 0.9]  # Full test

# Single model first for quick validation
config.SINGLE_MODEL_MULTI_RATIO = True
config.ENABLE_MULTI_MODEL = False  # Set to True for multi-model experiments

# Simpler models for quick testing
config.MODEL_NAME = "resnet18"  # Faster than resnet34
config.REFERENCE_MODEL_NAME = "resnet18"

# Optional: Test with multiple models
config.MODELS_TO_TEST = [
    ("resnet18", "resnet18"),
    ("resnet34", "resnet18"),
    # ("resnet50", "resnet34"),  # Uncomment for full test
    # ("efficientnet_b0", "mobilenetv3_small_100"),
]

# Evaluation settings
config.EVAL_FREQUENCY = 2  # Evaluate every 2 epochs
config.SELECTION_FREQUENCY = 3  # Re-select every 3 epochs

# Experiment naming
config.EXPERIMENT_NAME = f"cifar10_quick_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
config.RESULTS_DIR = "./experiment_results"

# Checkpointing
config.SAVE_CHECKPOINTS = True
config.CHECKPOINT_DIR = "./checkpoints"
config.SAVE_FREQUENCY = 5

# Visualization
config.GENERATE_PLOTS = True
config.PLOT_FORMAT = "png"

# Tracking
config.TRACK_SAMPLE_WASTE = True
config.TRACK_CONFIDENCE_EVOLUTION = True

# Device
config.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print("="*70)
print("CIFAR-10 QUICK TEST CONFIGURATION")
print("="*70)
print(f"Dataset: {config.DATASET}")
print(f"Epochs: {config.NUM_EPOCHS}")
print(f"Batch Size: {config.BATCH_SIZE}")
print(f"Device: {config.DEVICE}")
print(f"Ratios to test: {config.RATIOS_TO_TEST}")
print(f"Model: {config.MODEL_NAME}")
print(f"Multi-Model: {config.ENABLE_MULTI_MODEL}")
print("="*70)
print()

if __name__ == "__main__":
    main()
