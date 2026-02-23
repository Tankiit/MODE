#!/usr/bin/env python3
"""
Train classifier on a single percentage - allows parallel execution.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import torchvision
import torchvision.transforms as transforms
import numpy as np
import time
import json
import argparse
from pathlib import Path

# Import from comprehensive_evaluation
from comprehensive_evaluation import SmallResNet, train_and_evaluate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--percentage', type=float, required=True)
    parser.add_argument('--selection-type', type=str, required=True,
                       choices=['fixed', 'adaptive'])
    parser.add_argument('--dataset', type=str, default='cifar10')
    parser.add_argument('--device', type=str, default='mps')
    parser.add_argument('--epochs', type=int, default=30)

    args = parser.parse_args()

    # Determine indices file
    if args.selection_type == 'fixed':
        indices_file = f'selected_indices_combined_resnet18_cifar10_{int(args.percentage)}pct.npy'
    else:
        indices_file = f'adaptive_progressive_{int(args.percentage)}pct.npy'

    # Check if result already exists
    result_file = f'training_result_{args.selection_type}_{int(args.percentage)}pct.json'
    if Path(result_file).exists():
        print(f"Result already exists: {result_file}")
        with open(result_file, 'r') as f:
            result = json.load(f)
        print(f"Best test accuracy: {result['best_test_acc']:.2f}%")
        return

    # Train and evaluate
    result = train_and_evaluate(
        indices_file=indices_file,
        dataset_name=args.dataset,
        epochs=args.epochs,
        device=args.device,
        selection_type=args.selection_type
    )

    if result is not None:
        # Save result
        with open(result_file, 'w') as f:
            json.dump(result, f, indent=2)
        print(f"\nResult saved to: {result_file}")


if __name__ == '__main__':
    main()
