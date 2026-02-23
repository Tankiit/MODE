#!/usr/bin/env python3
"""Analyze sample selections across different budgets."""

import numpy as np
import torchvision
import torchvision.transforms as transforms
from collections import Counter
import matplotlib.pyplot as plt

# Load CIFAR-10 to get labels
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])

train_dataset = torchvision.datasets.CIFAR10(
    root='/Users/tanmoy/research/data',
    train=True,
    download=False,
    transform=transform
)

class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

budgets = [1000, 2500, 5000, 10000]

print("="*80)
print("CIFAR-10 Sample Selection Analysis")
print("="*80)
print(f"\nTotal training samples: {len(train_dataset)}")
print(f"Strategy: Combined (uncertainty + diversity + class_balance + boundary)")
print(f"Model: ResNet18\n")

results = {}

for budget in budgets:
    filename = f'selected_indices_combined_resnet18_cifar10_budget{budget}.npy'
    indices = np.load(filename)

    # Get labels for selected indices
    labels = [train_dataset.targets[i] for i in indices]
    label_counts = Counter(labels)

    results[budget] = {
        'indices': indices,
        'label_counts': label_counts,
        'total': len(indices)
    }

    print(f"Budget: {budget} samples ({budget/len(train_dataset)*100:.1f}% of training data)")
    print("-" * 80)
    print(f"{'Class':<15} {'Count':>10} {'Percentage':>12} {'Samples/Class':>15}")
    print("-" * 80)

    for class_idx in range(10):
        count = label_counts.get(class_idx, 0)
        percentage = count / budget * 100
        samples_per_class = 5000  # CIFAR-10 has 5000 samples per class
        print(f"{class_names[class_idx]:<15} {count:>10} {percentage:>11.1f}% {count/samples_per_class*100:>14.1f}%")

    # Statistics
    counts = [label_counts.get(i, 0) for i in range(10)]
    mean_count = np.mean(counts)
    std_count = np.std(counts)
    min_count = np.min(counts)
    max_count = np.max(counts)

    print("-" * 80)
    print(f"Statistics:")
    print(f"  Mean samples per class: {mean_count:.1f}")
    print(f"  Std deviation: {std_count:.1f}")
    print(f"  Min samples: {min_count}")
    print(f"  Max samples: {max_count}")
    print(f"  Balance ratio (min/max): {min_count/max_count:.3f}")
    print()

# Overlap analysis
print("="*80)
print("Overlap Analysis Between Budgets")
print("="*80)

for i, budget1 in enumerate(budgets):
    for budget2 in budgets[i+1:]:
        indices1 = set(results[budget1]['indices'])
        indices2 = set(results[budget2]['indices'])

        overlap = len(indices1 & indices2)
        overlap_pct = overlap / len(indices1) * 100

        print(f"Budget {budget1} ∩ Budget {budget2}: {overlap} samples ({overlap_pct:.1f}% of {budget1})")

print("\n" + "="*80)
print("Summary")
print("="*80)
print(f"{'Budget':<10} {'Samples':>10} {'% of Data':>12} {'Balance (min/max)':>20}")
print("-" * 80)

for budget in budgets:
    counts = [results[budget]['label_counts'].get(i, 0) for i in range(10)]
    min_count = np.min(counts)
    max_count = np.max(counts)
    balance = min_count / max_count
    pct_of_data = budget / len(train_dataset) * 100

    print(f"{budget:<10} {budget:>10} {pct_of_data:>11.1f}% {balance:>19.3f}")

print("="*80)
