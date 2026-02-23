#!/usr/bin/env python3
"""
Simplified adaptive selection - demonstrates the concept of adapting strategy weights.
"""

import torch
import numpy as np
from cachelib_feature_extraction import CacheLibStrategies, create_model
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import json
from pathlib import Path


def get_adaptive_weights(stage_pct, target_pct):
    """
    Simple adaptive weighting based on stage.
    Early stages: focus on uncertainty/diversity
    Late stages: focus on class_balance/boundary
    """
    progress = stage_pct / target_pct

    if progress < 0.3:  # Early - explore
        return {
            'uncertainty': 0.5,
            'diversity': 0.3,
            'class_balance': 0.1,
            'boundary': 0.1
        }
    elif progress < 0.7:  # Middle - balance
        return {
            'uncertainty': 0.3,
            'diversity': 0.3,
            'class_balance': 0.2,
            'boundary': 0.2
        }
    else:  # Late - refine
        return {
            'uncertainty': 0.2,
            'diversity': 0.2,
            'class_balance': 0.3,
            'boundary': 0.3
        }


def run_progressive_adaptive_selection(percentage, device='mps', dataset_name='cifar10'):
    """Progressive selection with adaptive strategy weighting."""

    print(f"\n{'='*80}")
    print(f"Progressive Adaptive Selection - {percentage}% target")
    print(f"{'='*80}\n")

    # Load dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    if dataset_name == 'cifar10':
        train_dataset = torchvision.datasets.CIFAR10(
            root='/Users/tanmoy/research/data', train=True, download=False, transform=transform
        )
        num_classes = 10

    # Create model
    model = create_model('resnet18', num_classes=num_classes, pretrained=False, device=device)
    model.eval()

    # Initialize strategies
    strategies = CacheLibStrategies(device=device)

    # Progressive stages
    total_samples = len(train_dataset)
    target_samples = int(total_samples * percentage / 100)

    stages = [0.5, 1.0, 2.0, 5.0, percentage]
    stages = [s for s in stages if s <= percentage]

    all_selected = set()
    stage_results = []

    for stage_pct in stages:
        stage_samples = int(total_samples * stage_pct / 100)

        print(f"\nStage {stage_pct}% - Target: {stage_samples} samples")

        # Get adaptive weights
        weights = get_adaptive_weights(stage_pct, percentage)
        print(f"  Adaptive weights: {weights}")

        # Create dataloader
        data_loader = DataLoader(train_dataset, batch_size=128, shuffle=False, num_workers=2)

        # Select with adaptive weights
        selected = strategies.combined_selection(
            model=model,
            data_loader=data_loader,
            dataset=train_dataset,
            n_select=stage_samples,
            weights=weights
        )

        all_selected.update(selected)

        stage_results.append({
            'percentage': stage_pct,
            'n_samples': len(all_selected),
            'weights': weights
        })

        print(f"  Total selected so far: {len(all_selected)}")

    # Save results
    output_file = f'adaptive_progressive_{int(percentage)}pct.npy'
    np.save(output_file, np.array(list(all_selected)))

    meta_file = f'adaptive_progressive_{int(percentage)}pct_meta.json'
    with open(meta_file, 'w') as f:
        json.dump({
            'target_percentage': percentage,
            'final_samples': len(all_selected),
            'stages': stage_results
        }, f, indent=2)

    print(f"\n{'='*80}")
    print(f"Progressive Adaptive Selection Complete!")
    print(f"{'='*80}")
    print(f"Total selected: {len(all_selected)} samples")
    print(f"Saved to: {output_file}")
    print(f"{'='*80}\n")

    return list(all_selected), stage_results


def compare_fixed_vs_adaptive(percentage, device='mps'):
    """Compare fixed weights vs adaptive weights."""

    print(f"\n{'='*80}")
    print(f"Comparing Fixed vs Adaptive Selection - {percentage}%")
    print(f"{'='*80}\n")

    # Fixed strategy
    print("Running FIXED strategy (uniform weights)...")
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    train_dataset = torchvision.datasets.CIFAR10(
        root='/Users/tanmoy/research/data', train=True, download=False, transform=transform
    )

    model = create_model('resnet18', num_classes=10, pretrained=False, device=device)
    model.eval()

    strategies = CacheLibStrategies(device=device)
    data_loader = DataLoader(train_dataset, batch_size=128, shuffle=False, num_workers=2)

    n_select = int(len(train_dataset) * percentage / 100)

    fixed_weights = {'uncertainty': 0.25, 'diversity': 0.25, 'class_balance': 0.25, 'boundary': 0.25}
    fixed_indices = strategies.combined_selection(model, data_loader, train_dataset, n_select, fixed_weights)

    print(f"Fixed selection: {len(fixed_indices)} samples\n")

    # Adaptive strategy
    print("Running ADAPTIVE strategy (progressive)...")
    adaptive_indices, adaptive_stages = run_progressive_adaptive_selection(percentage, device)

    # Analyze difference
    fixed_set = set(fixed_indices)
    adaptive_set = set(adaptive_indices)

    overlap = len(fixed_set & adaptive_set)
    only_fixed = len(fixed_set - adaptive_set)
    only_adaptive = len(adaptive_set - fixed_set)

    print(f"\n{'='*80}")
    print(f"Comparison Results")
    print(f"{'='*80}")
    print(f"Fixed selected:     {len(fixed_set)} samples")
    print(f"Adaptive selected:  {len(adaptive_set)} samples")
    print(f"Overlap:            {overlap} samples ({overlap/len(fixed_set)*100:.1f}%)")
    print(f"Only in fixed:      {only_fixed} samples")
    print(f"Only in adaptive:   {only_adaptive} samples")
    print(f"{'='*80}\n")

    # Save comparison
    comparison = {
        'percentage': percentage,
        'fixed': {
            'n_samples': len(fixed_set),
            'weights': fixed_weights
        },
        'adaptive': {
            'n_samples': len(adaptive_set),
            'stages': adaptive_stages
        },
        'overlap': overlap,
        'overlap_percentage': overlap/len(fixed_set)*100
    }

    with open(f'comparison_{int(percentage)}pct.json', 'w') as f:
        json.dump(comparison, f, indent=2)

    return comparison


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--percentage', type=float, required=True)
    parser.add_argument('--device', type=str, default='mps')
    parser.add_argument('--compare', action='store_true')

    args = parser.parse_args()

    if args.compare:
        compare_fixed_vs_adaptive(args.percentage, args.device)
    else:
        run_progressive_adaptive_selection(args.percentage, args.device)
