#!/usr/bin/env python3
"""
Example usage of CacheLib-powered feature extraction library.

This script demonstrates:
1. Multiple architecture support (ResNet, EfficientNet, ViT, etc.)
2. Different selection strategies
3. Cache performance benefits
4. Integration with existing MODE pipeline
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
import torchvision
import torchvision.transforms as transforms
import numpy as np
import time
from cachelib_feature_extraction import CacheLibStrategies, create_model


def example_1_basic_usage():
    """Example 1: Basic usage with a single model and strategy."""

    print("\n" + "="*80)
    print("Example 1: Basic Usage - Uncertainty Selection with ResNet18")
    print("="*80)

    # Setup
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Load CIFAR-10
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    train_dataset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform
    )

    # Create data loader
    train_loader = DataLoader(
        train_dataset, batch_size=128, shuffle=False, num_workers=4
    )

    # Create model
    print("Creating ResNet18...")
    model = create_model('resnet18', num_classes=10, pretrained=True, device=device)
    model.eval()

    # Initialize strategies
    strategies = CacheLibStrategies(device=device, cache_size_mb=1024)

    # Select 1000 samples using uncertainty
    print("\nSelecting 1000 samples using uncertainty strategy...")
    selected_indices = strategies.uncertainty_selection(model, train_loader, n_select=1000)

    print(f"Selected {len(selected_indices)} samples")
    print(f"Sample indices: {selected_indices[:10]}...")

    # Print cache stats
    strategies.print_cache_stats()


def example_2_multiple_architectures():
    """Example 2: Comparing multiple architectures."""

    print("\n" + "="*80)
    print("Example 2: Multiple Architectures Comparison")
    print("="*80)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Load dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    train_dataset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform
    )

    # Use subset for faster demo
    subset_indices = np.random.choice(len(train_dataset), 5000, replace=False)
    subset = Subset(train_dataset, subset_indices)
    train_loader = DataLoader(subset, batch_size=128, shuffle=False, num_workers=4)

    # Test different architectures
    architectures = ['resnet18', 'resnet34', 'efficientnet_b0']

    # Initialize strategies once
    strategies = CacheLibStrategies(device=device, cache_size_mb=2048)

    results = {}

    for arch_name in architectures:
        print(f"\n{'='*60}")
        print(f"Testing {arch_name}")
        print(f"{'='*60}")

        try:
            # Create model
            model = create_model(arch_name, num_classes=10, pretrained=True, device=device)
            model.eval()

            # Measure selection time
            start_time = time.time()
            selected = strategies.uncertainty_selection(model, train_loader, n_select=500)
            selection_time = time.time() - start_time

            results[arch_name] = {
                'selected_count': len(selected),
                'selection_time': selection_time
            }

            print(f"Selected: {len(selected)} samples")
            print(f"Time: {selection_time:.2f}s")

        except Exception as e:
            print(f"Error with {arch_name}: {e}")
            results[arch_name] = {'error': str(e)}

    # Print summary
    print("\n" + "="*80)
    print("Summary")
    print("="*80)
    for arch, res in results.items():
        if 'error' not in res:
            print(f"{arch:20s}: {res['selected_count']} samples in {res['selection_time']:.2f}s")
        else:
            print(f"{arch:20s}: ERROR - {res['error']}")

    strategies.print_cache_stats()


def example_3_all_strategies():
    """Example 3: Compare all selection strategies."""

    print("\n" + "="*80)
    print("Example 3: All Selection Strategies")
    print("="*80)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Load dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    train_dataset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform
    )

    # Use subset
    subset_indices = np.random.choice(len(train_dataset), 5000, replace=False)
    subset = Subset(train_dataset, subset_indices)
    train_loader = DataLoader(subset, batch_size=128, shuffle=False, num_workers=4)

    # Create model
    model = create_model('resnet18', num_classes=10, pretrained=True, device=device)
    model.eval()

    # Initialize strategies
    strategies = CacheLibStrategies(device=device, cache_size_mb=2048)

    n_select = 500

    # Test all strategies
    strategy_results = {}

    print(f"\nSelecting {n_select} samples with each strategy...\n")

    # Uncertainty
    print("1. Uncertainty Selection...")
    start = time.time()
    selected = strategies.uncertainty_selection(model, train_loader, n_select)
    strategy_results['uncertainty'] = {
        'indices': selected,
        'time': time.time() - start,
        'count': len(selected)
    }
    print(f"   Selected {len(selected)} samples in {strategy_results['uncertainty']['time']:.2f}s")

    # Diversity
    print("2. Diversity Selection...")
    start = time.time()
    selected = strategies.diversity_selection(model, train_loader, n_select)
    strategy_results['diversity'] = {
        'indices': selected,
        'time': time.time() - start,
        'count': len(selected)
    }
    print(f"   Selected {len(selected)} samples in {strategy_results['diversity']['time']:.2f}s")

    # Class Balance
    print("3. Class Balance Selection...")
    start = time.time()
    selected = strategies.class_balance_selection(subset, n_select)
    strategy_results['class_balance'] = {
        'indices': selected,
        'time': time.time() - start,
        'count': len(selected)
    }
    print(f"   Selected {len(selected)} samples in {strategy_results['class_balance']['time']:.2f}s")

    # Boundary
    print("4. Boundary Selection...")
    start = time.time()
    selected = strategies.boundary_selection(model, train_loader, n_select)
    strategy_results['boundary'] = {
        'indices': selected,
        'time': time.time() - start,
        'count': len(selected)
    }
    print(f"   Selected {len(selected)} samples in {strategy_results['boundary']['time']:.2f}s")

    # Combined
    print("5. Combined Selection...")
    weights = {
        'uncertainty': 0.3,
        'diversity': 0.3,
        'class_balance': 0.2,
        'boundary': 0.2
    }
    start = time.time()
    selected = strategies.combined_selection(model, train_loader, subset, n_select, weights)
    strategy_results['combined'] = {
        'indices': selected,
        'time': time.time() - start,
        'count': len(selected)
    }
    print(f"   Selected {len(selected)} samples in {strategy_results['combined']['time']:.2f}s")

    # Print summary
    print("\n" + "="*80)
    print("Strategy Performance Summary")
    print("="*80)
    for strategy, res in strategy_results.items():
        print(f"{strategy:20s}: {res['count']:4d} samples in {res['time']:6.2f}s")

    strategies.print_cache_stats()


def example_4_cache_performance():
    """Example 4: Demonstrate cache performance benefits."""

    print("\n" + "="*80)
    print("Example 4: Cache Performance Benefits")
    print("="*80)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Load dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    train_dataset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform
    )

    # Use subset
    subset_indices = np.random.choice(len(train_dataset), 5000, replace=False)
    subset = Subset(train_dataset, subset_indices)
    train_loader = DataLoader(subset, batch_size=128, shuffle=False, num_workers=4)

    # Create model
    model = create_model('resnet18', num_classes=10, pretrained=True, device=device)
    model.eval()

    # Initialize strategies
    strategies = CacheLibStrategies(device=device, cache_size_mb=2048)

    # First run - cache miss
    print("\nFirst run (cache miss)...")
    start = time.time()
    selected1 = strategies.uncertainty_selection(model, train_loader, 500)
    time1 = time.time() - start
    print(f"Time: {time1:.2f}s")

    # Second run - cache hit
    print("\nSecond run (cache hit)...")
    start = time.time()
    selected2 = strategies.uncertainty_selection(model, train_loader, 500)
    time2 = time.time() - start
    print(f"Time: {time2:.2f}s")

    # Calculate speedup
    speedup = time1 / time2 if time2 > 0 else float('inf')

    print(f"\n{'='*80}")
    print(f"Cache Performance")
    print(f"{'='*80}")
    print(f"First run (no cache):  {time1:.2f}s")
    print(f"Second run (cached):   {time2:.2f}s")
    print(f"Speedup:               {speedup:.2f}x")
    print(f"Time saved:            {time1 - time2:.2f}s ({(1 - time2/time1)*100:.1f}%)")

    strategies.print_cache_stats()


def example_5_integration_with_mode():
    """Example 5: Integration with MODE hypernetwork controller."""

    print("\n" + "="*80)
    print("Example 5: Integration with MODE Hypernetwork")
    print("="*80)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Load dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    train_dataset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform
    )

    # Use subset
    subset_indices = np.random.choice(len(train_dataset), 5000, replace=False)
    subset = Subset(train_dataset, subset_indices)
    train_loader = DataLoader(subset, batch_size=128, shuffle=False, num_workers=4)

    # Create model
    model = create_model('resnet18', num_classes=10, pretrained=True, device=device)
    model.eval()

    # Initialize strategies
    strategies = CacheLibStrategies(device=device, cache_size_mb=2048)

    # Simulate MODE's adaptive strategy weights
    # In practice, these would come from the hypernetwork controller
    print("\nSimulating MODE adaptive strategy selection...\n")

    # Early stage: Focus on diversity and class balance
    print("Early training stage:")
    weights_early = {
        'uncertainty': 0.1,
        'diversity': 0.4,
        'class_balance': 0.4,
        'boundary': 0.1
    }
    selected_early = strategies.combined_selection(
        model, train_loader, subset, 200, weights_early
    )
    print(f"  Weights: {weights_early}")
    print(f"  Selected: {len(selected_early)} samples")

    # Mid stage: Balance uncertainty and diversity
    print("\nMid training stage:")
    weights_mid = {
        'uncertainty': 0.3,
        'diversity': 0.3,
        'class_balance': 0.2,
        'boundary': 0.2
    }
    selected_mid = strategies.combined_selection(
        model, train_loader, subset, 200, weights_mid
    )
    print(f"  Weights: {weights_mid}")
    print(f"  Selected: {len(selected_mid)} samples")

    # Late stage: Focus on boundary and uncertainty
    print("\nLate training stage:")
    weights_late = {
        'uncertainty': 0.4,
        'diversity': 0.1,
        'class_balance': 0.1,
        'boundary': 0.4
    }
    selected_late = strategies.combined_selection(
        model, train_loader, subset, 200, weights_late
    )
    print(f"  Weights: {weights_late}")
    print(f"  Selected: {len(selected_late)} samples")

    print(f"\nTotal selected across stages: {len(set(selected_early + selected_mid + selected_late))} unique samples")

    strategies.print_cache_stats()


def example_6_vision_transformer():
    """Example 6: Using Vision Transformers (ViT)."""

    print("\n" + "="*80)
    print("Example 6: Vision Transformer Feature Extraction")
    print("="*80)

    try:
        import timm
    except ImportError:
        print("⚠️ timm not available. Skipping ViT example.")
        return

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Load dataset
    transform = transforms.Compose([
        transforms.Resize(224),  # ViT requires 224x224
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    train_dataset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform
    )

    # Use smaller subset for ViT (slower)
    subset_indices = np.random.choice(len(train_dataset), 1000, replace=False)
    subset = Subset(train_dataset, subset_indices)
    train_loader = DataLoader(subset, batch_size=32, shuffle=False, num_workers=4)

    # Create ViT model
    print("Creating Vision Transformer (vit_tiny_patch16_224)...")
    model = create_model('vit_tiny_patch16_224', num_classes=10, pretrained=True, device=device)
    model.eval()

    # Initialize strategies
    strategies = CacheLibStrategies(device=device, cache_size_mb=2048)

    # Select samples
    print("\nSelecting 200 samples using combined strategy...")
    selected = strategies.combined_selection(model, train_loader, subset, 200)

    print(f"Selected {len(selected)} samples")

    strategies.print_cache_stats()


def main():
    """Run all examples."""

    print("="*80)
    print("CacheLib Feature Extraction Library - Examples")
    print("="*80)

    examples = [
        ("Basic Usage", example_1_basic_usage),
        ("Multiple Architectures", example_2_multiple_architectures),
        ("All Strategies", example_3_all_strategies),
        ("Cache Performance", example_4_cache_performance),
        ("MODE Integration", example_5_integration_with_mode),
        ("Vision Transformer", example_6_vision_transformer),
    ]

    print("\nAvailable examples:")
    for i, (name, _) in enumerate(examples, 1):
        print(f"  {i}. {name}")

    print("\nRunning all examples...\n")

    for name, example_func in examples:
        try:
            example_func()
        except Exception as e:
            print(f"\n❌ Error in {name}: {e}")
            import traceback
            traceback.print_exc()

        print("\n" + "="*80)
        input("Press Enter to continue to next example...")


if __name__ == '__main__':
    main()