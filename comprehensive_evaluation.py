#!/usr/bin/env python3
"""
Comprehensive evaluation: Adaptive vs Fixed selection across percentages and datasets.
Trains classifiers and compares test accuracies.
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
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from simple_adaptive import run_progressive_adaptive_selection, compare_fixed_vs_adaptive


# Small ResNet for fast training
class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return torch.relu(out)


class SmallResNet(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.layer1 = self._make_layer(32, 32, 2, 1)
        self.layer2 = self._make_layer(32, 64, 2, 2)
        self.layer3 = self._make_layer(64, 128, 2, 2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(128, num_classes)

    def _make_layer(self, in_channels, out_channels, num_blocks, stride):
        layers = [BasicBlock(in_channels, out_channels, stride)]
        for _ in range(1, num_blocks):
            layers.append(BasicBlock(out_channels, out_channels, 1))
        return nn.Sequential(*layers)

    def forward(self, x):
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        return self.fc(out)


def train_and_evaluate(indices_file, dataset_name='cifar10', epochs=30,
                      batch_size=128, device='mps', selection_type='fixed'):
    """Train classifier on selected indices and evaluate."""

    print(f"\n{'='*80}")
    print(f"Training on {selection_type.upper()} selection")
    print(f"File: {indices_file}")
    print(f"{'='*80}\n")

    # Load indices
    if not Path(indices_file).exists():
        print(f"File not found: {indices_file}")
        return None

    indices = np.load(indices_file)
    print(f"Loaded {len(indices)} selected indices")

    # Data transforms
    if dataset_name == 'cifar10':
        mean, std = (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
        num_classes = 10
    else:  # cifar100
        mean, std = (0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)
        num_classes = 100

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    # Load datasets
    if dataset_name == 'cifar10':
        full_train = torchvision.datasets.CIFAR10(
            '/Users/tanmoy/research/data', train=True, download=False, transform=transform_train
        )
        test_dataset = torchvision.datasets.CIFAR10(
            '/Users/tanmoy/research/data', train=False, download=False, transform=transform_test
        )
    else:
        full_train = torchvision.datasets.CIFAR100(
            '/Users/tanmoy/research/data', train=True, download=False, transform=transform_train
        )
        test_dataset = torchvision.datasets.CIFAR100(
            '/Users/tanmoy/research/data', train=False, download=False, transform=transform_test
        )

    # Create subset
    train_dataset = Subset(full_train, indices)

    # Data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    print(f"Training samples: {len(train_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    print(f"Epochs: {epochs}\n")

    # Model
    model = SmallResNet(num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # Training
    best_test_acc = 0
    history = {'train_acc': [], 'test_acc': []}

    start_time = time.time()

    for epoch in range(epochs):
        # Train
        model.train()
        train_correct = 0
        train_total = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()

        train_acc = 100. * train_correct / train_total

        # Test
        model.eval()
        test_correct = 0
        test_total = 0

        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = outputs.max(1)
                test_total += labels.size(0)
                test_correct += predicted.eq(labels).sum().item()

        test_acc = 100. * test_correct / test_total

        history['train_acc'].append(train_acc)
        history['test_acc'].append(test_acc)

        if test_acc > best_test_acc:
            best_test_acc = test_acc

        scheduler.step()

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch [{epoch+1:3d}/{epochs}] Train: {train_acc:6.2f}% | Test: {test_acc:6.2f}% | Best: {best_test_acc:6.2f}%")

    training_time = time.time() - start_time

    print(f"\n{'='*80}")
    print(f"Training Complete - {selection_type.upper()}")
    print(f"{'='*80}")
    print(f"Best Test Accuracy: {best_test_acc:.2f}%")
    print(f"Final Test Accuracy: {test_acc:.2f}%")
    print(f"Training Time: {training_time:.1f}s ({training_time/60:.1f}m)")
    print(f"{'='*80}\n")

    return {
        'selection_type': selection_type,
        'n_samples': len(indices),
        'best_test_acc': best_test_acc,
        'final_test_acc': test_acc,
        'training_time': training_time,
        'history': history
    }


def run_full_comparison(percentages, dataset_name='cifar10', device='mps'):
    """Run full comparison across all percentages."""

    print(f"\n{'='*100}")
    print(f"COMPREHENSIVE EVALUATION: {dataset_name.upper()}")
    print(f"{'='*100}\n")

    all_results = []

    for pct in percentages:
        print(f"\n{'#'*100}")
        print(f"# PERCENTAGE: {pct}%")
        print(f"{'#'*100}\n")

        # Step 1: Run selection comparison if not already done
        comparison_file = f'comparison_{int(pct)}pct.json'
        if not Path(comparison_file).exists():
            print(f"Running selection comparison for {pct}%...")
            compare_fixed_vs_adaptive(pct, device)

        # Load comparison
        with open(comparison_file, 'r') as f:
            comparison = json.load(f)

        # Step 2: Train on fixed selection
        fixed_file = f'selected_indices_combined_resnet18_cifar10_{int(pct)}pct.npy'
        fixed_result = train_and_evaluate(
            fixed_file, dataset_name, epochs=30, device=device, selection_type='fixed'
        )

        # Step 3: Train on adaptive selection
        adaptive_file = f'adaptive_progressive_{int(pct)}pct.npy'
        adaptive_result = train_and_evaluate(
            adaptive_file, dataset_name, epochs=30, device=device, selection_type='adaptive'
        )

        # Combine results
        if fixed_result and adaptive_result:
            result = {
                'percentage': pct,
                'overlap': comparison.get('overlap_percentage', 0),
                'fixed': fixed_result,
                'adaptive': adaptive_result,
                'improvement': adaptive_result['best_test_acc'] - fixed_result['best_test_acc']
            }
            all_results.append(result)

    # Save all results
    with open(f'comprehensive_results_{dataset_name}.json', 'w') as f:
        json.dump(all_results, f, indent=2)

    return all_results


def plot_comparison_results(results, dataset_name='cifar10'):
    """Plot comparison results."""

    output_dir = 'comprehensive_plots'
    Path(output_dir).mkdir(exist_ok=True)

    percentages = [r['percentage'] for r in results]
    fixed_accs = [r['fixed']['best_test_acc'] for r in results]
    adaptive_accs = [r['adaptive']['best_test_acc'] for r in results]
    improvements = [r['improvement'] for r in results]
    overlaps = [r['overlap'] for r in results]

    # Plot 1: Test Accuracy Comparison
    fig, ax = plt.subplots(figsize=(12, 6))

    ax.plot(percentages, fixed_accs, 'o-', linewidth=2, markersize=10,
            label='Fixed (Uniform Weights)', color='steelblue')
    ax.plot(percentages, adaptive_accs, 's-', linewidth=2, markersize=10,
            label='Adaptive (Progressive)', color='coral')

    ax.set_xlabel('Percentage of Training Data (%)', fontsize=12)
    ax.set_ylabel('Test Accuracy (%)', fontsize=12)
    ax.set_title(f'{dataset_name.upper()}: Adaptive vs Fixed Selection\nTest Accuracy Comparison',
                fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    # Add value labels
    for pct, fixed, adaptive in zip(percentages, fixed_accs, adaptive_accs):
        ax.annotate(f'{fixed:.1f}%', xy=(pct, fixed), xytext=(0, -15),
                   textcoords='offset points', ha='center', fontsize=8, color='steelblue')
        ax.annotate(f'{adaptive:.1f}%', xy=(pct, adaptive), xytext=(0, 10),
                   textcoords='offset points', ha='center', fontsize=8, color='coral')

    plt.tight_layout()
    plt.savefig(f'{output_dir}/accuracy_comparison_{dataset_name}.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir}/accuracy_comparison_{dataset_name}.png")
    plt.close()

    # Plot 2: Improvement & Overlap
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    # Improvement
    colors = ['green' if x > 0 else 'red' for x in improvements]
    ax1.bar(range(len(percentages)), improvements, color=colors, alpha=0.7)
    ax1.set_xticks(range(len(percentages)))
    ax1.set_xticklabels([f'{p}%' for p in percentages])
    ax1.set_xlabel('Data Percentage', fontsize=12)
    ax1.set_ylabel('Improvement (Adaptive - Fixed) %', fontsize=12)
    ax1.set_title('Test Accuracy Improvement', fontsize=12, fontweight='bold')
    ax1.axhline(y=0, color='black', linestyle='--', linewidth=1)
    ax1.grid(axis='y', alpha=0.3)

    # Add value labels
    for i, imp in enumerate(improvements):
        ax1.text(i, imp, f'{imp:+.2f}%', ha='center',
                va='bottom' if imp > 0 else 'top', fontsize=9)

    # Overlap
    ax2.plot(percentages, overlaps, 'o-', linewidth=2, markersize=10, color='purple')
    ax2.set_xlabel('Data Percentage', fontsize=12)
    ax2.set_ylabel('Selection Overlap (%)', fontsize=12)
    ax2.set_title('Fixed vs Adaptive Selection Overlap', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)

    for pct, overlap in zip(percentages, overlaps):
        ax2.annotate(f'{overlap:.1f}%', xy=(pct, overlap), xytext=(0, 10),
                    textcoords='offset points', ha='center', fontsize=9)

    plt.tight_layout()
    plt.savefig(f'{output_dir}/improvement_overlap_{dataset_name}.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir}/improvement_overlap_{dataset_name}.png")
    plt.close()


def print_summary_table(results, dataset_name='cifar10'):
    """Print summary table."""

    print(f"\n{'='*120}")
    print(f"{dataset_name.upper()} - Comprehensive Results Summary")
    print(f"{'='*120}")
    print(f"{'%':<6} {'Fixed Acc':<12} {'Adaptive Acc':<14} {'Improvement':<14} {'Overlap':<12} {'Fixed Time':<12} {'Adaptive Time':<14}")
    print(f"{'-'*120}")

    for r in results:
        print(f"{r['percentage']:<6.0f} "
              f"{r['fixed']['best_test_acc']:<12.2f} "
              f"{r['adaptive']['best_test_acc']:<14.2f} "
              f"{r['improvement']:<14.2f} "
              f"{r['overlap']:<12.1f} "
              f"{r['fixed']['training_time']/60:<12.1f} "
              f"{r['adaptive']['training_time']/60:<14.1f}")

    print(f"{'='*120}\n")

    # Statistics
    avg_improvement = np.mean([r['improvement'] for r in results])
    avg_overlap = np.mean([r['overlap'] for r in results])

    print("Summary Statistics:")
    print(f"  Average Improvement: {avg_improvement:+.2f}%")
    print(f"  Average Overlap: {avg_overlap:.1f}%")
    print(f"  Wins (Adaptive > Fixed): {sum(1 for r in results if r['improvement'] > 0)}/{len(results)}")
    print()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--percentages', type=float, nargs='+',
                       default=[5, 10, 20],
                       help='Percentages to test')
    parser.add_argument('--dataset', type=str, default='cifar10',
                       choices=['cifar10', 'cifar100'])
    parser.add_argument('--device', type=str, default='mps')

    args = parser.parse_args()

    # Run comprehensive evaluation
    results = run_full_comparison(args.percentages, args.dataset, args.device)

    # Print summary
    print_summary_table(results, args.dataset)

    # Generate plots
    plot_comparison_results(results, args.dataset)

    print("\nComprehensive evaluation complete!")
    print(f"Results saved to: comprehensive_results_{args.dataset}.json")
    print(f"Plots saved to: comprehensive_plots/")
