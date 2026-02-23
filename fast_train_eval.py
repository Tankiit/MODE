#!/usr/bin/env python3
"""
Fast training and evaluation on selected subsets.
Uses a small ResNet and quick training for rapid results.
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


# Simple ResNet for CIFAR-10
class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = torch.relu(out)
        return out


class SmallResNet(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)

        self.layer1 = self._make_layer(32, 32, 2, stride=1)
        self.layer2 = self._make_layer(32, 64, 2, stride=2)
        self.layer3 = self._make_layer(64, 128, 2, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(128, num_classes)

    def _make_layer(self, in_channels, out_channels, num_blocks, stride):
        layers = []
        layers.append(BasicBlock(in_channels, out_channels, stride))
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
        out = self.fc(out)
        return out


def train_and_evaluate(percentage, epochs=50, batch_size=128, lr=0.1, device='mps'):
    """Train on selected subset and evaluate on test data."""

    print(f"\n{'='*80}")
    print(f"Training on {percentage}% of CIFAR-10")
    print(f"{'='*80}")

    # Load indices
    pct_int = int(percentage)
    indices_file = f'selected_indices_combined_resnet18_cifar10_{pct_int}pct.npy'

    if not Path(indices_file).exists():
        print(f"Error: {indices_file} not found")
        return None

    selected_indices = np.load(indices_file)
    print(f"Loaded {len(selected_indices)} selected indices")

    # Data transforms
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    # Load datasets
    full_train = torchvision.datasets.CIFAR10(
        root='/Users/tanmoy/research/data', train=True, download=False, transform=transform_train
    )

    test_dataset = torchvision.datasets.CIFAR10(
        root='/Users/tanmoy/research/data', train=False, download=False, transform=transform_test
    )

    # Create subset
    train_dataset = Subset(full_train, selected_indices)

    # Data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    # Create model
    model = SmallResNet(num_classes=10).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    print(f"Training samples: {len(train_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    print(f"Epochs: {epochs}")
    print(f"Batch size: {batch_size}")
    print(f"Device: {device}\n")

    # Training
    best_test_acc = 0
    best_epoch = 0
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

        # Evaluate
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
            best_epoch = epoch

        scheduler.step()

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch [{epoch+1:3d}/{epochs}] Train: {train_acc:6.2f}% | Test: {test_acc:6.2f}% | Best: {best_test_acc:6.2f}%")

    training_time = time.time() - start_time

    print(f"\n{'='*80}")
    print(f"Training Complete!")
    print(f"{'='*80}")
    print(f"Percentage: {percentage}%")
    print(f"Samples: {len(train_dataset)}")
    print(f"Best Test Accuracy: {best_test_acc:.2f}% (epoch {best_epoch+1})")
    print(f"Final Test Accuracy: {test_acc:.2f}%")
    print(f"Training Time: {training_time:.2f}s ({training_time/60:.2f}m)")
    print(f"{'='*80}\n")

    return {
        'percentage': percentage,
        'n_samples': len(train_dataset),
        'best_test_acc': best_test_acc,
        'final_test_acc': test_acc,
        'best_epoch': best_epoch,
        'training_time': training_time,
        'history': history
    }


def run_all_experiments(percentages, epochs=50, device='mps'):
    """Run experiments for all percentages."""

    results = []

    for pct in percentages:
        result = train_and_evaluate(pct, epochs=epochs, device=device)
        if result:
            results.append(result)

            # Save intermediate results
            with open('training_results.json', 'w') as f:
                json.dump(results, f, indent=2)

    return results


def plot_results(results, output_dir='plots'):
    """Plot test accuracy vs percentage."""

    Path(output_dir).mkdir(exist_ok=True)

    # Extract data
    percentages = [r['percentage'] for r in results]
    n_samples = [r['n_samples'] for r in results]
    best_accs = [r['best_test_acc'] for r in results]
    final_accs = [r['final_test_acc'] for r in results]

    # Plot 1: Test Accuracy vs Percentage
    fig, ax = plt.subplots(figsize=(12, 6))

    ax.plot(percentages, best_accs, 'o-', linewidth=2, markersize=10,
            label='Best Test Accuracy', color='steelblue')
    ax.plot(percentages, final_accs, 's--', linewidth=2, markersize=8,
            label='Final Test Accuracy', color='coral', alpha=0.7)

    ax.set_xlabel('Percentage of Training Data (%)', fontsize=12)
    ax.set_ylabel('Test Accuracy (%)', fontsize=12)
    ax.set_title('CIFAR-10 Test Accuracy vs Data Percentage\n(Combined Selection Strategy)',
                fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    # Add value labels
    for pct, acc in zip(percentages, best_accs):
        ax.annotate(f'{acc:.1f}%', xy=(pct, acc), xytext=(0, 10),
                   textcoords='offset points', ha='center', fontsize=9)

    plt.tight_layout()
    plt.savefig(f'{output_dir}/test_accuracy_vs_percentage.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir}/test_accuracy_vs_percentage.png")
    plt.close()

    # Plot 2: Accuracy vs Number of Samples
    fig, ax = plt.subplots(figsize=(12, 6))

    ax.plot(n_samples, best_accs, 'o-', linewidth=2, markersize=10, color='green')

    ax.set_xlabel('Number of Training Samples', fontsize=12)
    ax.set_ylabel('Best Test Accuracy (%)', fontsize=12)
    ax.set_title('CIFAR-10 Test Accuracy vs Training Set Size',
                fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)

    # Add value labels
    for n, acc, pct in zip(n_samples, best_accs, percentages):
        ax.annotate(f'{acc:.1f}%\n({pct}%)', xy=(n, acc), xytext=(0, 10),
                   textcoords='offset points', ha='center', fontsize=8)

    plt.tight_layout()
    plt.savefig(f'{output_dir}/test_accuracy_vs_samples.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir}/test_accuracy_vs_samples.png")
    plt.close()

    # Plot 3: Training curves
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    for idx, result in enumerate(results[:6]):  # Plot first 6
        if idx >= len(axes):
            break

        ax = axes[idx]
        history = result['history']
        epochs = range(1, len(history['train_acc']) + 1)

        ax.plot(epochs, history['train_acc'], label='Train', linewidth=2)
        ax.plot(epochs, history['test_acc'], label='Test', linewidth=2)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Accuracy (%)')
        ax.set_title(f"{result['percentage']}% ({result['n_samples']} samples)\nBest: {result['best_test_acc']:.1f}%")
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{output_dir}/training_curves.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir}/training_curves.png")
    plt.close()


def print_summary_table(results):
    """Print a summary table."""

    print("\n" + "="*100)
    print("CIFAR-10 Test Accuracy Summary")
    print("="*100)
    print(f"{'%':<6} {'Samples':<10} {'Best Test Acc':<15} {'Final Test Acc':<15} {'Time (s)':<12} {'Acc/1000 samples':<15}")
    print("-"*100)

    for r in results:
        acc_per_1k = r['best_test_acc'] / (r['n_samples'] / 1000)
        print(f"{r['percentage']:<6.0f} {r['n_samples']:<10} {r['best_test_acc']:<15.2f} "
              f"{r['final_test_acc']:<15.2f} {r['training_time']:<12.1f} {acc_per_1k:<15.2f}")

    print("="*100 + "\n")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--percentages', type=float, nargs='+',
                       default=[1, 2, 5, 10, 15, 20, 30],
                       help='Percentages to evaluate')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of training epochs')
    parser.add_argument('--device', type=str, default='mps',
                       choices=['cuda', 'cpu', 'mps'])

    args = parser.parse_args()

    print("="*100)
    print("CIFAR-10 Training Evaluation on Selected Subsets")
    print("="*100)
    print(f"Percentages: {args.percentages}")
    print(f"Epochs: {args.epochs}")
    print(f"Device: {args.device}")
    print("="*100 + "\n")

    # Run experiments
    results = run_all_experiments(args.percentages, epochs=args.epochs, device=args.device)

    # Save results
    with open('training_results_final.json', 'w') as f:
        json.dump(results, f, indent=2)

    # Print summary
    print_summary_table(results)

    # Generate plots
    plot_results(results)

    print("\nAll experiments completed!")
    print("Results saved to: training_results_final.json")
    print("Plots saved to: plots/")
