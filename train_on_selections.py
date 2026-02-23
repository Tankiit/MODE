#!/usr/bin/env python3
"""Train classifier on selected subsets and evaluate on test data."""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import torchvision
import torchvision.transforms as transforms
import numpy as np
import time
import argparse
import json
from pathlib import Path

# Simple ResNet18 for CIFAR-10
try:
    import timm
    TIMM_AVAILABLE = True
except ImportError:
    TIMM_AVAILABLE = False


def create_model(num_classes=10, device='cuda'):
    """Create ResNet18 model."""
    if TIMM_AVAILABLE:
        model = timm.create_model('resnet18', pretrained=False, num_classes=num_classes)
    else:
        import torchvision.models as models
        model = models.resnet18(pretrained=False)
        model.fc = nn.Linear(model.fc.in_features, num_classes)

    return model.to(device)


def train_epoch(model, train_loader, criterion, optimizer, device):
    """Train for one epoch."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100. * correct / total

    return epoch_loss, epoch_acc


def evaluate(model, test_loader, criterion, device):
    """Evaluate on test set."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    test_loss = running_loss / len(test_loader)
    test_acc = 100. * correct / total

    return test_loss, test_acc


def train_and_evaluate(budget, data_dir, device, epochs=100, batch_size=128, lr=0.1):
    """Train on selected subset and evaluate on test data."""

    print(f"\n{'='*80}")
    print(f"Training with Budget: {budget} samples")
    print(f"{'='*80}\n")

    # Load selected indices
    indices_file = f'selected_indices_combined_resnet18_cifar10_budget{budget}.npy'
    selected_indices = np.load(indices_file)
    print(f"Loaded {len(selected_indices)} selected indices from {indices_file}")

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

    # Load full dataset
    full_train_dataset = torchvision.datasets.CIFAR10(
        root=data_dir, train=True, download=False, transform=transform_train
    )

    test_dataset = torchvision.datasets.CIFAR10(
        root=data_dir, train=False, download=False, transform=transform_test
    )

    # Create subset from selected indices
    train_dataset = Subset(full_train_dataset, selected_indices)

    # Data loaders
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=4, pin_memory=True if device == 'cuda' else False
    )

    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=4, pin_memory=True if device == 'cuda' else False
    )

    print(f"Training set size: {len(train_dataset)}")
    print(f"Test set size: {len(test_dataset)}")
    print(f"Batch size: {batch_size}")
    print(f"Epochs: {epochs}")
    print(f"Learning rate: {lr}")
    print(f"Device: {device}\n")

    # Create model
    model = create_model(num_classes=10, device=device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)

    # Learning rate scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # Training loop
    best_test_acc = 0.0
    best_epoch = 0
    training_history = {
        'train_loss': [],
        'train_acc': [],
        'test_loss': [],
        'test_acc': [],
        'lr': []
    }

    start_time = time.time()

    for epoch in range(epochs):
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)

        # Evaluate
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)

        # Update scheduler
        scheduler.step()

        # Save history
        training_history['train_loss'].append(train_loss)
        training_history['train_acc'].append(train_acc)
        training_history['test_loss'].append(test_loss)
        training_history['test_acc'].append(test_acc)
        training_history['lr'].append(optimizer.param_groups[0]['lr'])

        # Track best
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            best_epoch = epoch

        # Print progress
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch [{epoch+1:3d}/{epochs}] "
                  f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:6.2f}% | "
                  f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:6.2f}% | "
                  f"Best: {best_test_acc:6.2f}%")

    total_time = time.time() - start_time

    print(f"\n{'='*80}")
    print(f"Training Complete!")
    print(f"{'='*80}")
    print(f"Budget: {budget} samples")
    print(f"Best Test Accuracy: {best_test_acc:.2f}% (epoch {best_epoch+1})")
    print(f"Final Test Accuracy: {test_acc:.2f}%")
    print(f"Training Time: {total_time:.2f}s ({total_time/60:.2f}m)")
    print(f"{'='*80}\n")

    # Save results
    results = {
        'budget': int(budget),
        'best_test_acc': float(best_test_acc),
        'final_test_acc': float(test_acc),
        'best_epoch': int(best_epoch),
        'total_epochs': int(epochs),
        'training_time': float(total_time),
        'training_history': training_history
    }

    results_file = f'training_results_budget{budget}.json'
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"Results saved to {results_file}")

    return results


def main():
    parser = argparse.ArgumentParser(description='Train on selected subsets')

    parser.add_argument('--budget', type=int, required=True,
                       help='Budget size (1000, 2500, 5000, 10000)')
    parser.add_argument('--data-dir', type=str, default='/Users/tanmoy/research/data',
                       help='Path to dataset directory')
    parser.add_argument('--device', type=str, default='mps',
                       choices=['cuda', 'cpu', 'mps'],
                       help='Device to use')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=128,
                       help='Batch size')
    parser.add_argument('--lr', type=float, default=0.1,
                       help='Learning rate')

    args = parser.parse_args()

    # Check device
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("Warning: CUDA not available, falling back to CPU")
        args.device = 'cpu'
    elif args.device == 'mps' and not (hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()):
        print("Warning: MPS not available, falling back to CPU")
        args.device = 'cpu'

    # Train and evaluate
    results = train_and_evaluate(
        budget=args.budget,
        data_dir=args.data_dir,
        device=args.device,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr
    )


if __name__ == '__main__':
    main()
