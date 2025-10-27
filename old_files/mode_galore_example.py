"""
MODE + GALORE Integration Example
=================================

This example demonstrates how to integrate MODE (Memory-Optimized Data Selection)
with the GALORE framework for RL-guided data selection on CIFAR datasets.

Based on the GALORE repository structure and CIFAR experiments.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
from torchvision.models import resnet18, vgg16
import numpy as np
import json
import os
from pathlib import Path
from tqdm import tqdm

# Import MODE components
from mode_integrated_galore import (
    MODEConfig, MODEController, select_data_with_mode,
    create_mode_config_for_cifar, save_mode_results
)


# ============================================================================
# CIFAR DATASET LOADERS
# ============================================================================

def load_cifar10_data(data_dir: str = './data', batch_size: int = 32, 
                     augment: bool = True):
    """
    Load CIFAR10 dataset with optional augmentation.
    Compatible with GALORE framework.
    """
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    # Load datasets
    trainset = torchvision.datasets.CIFAR10(
        root=data_dir, train=True, download=True, transform=transform_train if augment else transform_test
    )
    testset = torchvision.datasets.CIFAR10(
        root=data_dir, train=False, download=True, transform=transform_test
    )
    
    # Create data loaders
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    return trainloader, testloader


def load_cifar100_data(data_dir: str = './data', batch_size: int = 32,
                      augment: bool = True):
    """
    Load CIFAR100 dataset with optional augmentation.
    """
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])
    
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])
    
    trainset = torchvision.datasets.CIFAR100(
        root=data_dir, train=True, download=True, transform=transform_train if augment else transform_test
    )
    testset = torchvision.datasets.CIFAR100(
        root=data_dir, train=False, download=True, transform=transform_test
    )
    
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    return trainloader, testloader


# ============================================================================
# MODEL DEFINITIONS
# ============================================================================

class CIFARResNet(nn.Module):
    """
    ResNet model adapted for CIFAR datasets.
    Compatible with GALORE framework.
    """
    
    def __init__(self, num_classes: int = 10, depth: int = 18):
        super(CIFARResNet, self).__init__()
        
        if depth == 18:
            self.backbone = resnet18(pretrained=False)
            self.backbone.fc = nn.Linear(512, num_classes)
        elif depth == 34:
            from torchvision.models import resnet34
            self.backbone = resnet34(pretrained=False)
            self.backbone.fc = nn.Linear(512, num_classes)
        else:
            raise ValueError(f"Unsupported ResNet depth: {depth}")
    
    def forward(self, x):
        return self.backbone(x)


class CIFARVGG(nn.Module):
    """
    VGG model adapted for CIFAR datasets.
    """
    
    def __init__(self, num_classes: int = 10, depth: int = 16):
        super(CIFARVGG, self).__init__()
        
        if depth == 16:
            self.backbone = vgg16(pretrained=False)
            self.backbone.classifier[6] = nn.Linear(4096, num_classes)
        elif depth == 19:
            from torchvision.models import vgg19
            self.backbone = vgg19(pretrained=False)
            self.backbone.classifier[6] = nn.Linear(4096, num_classes)
        else:
            raise ValueError(f"Unsupported VGG depth: {depth}")
    
    def forward(self, x):
        return self.backbone(x)


# ============================================================================
# TRAINING FUNCTIONS
# ============================================================================

def train_with_mode(model, trainloader, testloader, config: MODEConfig, 
                   device, num_epochs: int = 50):
    """
    Train model using MODE for adaptive data selection.
    """
    # Initialize MODE controller
    mode_controller = MODEController(config, device)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=config.learning_rate, 
                         momentum=0.9, weight_decay=config.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    
    # Training history
    history = {
        'train_loss': [], 'train_acc': [], 'test_loss': [], 'test_acc': [],
        'selection_ratios': [], 'strategy_weights': []
    }
    
    print(f"Starting MODE training for {num_epochs} epochs")
    print(f"Device: {device}")
    print(f"Selection budget: {config.selection_budget:.1%}")
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        epoch_selection_ratios = []
        epoch_strategy_weights = []
        
        progress_bar = tqdm(trainloader, desc=f'Epoch {epoch+1}/{num_epochs}')
        
        for batch_idx, (images, labels) in enumerate(progress_bar):
            images, labels = images.to(device), labels.to(device)
            
            # Use MODE to select data
            selected_images, selected_labels, selection_info = select_data_with_mode(
                images, labels, mode_controller, config, model, criterion
            )
            
            # Forward pass
            optimizer.zero_grad()
            outputs = model(selected_images)
            loss = criterion(outputs, selected_labels)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Statistics
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += selected_labels.size(0)
            train_correct += (predicted == selected_labels).sum().item()
            
            # Update MODE controller
            accuracy = train_correct / train_total
            success_signal = np.exp(-loss.item())  # Convert loss to success signal
            
            mode_controller.update_training_state(
                loss.item(), accuracy, selection_info['selection_ratio'],
                selection_info['strategy_weights'], success_signal, labels
            )
            
            # Record selection info
            epoch_selection_ratios.append(selection_info['selection_ratio'])
            epoch_strategy_weights.append(selection_info['strategy_weights'])
            
            # Update progress bar
            progress_bar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{100.*train_correct/train_total:.1f}%',
                'Sel': f'{selection_info["selection_ratio"]:.1%}'
            })
        
        # Calculate epoch averages
        avg_train_loss = train_loss / len(trainloader)
        avg_train_acc = 100. * train_correct / train_total
        avg_selection_ratio = np.mean(epoch_selection_ratios)
        avg_strategy_weights = np.mean(epoch_strategy_weights, axis=0)
        
        # Evaluation
        test_loss, test_acc = evaluate_model(model, testloader, criterion, device)
        
        # Update scheduler
        scheduler.step()
        
        # Record history
        history['train_loss'].append(avg_train_loss)
        history['train_acc'].append(avg_train_acc)
        history['test_loss'].append(test_loss)
        history['test_acc'].append(test_acc)
        history['selection_ratios'].append(avg_selection_ratio)
        history['strategy_weights'].append(avg_strategy_weights.tolist())
        
        # Print epoch summary
        print(f'Epoch {epoch+1}: Train Loss: {avg_train_loss:.4f}, Train Acc: {avg_train_acc:.2f}%, '
              f'Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%, '
              f'Selection: {avg_selection_ratio:.1%}')
        
        # Strategy analysis
        strategy_interp = mode_controller.get_strategy_interpretation()
        if strategy_interp:
            print(f'  Dominant Strategy: {strategy_interp["dominant_strategy"]}')
            print(f'  Strategy Weights: {strategy_interp["current_weights"]}')
    
    return history


def evaluate_model(model, testloader, criterion, device):
    """
    Evaluate model on test set.
    """
    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            test_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    avg_test_loss = test_loss / len(testloader)
    avg_test_acc = 100. * correct / total
    
    return avg_test_loss, avg_test_acc


# ============================================================================
# EXPERIMENT FUNCTIONS
# ============================================================================

def run_cifar10_mode_experiment(data_dir: str = './data', device: str = 'auto'):
    """
    Run MODE experiment on CIFAR10 dataset.
    """
    # Setup device
    if device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(device)
    
    # Load data
    print("Loading CIFAR10 dataset...")
    trainloader, testloader = load_cifar10_data(data_dir, batch_size=32)
    
    # Create model
    model = CIFARResNet(num_classes=10, depth=18).to(device)
    
    # Create MODE config
    config = create_mode_config_for_cifar(
        num_classes=10,
        epochs=50,
        batch_size=32,
        selection_budget=0.3,
        learning_rate=1e-3
    )
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Train with MODE
    history = train_with_mode(model, trainloader, testloader, config, device)
    
    # Save results
    results = {
        'dataset': 'CIFAR10',
        'model': 'ResNet18',
        'config': config.__dict__,
        'history': history,
        'final_test_acc': history['test_acc'][-1],
        'best_test_acc': max(history['test_acc'])
    }
    
    save_mode_results(results, './results/cifar10_mode_results.json')
    
    print(f"\nCIFAR10 MODE Experiment Complete!")
    print(f"Final Test Accuracy: {results['final_test_acc']:.2f}%")
    print(f"Best Test Accuracy: {results['best_test_acc']:.2f}%")
    
    return results


def run_cifar100_mode_experiment(data_dir: str = './data', device: str = 'auto'):
    """
    Run MODE experiment on CIFAR100 dataset.
    """
    # Setup device
    if device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(device)
    
    # Load data
    print("Loading CIFAR100 dataset...")
    trainloader, testloader = load_cifar100_data(data_dir, batch_size=64)
    
    # Create model
    model = CIFARResNet(num_classes=100, depth=34).to(device)
    
    # Create MODE config
    config = create_mode_config_for_cifar(
        num_classes=100,
        epochs=100,
        batch_size=64,
        selection_budget=0.3,
        learning_rate=5e-4
    )
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Train with MODE
    history = train_with_mode(model, trainloader, testloader, config, device)
    
    # Save results
    results = {
        'dataset': 'CIFAR100',
        'model': 'ResNet34',
        'config': config.__dict__,
        'history': history,
        'final_test_acc': history['test_acc'][-1],
        'best_test_acc': max(history['test_acc'])
    }
    
    save_mode_results(results, './results/cifar100_mode_results.json')
    
    print(f"\nCIFAR100 MODE Experiment Complete!")
    print(f"Final Test Accuracy: {results['final_test_acc']:.2f}%")
    print(f"Best Test Accuracy: {results['best_test_acc']:.2f}%")
    
    return results


def run_comparison_experiment(data_dir: str = './data', device: str = 'auto'):
    """
    Run comparison experiment: MODE vs Random Selection vs Full Training.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') if device == 'auto' else torch.device(device)
    
    print("Running comparison experiment: MODE vs Random vs Full Training")
    
    # Load data
    trainloader, testloader = load_cifar10_data(data_dir, batch_size=32)
    
    results = {}
    
    # 1. Full Training (baseline)
    print("\n1. Full Training (Baseline)")
    model_full = CIFARResNet(num_classes=10, depth=18).to(device)
    config_full = create_mode_config_for_cifar(selection_budget=1.0)  # Use all data
    history_full = train_with_mode(model_full, trainloader, testloader, config_full, device, num_epochs=30)
    results['full_training'] = {
        'final_acc': history_full['test_acc'][-1],
        'best_acc': max(history_full['test_acc']),
        'history': history_full
    }
    
    # 2. Random Selection
    print("\n2. Random Selection")
    model_random = CIFARResNet(num_classes=10, depth=18).to(device)
    config_random = create_mode_config_for_cifar(selection_budget=0.3)
    # Modify to use random selection instead of MODE
    # (This would require modifying the training loop)
    results['random_selection'] = {
        'note': 'Random selection implementation needed'
    }
    
    # 3. MODE Selection
    print("\n3. MODE Selection")
    model_mode = CIFARResNet(num_classes=10, depth=18).to(device)
    config_mode = create_mode_config_for_cifar(selection_budget=0.3)
    history_mode = train_with_mode(model_mode, trainloader, testloader, config_mode, device, num_epochs=30)
    results['mode_selection'] = {
        'final_acc': history_mode['test_acc'][-1],
        'best_acc': max(history_mode['test_acc']),
        'history': history_mode
    }
    
    # Save comparison results
    save_mode_results(results, './results/comparison_results.json')
    
    print(f"\nComparison Results:")
    print(f"Full Training - Best Acc: {results['full_training']['best_acc']:.2f}%")
    print(f"MODE Selection - Best Acc: {results['mode_selection']['best_acc']:.2f}%")
    
    return results


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """
    Main function to run MODE experiments.
    Compatible with GALORE framework.
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='MODE + GALORE Integration Example')
    parser.add_argument('--experiment', type=str, default='cifar10',
                       choices=['cifar10', 'cifar100', 'comparison'],
                       help='Experiment to run')
    parser.add_argument('--data_dir', type=str, default='./data',
                       help='Directory for CIFAR datasets')
    parser.add_argument('--device', type=str, default='auto',
                       choices=['auto', 'cuda', 'cpu'],
                       help='Device to run on')
    parser.add_argument('--results_dir', type=str, default='./results',
                       help='Directory to save results')
    
    args = parser.parse_args()
    
    # Create results directory
    os.makedirs(args.results_dir, exist_ok=True)
    
    print("="*70)
    print("MODE + GALORE Integration Example")
    print("="*70)
    print(f"Experiment: {args.experiment}")
    print(f"Data Directory: {args.data_dir}")
    print(f"Results Directory: {args.results_dir}")
    print("="*70)
    
    # Run experiment
    if args.experiment == 'cifar10':
        results = run_cifar10_mode_experiment(args.data_dir, args.device)
    elif args.experiment == 'cifar100':
        results = run_cifar100_mode_experiment(args.data_dir, args.device)
    elif args.experiment == 'comparison':
        results = run_comparison_experiment(args.data_dir, args.device)
    else:
        raise ValueError(f"Unknown experiment: {args.experiment}")
    
    print("\nExperiment completed successfully!")
    print(f"Results saved to: {args.results_dir}")


if __name__ == "__main__":
    main()
