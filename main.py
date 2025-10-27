#!/usr/bin/env python3

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Subset, Dataset
import torchvision
import torchvision.transforms as transforms
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
import logging
from tqdm import tqdm
import timm

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ==============================================================================
# CONFIGURATION
# ==============================================================================

class SelectiveClassificationConfig:

    # Model settings
    MODEL_NAME = "resnet34"  # Main training model: resnet18, resnet34, resnet50, efficientnet_b0, vit_base_patch16_224
    REFERENCE_MODEL_NAME = "resnet18"  # Smaller reference model: resnet18 is smaller than resnet34
    PRETRAINED = True  # Use pretrained weights for better initialization
    
    # Alternative model combinations for different scenarios:
    # For stronger main model: MODEL_NAME="resnet50", REFERENCE_MODEL_NAME="resnet18"
    # For mobile/efficient: MODEL_NAME="mobilenetv3_large_100", REFERENCE_MODEL_NAME="mobilenetv3_small_100"  
    # For vision transformer: MODEL_NAME="vit_base_patch16_224", REFERENCE_MODEL_NAME="vit_small_patch16_224"
    # For EfficientNet: MODEL_NAME="efficientnet_b2", REFERENCE_MODEL_NAME="efficientnet_b0"

    # Data settings
    DATASET = "CIFAR10"  # Options: CIFAR10, CIFAR100, ImageNet, SVHN, STL10
    DATA_PATH = "/Users/tanmoy/research/data"
    IMAGE_SIZE = 32  # 32 for CIFAR, 224 for ImageNet
    NUM_CLASSES = 10  # 10 for CIFAR10, 100 for CIFAR100, 1000 for ImageNet

    # Selective training settings
    SELECTION_RATIO = 0.3  # Select top 30% of samples with highest excess loss
    USE_REFERENCE_MODEL = True  # If False, uses random baseline
    SELECTION_FREQUENCY = 1  # Update selection every N epochs

    # Caching settings
    CACHE_MODE = 'memory'  # Options: 'none', 'memory', 'disk'
    PRECOMPUTE_REFERENCE_LOSSES = True  # Precompute all reference losses before training

    # Training settings
    OUTPUT_DIR = "./selective_classification_output"
    NUM_EPOCHS = 5  # Quick test: 5 epochs (use 200 for full training)
    BATCH_SIZE = 32  # Smaller batch for CPU (use 128 for GPU)
    LEARNING_RATE = 0.1  # Standard SGD learning rate for ResNet
    MOMENTUM = 0.9
    WEIGHT_DECAY = 1e-4  # Slightly reduced for pretrained models
    LR_SCHEDULE = "cosine"  # Options: cosine, step, multistep
    WARMUP_EPOCHS = 5

    # Logging
    LOGGING_STEPS = 10  # Log more frequently for testing
    EVAL_FREQUENCY = 1  # Evaluate every epoch for testing
    SAVE_FREQUENCY = 5  # Save checkpoint every 5 epochs

    # Device
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

config = SelectiveClassificationConfig()

# ==============================================================================
# DATA PROCESSING (Your existing structure)
# ==============================================================================

class IndexedDataset(Dataset):

    def __init__(self, base_dataset):
        self.base_dataset = base_dataset

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        data, target = self.base_dataset[idx]
        return data, target, idx


def load_and_process_data():
    logger.info(f"Loading {config.DATASET} dataset...")
    
    # Define transforms
    if config.DATASET in ["CIFAR10", "CIFAR100"]:
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
    else:
        # ImageNet-style transforms
        transform_train = transforms.Compose([
            transforms.Resize(256),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        transform_test = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    # Load datasets
    if config.DATASET == "CIFAR10":
        train_ds = torchvision.datasets.CIFAR10(
            root=config.DATA_PATH, train=True, download=True, transform=transform_train
        )
        eval_ds = torchvision.datasets.CIFAR10(
            root=config.DATA_PATH, train=False, download=True, transform=transform_test
        )
    elif config.DATASET == "CIFAR100":
        train_ds = torchvision.datasets.CIFAR100(
            root=config.DATA_PATH, train=True, download=True, transform=transform_train
        )
        eval_ds = torchvision.datasets.CIFAR100(
            root=config.DATA_PATH, train=False, download=True, transform=transform_test
        )
    else:
        raise ValueError(f"Dataset {config.DATASET} not supported")
    
    logger.info(f"Train samples: {len(train_ds)}")
    logger.info(f"Eval samples: {len(eval_ds)}")
    
    return train_ds, eval_ds

# ==============================================================================
# RHO-1 SELECTIVE TRAINING COMPONENTS
# ==============================================================================

class SelectiveReferenceModel:

    def __init__(self, cache_mode='memory'):
        self.cache_mode = cache_mode
        self.loss_cache = {}  # For memory caching: {sample_idx: loss_value}
        self.cache_file = None  # For disk caching

        if config.USE_REFERENCE_MODEL:
            logger.info("Creating reference model...")
            # Load smaller reference model using timm (e.g., ResNet18 as reference for ResNet34 main model)
            self.model = timm.create_model(config.REFERENCE_MODEL_NAME,
                                         pretrained=config.PRETRAINED,
                                         num_classes=config.NUM_CLASSES)
            self.model = self.model.to(config.DEVICE)
            self.model.eval()

            logger.info(f"Reference model: {config.REFERENCE_MODEL_NAME}")
            logger.info(f"Reference model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
            logger.info(f"Cache mode: {cache_mode}")

            # Setup disk cache if needed
            if cache_mode == 'disk':
                import os
                self.cache_file = os.path.join(config.OUTPUT_DIR, 'reference_loss_cache.pt')
                if os.path.exists(self.cache_file):
                    logger.info(f"Loading cached reference losses from {self.cache_file}")
                    self.loss_cache = torch.load(self.cache_file)
                    logger.info(f"Loaded {len(self.loss_cache)} cached losses")

            # Quick pretraining on clean subset (optional)
            self._pretrain_reference_model()
        else:
            self.model = None
            logger.info("Using random baseline instead of reference model")

    def _pretrain_reference_model(self):
        logger.info("Pretraining reference model...")
        # For demo purposes, we'll skip this and use the pretrained weights
        # In practice, you'd train on a curated subset
        pass

    def precompute_all_losses(self, dataloader, use_indexed_data=False):
        if self.model is None:
            logger.warning("No reference model, skipping precomputation")
            return

        logger.info("Precomputing reference losses for entire dataset...")
        self.model.eval()

        sample_idx = 0
        with torch.no_grad():
            for batch_data in tqdm(dataloader, desc="Precomputing reference losses"):
                # Unpack batch (with or without indices)
                if use_indexed_data:
                    inputs, targets, indices = batch_data
                else:
                    inputs, targets = batch_data

                inputs, targets = inputs.to(config.DEVICE), targets.to(config.DEVICE)

                # Forward pass through reference model
                outputs = self.model(inputs)

                # Compute per-sample cross-entropy losses
                loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
                per_sample_losses = loss_fct(outputs, targets)

                # Cache losses
                if use_indexed_data:
                    # Use actual indices from the dataset
                    for idx, loss_value in zip(indices.cpu().tolist(), per_sample_losses):
                        self.loss_cache[idx] = loss_value.cpu().item()
                else:
                    # Use sequential indices
                    for i, loss_value in enumerate(per_sample_losses):
                        self.loss_cache[sample_idx + i] = loss_value.cpu().item()
                    sample_idx += len(targets)

        logger.info(f"Precomputed {len(self.loss_cache)} reference losses")

        # Save to disk if using disk cache
        if self.cache_mode == 'disk':
            torch.save(self.loss_cache, self.cache_file)
            logger.info(f"Saved cache to {self.cache_file}")

    def get_cached_losses(self, indices):
        if isinstance(indices, torch.Tensor):
            indices = indices.cpu().tolist()

        losses = []
        for idx in indices:
            if idx in self.loss_cache:
                losses.append(self.loss_cache[idx])
            else:
                # If not cached, return a default value
                losses.append(2.0)  # Average cross-entropy loss

        return torch.tensor(losses, device=config.DEVICE)

    def compute_reference_loss(self, inputs, targets, sample_indices=None):
        # Use cache if available
        if self.cache_mode in ['memory', 'disk'] and sample_indices is not None:
            if len(self.loss_cache) > 0:
                return self.get_cached_losses(sample_indices)

        # Otherwise compute on-the-fly
        if self.model is None:
            # Random baseline: return random losses
            batch_size = targets.shape[0]
            return torch.randn(batch_size, device=targets.device) * 0.5 + 2.0

        self.model.eval()
        with torch.no_grad():
            # Forward pass through reference model
            outputs = self.model(inputs)

            # Compute per-sample cross-entropy losses
            loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
            per_sample_losses = loss_fct(outputs, targets)

            return per_sample_losses

class SelectiveSampleSelector:
    
    def __init__(self, reference_model: SelectiveReferenceModel):
        self.reference_model = reference_model
        self.selection_ratio = config.SELECTION_RATIO
        
        # Statistics tracking
        self.total_samples_seen = 0
        self.total_samples_selected = 0
        self.excess_loss_history = []
    
    def compute_excess_loss(self, inputs, targets, training_logits, sample_indices=None):
        # Get reference model losses (uses cache if available)
        ref_sample_losses = self.reference_model.compute_reference_loss(
            inputs, targets, sample_indices=sample_indices
        )

        # Compute training model losses
        loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
        train_sample_losses = loss_fct(training_logits, targets)

        # Excess loss = training loss - reference loss
        excess_losses = train_sample_losses - ref_sample_losses

        # Track statistics
        self.excess_loss_history.append(excess_losses.mean().item())
        if len(self.excess_loss_history) > 1000:
            self.excess_loss_history = self.excess_loss_history[-500:]

        return excess_losses
    
    def select_samples(self, excess_losses):
        batch_size = excess_losses.shape[0]
        
        # Select top-k% samples
        k = max(1, int(self.selection_ratio * batch_size))
        
        if batch_size > 0:
            # Get top-k samples with highest excess loss
            topk_values, topk_indices = torch.topk(excess_losses, min(k, batch_size))
            
            # Create selection mask
            selection_mask = torch.zeros_like(excess_losses, dtype=torch.bool)
            selection_mask[topk_indices] = True
        else:
            selection_mask = torch.zeros_like(excess_losses, dtype=torch.bool)
        
        # Update statistics
        self.total_samples_seen += batch_size
        self.total_samples_selected += selection_mask.sum().item()
        
        return selection_mask
    
    def get_selection_stats(self):
        if self.total_samples_seen > 0:
            overall_ratio = self.total_samples_selected / self.total_samples_seen
        else:
            overall_ratio = 0.0
        
        avg_excess_loss = np.mean(self.excess_loss_history) if self.excess_loss_history else 0.0
        
        return {
            'overall_selection_ratio': overall_ratio,
            'total_samples_seen': self.total_samples_seen,
            'total_samples_selected': self.total_samples_selected,
            'avg_excess_loss': avg_excess_loss
        }

# ==============================================================================
# TRAINING FUNCTIONS
# ==============================================================================

def train_model_with_selection(model, train_loader, val_loader, sample_selector, criterion, optimizer, scheduler=None, use_indexed_data=False):

    logger.info("Starting selective training with caching...")
    model.train()

    for epoch in range(config.NUM_EPOCHS):
        epoch_loss = 0.0
        epoch_acc = 0.0
        num_batches = 0

        for batch_idx, batch_data in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.NUM_EPOCHS}")):
            # Unpack batch (with or without indices)
            if use_indexed_data:
                inputs, targets, sample_indices = batch_data
                sample_indices = sample_indices.to(config.DEVICE)
            else:
                inputs, targets = batch_data
                sample_indices = None

            inputs, targets = inputs.to(config.DEVICE), targets.to(config.DEVICE)

            # Forward pass
            outputs = model(inputs)

            # Compute excess loss and select samples (uses cache if available)
            excess_losses = sample_selector.compute_excess_loss(
                inputs, targets, outputs, sample_indices=sample_indices
            )
            selection_mask = sample_selector.select_samples(excess_losses)

            # Compute loss only on selected samples
            if selection_mask.sum() > 0:
                selected_outputs = outputs[selection_mask]
                selected_targets = targets[selection_mask]
                loss = criterion(selected_outputs, selected_targets)
            else:
                # Fallback: if no samples selected, use regular loss
                loss = criterion(outputs, targets)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Statistics
            epoch_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            epoch_acc += (predicted == targets).sum().item() / targets.size(0)
            num_batches += 1

            if batch_idx % config.LOGGING_STEPS == 0:
                stats = sample_selector.get_selection_stats()
                cache_info = f", Cached: {len(sample_selector.reference_model.loss_cache)}" if hasattr(sample_selector.reference_model, 'loss_cache') else ""
                logger.info(f"Batch {batch_idx}: Loss={loss.item():.4f}, "
                           f"Selection ratio={stats['overall_selection_ratio']:.3f}{cache_info}")

        # End of epoch
        epoch_loss /= num_batches
        epoch_acc /= num_batches

        if scheduler:
            scheduler.step()

        # Validation
        if epoch % config.EVAL_FREQUENCY == 0:
            val_acc = evaluate_model(model, val_loader, use_indexed_data=use_indexed_data)
            logger.info(f"Epoch {epoch+1}: Train Loss={epoch_loss:.4f}, Train Acc={epoch_acc:.4f}, Val Acc={val_acc:.4f}")

        # Save checkpoint
        if epoch % config.SAVE_FREQUENCY == 0:
            torch.save(model.state_dict(), f"{config.OUTPUT_DIR}/model_epoch_{epoch}.pth")

    return model

def evaluate_model(model, val_loader, use_indexed_data=False):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for batch_data in val_loader:
            # Unpack batch (with or without indices)
            if use_indexed_data:
                inputs, targets, _ = batch_data
            else:
                inputs, targets = batch_data

            inputs, targets = inputs.to(config.DEVICE), targets.to(config.DEVICE)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()

    model.train()
    return correct / total

def run_selective_training():

    logger.info("Starting Selective Classification Training with Caching")
    logger.info("=" * 70)
    logger.info(f"Cache mode: {config.CACHE_MODE}")
    logger.info(f"Precompute reference losses: {config.PRECOMPUTE_REFERENCE_LOSSES}")
    logger.info("=" * 70)

    # 1. Load and process data
    train_ds, eval_ds = load_and_process_data()

    # 2. Wrap datasets with IndexedDataset if caching is enabled
    use_indexed_data = config.CACHE_MODE in ['memory', 'disk']
    if use_indexed_data:
        logger.info("Wrapping datasets with index tracking for caching...")
        train_ds = IndexedDataset(train_ds)
        eval_ds = IndexedDataset(eval_ds)

    # 3. Create data loaders
    train_loader = DataLoader(train_ds, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader = DataLoader(eval_ds, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=2)

    # 4. Initialize reference model with caching
    reference_model = SelectiveReferenceModel(cache_mode=config.CACHE_MODE)

    # 5. Precompute reference losses if enabled
    if config.PRECOMPUTE_REFERENCE_LOSSES and config.USE_REFERENCE_MODEL:
        logger.info("\n" + "=" * 70)
        logger.info("PRECOMPUTING REFERENCE LOSSES")
        logger.info("=" * 70)
        reference_model.precompute_all_losses(train_loader, use_indexed_data=use_indexed_data)
        logger.info(f"Precomputed {len(reference_model.loss_cache)} reference losses")
        logger.info("These losses will be reused throughout training (no recomputation!)")
        logger.info("=" * 70 + "\n")

    # 6. Initialize sample selector
    sample_selector = SelectiveSampleSelector(reference_model)

    # 7. Initialize main model
    logger.info("Loading main training model...")
    model = timm.create_model(config.MODEL_NAME, pretrained=config.PRETRAINED, num_classes=config.NUM_CLASSES)
    model = model.to(config.DEVICE)

    # 8. Setup training components
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=config.LEARNING_RATE,
                         momentum=config.MOMENTUM, weight_decay=config.WEIGHT_DECAY)

    if config.LR_SCHEDULE == "cosine":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.NUM_EPOCHS)
    elif config.LR_SCHEDULE == "step":
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
    else:
        scheduler = None

    # 9. Create output directory
    import os
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)

    # 10. Train the model
    model = train_model_with_selection(
        model, train_loader, val_loader, sample_selector,
        criterion, optimizer, scheduler, use_indexed_data=use_indexed_data
    )

    # 11. Final evaluation
    final_acc = evaluate_model(model, val_loader, use_indexed_data=use_indexed_data)

    # 12. Print final statistics
    final_stats = sample_selector.get_selection_stats()
    logger.info("\n" + "=" * 70)
    logger.info("FINAL SELECTIVE TRAINING RESULTS")
    logger.info("=" * 70)
    logger.info(f"Final validation accuracy: {final_acc:.4f}")
    logger.info(f"Overall selection ratio: {final_stats['overall_selection_ratio']:.3f}")
    logger.info(f"Total samples processed: {final_stats['total_samples_seen']:,}")
    logger.info(f"Total samples selected: {final_stats['total_samples_selected']:,}")
    logger.info(f"Average excess loss: {final_stats['avg_excess_loss']:.4f}")

    if config.CACHE_MODE != 'none':
        logger.info(f"Reference losses cached: {len(reference_model.loss_cache):,}")
        logger.info(f"Cache hits saved {len(reference_model.loss_cache) * config.NUM_EPOCHS:,} forward passes!")

    # 13. Save the final model
    torch.save(model.state_dict(), f"{config.OUTPUT_DIR}/final_model.pth")
    logger.info(f"Model saved to {config.OUTPUT_DIR}")

    return model, final_stats

# ==============================================================================
# BASELINE COMPARISON
# ==============================================================================

def run_baseline_training():
    
    logger.info("Running baseline training for comparison...")
    
    # Load data
    train_ds, eval_ds = load_and_process_data()
    train_loader = DataLoader(train_ds, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader = DataLoader(eval_ds, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=2)
    
    # Model
    model = timm.create_model(config.MODEL_NAME, pretrained=config.PRETRAINED, num_classes=config.NUM_CLASSES)
    model = model.to(config.DEVICE)
    
    # Training components
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=config.LEARNING_RATE, 
                         momentum=config.MOMENTUM, weight_decay=config.WEIGHT_DECAY)
    
    if config.LR_SCHEDULE == "cosine":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.NUM_EPOCHS)
    else:
        scheduler = None
    
    # Training loop
    model.train()
    for epoch in range(config.NUM_EPOCHS):
        epoch_loss = 0.0
        num_batches = 0
        
        for inputs, targets in tqdm(train_loader, desc=f"Baseline Epoch {epoch+1}/{config.NUM_EPOCHS}"):
            inputs, targets = inputs.to(config.DEVICE), targets.to(config.DEVICE)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            num_batches += 1
        
        if scheduler:
            scheduler.step()
        
        epoch_loss /= num_batches
        
        if epoch % config.EVAL_FREQUENCY == 0:
            val_acc = evaluate_model(model, val_loader)
            logger.info(f"Baseline Epoch {epoch+1}: Train Loss={epoch_loss:.4f}, Val Acc={val_acc:.4f}")
    
    # Final evaluation
    final_acc = evaluate_model(model, val_loader)
    logger.info(f"Baseline final accuracy: {final_acc:.4f}")
    
    return model, final_acc

# ==============================================================================
# MAIN EXECUTION
# ==============================================================================

if __name__ == "__main__":
    # Run selective training
    selective_model, selective_stats = run_selective_training()
    
    # Optionally run baseline for comparison
    print("\n" + "Run baseline training for comparison? (y/n): ", end="")
    if input().lower().startswith('y'):
        baseline_model, baseline_acc = run_baseline_training()
        
        # Compare results
        logger.info("\n" + "=" * 50)
        logger.info("COMPARISON RESULTS")
        logger.info("=" * 50)
        
        # Re-evaluate selective model for fair comparison
        train_ds, eval_ds = load_and_process_data()
        val_loader = DataLoader(eval_ds, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=2)
        selective_acc = evaluate_model(selective_model, val_loader)
        
        logger.info(f"Selective training accuracy: {selective_acc:.4f}")
        logger.info(f"Baseline training accuracy:  {baseline_acc:.4f}")
        logger.info(f"Improvement:                 {selective_acc - baseline_acc:.4f}")
        logger.info(f"Selection ratio:             {selective_stats['overall_selection_ratio']:.3f}")
    
    logger.info("\nSelective classification experiment completed!")