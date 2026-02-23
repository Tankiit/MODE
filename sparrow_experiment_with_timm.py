#!/usr/bin/env python3
"""
MODE Coreset Selection with timm models support

This script implements MODE experiments using timm models for better performance
and flexibility in model selection.
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
import matplotlib.pyplot as plt
from tqdm import tqdm
import argparse
import json
import time
from collections import Counter
from datetime import datetime
import pandas as pd
import timm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform

# Import the hypernetwork controller
from enhanced_meta_controller import EnhancedMetaController

class TimmModelWrapper(nn.Module):
    """Wrapper for timm models to handle different architectures"""
    def __init__(self, model_name, num_classes, pretrained=True):
        super(TimmModelWrapper, self).__init__()
        self.model_name = model_name
        self.num_classes = num_classes
        
        # Create the base model
        self.model = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=num_classes
        )
        
        # Get the feature extractor
        self.feature_extractor = self._create_feature_extractor()
        
    def _create_feature_extractor(self):
        """Create a feature extractor from the timm model"""
        # Different timm models have different architectures
        # We'll try to create a feature extractor by removing the last layer(s)
        
        if hasattr(self.model, 'fc'):
            # Models like ResNet, EfficientNet, etc.
            feature_extractor = nn.Sequential(*list(self.model.children())[:-1])
        elif hasattr(self.model, 'head'):
            # Models like ViT, DeiT, etc.
            feature_extractor = nn.Sequential(*list(self.model.children())[:-1])
        elif hasattr(self.model, 'classifier'):
            # Some other models
            feature_extractor = nn.Sequential(*list(self.model.children())[:-1])
        else:
            # Fallback - use forward_features if available
            if hasattr(self.model, 'forward_features'):
                feature_extractor = lambda x: self.model.forward_features(x)
            else:
                # Create a custom wrapper
                feature_extractor = self._custom_feature_extractor()
        
        return feature_extractor
    
    def _custom_feature_extractor(self):
        """Custom feature extractor for models with non-standard architectures"""
        class FeatureExtractor(nn.Module):
            def __init__(self, model):
                super().__init__()
                self.model = model
                
            def forward(self, x):
                # Monkey patch the forward method to return features before the final layer
                original_forward = type(self.model).forward
                
                def custom_forward(self, x):
                    # This will vary based on the model architecture
                    # You might need to customize this for specific models
                    features = None
                    
                    # Try to intercept features before the final layer
                    if hasattr(self, 'features'):
                        features = self.features(x)
                    elif hasattr(self, 'forward_features'):
                        features = self.forward_features(x)
                    else:
                        # Fallback: run through all but last layer
                        layers = list(self.children())
                        x = x
                        for layer in layers[:-1]:
                            x = layer(x)
                        features = x
                    
                    return features
                
                # Temporarily replace the forward method
                type(self.model).forward = custom_forward
                features = self.model(x)
                type(self.model).forward = original_forward
                
                return features
        
        return FeatureExtractor(self.model)
    
    def forward(self, x):
        return self.model(x)
    
    def get_features(self, x):
        """Extract features using the feature extractor"""
        return self.feature_extractor(x)

class MetaExperimentWithTimm:
    """Experiment runner for MODE with timm models"""
    def __init__(self, args):
        self.args = args
        self.device = self._setup_device()
        self.results_dir = self._create_results_dir()
        
        # Initialize datasets with appropriate transforms for timm models
        self.train_dataset, self.test_dataset = self._setup_datasets()
        self.test_loader = DataLoader(
            self.test_dataset, 
            batch_size=128, 
            shuffle=False, 
            num_workers=args.workers
        )
        
        # Create validation set (10% of test set)
        val_indices = np.random.choice(len(self.test_dataset), len(self.test_dataset)//10, replace=False)
        self.val_dataset = Subset(self.test_dataset, val_indices)
        self.val_loader = DataLoader(
            self.val_dataset, 
            batch_size=128, 
            shuffle=False, 
            num_workers=args.workers
        )
        
        # Track results
        self.experiments = []
        
        # Determine number of classes for the dataset
        self.num_classes = self._get_num_classes()
        
        # Get model info
        self.model_info = self._get_model_info()
    
    def _get_num_classes(self):
        """Determine the number of classes for the dataset"""
        if self.args.dataset == 'cifar10':
            return 10
        elif self.args.dataset == 'cifar100':
            return 100
        elif self.args.dataset == 'svhn':
            return 10
        else:
            raise ValueError(f"Unknown dataset: {self.args.dataset}")
    
    def _setup_device(self):
        """Setup compute device"""
        if self.args.device:
            device = torch.device(self.args.device)
        else:
            # Try CUDA first
            if torch.cuda.is_available():
                device = torch.device("cuda")
            # Then try MPS (Apple Silicon)
            elif hasattr(torch, 'backends') and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                device = torch.device("mps")
            # Fall back to CPU
            else:
                device = torch.device("cpu")
        
        print(f"Using device: {device}")
        return device
    
    def _create_results_dir(self):
        """Create directory for results"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_dir = os.path.join(self.args.output_dir, f"sparrow_timm_{self.args.model}_{self.args.dataset}_{timestamp}")
        os.makedirs(results_dir, exist_ok=True)
        
        # Save arguments
        with open(os.path.join(results_dir, "args.json"), "w") as f:
            json.dump(vars(self.args), f, indent=4)
        
        return results_dir
    
    def _get_model_info(self):
        """Get information about the timm model"""
        # Create a dummy model to get config
        dummy_model = timm.create_model(self.args.model, pretrained=False)
        config = resolve_data_config({}, model=dummy_model)
        
        return {
            'model_name': self.args.model,
            'input_size': config['input_size'],
            'mean': config['mean'],
            'std': config['std'],
            'interpolation': config.get('interpolation', 'bilinear')
        }
    
    def _setup_datasets(self):
        """Setup dataset with timm-compatible transforms"""
        # Get timm model config for transforms
        dummy_model = timm.create_model(self.args.model, pretrained=False)
        config = resolve_data_config({}, model=dummy_model)
        
        # Create timm transforms
        transform_train = create_transform(
            input_size=config['input_size'],
            is_training=True,
            hflip=0.5,
            mean=config['mean'],
            std=config['std'],
            interpolation=config.get('interpolation', 'bilinear')
        )
        
        transform_test = create_transform(
            input_size=config['input_size'],
            is_training=False,
            mean=config['mean'],
            std=config['std'],
            interpolation=config.get('interpolation', 'bilinear')
        )
        
        # Dataset-specific loading
        if self.args.dataset == 'svhn':
            train_dataset = torchvision.datasets.SVHN(
                root=self.args.data_dir, split='train', download=True, transform=transform_train
            )
            test_dataset = torchvision.datasets.SVHN(
                root=self.args.data_dir, split='test', download=True, transform=transform_test
            )
        elif self.args.dataset == 'cifar10':
            train_dataset = torchvision.datasets.CIFAR10(
                root=self.args.data_dir, train=True, download=True, transform=transform_train
            )
            test_dataset = torchvision.datasets.CIFAR10(
                root=self.args.data_dir, train=False, download=True, transform=transform_test
            )
        elif self.args.dataset == 'cifar100':
            train_dataset = torchvision.datasets.CIFAR100(
                root=self.args.data_dir, train=True, download=True, transform=transform_train
            )
            test_dataset = torchvision.datasets.CIFAR100(
                root=self.args.data_dir, train=False, download=True, transform=transform_test
            )
        else:
            raise ValueError(f"Unknown dataset: {self.args.dataset}")
        
        return train_dataset, test_dataset
    
    def _create_model(self):
        """Create timm model appropriate for the dataset"""
        model = TimmModelWrapper(
            model_name=self.args.model,
            num_classes=self.num_classes,
            pretrained=True
        )
        return model.to(self.device)
    
    def _compute_feature_statistics(self, model, subset_indices):
        """Compute feature space statistics for the meta-controller"""
        model.eval()
        
        # Sample subset for efficiency
        if len(subset_indices) > 500:
            sample_indices = np.random.choice(subset_indices, 500, replace=False)
        else:
            sample_indices = subset_indices
        
        # Create subset and loader
        subset = Subset(self.train_dataset, sample_indices)
        loader = DataLoader(subset, batch_size=64, shuffle=False, num_workers=2)
        
        # Extract features
        features = []
        labels = []
        
        try:
            with torch.no_grad():
                for inputs, targets in loader:
                    inputs = inputs.to(self.device)
                    batch_features = model.get_features(inputs)
                    
                    # Ensure features are flattened
                    if len(batch_features.shape) > 2:
                        batch_features = batch_features.reshape(batch_features.size(0), -1)
                    
                    features.append(batch_features.cpu().numpy())
                    labels.extend(targets.numpy())
        except Exception as e:
            print(f"Error during feature extraction: {e}")
            # Return default statistics
            return {
                "feature_diversity": 0.5,
                "feature_redundancy": 0.5,
                "class_separation": 0.5
            }
            
        # Process features
        if not features:
            return {
                "feature_diversity": 0.5,
                "feature_redundancy": 0.5,
                "class_separation": 0.5
            }
        
        features = np.vstack(features)
        labels = np.array(labels)
        
        # Compute simple diversity metric
        distances = []
        for i in range(min(100, len(features))):
            for j in range(i+1, min(100, len(features))):
                distances.append(np.sqrt(np.sum((features[i] - features[j])**2)))
        
        feature_diversity = np.mean(distances) / 100.0 if distances else 0.5
        # Prevent NaN values
        if np.isnan(feature_diversity):
            feature_diversity = 0.5
            
        # Compute class separation
        class_features = {}
        for i, label in enumerate(labels):
            if label not in class_features:
                class_features[label] = []
            class_features[label].append(features[i])
        
        # Compute within-class distances
        within_dists = []
        for label, feat_list in class_features.items():
            if len(feat_list) > 1:
                for i in range(min(10, len(feat_list))):
                    for j in range(i+1, min(10, len(feat_list))):
                        within_dists.append(np.sqrt(np.sum((feat_list[i] - feat_list[j])**2)))
        
        # Compute between-class distances
        between_dists = []
        classes = list(class_features.keys())
        for i in range(len(classes)):
            for j in range(i+1, len(classes)):
                if not class_features[classes[i]] or not class_features[classes[j]]:
                    continue
                
                for _ in range(min(10, len(class_features[classes[i]]) * len(class_features[classes[j]]))):
                    idx1 = np.random.choice(len(class_features[classes[i]]))
                    idx2 = np.random.choice(len(class_features[classes[j]]))
                    between_dists.append(np.sqrt(np.sum(
                        (class_features[classes[i]][idx1] - class_features[classes[j]][idx2])**2
                    )))
        
        if within_dists and between_dists:
            class_separation = np.mean(between_dists) / np.mean(within_dists)
            class_separation = min(max(class_separation, 0.1), 10.0)
            # Prevent NaN values
            if np.isnan(class_separation):
                class_separation = 1.0
        else:
            class_separation = 1.0
        
        # Simple feature redundancy estimation
        if features.shape[1] > 1:
            # Sample random feature dimensions for efficiency
            if features.shape[1] > 100:
                rand_dims = np.random.choice(features.shape[1], 100, replace=False)
                feature_subset = features[:, rand_dims]
            else:
                feature_subset = features
            
            # Compute correlation
            try:
                corr_matrix = np.abs(np.corrcoef(feature_subset.T))
                np.fill_diagonal(corr_matrix, 0)  # Remove self-correlations
                feature_redundancy = np.mean(corr_matrix)
                # Prevent NaN values
                if np.isnan(feature_redundancy):
                    feature_redundancy = 0.5
            except:
                feature_redundancy = 0.5
        else:
            feature_redundancy = 0.5
        
        return {
            "feature_diversity": float(feature_diversity),
            "feature_redundancy": float(feature_redundancy),
            "class_separation": float(class_separation)
        }
    
    def _evaluate_strategies(self, model, coreset_indices, strategies, epoch):
        """Evaluate strategy performances for meta-controller"""
        # Create scoring functions
        scoring_fns = {
            "uncertainty": self._uncertainty_scores,
            "diversity": self._diversity_scores, 
            "class_balance": self._class_balance_scores,
            "boundary": self._boundary_scores
        }
        
        # Compute scores for each strategy
        strategy_scores = {}
        for strategy in strategies:
            if strategy == "S_U":
                strategy_scores[strategy] = scoring_fns["uncertainty"](model)
            elif strategy == "S_D":
                strategy_scores[strategy] = scoring_fns["diversity"](model, coreset_indices)
            elif strategy == "S_C":
                strategy_scores[strategy] = scoring_fns["class_balance"](coreset_indices)
            elif strategy == "S_B":
                strategy_scores[strategy] = scoring_fns["boundary"](model)
            elif strategy == "S_G":  # Approximate gradient score using uncertainty for efficiency
                strategy_scores[strategy] = scoring_fns["uncertainty"](model)
            elif strategy == "S_F":  # Approximate forgetting with inverse of boundary
                boundary_scores = scoring_fns["boundary"](model)
                strategy_scores[strategy] = 1.0 - boundary_scores
        
        # Normalize scores
        for strategy in strategy_scores:
            scores = strategy_scores[strategy]
            mean = np.mean(scores)
            std = np.std(scores)
            if std > 1e-8:
                strategy_scores[strategy] = (scores - mean) / std
            else:
                strategy_scores[strategy] = np.zeros_like(scores)
        
        # Evaluate strategy performances by selecting small test batches
        base_acc_result = self._evaluate_model(model, self.val_loader)
        # Extract just the accuracy from the result (which might be a tuple)
        if isinstance(base_acc_result, tuple):
            base_acc = base_acc_result[0]  # First element is accuracy
        else:
            base_acc = base_acc_result
        
        performances = {}
        
        for strategy, scores in strategy_scores.items():
            # Select top samples
            n_eval = min(100, len(self.train_dataset) // 100)
            available = np.ones(len(self.train_dataset), dtype=bool)
            available[coreset_indices] = False
            
            if not np.any(available):
                performances[strategy] = 0.0
                continue
                
            available_indices = np.where(available)[0]
            available_scores = scores[available]
            
            if len(available_indices) < n_eval:
                top_indices = available_indices
            else:
                top_idx = np.argsort(available_scores)[-n_eval:]
                top_indices = available_indices[top_idx]
            
            # Evaluate these samples
            eval_subset = Subset(self.train_dataset, top_indices)
            eval_loader = DataLoader(eval_subset, batch_size=64, shuffle=False, num_workers=2)
            eval_acc_result = self._quick_eval(model, eval_loader)
            
            # Handle potential tuple result from _quick_eval
            if isinstance(eval_acc_result, tuple):
                eval_acc = eval_acc_result[0]
            else:
                eval_acc = eval_acc_result
            
            # Performance is relative to base accuracy
            performances[strategy] = eval_acc - base_acc
        
        return performances
    
    def _uncertainty_scores(self, model):
        """Compute uncertainty scores for all samples"""
        model.eval()
        uncertainties = np.zeros(len(self.train_dataset))
        loader = DataLoader(self.train_dataset, batch_size=128, shuffle=False, num_workers=2)
        
        with torch.no_grad():
            start_idx = 0
            for inputs, _ in loader:
                inputs = inputs.to(self.device)
                outputs = model(inputs)
                probs = torch.softmax(outputs, dim=1)
                entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=1)
                
                batch_size = inputs.size(0)
                uncertainties[start_idx:start_idx+batch_size] = entropy.cpu().numpy()
                start_idx += batch_size
        
        return uncertainties
    
    def _diversity_scores(self, model, coreset_indices):
        """Compute diversity scores based on feature space"""
        model.eval()
        
        # Get features for current coreset
        if len(coreset_indices) > 500:
            ref_indices = np.random.choice(coreset_indices, 500, replace=False)
        else:
            ref_indices = coreset_indices
            
        ref_subset = Subset(self.train_dataset, ref_indices)
        ref_loader = DataLoader(ref_subset, batch_size=64, shuffle=False, num_workers=2)
        
        ref_features = []
        try:
            with torch.no_grad():
                for inputs, _ in ref_loader:
                    inputs = inputs.to(self.device)
                    features = model.get_features(inputs)
                    
                    # Ensure features are flattened
                    if len(features.shape) > 2:
                        features = features.reshape(features.size(0), -1)
                    
                    ref_features.append(features.cpu().numpy())
        except Exception as e:
            print(f"Error extracting reference features: {e}")
            return np.random.rand(len(self.train_dataset))
        
        if ref_features:
            try:
                ref_features = np.vstack(ref_features)
            except ValueError as e:
                print(f"Error stacking reference features: {e}")
                return np.random.rand(len(self.train_dataset))
        else:
            return np.random.rand(len(self.train_dataset))
        
        # Compute diversity scores
        diversity_scores = np.zeros(len(self.train_dataset))
        loader = DataLoader(self.train_dataset, batch_size=64, shuffle=False, num_workers=2)
        
        try:
            with torch.no_grad():
                start_idx = 0
                for inputs, _ in loader:
                    inputs = inputs.to(self.device)
                    features = model.get_features(inputs)
                    
                    # Ensure features are flattened
                    if len(features.shape) > 2:
                        features = features.reshape(features.size(0), -1)
                    
                    batch_features = features.cpu().numpy()
                    
                    batch_size = inputs.size(0)
                    
                    # Compute minimum distance to reference set
                    for i in range(batch_size):
                        sample_features = batch_features[i].reshape(1, -1)
                        dists = np.sum((ref_features - sample_features) ** 2, axis=1)
                        min_dist = np.min(dists) if len(dists) > 0 else 1.0
                        diversity_scores[start_idx + i] = min_dist
                    
                    start_idx += batch_size
        except Exception as e:
            print(f"Error computing diversity scores: {e}")
            return np.random.rand(len(self.train_dataset))
        
        # Fix NaN values
        if np.any(np.isnan(diversity_scores)):
            print("Warning: NaN values in diversity scores, replacing with random values")
            nan_mask = np.isnan(diversity_scores)
            diversity_scores[nan_mask] = np.random.rand(np.sum(nan_mask))
            
        return diversity_scores
    
    def _class_balance_scores(self, coreset_indices):
        """Compute class balance scores"""
        class_counts = Counter([self.train_dataset[i][1] for i in coreset_indices])
        
        # Higher score for underrepresented classes
        total = len(coreset_indices)
        class_weights = {c: total / (count + 1) for c, count in class_counts.items()}
        
        # Normalize
        max_weight = max(class_weights.values()) if class_weights else 1.0
        class_weights = {c: w / max_weight for c, w in class_weights.items()}
        
        # Assign scores
        balance_scores = np.zeros(len(self.train_dataset))
        
        for i in range(len(self.train_dataset)):
            _, label = self.train_dataset[i]
            balance_scores[i] = class_weights.get(label, 1.0)
        
        return balance_scores
    
    def _boundary_scores(self, model):
        """Compute boundary proximity scores"""
        model.eval()
        boundary_scores = np.zeros(len(self.train_dataset))
        loader = DataLoader(self.train_dataset, batch_size=128, shuffle=False, num_workers=2)
        
        with torch.no_grad():
            start_idx = 0
            for inputs, _ in loader:
                inputs = inputs.to(self.device)
                outputs = model(inputs)
                probs = torch.softmax(outputs, dim=1)
                
                # Get top-2 probabilities
                top_values, _ = torch.topk(probs, k=2, dim=1)
                
                # Margin as 1 - (p_max - p_second)
                margin = top_values[:, 0] - top_values[:, 1]
                scores = 1.0 - margin
                
                batch_size = inputs.size(0)
                boundary_scores[start_idx:start_idx+batch_size] = scores.cpu().numpy()
                start_idx += batch_size
        
        return boundary_scores
    
    def _quick_eval(self, model, loader):
        """Quick model evaluation - returns just accuracy as a float"""
        model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, targets in loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = model(inputs)
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        
        return correct / total if total > 0 else 0.0
    
    def _evaluate_model(self, model, loader):
        """Evaluate model with full metrics"""
        model.eval()
        correct = 0
        total = 0
        loss_sum = 0
        criterion = nn.CrossEntropyLoss()
        
        with torch.no_grad():
            for inputs, targets in loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                
                loss_sum += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        
        accuracy = correct / total if total > 0 else 0.0
        avg_loss = loss_sum / len(loader) if len(loader) > 0 else 0.0
        
        return accuracy, avg_loss
    
    def _select_samples(self, model, budget, meta_controller, mode='hypernetwork', ablation_config=None):
        """
        Select samples using specified selection mode with improved budget enforcement
        
        Args:
            model: Model to use for selection
            budget: Exact number of samples to select
            meta_controller: Meta-controller for hypernetwork mode
            mode: Selection mode ('hypernetwork', 'random', 'uncertainty')
            ablation_config: Configuration for ablation studies
        
        Returns:
            Dict with selection results including indices and performance metrics
        """
        # Base strategies
        all_strategies = ["S_U", "S_D", "S_C", "S_B", "S_G", "S_F"]
        
        # Apply ablation if specified
        if ablation_config and mode == 'hypernetwork':
            strategies = [s for s in all_strategies if s not in ablation_config.get('disabled_strategies', [])]
            print(f"Ablation: Using strategies {strategies} (disabled: {ablation_config.get('disabled_strategies', [])})")
        else:
            strategies = all_strategies
        
        if mode == 'hypernetwork':
            # Update strategies in meta-controller if using ablation
            if ablation_config and meta_controller is not None:
                meta_controller.strategies = strategies
                meta_controller.budget_ratio = budget / len(self.train_dataset)
            
            # Initialize with random samples
            initial_ratio = 0.1  # 10% initial sampling
            initial_size = int(budget * initial_ratio)
            
            # Initial random sampling with class balance
            class_indices = {c: [] for c in range(self.num_classes)}
            for i in range(len(self.train_dataset)):
                _, label = self.train_dataset[i]
                class_indices[label].append(i)
            
            # Select equal per class
            init_per_class = initial_size // self.num_classes
            selected_indices = []
            
            for c in range(self.num_classes):
                if len(class_indices[c]) > 0:
                    indices = np.random.choice(
                        class_indices[c], 
                        min(init_per_class, len(class_indices[c])), 
                        replace=False
                    )
                    selected_indices.extend(indices)
            
            # Add random samples if needed
            if len(selected_indices) < initial_size:
                remaining = initial_size - len(selected_indices)
                avail_indices = list(set(range(len(self.train_dataset))) - set(selected_indices))
                if avail_indices:
                    extra = np.random.choice(avail_indices, min(remaining, len(avail_indices)), replace=False)
                    selected_indices.extend(extra)
            
            selected_indices = np.array(selected_indices)
            
            # Iteratively add samples using hypernetwork
            total_epochs = self.args.epochs
            model = self._create_model()
            optimizer = optim.Adam(model.parameters(), lr=0.001)
            
            # Initial training
            subset = Subset(self.train_dataset, selected_indices)
            train_loader = DataLoader(subset, batch_size=128, shuffle=True, num_workers=2)
            
            # Track metrics
            train_accs = []
            test_accs = []
            coreset_sizes = [len(selected_indices)]
            
            # Calculate number of selection rounds and samples per round
            selection_rounds = total_epochs // 5  # Selection every 5 epochs
            remaining_budget = budget - len(selected_indices)
            samples_per_round = remaining_budget // max(1, selection_rounds)
            
            # Adjust samples per round to ensure exact budget by the end
            samples_per_round_adjusted = []
            for i in range(selection_rounds):
                if i == selection_rounds - 1:
                    # In the last round, select exactly what's left to reach the budget
                    samples = budget - len(selected_indices) - sum(samples_per_round_adjusted)
                else:
                    samples = samples_per_round
                samples_per_round_adjusted.append(max(0, samples))
            
            print(f"Initial selection: {len(selected_indices)} samples")
            print(f"Selection schedule: {samples_per_round_adjusted}")
            
            for epoch in range(total_epochs):
                # Train epoch
                model.train()
                for inputs, targets in train_loader:
                    inputs, targets = inputs.to(self.device), targets.to(self.device)
                    optimizer.zero_grad()
                    outputs = model(inputs)
                    loss = nn.CrossEntropyLoss()(outputs, targets)
                    loss.backward()
                    optimizer.step()
                
                # Evaluate
                train_acc, train_loss = self._evaluate_model(model, train_loader)
                test_acc, test_loss = self._evaluate_model(model, self.test_loader)
                
                train_accs.append(train_acc)
                test_accs.append(test_acc)
                
                print(f"Epoch {epoch+1}/{total_epochs}, "
                      f"Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}, "
                      f"Coreset: {len(selected_indices)}/{budget}")
                
                # Add samples if budget allows and on specific epochs
                round_idx = epoch // 5
                if len(selected_indices) < budget and epoch % 5 == 0 and epoch > 0 and round_idx < len(samples_per_round_adjusted):
                    # Get number of samples to add in this round
                    n_samples = samples_per_round_adjusted[round_idx]
                    if n_samples <= 0:
                        continue
                    
                    print(f"Selection round {round_idx+1}: Adding {n_samples} samples")
                    
                    # Compute feature statistics
                    feature_stats = self._compute_feature_statistics(model, selected_indices)
                    
                    # Evaluate strategies
                    strategy_perfs = self._evaluate_strategies(model, selected_indices, strategies, epoch)
                    
                    # Prepare distribution info
                    class_dist = Counter([self.train_dataset[i][1] for i in selected_indices])
                    
                    # Update meta-controller state
                    state = meta_controller.update_state(
                        epoch=epoch,
                        accuracy=test_acc,
                        loss=test_loss,
                        class_distribution=class_dist,
                        dataset_size=len(selected_indices),
                        budget_size=budget,
                        strategy_performances=strategy_perfs,
                        feature_statistics=feature_stats
                    )
                    
                    # Get allocation for next batch
                    allocation = meta_controller.get_sample_allocation(n_samples)
                    
                    # Track how many samples we've successfully added with strategies
                    new_indices = []
                    strategy_selected_counts = {}
                    
                    # First pass: Select samples according to allocation
                    for strategy, count in allocation.items():
                        if count <= 0:
                            strategy_selected_counts[strategy] = 0
                            continue
                            
                        # Compute strategy scores
                        if strategy == "S_U":
                            scores = self._uncertainty_scores(model)
                        elif strategy == "S_D":
                            scores = self._diversity_scores(model, selected_indices)
                        elif strategy == "S_C":
                            scores = self._class_balance_scores(selected_indices)
                        elif strategy == "S_B":
                            scores = self._boundary_scores(model)
                        elif strategy == "S_G":
                            scores = self._uncertainty_scores(model)  # Approximate
                        elif strategy == "S_F":
                            boundary = self._boundary_scores(model)
                            scores = 1.0 - boundary  # Approximate
                        else:
                            strategy_selected_counts[strategy] = 0
                            continue
                        
                        # Select top scoring samples
                        available = np.ones(len(self.train_dataset), dtype=bool)
                        available[selected_indices] = False
                        available[new_indices] = False
                        
                        if not np.any(available):
                            strategy_selected_counts[strategy] = 0
                            continue
                            
                        avail_indices = np.where(available)[0]
                        avail_scores = scores[available]
                        
                        if len(avail_indices) <= count:
                            strat_indices = avail_indices
                        else:
                            top_idx = np.argsort(avail_scores)[-count:]
                            strat_indices = avail_indices[top_idx]
                        
                        new_indices.extend(strat_indices)
                        strategy_selected_counts[strategy] = len(strat_indices)
                    
                    # Calculate shortfall - how many more samples we need
                    shortfall = n_samples - len(new_indices)
                    
                    # Second pass: Reallocate any shortfall
                    if shortfall > 0:
                        print(f"Shortfall detected: {shortfall} samples. Reallocating...")
                        
                        # Combine scores from all strategies based on their weights
                        combined_scores = np.zeros(len(self.train_dataset))
                        weights = meta_controller.get_current_weights()
                        
                        for strategy, weight in weights.items():
                            if weight <= 0.001:  # Skip negligible weights
                                continue
                                
                            if strategy == "S_U":
                                scores = self._uncertainty_scores(model)
                            elif strategy == "S_D":
                                scores = self._diversity_scores(model, selected_indices)
                            elif strategy == "S_C":
                                scores = self._class_balance_scores(selected_indices)
                            elif strategy == "S_B":
                                scores = self._boundary_scores(model)
                            elif strategy == "S_G":
                                scores = self._uncertainty_scores(model)
                            elif strategy == "S_F":
                                boundary = self._boundary_scores(model)
                                scores = 1.0 - boundary
                            else:
                                continue
                            
                            # Normalize scores to [0, 1] range
                            score_min = np.min(scores)
                            score_max = np.max(scores)
                            if score_max > score_min:
                                normalized = (scores - score_min) / (score_max - score_min)
                            else:
                                normalized = np.zeros_like(scores)
                                
                            # Add weighted scores
                            combined_scores += weight * normalized
                        
                        # Select additional samples based on combined score
                        available = np.ones(len(self.train_dataset), dtype=bool)
                        available[selected_indices] = False
                        available[new_indices] = False
                        
                        if np.any(available):
                            avail_indices = np.where(available)[0]
                            avail_scores = combined_scores[available]
                            
                            if len(avail_indices) <= shortfall:
                                extra_indices = avail_indices
                            else:
                                top_idx = np.argsort(avail_scores)[-shortfall:]
                                extra_indices = avail_indices[top_idx]
                            
                            print(f"Adding {len(extra_indices)} extra samples to address shortfall")
                            new_indices.extend(extra_indices)
                    
                    # Check if we have the exact number of requested samples
                    if len(new_indices) != n_samples:
                        print(f"Warning: Selected {len(new_indices)} samples instead of {n_samples} requested")
                    
                    # Update selected indices
                    if new_indices:
                        selected_indices = np.concatenate([selected_indices, new_indices])
                        
                        # Update training loader
                        subset = Subset(self.train_dataset, selected_indices)
                        train_loader = DataLoader(subset, batch_size=128, shuffle=True, num_workers=2)
                        
                        # Update coreset size history
                        coreset_sizes.append(len(selected_indices))
                        
                        # Calculate diversity score for reward
                        diversity_score = np.mean(self._diversity_scores(model, new_indices))
                        
                        # Update hypernetwork
                        meta_controller.update_weights(diversity_score)
                        
                        print(f"Round {round_idx+1}: Added {len(new_indices)} samples, "
                              f"Strategy counts: {strategy_selected_counts}, "
                              f"Current coreset size: {len(selected_indices)}/{budget}")
            
            # Final adjustment to ensure exact budget
            final_diff = budget - len(selected_indices)
            if final_diff > 0:
                # Need to add more samples
                print(f"Final adjustment: Adding {final_diff} samples to reach exact budget")
                
                # Compute combined scores from all strategies
                combined_scores = np.zeros(len(self.train_dataset))
                weights = meta_controller.get_current_weights()
                
                for strategy, weight in weights.items():
                    if strategy == "S_U":
                        scores = self._uncertainty_scores(model)
                    elif strategy == "S_D":
                        scores = self._diversity_scores(model, selected_indices)
                    elif strategy == "S_C":
                        scores = self._class_balance_scores(selected_indices)
                    elif strategy == "S_B":
                        scores = self._boundary_scores(model)
                    elif strategy == "S_G":
                        scores = self._uncertainty_scores(model)
                    elif strategy == "S_F":
                        boundary = self._boundary_scores(model)
                        scores = 1.0 - boundary
                    else:
                        continue
                    
                    # Normalize scores
                    score_min = np.min(scores)
                    score_max = np.max(scores)
                    if score_max > score_min:
                        normalized = (scores - score_min) / (score_max - score_min)
                    else:
                        normalized = np.zeros_like(scores)
                    
                    combined_scores += weight * normalized
                
                # Select additional samples
                available = np.ones(len(self.train_dataset), dtype=bool)
                available[selected_indices] = False
                
                if np.any(available):
                    avail_indices = np.where(available)[0]
                    avail_scores = combined_scores[available]
                    
                    if len(avail_indices) <= final_diff:
                        extra = avail_indices
                    else:
                        top_idx = np.argsort(avail_scores)[-final_diff:]
                        extra = avail_indices[top_idx]
                    
                    selected_indices = np.append(selected_indices, extra)
                    coreset_sizes.append(len(selected_indices))
                    
                    # Update training data for final evaluation
                    subset = Subset(self.train_dataset, selected_indices)
                    train_loader = DataLoader(subset, batch_size=128, shuffle=True, num_workers=2)
            
            elif final_diff < 0:
                # Need to remove some samples
                print(f"Final adjustment: Removing {-final_diff} samples to reach exact budget")
                
                # Calculate importance scores
                importance_scores = {}
                weights = meta_controller.get_current_weights()
                
                for idx in selected_indices:
                    # Calculate weighted average of all strategy scores
                    score = 0.0
                    for strategy, weight in weights.items():
                        if strategy == "S_U":
                            s = self._uncertainty_scores(model)[idx]
                        elif strategy == "S_D":
                            # Use average feature distance to other selected samples
                            features = self._extract_features(model, idx)
                            other_indices = np.setdiff1d(selected_indices, [idx])
                            other_features = self._extract_features_batch(model, other_indices[:min(100, len(other_indices))])
                            dists = np.mean(np.sum((other_features - features) ** 2, axis=1))
                            s = dists
                        elif strategy == "S_C":
                            # Higher importance for underrepresented classes
                            label = self.train_dataset[idx][1]
                            class_counts = Counter([self.train_dataset[i][1] for i in selected_indices])
                            s = 1.0 / (class_counts.get(label, 1) + 1)
                        elif strategy == "S_B":
                            s = self._boundary_scores(model)[idx]
                        elif strategy == "S_G":
                            s = self._uncertainty_scores(model)[idx]
                        elif strategy == "S_F":
                            s = 1.0 - self._boundary_scores(model)[idx]
                        else:
                            s = 0.0
                        
                        score += weight * s
                    
                    importance_scores[idx] = score
                
                # Keep the most important samples
                sorted_indices = sorted(importance_scores.items(), key=lambda x: x[1], reverse=True)
                keep_indices = [idx for idx, _ in sorted_indices[:budget]]
                
                selected_indices = np.array(keep_indices)
                coreset_sizes.append(len(selected_indices))
                
                # Update training data for final evaluation
                subset = Subset(self.train_dataset, selected_indices)
                train_loader = DataLoader(subset, batch_size=128, shuffle=True, num_workers=2)
            
            # Final evaluation with the exact budget
            final_train_acc, final_train_loss = self._evaluate_model(model, train_loader)
            final_test_acc, final_test_loss = self._evaluate_model(model, self.test_loader)
            
            train_accs.append(final_train_acc)
            test_accs.append(final_test_acc)
            
            print(f"Final coreset size: {len(selected_indices)}/{budget}")
            print(f"Final metrics - Train: {final_train_acc:.4f}, Test: {final_test_acc:.4f}")
            
            # Return final indices and metrics
            results = {
                'indices': selected_indices,
                'train_accs': train_accs,
                'test_accs': test_accs,
                'coreset_sizes': coreset_sizes,
                'weight_history': meta_controller.get_weight_history(),
                'temp_history': meta_controller.get_temperature_history(),
                'explanations': meta_controller.get_all_explanations(),
                'ablation_config': ablation_config,
                'model_info': self.model_info
            }
            
            return results
        
        elif mode == 'random':
            # Simple random sampling to exact budget
            indices = np.random.choice(len(self.train_dataset), budget, replace=False)
            
            # Track metrics with this fixed set
            train_accs, test_accs = self._train_with_coreset(indices)
            
            results = {
                'indices': indices,
                'train_accs': train_accs,
                'test_accs': test_accs,
                'coreset_sizes': [budget] * len(train_accs),
                'model_info': self.model_info
            }
            
            return results
        
        elif mode == 'uncertainty':
            # Initialize with random samples
            initial_ratio = 0.1  # 10% initial sampling
            initial_size = int(budget * initial_ratio)
            indices = np.random.choice(len(self.train_dataset), initial_size, replace=False)
            
            # Train initial model
            subset = Subset(self.train_dataset, indices)
            train_loader = DataLoader(subset, batch_size=128, shuffle=True, num_workers=2)
            
            model = self._create_model()
            optimizer = optim.Adam(model.parameters(), lr=0.001)
            
            # Initial training
            for epoch in range(5):  # Just a few epochs to get started
                model.train()
                for inputs, targets in train_loader:
                    inputs, targets = inputs.to(self.device), targets.to(self.device)
                    optimizer.zero_grad()
                    outputs = model(inputs)
                    loss = nn.CrossEntropyLoss()(outputs, targets)
                    loss.backward()
                    optimizer.step()
            
            # Select remaining samples by uncertainty to reach exact budget
            remaining = budget - initial_size
            if remaining > 0:
                uncertainties = self._uncertainty_scores(model)
                
                # Mask already selected indices
                available = np.ones(len(self.train_dataset), dtype=bool)
                available[indices] = False
                
                if np.any(available):
                    avail_indices = np.where(available)[0]
                    avail_scores = uncertainties[available]
                    
                    # Select top uncertain samples
                    top_idx = np.argsort(avail_scores)[-remaining:]
                    additional = avail_indices[top_idx]
                    
                    # Update indices
                    indices = np.concatenate([indices, additional])
            
            # Train with final coreset
            train_accs, test_accs = self._train_with_coreset(indices)
            
            results = {
                'indices': indices,
                'train_accs': train_accs,
                'test_accs': test_accs,
                'coreset_sizes': [budget] * len(train_accs),
                'model_info': self.model_info
            }
            
            return results
        
        else:
            raise ValueError(f"Unknown selection mode: {mode}")

    def _extract_features(self, model, idx):
        """
        Extract features for a single sample
        
        Args:
            model: Model to use for feature extraction
            idx: Index of the sample to extract features from
            
        Returns:
            Numpy array of features
        """
        model.eval()
        
        # Get sample data
        inputs, _ = self.train_dataset[idx]
        inputs = inputs.unsqueeze(0).to(self.device)  # Add batch dimension
        
        # Extract features
        with torch.no_grad():
            if isinstance(model, TimmModelWrapper):
                features = model.get_features(inputs)
            else:
                # Try to use model's feature extraction capabilities
                try:
                    # Get intermediate representation by running forward pass but extracting
                    # features from the penultimate layer
                    features = model.feature_extractor(inputs)
                except Exception as e:
                    print(f"Error extracting features: {e}")
                    return np.zeros(512)  # Default fallback feature dimension
            
            # Ensure features are flattened
            if len(features.shape) > 2:
                features = features.view(features.size(0), -1)
            
            features = features.cpu().numpy()
        
        return features[0]  # Return as 1D array

    def _extract_features_batch(self, model, indices, batch_size=64):
        """
        Extract features for a batch of samples
        
        Args:
            model: Model to use for feature extraction
            indices: Indices of samples to extract features from
            batch_size: Batch size for processing
            
        Returns:
            Numpy array of features for all samples
        """
        model.eval()
        
        # Create subset and loader
        subset = Subset(self.train_dataset, indices)
        loader = DataLoader(subset, batch_size=batch_size, shuffle=False, num_workers=2)
        
        # Extract features
        all_features = []
        
        with torch.no_grad():
            for inputs, _ in loader:
                inputs = inputs.to(self.device)
                
                if isinstance(model, TimmModelWrapper):
                    features = model.get_features(inputs)
                else:
                    # Try to use model's feature extraction capabilities
                    try:
                        # Get intermediate representation
                        features = model.feature_extractor(inputs)
                    except Exception as e:
                        print(f"Error extracting features: {e}")
                        return np.zeros((len(indices), 512))  # Default fallback
                
                # Ensure features are flattened
                if len(features.shape) > 2:
                    features = features.view(features.size(0), -1)
                
                all_features.append(features.cpu().numpy())
        
        if not all_features:
            return np.zeros((len(indices), 512))  # Return empty features
        
        return np.vstack(all_features)
    
    def _train_with_coreset(self, indices, epochs=None):
        """Train a model using the selected coreset"""
        # Set epochs
        if epochs is None:
            epochs = self.args.epochs
            
        # Create dataset and loader
        subset = Subset(self.train_dataset, indices)
        train_loader = DataLoader(subset, batch_size=self.args.batch_size, shuffle=True, num_workers=2)
        
        # Initialize model
        model = self._create_model()
        optimizer = optim.Adam(model.parameters(), lr=self.args.lr)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        
        # Training loop
        train_accs = []
        test_accs = []
        
        for epoch in range(epochs):
            # Train
            model.train()
            for inputs, targets in train_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = nn.CrossEntropyLoss()(outputs, targets)
                loss.backward()
                optimizer.step()
            
            # Evaluate
            train_acc, _ = self._evaluate_model(model, train_loader)
            test_acc, _ = self._evaluate_model(model, self.test_loader)
            
            train_accs.append(train_acc)
            test_accs.append(test_acc)
            
            # Update scheduler
            scheduler.step()
            
            # Print progress
            if (epoch + 1) % 5 == 0:
                print(f"Epoch {epoch+1}/{epochs}, "
                      f"Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}")
        
        return train_accs, test_accs
    
    def run_experiment(self):
        """Run the full experiment with different budget ratios and methods"""
        # Available selection methods
        methods = {
            'hypernetwork': f'MODE-{self.args.model}',
            'random': 'Random Sampling',
            'uncertainty': 'Uncertainty Sampling'
        }
        
        # Budget ratios to test
        budget_ratios = [0.1, 0.2, 0.3, 0.5, 0.7]
        
        # Results container
        results = {}
        
        for method_key, method_name in methods.items():
            method_results = {}
            
            for ratio in budget_ratios:
                print(f"\n\n=== Running {method_name} with {ratio*100:.0f}% budget on {self.args.dataset} ===\n")
                
                # Calculate budget
                budget = int(len(self.train_dataset) * ratio)
                
                # Create meta-controller if needed
                meta_controller = None
                if method_key == 'hypernetwork':
                    strategies = ["S_U", "S_D", "S_C", "S_B", "S_G", "S_F"]
                    
                    # Create enhanced controller with the budget ratio
                    meta_controller = EnhancedMetaController(
                        strategies=strategies,
                        total_epochs=self.args.epochs,
                        budget_ratio=ratio,  # Pass the budget ratio explicitly
                        device=self.device
                    )
                    
                    print(f"Initialized meta-controller with budget ratio {ratio}")
                    print(f"Initial weights: {meta_controller.get_current_weights()}")
                    try:
                        if meta_controller.temperature_history and len(meta_controller.temperature_history) > 0:
                            print(f"Initial temperature: {meta_controller.temperature_history[0]}")
                        else:
                            print("Initial temperature not yet set")
                    except (AttributeError, IndexError) as e:
                        print("Initial temperature not available")
                
                # Run selection and training with the appropriate method
                experiment_results = self._select_samples(
                    model=self._create_model(),
                    budget=budget,
                    meta_controller=meta_controller,
                    mode=method_key
                )
                
                # Save results
                method_results[ratio] = experiment_results
                
                # Create and save plots
                self._save_experiment_plots(
                    method_name=method_name,
                    budget_ratio=ratio,
                    results=experiment_results
                )
                
                # For hypernetwork, save additional diagnostics
                if method_key == 'hypernetwork' and meta_controller is not None:
                    try:
                        # Save reward and loss trends
                        plt.figure(figsize=(10, 6))
                        plt.plot(meta_controller.trainer.train_losses)
                        plt.title(f"{method_name} Training Loss - {ratio*100:.0f}% Budget")
                        plt.xlabel('Update')
                        plt.ylabel('Loss')
                        plt.yscale('log')
                        plt.grid(True)
                        plt.savefig(os.path.join(self.results_dir, 
                            f"{method_name.lower().replace(' ', '_')}_{int(ratio*100)}/training_loss.png"))
                        plt.close()
                        
                        # Save hypernetwork model for later analysis
                        os.makedirs(os.path.join(self.results_dir, f"models"), exist_ok=True)
                        meta_controller.save_model(os.path.join(self.results_dir, 
                            f"models/{method_name.lower().replace(' ', '_')}_{int(ratio*100)}.pth"))
                    except Exception as e:
                        print(f"Error saving diagnostics: {e}")
            
            results[method_key] = method_results
        
        # Save all results
        self._save_results(results)
        
        # Generate summary plots
        self._generate_summary_plots(results)
        
        # Generate comparative analysis
        self._generate_comparative_analysis(results)
        
        return results
    
    def run_ablation_study(self):
        """Run ablation study to test the effect of removing components"""
        print(f"\n\n=== Running Ablation Study on {self.args.dataset} with {self.args.model} ===\n")
        
        # Define ablation configurations
        ablation_configs = [
            # Baseline (all strategies)
            {
                'name': 'baseline',
                'disabled_strategies': []
            },
            # Remove uncertainty
            {
                'name': 'no_uncertainty',
                'disabled_strategies': ['S_U']
            },
            # Remove diversity
            {
                'name': 'no_diversity',
                'disabled_strategies': ['S_D']
            },
            # Remove class balance
            {
                'name': 'no_class_balance',
                'disabled_strategies': ['S_C']
            },
            # Remove boundary
            {
                'name': 'no_boundary',
                'disabled_strategies': ['S_B']
            },
            # Remove gradient (approximated)
            {
                'name': 'no_gradient',
                'disabled_strategies': ['S_G']
            },
            # Remove forgetting (approximated)
            {
                'name': 'no_forgetting',
                'disabled_strategies': ['S_F']
            },
            # Only uncertainty + diversity
            {
                'name': 'only_uncertainty_diversity',
                'disabled_strategies': ['S_C', 'S_B', 'S_G', 'S_F']
            },
            # Only class balance + boundary
            {
                'name': 'only_balance_boundary',
                'disabled_strategies': ['S_U', 'S_D', 'S_G', 'S_F']
            }
        ]
        
        # Test budget ratio (30% for ablation)
        budget_ratio = 0.3
        budget = int(len(self.train_dataset) * budget_ratio)
        
        # Results container
        ablation_results = {}
        
        for config in ablation_configs:
            print(f"\n--- Testing {config['name']} ---")
            print(f"Disabled strategies: {config['disabled_strategies']}")
            
            # Create meta-controller
            all_strategies = ["S_U", "S_D", "S_C", "S_B", "S_G", "S_F"]
            strategies = [s for s in all_strategies if s not in config['disabled_strategies']]
            
            meta_controller = EnhancedMetaController(
                strategies=strategies,
                total_epochs=self.args.epochs,
                budget_ratio=budget_ratio,
                device=self.device
            )
            
            # Run experiment
            results = self._select_samples(
                model=self._create_model(),
                budget=budget,
                meta_controller=meta_controller,
                mode='hypernetwork',
                ablation_config=config
            )
            
            ablation_results[config['name']] = results
            
            # Save individual results
            self._save_ablation_results(config['name'], results, budget_ratio)
        
        # Generate ablation analysis plots
        self._generate_ablation_analysis(ablation_results, budget_ratio)
        
        return ablation_results
    
    def _save_ablation_results(self, config_name, results, budget_ratio):
        """Save results for individual ablation configuration"""
        ablation_dir = os.path.join(self.results_dir, f"ablation_{config_name}_{int(budget_ratio*100)}")
        os.makedirs(ablation_dir, exist_ok=True)
        
        # Save metrics plot
        plt.figure(figsize=(10, 6))
        plt.plot(results['train_accs'], label='Train Accuracy')
        plt.plot(results['test_accs'], label='Test Accuracy')
        plt.title(f"Ablation: {config_name} - {budget_ratio*100:.0f}% Budget")
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(ablation_dir, 'accuracy.png'))
        plt.close()
        
        # Save strategy weights plot
        if 'weight_history' in results:
            plt.figure(figsize=(10, 6))
            weight_history = results['weight_history']
            
            for strategy in weight_history[0].keys():
                weights = [w.get(strategy, 0.0) for w in weight_history]
                plt.plot(weights, label=strategy)
            
            plt.title(f"Ablation: {config_name} - Strategy Weights")
            plt.xlabel('Update Step')
            plt.ylabel('Weight')
            plt.legend()
            plt.grid(True)
            plt.savefig(os.path.join(ablation_dir, 'strategy_weights.png'))
            plt.close()
        
        # Save numerical results
        with open(os.path.join(ablation_dir, 'results.json'), 'w') as f:
            json_results = {k: v.tolist() if isinstance(v, np.ndarray) else v 
                           for k, v in results.items() if k != 'ablation_config'}
            json_results['ablation_config'] = results.get('ablation_config', {})
            json_results['model_info'] = results.get('model_info', {})
            json.dump(json_results, f, indent=4)
    
    def _generate_ablation_analysis(self, ablation_results, budget_ratio):
        """Generate comprehensive ablation analysis"""
        ablation_dir = os.path.join(self.results_dir, f"ablation_analysis_{int(budget_ratio*100)}")
        os.makedirs(ablation_dir, exist_ok=True)
        
        # Compare final accuracies
        plt.figure(figsize=(12, 8))
        
        config_names = []
        final_accs = []
        
        for config_name, results in ablation_results.items():
            config_names.append(config_name)
            final_accs.append(results['test_accs'][-1])
        
        # Sort by accuracy
        sorted_pairs = sorted(zip(config_names, final_accs), key=lambda x: x[1], reverse=True)
        config_names, final_accs = zip(*sorted_pairs)
        
        bars = plt.bar(range(len(config_names)), final_accs)
        plt.xticks(range(len(config_names)), config_names, rotation=45, ha='right')
        plt.ylabel('Final Test Accuracy')
        plt.title(f'Ablation Study Results - {budget_ratio*100:.0f}% Budget on {self.args.dataset} with {self.args.model}')
        plt.grid(True, axis='y')
        
        # Add value labels on bars
        for bar, acc in zip(bars, final_accs):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001, 
                    f'{acc:.4f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(os.path.join(ablation_dir, 'comparison.png'))
        plt.close()
        
        # Generate comparison table
        comparison_data = {
            'Configuration': config_names,
            'Final Accuracy': [f"{acc:.4f}" for acc in final_accs],
            'Difference from Baseline': [
                f"{(acc - ablation_results['baseline']['test_accs'][-1])*100:.2f}%" 
                for acc in final_accs
            ]
        }
        
        df = pd.DataFrame(comparison_data)
        df.to_csv(os.path.join(ablation_dir, 'ablation_comparison.csv'), index=False)
        
        # Generate detailed analysis report
        with open(os.path.join(ablation_dir, 'ablation_report.txt'), 'w') as f:
            f.write(f"Ablation Study Report - {self.args.dataset} with {self.args.model}\n")
            f.write(f"Budget Ratio: {budget_ratio*100:.0f}%\n")
            f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            baseline_acc = ablation_results['baseline']['test_accs'][-1]
            f.write(f"Baseline Accuracy (all strategies): {baseline_acc:.4f}\n\n")
            
            f.write("Impact of Removing Individual Strategies:\n")
            f.write("-" * 40 + "\n")
            
            # Single strategy removals
            single_removals = {
                'no_uncertainty': 'Uncertainty (S_U)',
                'no_diversity': 'Diversity (S_D)',
                'no_class_balance': 'Class Balance (S_C)',
                'no_boundary': 'Boundary (S_B)',
                'no_gradient': 'Gradient (S_G)',
                'no_forgetting': 'Forgetting (S_F)'
            }
            
            for config_name, strategy_name in single_removals.items():
                if config_name in ablation_results:
                    acc = ablation_results[config_name]['test_accs'][-1]
                    diff = acc - baseline_acc
                    pct_diff = diff * 100
                    f.write(f"{strategy_name}: {acc:.4f} ({pct_diff:+.2f}%)\n")
            
            f.write("\nCombination Experiments:\n")
            f.write("-" * 40 + "\n")
            
            # Combination experiments
            combinations = {
                'only_uncertainty_diversity': 'Only Uncertainty + Diversity',
                'only_balance_boundary': 'Only Class Balance + Boundary'
            }
            
            for config_name, desc in combinations.items():
                if config_name in ablation_results:
                    acc = ablation_results[config_name]['test_accs'][-1]
                    diff = acc - baseline_acc
                    pct_diff = diff * 100
                    f.write(f"{desc}: {acc:.4f} ({pct_diff:+.2f}%)\n")
            
            f.write("\nKey Findings:\n")
            f.write("-" * 40 + "\n")
            
            # Find most important strategies
            max_impact = 0
            most_important = ""
            for config_name, strategy_name in single_removals.items():
                if config_name in ablation_results:
                    acc = ablation_results[config_name]['test_accs'][-1]
                    impact = baseline_acc - acc
                    if impact > max_impact:
                        max_impact = impact
                        most_important = strategy_name
            
            f.write(f"1. {most_important} has the highest impact (removal causes {max_impact*100:.2f}% drop)\n")
            
            # Find least important strategies
            min_impact = float('inf')
            least_important = ""
            for config_name, strategy_name in single_removals.items():
                if config_name in ablation_results:
                    acc = ablation_results[config_name]['test_accs'][-1]
                    impact = baseline_acc - acc
                    if impact < min_impact:
                        min_impact = impact
                        least_important = strategy_name
            
            f.write(f"2. {least_important} has the lowest impact (removal causes {min_impact*100:.2f}% drop)\n")
            
            # Check if combinations perform well
            best_combo = ""
            best_combo_acc = 0
            for config_name, desc in combinations.items():
                if config_name in ablation_results:
                    acc = ablation_results[config_name]['test_accs'][-1]
                    if acc > best_combo_acc:
                        best_combo_acc = acc
                        best_combo = desc
            
            diff_from_baseline = best_combo_acc - baseline_acc
            f.write(f"3. {best_combo} performs best among combinations (difference: {diff_from_baseline*100:+.2f}%)\n")
    
    def _save_experiment_plots(self, method_name, budget_ratio, results):
        """Save plots for individual experiment"""
        plot_dir = os.path.join(self.results_dir, f"{method_name.lower().replace(' ', '_')}_{int(budget_ratio*100)}")
        os.makedirs(plot_dir, exist_ok=True)
        
        # Accuracy plot
        plt.figure(figsize=(10, 6))
        plt.plot(results['train_accs'], label='Train Accuracy')
        plt.plot(results['test_accs'], label='Test Accuracy')
        plt.title(f"{method_name} Accuracy - {budget_ratio*100:.0f}% Budget on {self.args.dataset}")
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(plot_dir, 'accuracy.png'))
        plt.close()
        
        # Coreset growth
        plt.figure(figsize=(10, 6))
        plt.plot(results['coreset_sizes'])
        plt.title(f"{method_name} Coreset Growth - {budget_ratio*100:.0f}% Budget on {self.args.dataset}")
        plt.xlabel('Update Step')
        plt.ylabel('Coreset Size')
        plt.grid(True)
        plt.savefig(os.path.join(plot_dir, 'coreset_growth.png'))
        plt.close()
        
        # Strategy weights for hypernetwork
        if 'weight_history' in results:
            plt.figure(figsize=(10, 6))
            weight_history = results['weight_history']
            
            for strategy in weight_history[0].keys():
                weights = [w.get(strategy, 0.0) for w in weight_history]
                plt.plot(weights, label=strategy)
            
            plt.title(f"{method_name} Strategy Weights - {budget_ratio*100:.0f}% Budget on {self.args.dataset}")
            plt.xlabel('Update Step')
            plt.ylabel('Weight')
            plt.legend()
            plt.grid(True)
            plt.savefig(os.path.join(plot_dir, 'strategy_weights.png'))
            plt.close()
        
        # Temperature for hypernetwork
        if 'temp_history' in results:
            plt.figure(figsize=(10, 6))
            plt.plot(results['temp_history'])
            plt.title(f"{method_name} Temperature (Exploration) - {budget_ratio*100:.0f}% Budget on {self.args.dataset}")
            plt.xlabel('Update Step')
            plt.ylabel('Temperature')
            plt.grid(True)
            plt.savefig(os.path.join(plot_dir, 'temperature.png'))
            plt.close()
        
        # Save numerical results
        with open(os.path.join(plot_dir, 'results.json'), 'w') as f:
            # Convert numpy arrays to lists for JSON serialization
            json_results = {k: v.tolist() if isinstance(v, np.ndarray) else v 
                           for k, v in results.items()}
            json.dump(json_results, f, indent=4)
    
    def _save_results(self, results):
        """Save all experiment results"""
        # Save to JSON
        json_results = {}
        
        for method, method_results in results.items():
            json_results[method] = {}
            for ratio, experiment in method_results.items():
                json_results[method][str(ratio)] = {
                    'test_accs': experiment['test_accs'],
                    'final_accuracy': experiment['test_accs'][-1],
                    'coreset_size': len(experiment['indices']),
                    'model_info': experiment.get('model_info', {})
                }
        
        with open(os.path.join(self.results_dir, 'all_results.json'), 'w') as f:
            json.dump(json_results, f, indent=4)
        
        # Save to CSV for easy analysis
        rows = []
        for method, method_results in results.items():
            for ratio, experiment in method_results.items():
                row = {
                    'method': method,
                    'budget_ratio': ratio,
                    'final_accuracy': experiment['test_accs'][-1],
                    'coreset_size': len(experiment['indices']),
                    'max_accuracy': max(experiment['test_accs']),
                    'dataset': self.args.dataset,
                    'model': self.args.model
                }
                rows.append(row)
        
        df = pd.DataFrame(rows)
        df.to_csv(os.path.join(self.results_dir, 'results_summary.csv'), index=False)
    
    def _generate_summary_plots(self, results):
        """Generate summary plots comparing all methods and budgets"""
        # Accuracy vs Budget plot
        plt.figure(figsize=(12, 8))
        
        for method, method_results in results.items():
            method_name = {
                'hypernetwork': f'SPARROW-{self.args.model}',
                'random': 'Random Sampling',
                'uncertainty': 'Uncertainty Sampling'
            }.get(method, method)
            
            ratios = []
            final_accs = []
            
            for ratio, experiment in method_results.items():
                ratios.append(ratio)
                final_accs.append(experiment['test_accs'][-1])
            
            plt.plot(ratios, final_accs, 'o-', label=method_name)
        
        plt.title(f'Final Accuracy vs Budget Ratio on {self.args.dataset} with {self.args.model}')
        plt.xlabel('Budget Ratio')
        plt.ylabel('Final Test Accuracy')
        plt.grid(True)
        plt.legend()
        plt.savefig(os.path.join(self.results_dir, 'accuracy_vs_budget.png'))
        plt.close()
        
        # Create learning curve comparison for each budget
        for ratio in [0.1, 0.3, 0.5]:
            plt.figure(figsize=(12, 8))
            
            for method, method_results in results.items():
                if ratio not in method_results:
                    continue
                    
                method_name = {
                    'hypernetwork': f'SPARROW-{self.args.model}',
                    'random': 'Random Sampling',
                    'uncertainty': 'Uncertainty Sampling'
                }.get(method, method)
                
                plt.plot(method_results[ratio]['test_accs'], label=method_name)
            
            plt.title(f'Learning Curves - {ratio*100:.0f}% Budget on {self.args.dataset} with {self.args.model}')
            plt.xlabel('Epoch')
            plt.ylabel('Test Accuracy')
            plt.grid(True)
            plt.legend()
            plt.savefig(os.path.join(self.results_dir, f'learning_curves_{int(ratio*100)}.png'))
            plt.close()
    
    def _generate_comparative_analysis(self, results):
        """Generate comparative analysis of different methods"""
        # Prepare data for direct comparison
        final_accuracies = {method: [] for method in results.keys()}
        ratios = []
        
        for ratio in [0.1, 0.2, 0.3, 0.5, 0.7]:
            ratios.append(ratio)
            
            for method in results.keys():
                if ratio in results[method]:
                    final_acc = results[method][ratio]['test_accs'][-1]
                    final_accuracies[method].append(final_acc)
                else:
                    final_accuracies[method].append(None)
        
        # Create comparison table
        comparison_data = {
            'Budget Ratio': [f"{int(r*100)}%" for r in ratios]
        }
        
        method_names = {
            'hypernetwork': f'MODE-{self.args.model}',
            'random': 'Random Sampling',
            'uncertainty': 'Uncertainty Sampling'
        }
        
        for method, accs in final_accuracies.items():
            comparison_data[method_names.get(method, method)] = [
                f"{acc*100:.2f}%" if acc is not None else "N/A" for acc in accs
            ]
        
        # Write comparison to CSV
        df = pd.DataFrame(comparison_data)
        df.to_csv(os.path.join(self.results_dir, 'method_comparison.csv'), index=False)
        
        # Calculate improvement of hypernetwork over uncertainty sampling
        if 'hypernetwork' in final_accuracies and 'uncertainty' in final_accuracies:
            hypernetwork_accs = final_accuracies['hypernetwork']
            uncertainty_accs = final_accuracies['uncertainty']
            
            improvements = []
            for h_acc, u_acc in zip(hypernetwork_accs, uncertainty_accs):
                if h_acc is not None and u_acc is not None:
                    rel_improvement = (h_acc - u_acc) / u_acc * 100  # percentage improvement
                    improvements.append(rel_improvement)
                else:
                    improvements.append(None)
            
            # Write improvement analysis to text file
            with open(os.path.join(self.results_dir, 'hypernetwork_improvement.txt'), 'w') as f:
                f.write(f"Relative improvement of MODE-{self.args.model} over Uncertainty Sampling on {self.args.dataset}:\n\n")
                for i, ratio in enumerate(ratios):
                    if improvements[i] is not None:
                        f.write(f"Budget {int(ratio*100)}%: {improvements[i]:.2f}% relative improvement\n")
                    else:
                        f.write(f"Budget {int(ratio*100)}%: N/A\n")
                
                # Add overall analysis
                valid_improvements = [imp for imp in improvements if imp is not None]
                if valid_improvements:
                    avg_improvement = sum(valid_improvements) / len(valid_improvements)
                    f.write(f"\nAverage improvement: {avg_improvement:.2f}%\n")
                    
                    if avg_improvement > 0:
                        f.write(f"\nThe MODE with {self.args.model} successfully outperformed uncertainty sampling.\n")
                    else:
                        f.write(f"\nThe MODE with {self.args.model} did not outperform uncertainty sampling. Further improvements may be needed.\n")

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='MODE Experiments with timm models')
    
    # Dataset parameters
    parser.add_argument('--dataset', type=str, default='cifar10', 
                        choices=['cifar10', 'cifar100', 'svhn'],
                        help='Dataset to use')
    parser.add_argument('--data-dir', type=str, default='./data',
                        help='Path to dataset')
    
    # Model parameters
    parser.add_argument('--model', type=str, default='resnet18',
                        help='timm model name (e.g., resnet18, efficientnet_b0, vit_base_patch16_224)')
    
    # Experiment parameters
    parser.add_argument('--output-dir', type=str, default='./results',
                        help='Output directory for results')
    parser.add_argument('--epochs', type=int, default=30,
                        help='Number of epochs for training')
    parser.add_argument('--batch-size', type=int, default=128,
                        help='Batch size for training')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--workers', type=int, default=2,
                        help='Number of worker threads for data loading')
    
    # Device parameters
    parser.add_argument('--device', type=str, default=None,
                        help='Device to use (cuda, mps, cpu)')
    
    # Experiment type
    parser.add_argument('--mode', type=str, default='full', 
                        choices=['full', 'ablation', 'both'],
                        help='Type of experiment to run')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Create experiment runner
    experiment = MetaExperimentWithTimm(args)
    
    # Run experiments based on mode
    if args.mode == 'full':
        results = experiment.run_experiment()
    elif args.mode == 'ablation':
        results = experiment.run_ablation_study()
    elif args.mode == 'both':
        # Run full experiments
        results = experiment.run_experiment()
        # Run ablation study
        ablation_results = experiment.run_ablation_study()
        # Combine results
        results['ablation'] = ablation_results
    
    print("Experiments completed!")
    print(f"Results saved to {experiment.results_dir}")

if __name__ == "__main__":
    main()