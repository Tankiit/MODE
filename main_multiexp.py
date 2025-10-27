#!/usr/bin/env python3
"""
Selective Multi-Classifier Coreset Selection with Multi-Ratio/Multi-Model Experiments
======================================================================================

Extends the selective training with:
1. Multiple scoring functions (uncertainty, diversity, boundary, loss)
2. Confidence estimation per scorer
3. Learned gating mechanism
4. Selective rejection based on confidence
5. PAC-style risk-coverage guarantees
6. Multi-ratio experimentation (test different selection ratios)
7. Multi-model experimentation (test different model architectures)
8. Comprehensive result aggregation and visualization

For CVPR 2026: "Selective Multi-Classifier Coreset Selection"
"""

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
from collections import defaultdict
from scipy import optimize
from scipy.special import comb
import json
import os
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ==============================================================================
# CONFIGURATION (Extended for Multi-Experiment)
# ==============================================================================

class SelectiveMultiClassifierConfig:
    """Configuration for selective multi-classifier coreset selection"""

    # Model settings
    MODEL_NAME = "resnet34"
    REFERENCE_MODEL_NAME = "resnet18"  # For excess loss scorer
    PRETRAINED = True

    # Data settings
    DATASET = "CIFAR10"
    DATA_PATH = "./data"
    IMAGE_SIZE = 32
    NUM_CLASSES = 10

    # Multi-Classifier Settings
    NUM_SCORERS = 4  # Uncertainty, Diversity, Boundary, Excess Loss
    USE_GATING = True  # Learn adaptive gating weights
    USE_CONFIDENCE_REJECTION = True  # Reject low-confidence scorers
    CONFIDENCE_THRESHOLD_INIT = 0.5  # Initial rejection threshold

    # Gating network settings
    GATE_STATE_DIM = 6  # [acc, loss, grad_norm, epoch_progress, coverage, entropy]
    GATE_HIDDEN_DIM = 64
    TEMPERATURE_INIT = 1.0

    # Selection settings
    SELECTION_RATIO = 0.5  # Default, will be overridden in experiments
    SELECTION_FREQUENCY = 5  # Re-select every N epochs
    ADAPTIVE_BUDGET = False  # Disable for controlled experiments

    # Risk-Coverage Guarantee Settings
    DELTA = 0.001  # Confidence parameter for PAC bounds
    TARGET_RISK = 0.05  # Target risk level (5% error)
    COMPUTE_RISK_BOUNDS = True  # Compute theoretical guarantees

    # Training settings
    OUTPUT_DIR = "./selective_mc_output"
    NUM_EPOCHS = 200
    BATCH_SIZE = 128
    LEARNING_RATE = 0.1
    MOMENTUM = 0.9
    WEIGHT_DECAY = 1e-4
    LR_SCHEDULE = "cosine"

    # Multi-Experiment Settings (NEW)
    ENABLE_MULTI_RATIO = True  # Test multiple ratios
    ENABLE_MULTI_MODEL = True  # Test multiple models

    RATIOS_TO_TEST = [0.1, 0.2, 0.3, 0.5, 0.7, 0.9]  # Selection ratios
    MODELS_TO_TEST = [
        ("resnet18", "resnet18"),  # (main_model, reference_model)
        ("resnet34", "resnet18"),
        ("resnet50", "resnet34"),
        ("efficientnet_b0", "mobilenetv3_small_100"),
        ("mobilenetv3_large_100", "mobilenetv3_small_100"),
    ]

    # For single model experiments with multiple ratios
    SINGLE_MODEL_MULTI_RATIO = True  # Test one model with multiple ratios first

    # Experiment tracking
    EXPERIMENT_NAME = f"multi_exp_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    RESULTS_DIR = "./experiment_results"
    SAVE_CHECKPOINTS = True
    CHECKPOINT_DIR = "./checkpoints"

    # Logging and analysis
    LOGGING_STEPS = 100
    EVAL_FREQUENCY = 5
    SAVE_FREQUENCY = 20
    TRACK_SAMPLE_WASTE = True
    TRACK_CONFIDENCE_EVOLUTION = True

    # Visualization settings
    GENERATE_PLOTS = True
    PLOT_FORMAT = "png"  # png, pdf, svg

    # Device
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

config = SelectiveMultiClassifierConfig()

# ==============================================================================
# CONFIDENCE ESTIMATORS
# ==============================================================================

class ConfidenceEstimator:
    """Base class for confidence estimation"""

    def estimate(self, x: torch.Tensor, y: torch.Tensor, model: nn.Module) -> torch.Tensor:
        """Estimate confidence for a batch of samples"""
        raise NotImplementedError

class EntropyConfidence(ConfidenceEstimator):
    """Confidence based on prediction entropy (for uncertainty scorer)"""

    def estimate(self, x: torch.Tensor, y: torch.Tensor, model: nn.Module) -> torch.Tensor:
        with torch.no_grad():
            logits = model(x)
            probs = F.softmax(logits, dim=-1)
            entropy = -(probs * torch.log(probs + 1e-10)).sum(dim=-1)
            max_entropy = np.log(probs.shape[-1])
            normalized_entropy = entropy / max_entropy

            # U-shaped confidence: confident when entropy very high or very low
            confidence = 1.0 - 4.0 * torch.abs(normalized_entropy - 0.5)
            confidence = torch.clamp(confidence, 0, 1)

        return confidence

class MarginConfidence(ConfidenceEstimator):
    """Confidence based on margin between top-2 predictions"""

    def estimate(self, x: torch.Tensor, y: torch.Tensor, model: nn.Module) -> torch.Tensor:
        with torch.no_grad():
            logits = model(x)
            probs = F.softmax(logits, dim=-1)
            top2_probs = torch.topk(probs, k=2, dim=-1)[0]
            margin = top2_probs[:, 0] - top2_probs[:, 1]

            # Confident if margin clearly high or low
            confidence = 1.0 - 2.0 * torch.abs(margin - 0.35)
            confidence = torch.clamp(confidence, 0, 1)

        return confidence

class StabilityConfidence(ConfidenceEstimator):
    """Confidence based on prediction stability under dropout"""

    def __init__(self, n_samples=5):
        self.n_samples = n_samples

    def estimate(self, x: torch.Tensor, y: torch.Tensor, model: nn.Module) -> torch.Tensor:
        model.train()  # Enable dropout

        predictions = []
        for _ in range(self.n_samples):
            with torch.no_grad():
                logits = model(x)
                pred = logits.argmax(dim=-1)
                predictions.append(pred)

        model.eval()

        # Confidence = agreement across samples
        predictions = torch.stack(predictions, dim=0)
        mode_pred = torch.mode(predictions, dim=0)[0]
        agreement = (predictions == mode_pred.unsqueeze(0)).float().mean(dim=0)

        return agreement

class DistanceConfidence(ConfidenceEstimator):
    """Confidence based on distance to selected samples"""

    def __init__(self):
        self.selected_features = []

    def update_selected(self, features: torch.Tensor):
        """Update set of selected sample features"""
        self.selected_features.append(features.detach().cpu())

    def estimate(self, x: torch.Tensor, y: torch.Tensor, model: nn.Module) -> torch.Tensor:
        if len(self.selected_features) == 0:
            return torch.ones(x.shape[0], device=x.device)

        with torch.no_grad():
            # Extract features
            features = model.forward_features(x) if hasattr(model, 'forward_features') else model(x)
            features = features.view(features.size(0), -1)

            # Compute distances to all selected samples
            selected = torch.cat(self.selected_features, dim=0).to(x.device)
            distances = torch.cdist(features, selected, p=2)
            min_distances = distances.min(dim=1)[0]

            # Normalize distances
            median_dist = min_distances.median()
            normalized_dist = min_distances / (median_dist + 1e-6)

            # Confident if very far or very close
            confidence = torch.abs(normalized_dist - 1.0)
            confidence = torch.clamp(confidence, 0, 1)

        return confidence

# ==============================================================================
# SCORING FUNCTIONS
# ==============================================================================

class ScoringFunction:
    """Base class for scoring functions"""

    def __init__(self, confidence_estimator: ConfidenceEstimator):
        self.confidence_estimator = confidence_estimator

    def score(self, x: torch.Tensor, y: torch.Tensor, model: nn.Module) -> torch.Tensor:
        """Compute score for a batch of samples"""
        raise NotImplementedError

    def estimate_confidence(self, x: torch.Tensor, y: torch.Tensor, model: nn.Module) -> torch.Tensor:
        """Estimate confidence in the scoring"""
        return self.confidence_estimator.estimate(x, y, model)

class UncertaintyScorer(ScoringFunction):
    """Score based on prediction uncertainty (entropy)"""

    def __init__(self):
        super().__init__(EntropyConfidence())

    def score(self, x: torch.Tensor, y: torch.Tensor, model: nn.Module) -> torch.Tensor:
        with torch.no_grad():
            logits = model(x)
            probs = F.softmax(logits, dim=-1)
            entropy = -(probs * torch.log(probs + 1e-10)).sum(dim=-1)
            max_entropy = np.log(probs.shape[-1])
            normalized_entropy = entropy / max_entropy
        return normalized_entropy

class DiversityScorer(ScoringFunction):
    """Score based on diversity (distance to selected samples)"""

    def __init__(self):
        distance_conf = DistanceConfidence()
        super().__init__(distance_conf)
        self.selected_features = []

    def score(self, x: torch.Tensor, y: torch.Tensor, model: nn.Module) -> torch.Tensor:
        with torch.no_grad():
            features = model.forward_features(x) if hasattr(model, 'forward_features') else model(x)
            features = features.view(features.size(0), -1)

            if len(self.selected_features) == 0:
                return torch.ones(x.shape[0], device=x.device)

            selected = torch.cat(self.selected_features, dim=0).to(x.device)
            distances = torch.cdist(features, selected, p=2)
            min_distances = distances.min(dim=1)[0]

            max_dist = min_distances.max()
            normalized_distances = min_distances / (max_dist + 1e-6)

        return normalized_distances

    def update_selected(self, x: torch.Tensor, model: nn.Module):
        """Update selected samples for diversity computation"""
        with torch.no_grad():
            features = model.forward_features(x) if hasattr(model, 'forward_features') else model(x)
            features = features.view(features.size(0), -1)
            self.selected_features.append(features.detach().cpu())
            if len(self.selected_features) > 10:
                self.selected_features = self.selected_features[-10:]

class BoundaryScorer(ScoringFunction):
    """Score based on proximity to decision boundary"""

    def __init__(self):
        super().__init__(MarginConfidence())

    def score(self, x: torch.Tensor, y: torch.Tensor, model: nn.Module) -> torch.Tensor:
        with torch.no_grad():
            logits = model(x)
            probs = F.softmax(logits, dim=-1)
            top2_probs = torch.topk(probs, k=2, dim=-1)[0]
            margin = top2_probs[:, 0] - top2_probs[:, 1]
            boundary_score = 1.0 - margin
        return boundary_score

class ExcessLossScorer(ScoringFunction):
    """Score based on excess loss (from reference model)"""

    def __init__(self, reference_model: nn.Module):
        super().__init__(StabilityConfidence(n_samples=5))
        self.reference_model = reference_model

    def score(self, x: torch.Tensor, y: torch.Tensor, model: nn.Module) -> torch.Tensor:
        with torch.no_grad():
            main_logits = model(x)
            main_loss = F.cross_entropy(main_logits, y, reduction='none')

            ref_logits = self.reference_model(x)
            ref_loss = F.cross_entropy(ref_logits, y, reduction='none')

            excess_loss = main_loss - ref_loss
            normalized_excess = torch.sigmoid(excess_loss)

        return normalized_excess

# ==============================================================================
# GATING NETWORK
# ==============================================================================

class AdaptiveGatingNetwork(nn.Module):
    """Learns when to trust each scoring function"""

    def __init__(self, state_dim: int, num_scorers: int, hidden_dim: int = 64):
        super().__init__()
        self.num_scorers = num_scorers

        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, num_scorers)

        self.temperature = nn.Parameter(torch.tensor(config.TEMPERATURE_INIT))

        self.fc3.bias.data.zero_()

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        logits = self.fc3(x)

        weights = F.softmax(logits / torch.clamp(self.temperature, min=0.1), dim=-1)

        return weights

class TrainingStateComputer:
    """Computes training state for gating network"""

    def __init__(self):
        self.history = defaultdict(list)

    def compute_state(self, model: nn.Module, train_loader: DataLoader,
                     epoch: int, max_epochs: int, current_coverage: float) -> torch.Tensor:
        model.eval()

        correct = 0
        total = 0
        losses = []
        entropies = []

        with torch.no_grad():
            for i, batch in enumerate(train_loader):
                if i >= 10:
                    break

                if len(batch) == 3:
                    inputs, targets, _ = batch
                else:
                    inputs, targets = batch

                inputs, targets = inputs.to(config.DEVICE), targets.to(config.DEVICE)
                outputs = model(inputs)
                loss = F.cross_entropy(outputs, targets, reduction='none')

                _, predicted = torch.max(outputs.data, 1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()
                losses.extend(loss.cpu().numpy())

                probs = F.softmax(outputs, dim=-1)
                entropy = -(probs * torch.log(probs + 1e-10)).sum(dim=-1)
                entropies.extend(entropy.cpu().numpy())

        acc = correct / total if total > 0 else 0.0
        avg_loss = np.mean(losses) if losses else 0.0
        avg_entropy = np.mean(entropies) if entropies else 0.0

        if epoch % 5 == 0:
            grad_norms = []
            for i, batch in enumerate(train_loader):
                if i >= 5:
                    break

                if len(batch) == 3:
                    inputs, targets, _ = batch
                else:
                    inputs, targets = batch

                inputs, targets = inputs.to(config.DEVICE), targets.to(config.DEVICE)

                model.zero_grad()
                outputs = model(inputs)
                loss = F.cross_entropy(outputs, targets)
                loss.backward()

                grad_norm = torch.cat([p.grad.flatten() for p in model.parameters()
                                      if p.grad is not None]).norm().item()
                grad_norms.append(grad_norm)

            avg_grad_norm = np.mean(grad_norms) if grad_norms else 0.0
            self.history['grad_norm'].append(avg_grad_norm)
        else:
            avg_grad_norm = self.history['grad_norm'][-1] if self.history['grad_norm'] else 0.0

        epoch_progress = epoch / max_epochs
        coverage = current_coverage

        state = torch.tensor([
            acc,
            avg_loss,
            avg_grad_norm / 1000.0,
            epoch_progress,
            coverage,
            avg_entropy
        ], dtype=torch.float32).to(config.DEVICE)

        model.train()

        return state

# ==============================================================================
# SELECTIVE MULTI-CLASSIFIER CORESET SELECTOR
# ==============================================================================

class SelectiveMultiClassifierSelector:
    """Main class for selective multi-classifier coreset selection"""

    def __init__(self, model: nn.Module, reference_model: Optional[nn.Module] = None):
        self.model = model
        self.reference_model = reference_model

        # Initialize scorers
        self.scorers = [
            UncertaintyScorer(),
            DiversityScorer(),
            BoundaryScorer(),
        ]

        if reference_model is not None:
            self.scorers.append(ExcessLossScorer(reference_model))

        # Gating network
        if config.USE_GATING:
            self.gating_network = AdaptiveGatingNetwork(
                state_dim=config.GATE_STATE_DIM,
                num_scorers=len(self.scorers),
                hidden_dim=config.GATE_HIDDEN_DIM
            ).to(config.DEVICE)

            self.gate_optimizer = optim.Adam(
                self.gating_network.parameters(),
                lr=0.001
            )
        else:
            self.gating_network = None
            self.fixed_weights = torch.ones(len(self.scorers)) / len(self.scorers)

        # Rejection thresholds
        if config.USE_CONFIDENCE_REJECTION:
            self.rejection_thresholds = nn.Parameter(
                torch.ones(len(self.scorers)) * config.CONFIDENCE_THRESHOLD_INIT
            ).to(config.DEVICE)
        else:
            self.rejection_thresholds = torch.zeros(len(self.scorers)).to(config.DEVICE)

        self.state_computer = TrainingStateComputer()

        # Statistics tracking
        self.stats = {
            'gate_weights_history': [],
            'rejection_rates_history': [],
            'confidence_history': defaultdict(list),
            'selection_attribution': defaultdict(int),
            'waste_tracking': {
                'mastered_early': [],
                'stagnant': [],
                'harmful': []
            }
        }

        self.sample_confidences = defaultdict(list)
        self.sample_selected = defaultdict(list)

    def select_coreset(self, dataset: Dataset, epoch: int, budget: float) -> Tuple[torch.Tensor, Dict]:
        """Select coreset using multi-classifier ensemble"""
        n_samples = len(dataset)
        k = int(budget * n_samples)

        loader = DataLoader(dataset, batch_size=config.BATCH_SIZE,
                          shuffle=False, num_workers=2)

        # Compute training state
        state = self.state_computer.compute_state(
            self.model, loader, epoch, config.NUM_EPOCHS, budget
        )

        # Get gating weights
        if config.USE_GATING:
            gate_weights = self.gating_network(state)
        else:
            gate_weights = self.fixed_weights.to(config.DEVICE)

        self.stats['gate_weights_history'].append(gate_weights.detach().cpu().numpy())

        # Compute scores from all scorers
        all_scores = []
        all_confidences = []
        all_sample_indices = []

        self.model.eval()

        for batch_idx, batch_data in enumerate(tqdm(loader, desc=f"Scoring (Epoch {epoch})", leave=False)):
            if len(batch_data) == 3:
                inputs, targets, indices = batch_data
            else:
                inputs, targets = batch_data
                indices = torch.arange(batch_idx * config.BATCH_SIZE,
                                     min((batch_idx + 1) * config.BATCH_SIZE, n_samples))

            inputs, targets = inputs.to(config.DEVICE), targets.to(config.DEVICE)

            batch_scores = []
            batch_confidences = []

            for scorer in self.scorers:
                score = scorer.score(inputs, targets, self.model)
                confidence = scorer.estimate_confidence(inputs, targets, self.model)

                batch_scores.append(score.cpu())
                batch_confidences.append(confidence.cpu())

            all_scores.append(torch.stack(batch_scores, dim=0))
            all_confidences.append(torch.stack(batch_confidences, dim=0))
            all_sample_indices.append(indices.cpu())

        all_scores = torch.cat(all_scores, dim=1)
        all_confidences = torch.cat(all_confidences, dim=1)
        all_sample_indices = torch.cat(all_sample_indices, dim=0)

        # Apply selective rejection
        rejection_masks = all_confidences < self.rejection_thresholds.unsqueeze(1).cpu()
        rejection_rates = rejection_masks.float().mean(dim=1)
        self.stats['rejection_rates_history'].append(rejection_rates.numpy())

        all_scores_filtered = all_scores.clone()
        all_scores_filtered[rejection_masks] = 0.0

        # Compute ensemble scores
        gate_weights_cpu = gate_weights.cpu().unsqueeze(1)
        ensemble_scores = (gate_weights_cpu * all_confidences * all_scores_filtered).sum(dim=0)

        # Top-k selection
        _, selected_indices = torch.topk(ensemble_scores, k)
        selected_indices = all_sample_indices[selected_indices]

        # Update diversity scorer
        if isinstance(self.scorers[1], DiversityScorer):
            selected_dataset = Subset(dataset, selected_indices.tolist())
            selected_loader = DataLoader(selected_dataset, batch_size=config.BATCH_SIZE,
                                        shuffle=False, num_workers=2)
            for batch in selected_loader:
                if len(batch) == 3:
                    inputs, _, _ = batch
                else:
                    inputs, _ = batch
                inputs = inputs.to(config.DEVICE)
                self.scorers[1].update_selected(inputs, self.model)

        # Track sample selection
        if config.TRACK_SAMPLE_WASTE:
            self._track_sample_selection(all_sample_indices, selected_indices,
                                        all_confidences, epoch)

        # Compute selection attribution
        selection_attribution = self._compute_selection_attribution(
            selected_indices, all_scores, gate_weights_cpu
        )

        selection_info = {
            'gate_weights': gate_weights.detach().cpu().numpy(),
            'rejection_rates': rejection_rates.numpy(),
            'avg_confidences': all_confidences.mean(dim=1).numpy(),
            'ensemble_score_stats': {
                'mean': ensemble_scores.mean().item(),
                'std': ensemble_scores.std().item(),
                'min': ensemble_scores.min().item(),
                'max': ensemble_scores.max().item()
            },
            'selection_attribution': selection_attribution,
            'num_selected': len(selected_indices)
        }

        self.model.train()

        return selected_indices, selection_info

    def _track_sample_selection(self, all_indices: torch.Tensor,
                                selected_indices: torch.Tensor,
                                confidences: torch.Tensor, epoch: int):
        """Track sample confidence and selection for waste analysis"""
        avg_confidences = confidences.mean(dim=0)

        for i, idx in enumerate(all_indices):
            idx = idx.item()
            conf = avg_confidences[i].item()
            selected = idx in selected_indices

            self.sample_confidences[idx].append((epoch, conf))
            self.sample_selected[idx].append((epoch, selected))

    def _compute_selection_attribution(self, selected_indices: torch.Tensor,
                                      all_scores: torch.Tensor,
                                      gate_weights: torch.Tensor) -> Dict:
        """Determine which scorer was most responsible for each selection"""
        attribution = {i: 0 for i in range(len(self.scorers))}

        for idx in selected_indices:
            weighted_scores = gate_weights.squeeze() * all_scores[:, idx]
            primary_scorer = weighted_scores.argmax().item()
            attribution[primary_scorer] += 1
            self.stats['selection_attribution'][primary_scorer] += 1

        return attribution

    def compute_sample_waste(self, epoch: int) -> Dict:
        """Analyze sample waste based on confidence evolution"""
        waste_stats = {
            'mastered_early': 0,
            'stagnant': 0,
            'potentially_harmful': 0,
            'useful': 0
        }

        if epoch < 20:
            return waste_stats

        for idx, history in self.sample_confidences.items():
            if len(history) < 10:
                continue

            epochs, confs = zip(*history)

            # Check if mastered early
            for e, c in zip(epochs, confs):
                if c > 0.95 and e < min(50, epoch * 0.5):
                    waste_stats['mastered_early'] += 1
                    break

            # Check if stagnant
            if len(confs) >= 20:
                recent_confs = confs[-20:]
                improvement = max(recent_confs) - min(recent_confs)
                if improvement < 0.1:
                    waste_stats['stagnant'] += 1

            # Potentially harmful
            if len(confs) >= 10:
                recent_confs = confs[-10:]
                if recent_confs[-1] < recent_confs[0] - 0.15:
                    waste_stats['potentially_harmful'] += 1

        total_tracked = len(self.sample_confidences)
        waste_stats['useful'] = max(0, total_tracked - sum([
            waste_stats['mastered_early'],
            waste_stats['stagnant'],
            waste_stats['potentially_harmful']
        ]))

        self.stats['waste_tracking']['mastered_early'].append(waste_stats['mastered_early'])
        self.stats['waste_tracking']['stagnant'].append(waste_stats['stagnant'])
        self.stats['waste_tracking']['harmful'].append(waste_stats['potentially_harmful'])

        return waste_stats

    def get_statistics(self) -> Dict:
        """Get comprehensive statistics"""
        return self.stats

# ==============================================================================
# DATA LOADING
# ==============================================================================

class IndexedDataset(Dataset):
    """Wrapper dataset that returns sample index along with data"""

    def __init__(self, base_dataset):
        self.base_dataset = base_dataset

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        data, target = self.base_dataset[idx]
        return data, target, idx

def load_data():
    """Load CIFAR datasets"""
    logger.info(f"Loading {config.DATASET} dataset...")

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

    if config.DATASET == "CIFAR10":
        train_ds = torchvision.datasets.CIFAR10(
            root=config.DATA_PATH, train=True, download=True, transform=transform_train
        )
        val_ds = torchvision.datasets.CIFAR10(
            root=config.DATA_PATH, train=False, download=True, transform=transform_test
        )
    elif config.DATASET == "CIFAR100":
        train_ds = torchvision.datasets.CIFAR100(
            root=config.DATA_PATH, train=True, download=True, transform=transform_train
        )
        val_ds = torchvision.datasets.CIFAR100(
            root=config.DATA_PATH, train=False, download=True, transform=transform_test
        )
    else:
        raise ValueError(f"Dataset {config.DATASET} not supported")

    train_ds = IndexedDataset(train_ds)
    val_ds = IndexedDataset(val_ds)

    logger.info(f"Train samples: {len(train_ds)}")
    logger.info(f"Val samples: {len(val_ds)}")

    return train_ds, val_ds

# ==============================================================================
# TRAINING LOOP
# ==============================================================================

def train_with_selective_multiclassifier(
    model: nn.Module,
    train_dataset: Dataset,
    val_dataset: Dataset,
    selector: SelectiveMultiClassifierSelector,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    scheduler: Optional[optim.lr_scheduler._LRScheduler] = None,
    selection_ratio: float = 0.5
):
    """Training loop with selective multi-classifier coreset selection"""

    train_loader_full = DataLoader(train_dataset, batch_size=config.BATCH_SIZE,
                                   shuffle=False, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE,
                           shuffle=False, num_workers=2)

    best_val_acc = 0.0
    training_history = {
        'train_loss': [],
        'train_acc': [],
        'val_acc': [],
        'selection_ratios': [],
        'waste_stats': []
    }

    for epoch in range(config.NUM_EPOCHS):
        # Select coreset periodically
        if epoch % config.SELECTION_FREQUENCY == 0:
            logger.info(f"Selecting coreset with budget={selection_ratio:.2f}...")
            selected_indices, selection_info = selector.select_coreset(
                train_dataset, epoch, selection_ratio
            )

            logger.info(f"Selected {len(selected_indices)} samples")
            logger.info(f"Gate weights: {selection_info['gate_weights']}")

            train_subset = Subset(train_dataset, selected_indices.tolist())
            train_loader = DataLoader(train_subset, batch_size=config.BATCH_SIZE,
                                     shuffle=True, num_workers=2)

        # Train on selected coreset
        model.train()
        epoch_loss = 0.0
        num_batches = 0
        correct = 0
        total = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.NUM_EPOCHS}", leave=False)
        for batch_data in pbar:
            if len(batch_data) == 3:
                inputs, targets, _ = batch_data
            else:
                inputs, targets = batch_data

            inputs, targets = inputs.to(config.DEVICE), targets.to(config.DEVICE)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1
            _, predicted = torch.max(outputs, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()

            pbar.set_postfix({'loss': f'{loss.item():.4f}', 'acc': f'{100.*correct/total:.2f}%'})

        if scheduler:
            scheduler.step()

        epoch_loss /= num_batches
        train_acc = correct / total

        training_history['train_loss'].append(epoch_loss)
        training_history['train_acc'].append(train_acc)
        training_history['selection_ratios'].append(selection_ratio)

        # Validation
        if epoch % config.EVAL_FREQUENCY == 0:
            val_acc = evaluate_model(model, val_loader)
            training_history['val_acc'].append(val_acc)
            logger.info(f"Epoch {epoch+1}: Loss={epoch_loss:.4f}, "
                       f"Train Acc={train_acc:.4f}, Val Acc={val_acc:.4f}")

            if val_acc > best_val_acc:
                best_val_acc = val_acc

        # Compute sample waste periodically
        if config.TRACK_SAMPLE_WASTE and epoch % 10 == 0 and epoch > 0:
            waste_stats = selector.compute_sample_waste(epoch)
            training_history['waste_stats'].append(waste_stats)

    return model, training_history, best_val_acc

def evaluate_model(model: nn.Module, val_loader: DataLoader) -> float:
    """Evaluate model on validation set"""
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for batch_data in val_loader:
            if len(batch_data) == 3:
                inputs, targets, _ = batch_data
            else:
                inputs, targets = batch_data

            inputs, targets = inputs.to(config.DEVICE), targets.to(config.DEVICE)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()

    model.train()
    return correct / total

# ==============================================================================
# EXPERIMENT RUNNER (Multi-Ratio, Multi-Model)
# ==============================================================================

class ExperimentRunner:
    """Runs experiments across multiple ratios and models"""

    def __init__(self):
        self.results = []
        self.results_dir = os.path.join(config.RESULTS_DIR, config.EXPERIMENT_NAME)
        os.makedirs(self.results_dir, exist_ok=True)

        if config.SAVE_CHECKPOINTS:
            self.checkpoint_dir = os.path.join(config.CHECKPOINT_DIR, config.EXPERIMENT_NAME)
            os.makedirs(self.checkpoint_dir, exist_ok=True)

    def run_single_experiment(self, model_name: str, reference_model_name: str,
                            selection_ratio: float, train_dataset: Dataset,
                            val_dataset: Dataset, exp_id: int) -> Dict:
        """Run a single experiment with given configuration"""

        logger.info("="*70)
        logger.info(f"EXPERIMENT {exp_id}")
        logger.info(f"Model: {model_name}, Reference: {reference_model_name}")
        logger.info(f"Selection Ratio: {selection_ratio}")
        logger.info("="*70)

        # Initialize models
        model = timm.create_model(model_name, pretrained=config.PRETRAINED,
                                 num_classes=config.NUM_CLASSES)
        model = model.to(config.DEVICE)

        reference_model = None
        if reference_model_name:
            reference_model = timm.create_model(reference_model_name,
                                               pretrained=config.PRETRAINED,
                                               num_classes=config.NUM_CLASSES)
            reference_model = reference_model.to(config.DEVICE)
            reference_model.eval()

        # Initialize selector
        selector = SelectiveMultiClassifierSelector(model, reference_model)

        # Setup training
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=config.LEARNING_RATE,
                             momentum=config.MOMENTUM, weight_decay=config.WEIGHT_DECAY)

        if config.LR_SCHEDULE == "cosine":
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.NUM_EPOCHS)
        else:
            scheduler = None

        # Train
        model, training_history, best_val_acc = train_with_selective_multiclassifier(
            model, train_dataset, val_dataset, selector,
            criterion, optimizer, scheduler, selection_ratio
        )

        # Final evaluation
        val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE,
                               shuffle=False, num_workers=2)
        final_acc = evaluate_model(model, val_loader)

        # Get statistics
        stats = selector.get_statistics()

        # Package results
        result = {
            'exp_id': exp_id,
            'model_name': model_name,
            'reference_model_name': reference_model_name,
            'selection_ratio': selection_ratio,
            'final_acc': final_acc,
            'best_val_acc': best_val_acc,
            'training_history': training_history,
            'stats': stats,
            'timestamp': datetime.now().isoformat()
        }

        # Save checkpoint
        if config.SAVE_CHECKPOINTS:
            checkpoint_path = os.path.join(
                self.checkpoint_dir,
                f"exp_{exp_id}_model_{model_name}_ratio_{selection_ratio:.2f}.pth"
            )
            torch.save({
                'model_state_dict': model.state_dict(),
                'result': result
            }, checkpoint_path)
            logger.info(f"Saved checkpoint to {checkpoint_path}")

        logger.info(f"Experiment {exp_id} completed: Final Acc = {final_acc:.4f}")

        return result

    def run_multi_ratio_experiments(self, train_dataset: Dataset, val_dataset: Dataset):
        """Run experiments across multiple selection ratios"""

        logger.info("\n" + "="*70)
        logger.info("MULTI-RATIO EXPERIMENTS")
        logger.info("="*70)

        model_name = config.MODEL_NAME
        reference_model_name = config.REFERENCE_MODEL_NAME

        exp_id = 0
        for ratio in config.RATIOS_TO_TEST:
            result = self.run_single_experiment(
                model_name, reference_model_name, ratio,
                train_dataset, val_dataset, exp_id
            )
            self.results.append(result)
            exp_id += 1

            # Save intermediate results
            self._save_results()

    def run_multi_model_experiments(self, train_dataset: Dataset, val_dataset: Dataset):
        """Run experiments across multiple model architectures"""

        logger.info("\n" + "="*70)
        logger.info("MULTI-MODEL EXPERIMENTS")
        logger.info("="*70)

        exp_id = len(self.results)
        for model_name, reference_model_name in config.MODELS_TO_TEST:
            for ratio in config.RATIOS_TO_TEST:
                result = self.run_single_experiment(
                    model_name, reference_model_name, ratio,
                    train_dataset, val_dataset, exp_id
                )
                self.results.append(result)
                exp_id += 1

                # Save intermediate results
                self._save_results()

    def run_all_experiments(self, train_dataset: Dataset, val_dataset: Dataset):
        """Run complete experiment suite"""

        if config.SINGLE_MODEL_MULTI_RATIO:
            self.run_multi_ratio_experiments(train_dataset, val_dataset)

        if config.ENABLE_MULTI_MODEL:
            self.run_multi_model_experiments(train_dataset, val_dataset)

        # Generate comprehensive report
        self._generate_report()

        # Generate visualizations
        if config.GENERATE_PLOTS:
            self._generate_plots()

    def _save_results(self):
        """Save results to JSON file"""
        results_file = os.path.join(self.results_dir, "results.json")

        # Convert results to JSON-serializable format
        serializable_results = []
        for result in self.results:
            ser_result = result.copy()
            # Convert numpy arrays to lists
            if 'stats' in ser_result:
                stats = ser_result['stats']
                for key in ['gate_weights_history', 'rejection_rates_history']:
                    if key in stats:
                        stats[key] = [w.tolist() if isinstance(w, np.ndarray) else w
                                     for w in stats[key]]
            serializable_results.append(ser_result)

        with open(results_file, 'w') as f:
            json.dump(serializable_results, f, indent=2)

        logger.info(f"Results saved to {results_file}")

    def _generate_report(self):
        """Generate comprehensive experiment report"""

        logger.info("\n" + "="*70)
        logger.info("EXPERIMENT SUMMARY")
        logger.info("="*70)

        # Create DataFrame
        df_data = []
        for result in self.results:
            df_data.append({
                'Exp ID': result['exp_id'],
                'Model': result['model_name'],
                'Reference': result['reference_model_name'],
                'Ratio': result['selection_ratio'],
                'Final Acc': result['final_acc'],
                'Best Val Acc': result['best_val_acc']
            })

        df = pd.DataFrame(df_data)

        # Print summary
        logger.info("\n" + str(df))

        # Save to CSV
        csv_path = os.path.join(self.results_dir, "results_summary.csv")
        df.to_csv(csv_path, index=False)
        logger.info(f"\nSummary saved to {csv_path}")

        # Generate statistics
        logger.info("\n" + "="*70)
        logger.info("STATISTICS BY RATIO")
        logger.info("="*70)
        logger.info(df.groupby('Ratio')['Final Acc'].agg(['mean', 'std', 'min', 'max']))

        if config.ENABLE_MULTI_MODEL:
            logger.info("\n" + "="*70)
            logger.info("STATISTICS BY MODEL")
            logger.info("="*70)
            logger.info(df.groupby('Model')['Final Acc'].agg(['mean', 'std', 'min', 'max']))

        # Best configuration
        best_result = df.loc[df['Final Acc'].idxmax()]
        logger.info("\n" + "="*70)
        logger.info("BEST CONFIGURATION")
        logger.info("="*70)
        logger.info(f"Model: {best_result['Model']}")
        logger.info(f"Selection Ratio: {best_result['Ratio']}")
        logger.info(f"Final Accuracy: {best_result['Final Acc']:.4f}")

    def _generate_plots(self):
        """Generate visualization plots"""

        logger.info("\n" + "="*70)
        logger.info("GENERATING PLOTS")
        logger.info("="*70)

        plots_dir = os.path.join(self.results_dir, "plots")
        os.makedirs(plots_dir, exist_ok=True)

        # 1. Accuracy vs Ratio
        self._plot_acc_vs_ratio(plots_dir)

        # 2. Accuracy vs Model (if multi-model)
        if config.ENABLE_MULTI_MODEL:
            self._plot_acc_vs_model(plots_dir)

        # 3. Training curves
        self._plot_training_curves(plots_dir)

        # 4. Sample waste analysis
        if config.TRACK_SAMPLE_WASTE:
            self._plot_waste_analysis(plots_dir)

        logger.info(f"Plots saved to {plots_dir}")

    def _plot_acc_vs_ratio(self, plots_dir: str):
        """Plot accuracy vs selection ratio"""
        plt.figure(figsize=(10, 6))

        # Group by model
        models = set(r['model_name'] for r in self.results)

        for model in models:
            model_results = [r for r in self.results if r['model_name'] == model]
            ratios = [r['selection_ratio'] for r in model_results]
            accs = [r['final_acc'] for r in model_results]

            # Sort by ratio
            sorted_pairs = sorted(zip(ratios, accs))
            ratios, accs = zip(*sorted_pairs)

            plt.plot(ratios, accs, marker='o', label=model, linewidth=2, markersize=8)

        plt.xlabel('Selection Ratio', fontsize=12)
        plt.ylabel('Final Accuracy', fontsize=12)
        plt.title('Final Accuracy vs Selection Ratio', fontsize=14, fontweight='bold')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        plt.savefig(os.path.join(plots_dir, f'acc_vs_ratio.{config.PLOT_FORMAT}'), dpi=300)
        plt.close()

    def _plot_acc_vs_model(self, plots_dir: str):
        """Plot accuracy comparison across models"""
        plt.figure(figsize=(12, 6))

        # Group results by model
        model_accs = defaultdict(list)
        for result in self.results:
            model_accs[result['model_name']].append(result['final_acc'])

        models = list(model_accs.keys())
        mean_accs = [np.mean(model_accs[m]) for m in models]
        std_accs = [np.std(model_accs[m]) for m in models]

        x = np.arange(len(models))
        plt.bar(x, mean_accs, yerr=std_accs, capsize=5, alpha=0.7)
        plt.xticks(x, models, rotation=45, ha='right')
        plt.ylabel('Final Accuracy', fontsize=12)
        plt.title('Model Comparison (Mean ± Std)', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()

        plt.savefig(os.path.join(plots_dir, f'acc_vs_model.{config.PLOT_FORMAT}'), dpi=300)
        plt.close()

    def _plot_training_curves(self, plots_dir: str):
        """Plot training curves for each experiment"""

        for result in self.results[:min(6, len(self.results))]:  # Plot first 6 experiments
            exp_id = result['exp_id']
            history = result['training_history']

            fig, axes = plt.subplots(1, 2, figsize=(14, 5))

            # Loss curve
            axes[0].plot(history['train_loss'], label='Train Loss', linewidth=2)
            axes[0].set_xlabel('Epoch', fontsize=12)
            axes[0].set_ylabel('Loss', fontsize=12)
            axes[0].set_title(f'Training Loss (Exp {exp_id})', fontsize=12, fontweight='bold')
            axes[0].legend()
            axes[0].grid(True, alpha=0.3)

            # Accuracy curve
            axes[1].plot(history['train_acc'], label='Train Acc', linewidth=2)
            if 'val_acc' in history and len(history['val_acc']) > 0:
                val_epochs = np.arange(0, len(history['train_acc']), config.EVAL_FREQUENCY)[:len(history['val_acc'])]
                axes[1].plot(val_epochs, history['val_acc'], label='Val Acc', linewidth=2, marker='o')
            axes[1].set_xlabel('Epoch', fontsize=12)
            axes[1].set_ylabel('Accuracy', fontsize=12)
            axes[1].set_title(f'Training Accuracy (Exp {exp_id})', fontsize=12, fontweight='bold')
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)

            plt.tight_layout()
            plt.savefig(os.path.join(plots_dir, f'training_curve_exp_{exp_id}.{config.PLOT_FORMAT}'), dpi=300)
            plt.close()

    def _plot_waste_analysis(self, plots_dir: str):
        """Plot sample waste analysis"""

        for result in self.results[:min(3, len(self.results))]:  # Plot first 3
            exp_id = result['exp_id']

            if 'waste_tracking' in result['stats']:
                waste = result['stats']['waste_tracking']

                if len(waste['mastered_early']) > 0:
                    fig, ax = plt.subplots(figsize=(10, 6))

                    epochs = np.arange(len(waste['mastered_early'])) * 10
                    ax.plot(epochs, waste['mastered_early'], label='Mastered Early', marker='o')
                    ax.plot(epochs, waste['stagnant'], label='Stagnant', marker='s')
                    ax.plot(epochs, waste['harmful'], label='Harmful', marker='^')

                    ax.set_xlabel('Epoch', fontsize=12)
                    ax.set_ylabel('Number of Samples', fontsize=12)
                    ax.set_title(f'Sample Waste Analysis (Exp {exp_id})', fontsize=14, fontweight='bold')
                    ax.legend()
                    ax.grid(True, alpha=0.3)

                    plt.tight_layout()
                    plt.savefig(os.path.join(plots_dir, f'waste_exp_{exp_id}.{config.PLOT_FORMAT}'), dpi=300)
                    plt.close()

# ==============================================================================
# MAIN EXECUTION
# ==============================================================================

def main():
    """Main execution function"""

    logger.info("="*70)
    logger.info("SELECTIVE MULTI-CLASSIFIER CORESET SELECTION")
    logger.info("MULTI-RATIO AND MULTI-MODEL EXPERIMENTS")
    logger.info("="*70)
    logger.info(f"Dataset: {config.DATASET}")
    logger.info(f"Device: {config.DEVICE}")
    logger.info(f"Multi-Ratio: {config.ENABLE_MULTI_RATIO}")
    logger.info(f"Multi-Model: {config.ENABLE_MULTI_MODEL}")
    logger.info(f"Ratios to test: {config.RATIOS_TO_TEST}")
    if config.ENABLE_MULTI_MODEL:
        logger.info(f"Models to test: {[m[0] for m in config.MODELS_TO_TEST]}")
    logger.info("="*70)

    # Load data once
    logger.info("\nLoading data...")
    train_dataset, val_dataset = load_data()

    # Initialize experiment runner
    runner = ExperimentRunner()

    # Run all experiments
    logger.info("\nStarting experiments...")
    runner.run_all_experiments(train_dataset, val_dataset)

    logger.info("\n" + "="*70)
    logger.info("ALL EXPERIMENTS COMPLETED!")
    logger.info(f"Results saved to: {runner.results_dir}")
    logger.info("="*70)

if __name__ == "__main__":
    main()
