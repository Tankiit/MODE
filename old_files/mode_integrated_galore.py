"""
MODE (Memory-Optimized Data Selection) Integration for GALORE
============================================================

This module provides an integrated MODE implementation that can work with
the GALORE framework for RL-guided data selection on CIFAR datasets.

Based on the enhanced MODE implementation with multiple encoders:
- BinaryStateEncoder
- TokenEncoder  
- ContextEncoder
- StrategyEncoder
- PerformanceEncoder

Compatible with the GALORE repository structure and CIFAR experiments.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple, Union
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict, deque
import json
import os

try:
    from torch.utils.tensorboard import SummaryWriter
    HAS_TB = True
except:
    HAS_TB = False

try:
    import faiss
    HAS_FAISS = True
except:
    HAS_FAISS = False
    print("WARNING: FAISS not available, using fallback retrieval")


# ============================================================================
# MODE CONFIGURATION FOR GALORE
# ============================================================================

@dataclass
class MODEConfig:
    """
    Configuration for MODE integration with GALORE framework.
    Adapted for CIFAR datasets and computer vision tasks.
    """
    # Model Architecture (optimized for CIFAR)
    n_layer: int = 8
    n_embd: int = 512
    n_head: int = 8
    dropout: float = 0.1
    max_seq_length: int = 512

    # Training
    batch_size: int = 32  # Increased for CIFAR
    grad_accumulation: int = 1
    epochs: int = 100
    learning_rate: float = 1e-3  # Adjusted for vision tasks
    warmup_steps: int = 100
    warmup_epochs: int = 5
    weight_decay: float = 1e-4
    max_grad_norm: float = 1.0
    patience: int = 10

    # Data Selection
    selection_budget: float = 0.30  # 30% of data
    n_strategies: int = 4
    
    # MODE Memory
    max_memory_contexts: int = 1000
    retrieval_k: int = 15
    mode_sharpness: float = 2.0
    context_window: int = 5

    # Binary Hypernetwork
    state_dim: int = 19  # Binary state + performance features
    memory_size: int = 1000
    attention_heads: int = 4
    hypernetwork_hidden: int = 128
    lookback_window: int = 10
    update_threshold: float = 0.7

    # CIFAR-specific
    num_classes: int = 10  # CIFAR10 by default
    image_size: int = 32
    channels: int = 3

    # Evaluation
    eval_interval: int = 10
    save_interval: int = 20

    # Logging
    log_interval: int = 50
    log_dir: str = './runs_mode_galore'
    
    # Integration with GALORE
    use_galore: bool = True
    galore_rank: int = 16
    adaptive_rank: bool = True


# ============================================================================
# BINARY STATE ENCODER
# ============================================================================

class BinaryStateEncoder:
    """
    Lightweight binary state encoder for training state.
    Adapted for computer vision tasks and CIFAR datasets.
    """

    def __init__(self, lookback_window: int = 10):
        self.lookback_window = lookback_window
        self.history = {
            'loss': deque(maxlen=lookback_window * 2),
            'accuracy': deque(maxlen=lookback_window * 2),
            'selection_ratio': deque(maxlen=lookback_window * 2),
            'strategy_entropy': deque(maxlen=lookback_window * 2)
        }

        # Binary state names for interpretability
        self.state_names = [
            'progress_early',        # 0: < 30% progress
            'progress_middle',       # 1: 30-70% progress
            'progress_late',         # 2: > 70% progress
            'loss_improving',        # 3: Loss decreasing
            'loss_low',             # 4: Loss < threshold
            'loss_stable',          # 5: Loss variance low
            'accuracy_high',        # 6: Accuracy > threshold
            'accuracy_improving',   # 7: Accuracy increasing
            'selection_high',       # 8: High selection ratio
            'selection_stable',     # 9: Selection ratio stable
            'strategy_diverse',     # 10: High strategy entropy
            'strategy_focused',     # 11: Low strategy entropy
            'performance_good',     # 12: Recent performance good
            'model_experienced'     # 13: Sufficient training
        ]

    def update_history(self, loss: float, accuracy: float, selection_ratio: float,
                      strategy_weights: np.ndarray):
        """Update training history for trend computation"""
        self.history['loss'].append(loss)
        self.history['accuracy'].append(accuracy)
        self.history['selection_ratio'].append(selection_ratio)

        # Compute strategy entropy
        if strategy_weights.sum() > 0:
            normalized = strategy_weights / strategy_weights.sum()
            entropy = -np.sum(normalized * np.log(normalized + 1e-8))
            self.history['strategy_entropy'].append(entropy)
        else:
            self.history['strategy_entropy'].append(0.0)

    def encode_binary_state(self, epoch: int, total_epochs: int,
                           current_loss: float, current_accuracy: float,
                           current_selection: float) -> torch.Tensor:
        """
        Encode current state as binary vector for CIFAR training.
        """
        progress = epoch / total_epochs

        # Progress indicators (mutually exclusive)
        progress_early = float(progress < 0.3)
        progress_middle = float(0.3 <= progress < 0.7)
        progress_late = float(progress >= 0.7)

        # Loss signals
        loss_improving = self._compute_trend('loss', lower_better=True)
        loss_low = float(current_loss < 1.0)  # Adjusted for CIFAR
        loss_stable = self._compute_stability('loss')

        # Accuracy signals
        accuracy_high = float(current_accuracy > 0.8)  # 80% accuracy
        accuracy_improving = self._compute_trend('accuracy', lower_better=False)

        # Selection signals
        selection_high = float(current_selection > 0.25)
        selection_stable = self._compute_stability('selection_ratio')

        # Strategy diversity signals
        strategy_diverse = float(self._get_recent_entropy() > 1.0)
        strategy_focused = float(self._get_recent_entropy() < 0.5)

        # Overall performance
        performance_good = float(loss_improving and accuracy_improving)
        model_experienced = float(progress > 0.2)

        return torch.tensor([
            progress_early, progress_middle, progress_late,
            loss_improving, loss_low, loss_stable,
            accuracy_high, accuracy_improving,
            selection_high, selection_stable,
            strategy_diverse, strategy_focused,
            performance_good, model_experienced
        ], dtype=torch.float32)

    def _compute_trend(self, metric: str, lower_better: bool = False) -> float:
        """Fast trend computation"""
        history = self.history[metric]
        if len(history) < self.lookback_window:
            return 0.5

        recent = np.mean(list(history)[-self.lookback_window:])
        older = np.mean(list(history)[-self.lookback_window*2:-self.lookback_window])

        if lower_better:
            improvement = older - recent
        else:
            improvement = recent - older

        return float(improvement > 0.05)

    def _compute_stability(self, metric: str) -> float:
        """Fast stability computation"""
        history = self.history[metric]
        if len(history) < 3:
            return 0.0

        recent_std = np.std(list(history)[-3:])
        return float(recent_std < 0.1)

    def _get_recent_entropy(self) -> float:
        """Get recent strategy entropy"""
        if len(self.history['strategy_entropy']) < 2:
            return np.log(4)  # Max entropy for 4 strategies
        return np.mean(list(self.history['strategy_entropy'])[-3:])


# ============================================================================
# FEATURE ENCODER FOR CIFAR
# ============================================================================

class CIFARFeatureEncoder:
    """
    Feature encoder specifically designed for CIFAR datasets.
    Encodes image-level and batch-level features.
    """

    def __init__(self, num_classes: int = 10, image_size: int = 32):
        self.num_classes = num_classes
        self.image_size = image_size
        
        # Feature tracking
        self.class_frequencies = defaultdict(int)
        self.total_samples = 0
        
        # Position encoding for spatial features
        self.position_encoding = self._create_spatial_encoding()

    def _create_spatial_encoding(self) -> torch.Tensor:
        """Create spatial position encoding for CIFAR images"""
        pe = torch.zeros(self.image_size, self.image_size, 64)
        
        for i in range(self.image_size):
            for j in range(self.image_size):
                # 2D sinusoidal encoding
                pe[i, j, 0::4] = torch.sin(torch.tensor(i) / (10000 ** (torch.arange(0, 16) / 64)))
                pe[i, j, 1::4] = torch.cos(torch.tensor(i) / (10000 ** (torch.arange(0, 16) / 64)))
                pe[i, j, 2::4] = torch.sin(torch.tensor(j) / (10000 ** (torch.arange(0, 16) / 64)))
                pe[i, j, 3::4] = torch.cos(torch.tensor(j) / (10000 ** (torch.arange(0, 16) / 64)))
        
        return pe

    def update_class_frequencies(self, labels: torch.Tensor):
        """Update class frequency statistics"""
        for label in labels:
            self.class_frequencies[label.item()] += 1
            self.total_samples += 1

    def encode_image_features(self, images: torch.Tensor, labels: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Encode image-level features for CIFAR data.
        """
        batch_size = images.shape[0]
        device = images.device
        
        features = {}
        
        # 1. Class frequency features
        class_freq_scores = torch.zeros(batch_size, device=device)
        for i, label in enumerate(labels):
            freq = self.class_frequencies.get(label.item(), 0)
            class_freq_scores[i] = freq / max(self.total_samples, 1)
        features['class_frequency'] = class_freq_scores
        
        # 2. Image complexity (variance)
        complexity_scores = torch.var(images.view(batch_size, -1), dim=1)
        features['image_complexity'] = complexity_scores
        
        # 3. Spatial features (center vs edge)
        center_mask = torch.zeros_like(images[:, 0])
        center_start = self.image_size // 4
        center_end = 3 * self.image_size // 4
        center_mask[:, center_start:center_end, center_start:center_end] = 1.0
        
        center_intensity = torch.sum(images * center_mask.unsqueeze(1), dim=(2, 3))
        edge_intensity = torch.sum(images * (1 - center_mask).unsqueeze(1), dim=(2, 3))
        spatial_ratio = center_intensity / (edge_intensity + 1e-8)
        features['spatial_center_ratio'] = spatial_ratio
        
        # 4. Color distribution
        color_mean = torch.mean(images, dim=(2, 3))
        color_std = torch.std(images, dim=(2, 3))
        features['color_mean'] = color_mean
        features['color_std'] = color_std
        
        # 5. Class rarity
        class_rarity = 1.0 / (class_freq_scores + 1e-8)
        features['class_rarity'] = class_rarity
        
        return features


# ============================================================================
# CONTEXT ENCODER FOR CIFAR
# ============================================================================

class CIFARContextEncoder:
    """
    Context encoder for CIFAR datasets.
    Analyzes batch-level and training context.
    """

    def __init__(self, context_window: int = 5, hidden_dim: int = 512):
        self.context_window = context_window
        self.hidden_dim = hidden_dim
        
        # Context tracking
        self.batch_patterns = deque(maxlen=100)
        self.class_distributions = deque(maxlen=50)

    def encode_batch_features(self, images: torch.Tensor, labels: torch.Tensor,
                            hidden_states: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Encode batch-level context features.
        """
        batch_size = images.shape[0]
        device = images.device
        
        features = {}
        
        # 1. Batch class diversity
        unique_classes = torch.unique(labels)
        class_diversity = len(unique_classes) / self.num_classes
        diversity_tensor = torch.full((batch_size,), class_diversity, device=device)
        features['batch_diversity'] = diversity_tensor
        
        # 2. Class distribution entropy
        class_counts = torch.bincount(labels, minlength=self.num_classes).float()
        class_probs = class_counts / class_counts.sum()
        class_entropy = -torch.sum(class_probs * torch.log(class_probs + 1e-8))
        entropy_tensor = torch.full((batch_size,), class_entropy, device=device)
        features['class_entropy'] = entropy_tensor
        
        # 3. Batch similarity (if hidden states available)
        if hidden_states is not None:
            # Compute pairwise similarities
            similarities = F.cosine_similarity(
                hidden_states.unsqueeze(1), hidden_states.unsqueeze(0), dim=2
            )
            avg_similarity = torch.mean(similarities, dim=1)
            features['batch_similarity'] = avg_similarity
        
        # 4. Image quality indicators
        brightness = torch.mean(images, dim=(1, 2, 3))
        contrast = torch.std(images, dim=(1, 2, 3))
        features['brightness'] = brightness
        features['contrast'] = contrast
        
        return features


# ============================================================================
# STRATEGY ENCODER
# ============================================================================

class StrategyEncoder:
    """
    Strategy encoder for adaptive data selection strategies.
    """

    def __init__(self, n_strategies: int = 4):
        self.n_strategies = n_strategies
        
        # Strategy performance tracking
        self.strategy_performance = defaultdict(list)
        self.strategy_history = deque(maxlen=100)
        
        # Strategy names adapted for CIFAR
        self.strategy_names = ['uncertainty', 'gradient_magnitude', 'diversity', 'class_balance']

    def update_strategy_performance(self, strategy_weights: np.ndarray, 
                                  success_signal: float):
        """Update strategy performance history"""
        strategy_dict = dict(zip(self.strategy_names, strategy_weights))
        strategy_dict['success'] = success_signal
        self.strategy_history.append(strategy_dict)
        
        # Update individual strategy performance
        for i, name in enumerate(self.strategy_names):
            self.strategy_performance[name].append(strategy_weights[i] * success_signal)

    def encode_strategy_features(self, features: Dict[str, torch.Tensor],
                               strategy_weights: np.ndarray) -> Dict[str, torch.Tensor]:
        """
        Encode strategy-specific features.
        """
        batch_size = next(iter(features.values())).shape[0]
        device = next(iter(features.values())).device
        
        strategy_features = {}
        
        # 1. Strategy effectiveness scores
        effectiveness_scores = torch.zeros(batch_size, device=device)
        for i, name in enumerate(self.strategy_names):
            if name in features:
                effectiveness_scores += strategy_weights[i] * features[name]
        strategy_features['strategy_effectiveness'] = effectiveness_scores
        
        # 2. Strategy diversity (entropy of weights)
        strategy_entropy = -np.sum(strategy_weights * np.log(strategy_weights + 1e-8))
        entropy_tensor = torch.full((batch_size,), strategy_entropy, device=device)
        strategy_features['strategy_diversity'] = entropy_tensor
        
        # 3. Dominant strategy indicator
        dominant_idx = np.argmax(strategy_weights)
        dominant_weight = strategy_weights[dominant_idx]
        dominant_tensor = torch.full((batch_size,), dominant_weight, device=device)
        strategy_features['dominant_strategy_weight'] = dominant_tensor
        
        return strategy_features


# ============================================================================
# PERFORMANCE ENCODER
# ============================================================================

class PerformanceEncoder:
    """
    Performance encoder for training metrics and progress indicators.
    """

    def __init__(self, lookback_window: int = 10):
        self.lookback_window = lookback_window
        
        # Performance history
        self.loss_history = deque(maxlen=lookback_window * 2)
        self.accuracy_history = deque(maxlen=lookback_window * 2)
        self.selection_history = deque(maxlen=lookback_window * 2)
        self.epoch_history = deque(maxlen=lookback_window * 2)
        
        # Performance thresholds for CIFAR
        self.loss_thresholds = {'low': 0.5, 'medium': 1.0, 'high': 2.0}
        self.accuracy_thresholds = {'low': 0.7, 'medium': 0.85, 'high': 0.95}
        self.selection_thresholds = {'low': 0.1, 'medium': 0.3, 'high': 0.5}

    def update_performance_history(self, loss: float, accuracy: float, 
                                 selection_ratio: float, epoch: int):
        """Update performance history"""
        self.loss_history.append(loss)
        self.accuracy_history.append(accuracy)
        self.selection_history.append(selection_ratio)
        self.epoch_history.append(epoch)

    def encode_performance_features(self, current_loss: float, current_accuracy: float,
                                  current_selection: float, epoch: int, 
                                  total_epochs: int) -> Dict[str, float]:
        """
        Encode performance-based features.
        """
        progress_ratio = epoch / max(total_epochs, 1)
        
        # Performance trends
        loss_trend = self._compute_trend(self.loss_history, lower_better=True)
        accuracy_trend = self._compute_trend(self.accuracy_history, lower_better=False)
        selection_trend = self._compute_trend(self.selection_history, lower_better=False)
        
        # Performance stability
        loss_stability = self._compute_stability(self.loss_history)
        accuracy_stability = self._compute_stability(self.accuracy_history)
        selection_stability = self._compute_stability(self.selection_history)
        
        # Performance levels
        loss_level = self._get_performance_level(current_loss, self.loss_thresholds)
        accuracy_level = self._get_performance_level(current_accuracy, self.accuracy_thresholds)
        selection_level = self._get_performance_level(current_selection, self.selection_thresholds)
        
        return {
            'progress_ratio': progress_ratio,
            'loss_trend': loss_trend,
            'accuracy_trend': accuracy_trend,
            'selection_trend': selection_trend,
            'loss_stability': loss_stability,
            'accuracy_stability': accuracy_stability,
            'selection_stability': selection_stability,
            'loss_level': loss_level,
            'accuracy_level': accuracy_level,
            'selection_level': selection_level,
            'current_loss': current_loss,
            'current_accuracy': current_accuracy,
            'current_selection': current_selection
        }

    def _compute_trend(self, history: deque, lower_better: bool = False) -> float:
        """Compute trend from history"""
        if len(history) < self.lookback_window:
            return 0.5
        
        recent = np.mean(list(history)[-self.lookback_window:])
        older = np.mean(list(history)[-self.lookback_window*2:-self.lookback_window])
        
        if lower_better:
            improvement = older - recent
        else:
            improvement = recent - older
        
        return float(improvement > 0.05)

    def _compute_stability(self, history: deque) -> float:
        """Compute stability from history"""
        if len(history) < 3:
            return 0.0
        
        recent_std = np.std(list(history)[-3:])
        return float(recent_std < 0.1)

    def _get_performance_level(self, value: float, thresholds: Dict[str, float]) -> float:
        """Get performance level based on thresholds"""
        if value <= thresholds['low']:
            return 0.0  # Low level
        elif value <= thresholds['medium']:
            return 0.5  # Medium level
        else:
            return 1.0  # High level


# ============================================================================
# FAST MEMORY HYPERNETWORK
# ============================================================================

class FastMemoryHypernetwork(nn.Module):
    """
    Lightweight memory-augmented hypernetwork for strategy prediction.
    Adapted for CIFAR and computer vision tasks.
    """

    def __init__(self, config: MODEConfig):
        super().__init__()
        self.config = config

        # Memory components
        self.memory_states = nn.Parameter(torch.randn(config.memory_size, config.state_dim) * 0.1)
        self.memory_strategies = nn.Parameter(torch.randn(config.memory_size, config.n_strategies) * 0.1)
        self.memory_values = nn.Parameter(torch.zeros(config.memory_size))

        # Fast attention
        self.query_proj = nn.Linear(config.state_dim, config.hypernetwork_hidden)
        self.key_proj = nn.Linear(config.state_dim, config.hypernetwork_hidden)
        self.value_proj = nn.Linear(config.n_strategies, config.hypernetwork_hidden)

        # Strategy predictor
        self.strategy_net = nn.Sequential(
            nn.Linear(config.hypernetwork_hidden + config.state_dim, config.hypernetwork_hidden),
            nn.ReLU(),
            nn.Linear(config.hypernetwork_hidden, config.n_strategies)
        )

        # Memory update gate
        self.update_gate = nn.Sequential(
            nn.Linear(config.state_dim + config.n_strategies + 1, config.hypernetwork_hidden // 2),
            nn.ReLU(),
            nn.Linear(config.hypernetwork_hidden // 2, 1),
            nn.Sigmoid()
        )

        self._initialize_memory()

    def _initialize_memory(self):
        """Initialize with diverse strategy patterns for CIFAR"""
        with torch.no_grad():
            for i in range(self.config.memory_size):
                # Create diverse binary states
                state = torch.randint(0, 2, (self.config.state_dim,), dtype=torch.float32)

                # Ensure valid progress state (exactly one active)
                progress_idx = torch.randint(0, 3, (1,)).item()
                state[0:3] = 0
                state[progress_idx] = 1

                self.memory_states[i] = state

                # Create reasonable strategy patterns for CIFAR
                if progress_idx == 0:  # Early
                    strategy = torch.tensor([0.4, 0.3, 0.2, 0.1])  # uncertainty, gradient, diversity, balance
                elif progress_idx == 1:  # Middle
                    strategy = torch.tensor([0.2, 0.3, 0.3, 0.2])  # balanced
                else:  # Late
                    strategy = torch.tensor([0.2, 0.2, 0.3, 0.3])  # diversity and balance focus

                strategy += torch.randn_like(strategy) * 0.05
                strategy = F.softmax(strategy, dim=0)

                self.memory_strategies[i] = strategy
                self.memory_values[i] = torch.rand(1).item() * 0.5 + 0.5

    def forward(self, binary_state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Fast strategy prediction using memory attention"""
        if binary_state.dim() == 1:
            binary_state = binary_state.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False

        batch_size = binary_state.size(0)

        # Fast single-head attention
        queries = self.query_proj(binary_state)

        # Memory attention
        keys = self.key_proj(self.memory_states)
        values = self.value_proj(self.memory_strategies)

        # Compute attention weights
        scores = torch.matmul(queries, keys.T)
        attention_weights = F.softmax(scores, dim=-1)

        # Weighted memory retrieval
        attended_memory = torch.matmul(attention_weights, values)

        # Combine with current state
        combined = torch.cat([attended_memory, binary_state], dim=-1)

        # Predict strategies
        strategy_logits = self.strategy_net(combined)
        strategy_weights = F.softmax(strategy_logits, dim=-1)

        if squeeze_output:
            strategy_weights = strategy_weights.squeeze(0)
            attention_weights = attention_weights.squeeze(0)

        return strategy_weights, attention_weights

    def update_memory(self, state: torch.Tensor, strategy: torch.Tensor, success: float):
        """Update memory if gated and successful"""
        if success < self.config.update_threshold:
            return

        # Compute gate
        gate_input = torch.cat([state.flatten(), strategy.flatten(), torch.tensor([success])])
        gate_value = self.update_gate(gate_input).item()

        if gate_value > 0.5:
            # Find worst memory slot
            worst_idx = torch.argmin(self.memory_values).item()

            with torch.no_grad():
                self.memory_states[worst_idx] = state.clone()
                self.memory_strategies[worst_idx] = strategy.clone()
                self.memory_values[worst_idx] = success


# ============================================================================
# MODE CONTROLLER FOR GALORE
# ============================================================================

class MODEController:
    """
    MODE controller for GALORE integration.
    Combines multiple encoders for comprehensive data selection.
    """

    def __init__(self, config: MODEConfig, device):
        self.config = config
        self.device = device

        # Multiple encoders
        self.binary_state_encoder = BinaryStateEncoder(config.lookback_window)
        self.cifar_feature_encoder = CIFARFeatureEncoder(config.num_classes, config.image_size)
        self.context_encoder = CIFARContextEncoder(config.context_window, config.n_embd)
        self.strategy_encoder = StrategyEncoder(config.n_strategies)
        self.performance_encoder = PerformanceEncoder(config.lookback_window)

        # Memory hypernetwork
        self.hypernetwork = FastMemoryHypernetwork(config).to(device)

        # Strategy names
        self.strategy_names = ['uncertainty', 'gradient_magnitude', 'diversity', 'class_balance']

        # Training state
        self.epoch = 0
        self.total_epochs = config.epochs
        self.is_trained = False

        # History for analysis
        self.strategy_history = []

    def predict_strategy_weights(self, current_loss: float, current_accuracy: float,
                               current_selection: float) -> np.ndarray:
        """
        Enhanced strategy weight prediction using multiple encoders.
        """
        # Encode binary state
        binary_state = self.binary_state_encoder.encode_binary_state(
            self.epoch, self.total_epochs, current_loss, current_accuracy, current_selection
        ).to(self.device)

        # Encode performance features
        performance_features = self.performance_encoder.encode_performance_features(
            current_loss, current_accuracy, current_selection, self.epoch, self.total_epochs
        )

        # Enhanced state with performance features
        performance_values = torch.tensor([
            performance_features['progress_ratio'],
            performance_features['loss_trend'],
            performance_features['accuracy_trend'],
            performance_features['selection_trend'],
            performance_features['loss_stability'],
            performance_features['accuracy_stability'],
            performance_features['selection_stability']
        ], device=self.device)

        enhanced_state = torch.cat([binary_state, performance_values], dim=0)

        # Predict using memory hypernetwork
        with torch.no_grad():
            strategy_weights, _ = self.hypernetwork(enhanced_state)

        weights_np = strategy_weights.cpu().numpy()
        self.strategy_history.append(weights_np.copy())

        return weights_np

    def update_training_state(self, loss: float, accuracy: float, selection_ratio: float,
                            strategy_weights: np.ndarray, success_signal: float,
                            labels: Optional[torch.Tensor] = None):
        """Update state and potentially memory using multiple encoders"""
        # Update encoder histories
        self.binary_state_encoder.update_history(loss, accuracy, selection_ratio, strategy_weights)
        self.performance_encoder.update_performance_history(loss, accuracy, selection_ratio, self.epoch)
        self.strategy_encoder.update_strategy_performance(strategy_weights, success_signal)
        
        # Update feature encoder
        if labels is not None:
            self.cifar_feature_encoder.update_class_frequencies(labels)

        # Update memory if successful
        if success_signal > self.config.update_threshold:
            binary_state = self.binary_state_encoder.encode_binary_state(
                self.epoch, self.total_epochs, loss, accuracy, selection_ratio
            ).to(self.device)
            
            performance_features = self.performance_encoder.encode_performance_features(
                loss, accuracy, selection_ratio, self.epoch, self.total_epochs
            )
            
            performance_values = torch.tensor([
                performance_features['progress_ratio'],
                performance_features['loss_trend'],
                performance_features['accuracy_trend'],
                performance_features['selection_trend'],
                performance_features['loss_stability'],
                performance_features['accuracy_stability'],
                performance_features['selection_stability']
            ], device=self.device)
            
            enhanced_state = torch.cat([binary_state, performance_values], dim=0)
            strategy_tensor = torch.tensor(strategy_weights, dtype=torch.float32, device=self.device)
            self.hypernetwork.update_memory(enhanced_state, strategy_tensor, success_signal)

        self.epoch += 1
        self.is_trained = True

    def get_strategy_interpretation(self) -> Dict:
        """Get interpretable strategy analysis"""
        if not self.strategy_history:
            return {}

        recent_weights = np.mean(self.strategy_history[-10:], axis=0)

        interpretation = {
            'current_weights': dict(zip(self.strategy_names, recent_weights)),
            'dominant_strategy': self.strategy_names[np.argmax(recent_weights)],
            'strategy_diversity': -np.sum(recent_weights * np.log(recent_weights + 1e-8)),
            'memory_size': len(self.hypernetwork.memory_values)
        }

        return interpretation


# ============================================================================
# DATA SELECTION FUNCTIONS
# ============================================================================

def select_data_with_mode(images: torch.Tensor, labels: torch.Tensor,
                         mode_controller: MODEController, config: MODEConfig,
                         model: nn.Module, criterion: nn.Module) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
    """
    Select data using MODE controller for GALORE integration.
    
    Args:
        images: Input images
        labels: Input labels
        mode_controller: MODE controller instance
        config: MODE configuration
        model: Current model
        criterion: Loss criterion
        
    Returns:
        selected_images, selected_labels, selection_info
    """
    batch_size = images.shape[0]
    device = images.device
    
    # Get current training metrics
    model.eval()
    with torch.no_grad():
        outputs = model(images)
        current_loss = criterion(outputs, labels).item()
        
        # Compute accuracy
        _, predicted = torch.max(outputs, 1)
        current_accuracy = (predicted == labels).float().mean().item()
    
    model.train()
    
    # Get strategy weights
    strategy_weights = mode_controller.predict_strategy_weights(
        current_loss, current_accuracy, config.selection_budget
    )
    
    # Encode features
    image_features = mode_controller.cifar_feature_encoder.encode_image_features(images, labels)
    batch_features = mode_controller.context_encoder.encode_batch_features(images, labels)
    
    # Combine all features
    all_features = {**image_features, **batch_features}
    
    # Compute selection scores
    selection_scores = torch.zeros(batch_size, device=device)
    
    # Core strategy features
    strategy_names = ['uncertainty', 'gradient_magnitude', 'diversity', 'class_balance']
    for i, name in enumerate(strategy_names):
        if name in all_features:
            selection_scores += strategy_weights[i] * all_features[name]
    
    # Enhanced features with adaptive weighting
    enhanced_features = {
        'class_frequency': 0.1,
        'image_complexity': 0.2,
        'spatial_center_ratio': 0.15,
        'class_rarity': 0.2,
        'batch_diversity': 0.15,
        'class_entropy': 0.1,
        'brightness': 0.05,
        'contrast': 0.05
    }
    
    for feature_name, weight in enhanced_features.items():
        if feature_name in all_features:
            feature = all_features[feature_name]
            # Normalize feature to [0, 1] range
            feature_min = feature.min()
            feature_max = feature.max()
            if feature_max > feature_min:
                feature_normalized = (feature - feature_min) / (feature_max - feature_min)
            else:
                feature_normalized = feature
            selection_scores += weight * feature_normalized
    
    # Strategy-specific features
    strategy_features = mode_controller.strategy_encoder.encode_strategy_features(
        all_features, strategy_weights
    )
    
    if 'strategy_effectiveness' in strategy_features:
        selection_scores += 0.2 * strategy_features['strategy_effectiveness']
    
    # Select top-k samples
    n_select = max(1, int(config.selection_budget * batch_size))
    
    if batch_size > n_select:
        _, selected_indices = torch.topk(selection_scores, n_select)
        selected_images = images[selected_indices]
        selected_labels = labels[selected_indices]
    else:
        selected_images = images
        selected_labels = labels
    
    # Selection info
    selection_info = {
        'strategy_weights': strategy_weights,
        'selection_scores': selection_scores.cpu().numpy(),
        'n_selected': len(selected_images),
        'selection_ratio': len(selected_images) / batch_size
    }
    
    return selected_images, selected_labels, selection_info


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def create_mode_config_for_cifar(num_classes: int = 10, **kwargs) -> MODEConfig:
    """
    Create MODE configuration optimized for CIFAR datasets.
    
    Args:
        num_classes: Number of classes (10 for CIFAR10, 100 for CIFAR100)
        **kwargs: Additional configuration parameters
        
    Returns:
        MODEConfig instance
    """
    config = MODEConfig()
    config.num_classes = num_classes
    
    # Adjust parameters based on dataset
    if num_classes == 100:  # CIFAR100
        config.epochs = 150
        config.learning_rate = 5e-4
        config.batch_size = 64
        config.patience = 15
    
    # Update with any provided parameters
    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)
    
    return config


def save_mode_results(results: Dict, save_path: str):
    """Save MODE experiment results"""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)


def load_mode_results(load_path: str) -> Dict:
    """Load MODE experiment results"""
    with open(load_path, 'r') as f:
        return json.load(f)


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

def example_usage():
    """
    Example usage of MODE integration with GALORE.
    """
    # Create configuration
    config = create_mode_config_for_cifar(num_classes=10)
    
    # Create MODE controller
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    mode_controller = MODEController(config, device)
    
    # Example data (replace with actual CIFAR data)
    batch_size = 32
    images = torch.randn(batch_size, 3, 32, 32)
    labels = torch.randint(0, 10, (batch_size,))
    
    # Example model and criterion
    model = nn.Sequential(
        nn.Flatten(),
        nn.Linear(3 * 32 * 32, 128),
        nn.ReLU(),
        nn.Linear(128, 10)
    ).to(device)
    criterion = nn.CrossEntropyLoss()
    
    # Select data using MODE
    selected_images, selected_labels, selection_info = select_data_with_mode(
        images, labels, mode_controller, config, model, criterion
    )
    
    print(f"Selected {len(selected_images)} samples out of {batch_size}")
    print(f"Strategy weights: {selection_info['strategy_weights']}")
    print(f"Selection ratio: {selection_info['selection_ratio']:.2f}")
    
    return selected_images, selected_labels, selection_info


if __name__ == "__main__":
    example_usage()
