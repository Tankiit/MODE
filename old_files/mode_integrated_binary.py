
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, GPT2Config
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict, deque

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
# MODE CONFIGURATION
# ============================================================================

@dataclass
class MODEConfig:
    # Model Architecture (optimized for speed)
    n_layer: int = 8
    n_embd: int = 512
    n_head: int = 8
    dropout: float = 0.1
    max_seq_length: int = 512

    # Training
    batch_size: int = 2
    grad_accumulation: int = 1
    epochs: int = 50
    learning_rate: float = 5e-5
    warmup_steps: int = 100
    warmup_epochs: int = 1
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    patience: int = 4

    # Token Selection
    token_budget: float = 0.30
    n_strategies: int = 4  # Reduced from 5 for efficiency

    # Token MODE Memory (reduced for MPS memory constraints)
    max_memory_contexts: int = 500  # Reduced from 10000 for MPS
    retrieval_k: int = 10  # Reduced from 25 for speed
    mode_sharpness: float = 2.0
    context_window: int = 5

    # Binary Hypernetwork (NEW)
    state_dim: int = 19  # Binary state features + performance features (12 + 7)
    memory_size: int = 500  # Smaller memory for speed
    attention_heads: int = 4
    hypernetwork_hidden: int = 64  # Reduced from 128
    lookback_window: int = 5
    update_threshold: float = 0.7  # Only update memory on success

    # Data
    train_samples: int = 200
    val_samples: int = 50
    min_text_length: int = 100

    # Evaluation
    eval_stride: int = 128
    eval_max_length: int = 256

    # Logging
    log_interval: int = 50
    log_dir: str = './runs_integrated_mode'

    # Initialization
    use_pretrained_init: bool = True
    pretrained_model: str = 'distilgpt2'


# ============================================================================
# BINARY STATE ENCODER
# ============================================================================

class BinaryStateEncoder:
    """
    Lightweight binary state encoder for training state.
    Much faster than continuous feature extraction.
    """

    def __init__(self, lookback_window: int = 5):
        self.lookback_window = lookback_window
        self.history = {
            'loss': deque(maxlen=lookback_window * 2),
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
            'selection_high',       # 6: High selection ratio
            'selection_stable',     # 7: Selection ratio stable
            'strategy_diverse',     # 8: High strategy entropy
            'strategy_focused',     # 9: Low strategy entropy
            'performance_good',     # 10: Recent performance good
            'model_experienced'     # 11: Sufficient training
        ]

    def update_history(self, loss: float, selection_ratio: float,
                      strategy_weights: np.ndarray):
        """Update training history for trend computation"""
        self.history['loss'].append(loss)
        self.history['selection_ratio'].append(selection_ratio)

        # Compute strategy entropy
        if strategy_weights.sum() > 0:
            normalized = strategy_weights / strategy_weights.sum()
            entropy = -np.sum(normalized * np.log(normalized + 1e-8))
            self.history['strategy_entropy'].append(entropy)
        else:
            self.history['strategy_entropy'].append(0.0)

    def encode_binary_state(self, epoch: int, total_epochs: int,
                           current_loss: float, current_selection: float) -> torch.Tensor:
        """
        Encode current state as 12D binary vector.
        Very fast - just thresholding and trend computation.
        """
        progress = epoch / total_epochs

        # Progress indicators (mutually exclusive)
        progress_early = float(progress < 0.3)
        progress_middle = float(0.3 <= progress < 0.7)
        progress_late = float(progress >= 0.7)

        # Loss signals
        loss_improving = self._compute_trend('loss', lower_better=True)
        loss_low = float(current_loss < 2.0)  # Threshold
        loss_stable = self._compute_stability('loss')

        # Selection signals
        selection_high = float(current_selection > 0.25)
        selection_stable = self._compute_stability('selection_ratio')

        # Strategy diversity signals
        strategy_diverse = float(self._get_recent_entropy() > 1.0)
        strategy_focused = float(self._get_recent_entropy() < 0.5)

        # Overall performance
        performance_good = float(loss_improving and selection_stable)
        model_experienced = float(progress > 0.2)

        return torch.tensor([
            progress_early, progress_middle, progress_late,
            loss_improving, loss_low, loss_stable,
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
# TOKEN ENCODER
# ============================================================================

class TokenEncoder:
    """
    Encoder for token-level features including position, frequency, and semantic properties.
    """

    def __init__(self, vocab_size: int = 50257, max_seq_length: int = 512):
        self.vocab_size = vocab_size
        self.max_seq_length = max_seq_length
        
        # Token frequency tracking
        self.token_frequencies = defaultdict(int)
        self.total_tokens = 0
        
        # Position encoding
        self.position_embeddings = self._create_position_embeddings()

    def _create_position_embeddings(self) -> torch.Tensor:
        """Create sinusoidal position embeddings"""
        pe = torch.zeros(self.max_seq_length, 64)  # 64-dim position encoding
        position = torch.arange(0, self.max_seq_length).unsqueeze(1).float()
        
        div_term = torch.exp(torch.arange(0, 64, 2).float() * 
                           -(np.log(10000.0) / 64))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        return pe

    def update_token_frequencies(self, input_ids: torch.Tensor):
        """Update token frequency statistics"""
        for token_id in input_ids.flatten():
            self.token_frequencies[token_id.item()] += 1
            self.total_tokens += 1

    def encode_token_features(self, input_ids: torch.Tensor, 
                            attention_mask: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Encode token-level features for the sequence.
        Returns dictionary of encoded features.
        """
        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        
        features = {}
        
        # 1. Token frequency features
        freq_scores = torch.zeros(batch_size, seq_len, device=device)
        for b in range(batch_size):
            for pos in range(seq_len):
                if attention_mask[b, pos] > 0.5:
                    token_id = input_ids[b, pos].item()
                    freq = self.token_frequencies.get(token_id, 0)
                    freq_scores[b, pos] = freq / max(self.total_tokens, 1)
        features['token_frequency'] = freq_scores
        
        # 2. Position features
        pos_features = torch.zeros(batch_size, seq_len, 64, device=device)
        for b in range(batch_size):
            seq_length = attention_mask[b].sum().item()
            pos_features[b, :seq_length] = self.position_embeddings[:seq_length].to(device)
        features['position_encoding'] = pos_features
        
        # 3. Token rarity (inverse frequency)
        rarity_scores = 1.0 / (freq_scores + 1e-8)
        features['token_rarity'] = rarity_scores
        
        # 4. Sequence position ratio
        position_ratios = torch.zeros(batch_size, seq_len, device=device)
        for b in range(batch_size):
            seq_length = attention_mask[b].sum().item()
            for pos in range(seq_length):
                position_ratios[b, pos] = pos / max(seq_length - 1, 1)
        features['position_ratio'] = position_ratios
        
        # 5. Token type features (special tokens)
        special_token_mask = torch.zeros(batch_size, seq_len, device=device)
        special_tokens = {0, 1, 2, 50256}  # Common special tokens
        for b in range(batch_size):
            for pos in range(seq_len):
                if attention_mask[b, pos] > 0.5:
                    token_id = input_ids[b, pos].item()
                    special_token_mask[b, pos] = float(token_id in special_tokens)
        features['special_token'] = special_token_mask
        
        return features


# ============================================================================
# CONTEXT ENCODER
# ============================================================================

class ContextEncoder:
    """
    Encoder for contextual information including semantic coherence and context windows.
    """

    def __init__(self, context_window: int = 5, hidden_dim: int = 512):
        self.context_window = context_window
        self.hidden_dim = hidden_dim
        
        # Context similarity tracking
        self.context_patterns = []
        self.max_patterns = 1000

    def encode_context_features(self, hidden_states: torch.Tensor,
                              attention_mask: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Encode contextual features from hidden states.
        """
        batch_size, seq_len, hidden_dim = hidden_states.shape
        device = hidden_states.device
        
        features = {}
        
        # 1. Local context coherence (within window)
        coherence_scores = torch.zeros(batch_size, seq_len, device=device)
        for b in range(batch_size):
            seq_length = attention_mask[b].sum().item()
            for pos in range(seq_length):
                start_pos = max(0, pos - self.context_window // 2)
                end_pos = min(seq_length, pos + self.context_window // 2 + 1)
                
                if end_pos - start_pos > 1:
                    context_hidden = hidden_states[b, start_pos:end_pos]
                    current_hidden = hidden_states[b, pos]
                    
                    # Compute cosine similarity with context
                    similarities = F.cosine_similarity(
                        current_hidden.unsqueeze(0), context_hidden, dim=1
                    )
                    coherence_scores[b, pos] = similarities.mean()
        
        features['local_coherence'] = coherence_scores
        
        # 2. Global context similarity (with sequence mean)
        global_coherence = torch.zeros(batch_size, seq_len, device=device)
        for b in range(batch_size):
            seq_length = attention_mask[b].sum().item()
            if seq_length > 0:
                seq_mean = hidden_states[b, :seq_length].mean(dim=0)
                similarities = F.cosine_similarity(
                    hidden_states[b], seq_mean.unsqueeze(0), dim=1
                )
                global_coherence[b] = similarities
        features['global_coherence'] = global_coherence
        
        # 3. Context diversity (variance within window)
        diversity_scores = torch.zeros(batch_size, seq_len, device=device)
        for b in range(batch_size):
            seq_length = attention_mask[b].sum().item()
            for pos in range(seq_length):
                start_pos = max(0, pos - self.context_window // 2)
                end_pos = min(seq_length, pos + self.context_window // 2 + 1)
                
                if end_pos - start_pos > 1:
                    context_hidden = hidden_states[b, start_pos:end_pos]
                    diversity = torch.var(context_hidden, dim=0).mean()
                    diversity_scores[b, pos] = diversity
        
        features['context_diversity'] = diversity_scores
        
        # 4. Semantic distance from sequence start
        distance_scores = torch.zeros(batch_size, seq_len, device=device)
        for b in range(batch_size):
            seq_length = attention_mask[b].sum().item()
            if seq_length > 0:
                start_hidden = hidden_states[b, 0]
                distances = torch.norm(hidden_states[b] - start_hidden, dim=1)
                distance_scores[b] = distances
        features['semantic_distance'] = distance_scores
        
        return features


# ============================================================================
# STRATEGY ENCODER
# ============================================================================

class StrategyEncoder:
    """
    Encoder for strategy-specific features and their effectiveness.
    """

    def __init__(self, n_strategies: int = 4):
        self.n_strategies = n_strategies
        
        # Strategy performance tracking
        self.strategy_performance = defaultdict(list)
        self.strategy_history = deque(maxlen=100)
        
        # Strategy names
        self.strategy_names = ['uncertainty', 'loss', 'coherence', 'token_mode']

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
        Encode strategy-specific features based on current strategy weights.
        """
        batch_size, seq_len = next(iter(features.values())).shape[:2]
        device = next(iter(features.values())).device
        
        strategy_features = {}
        
        # 1. Strategy effectiveness scores
        effectiveness_scores = torch.zeros(batch_size, seq_len, device=device)
        for i, name in enumerate(self.strategy_names):
            if name in features:
                effectiveness_scores += strategy_weights[i] * features[name]
        strategy_features['strategy_effectiveness'] = effectiveness_scores
        
        # 2. Strategy diversity (entropy of weights)
        strategy_entropy = -np.sum(strategy_weights * np.log(strategy_weights + 1e-8))
        entropy_tensor = torch.full((batch_size, seq_len), strategy_entropy, device=device)
        strategy_features['strategy_diversity'] = entropy_tensor
        
        # 3. Dominant strategy indicator
        dominant_idx = np.argmax(strategy_weights)
        dominant_name = self.strategy_names[dominant_idx]
        dominant_weight = strategy_weights[dominant_idx]
        
        dominant_tensor = torch.full((batch_size, seq_len), dominant_weight, device=device)
        strategy_features['dominant_strategy_weight'] = dominant_tensor
        
        # 4. Strategy confidence (max weight)
        confidence_tensor = torch.full((batch_size, seq_len), dominant_weight, device=device)
        strategy_features['strategy_confidence'] = confidence_tensor
        
        # 5. Historical strategy performance
        historical_scores = torch.zeros(batch_size, seq_len, device=device)
        if len(self.strategy_history) > 0:
            recent_performance = np.mean([
                entry['success'] for entry in list(self.strategy_history)[-10:]
            ])
            historical_tensor = torch.full((batch_size, seq_len), recent_performance, device=device)
            strategy_features['historical_performance'] = historical_tensor
        
        return strategy_features


# ============================================================================
# PERFORMANCE ENCODER
# ============================================================================

class PerformanceEncoder:
    """
    Encoder for training performance metrics and progress indicators.
    """

    def __init__(self, lookback_window: int = 10):
        self.lookback_window = lookback_window
        
        # Performance history
        self.loss_history = deque(maxlen=lookback_window * 2)
        self.selection_history = deque(maxlen=lookback_window * 2)
        self.epoch_history = deque(maxlen=lookback_window * 2)
        
        # Performance thresholds
        self.loss_thresholds = {'low': 1.0, 'medium': 2.0, 'high': 3.0}
        self.selection_thresholds = {'low': 0.1, 'medium': 0.3, 'high': 0.5}

    def update_performance_history(self, loss: float, selection_ratio: float, epoch: int):
        """Update performance history"""
        self.loss_history.append(loss)
        self.selection_history.append(selection_ratio)
        self.epoch_history.append(epoch)

    def encode_performance_features(self, current_loss: float, current_selection: float,
                                  epoch: int, total_epochs: int) -> Dict[str, torch.Tensor]:
        """
        Encode performance-based features.
        """
        # Single values that will be broadcast to sequence length
        progress_ratio = epoch / max(total_epochs, 1)
        
        # Performance trends
        loss_trend = self._compute_trend(self.loss_history, lower_better=True)
        selection_trend = self._compute_trend(self.selection_history, lower_better=False)
        
        # Performance stability
        loss_stability = self._compute_stability(self.loss_history)
        selection_stability = self._compute_stability(self.selection_history)
        
        # Performance levels
        loss_level = self._get_performance_level(current_loss, self.loss_thresholds)
        selection_level = self._get_performance_level(current_selection, self.selection_thresholds)
        
        # Return as single values (will be broadcast when needed)
        return {
            'progress_ratio': progress_ratio,
            'loss_trend': loss_trend,
            'selection_trend': selection_trend,
            'loss_stability': loss_stability,
            'selection_stability': selection_stability,
            'loss_level': loss_level,
            'selection_level': selection_level,
            'current_loss': current_loss,
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
    Optimized for speed over the original complex meta-controller.
    """

    def __init__(self, config: IntegratedMODEConfig):
        super().__init__()
        self.config = config

        # Memory components (smaller for speed)
        self.memory_states = nn.Parameter(torch.randn(config.memory_size, config.state_dim) * 0.1)
        self.memory_strategies = nn.Parameter(torch.randn(config.memory_size, config.n_strategies) * 0.1)
        self.memory_values = nn.Parameter(torch.zeros(config.memory_size))

        # Fast attention (single head for speed)
        self.query_proj = nn.Linear(config.state_dim, config.hypernetwork_hidden)
        self.key_proj = nn.Linear(config.state_dim, config.hypernetwork_hidden)
        self.value_proj = nn.Linear(config.n_strategies, config.hypernetwork_hidden)

        # Strategy predictor (smaller network)
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
        """Initialize with diverse strategy patterns"""
        with torch.no_grad():
            for i in range(self.config.memory_size):
                # Create diverse binary states
                state = torch.randint(0, 2, (self.config.state_dim,), dtype=torch.float32)

                # Ensure valid progress state (exactly one active)
                progress_idx = torch.randint(0, 3, (1,)).item()
                state[0:3] = 0
                state[progress_idx] = 1

                self.memory_states[i] = state

                # Create reasonable strategy patterns
                if progress_idx == 0:  # Early
                    strategy = torch.tensor([0.5, 0.3, 0.1, 0.1])  # uncertainty, loss focus
                elif progress_idx == 1:  # Middle
                    strategy = torch.tensor([0.2, 0.4, 0.3, 0.1])  # diversity, coherence
                else:  # Late
                    strategy = torch.tensor([0.2, 0.2, 0.3, 0.3])  # coherence, balanced

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
        queries = self.query_proj(binary_state)  # [batch, hidden]

        # Memory attention
        keys = self.key_proj(self.memory_states)  # [memory_size, hidden]
        values = self.value_proj(self.memory_strategies)  # [memory_size, hidden]

        # Compute attention weights
        scores = torch.matmul(queries, keys.T)  # [batch, memory_size]
        attention_weights = F.softmax(scores, dim=-1)

        # Weighted memory retrieval
        attended_memory = torch.matmul(attention_weights, values)  # [batch, hidden]

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
# INTEGRATED MODE CONTROLLER
# ============================================================================

class IntegratedMODEController:
    """
    Fast MODE controller combining multiple encoders + memory hypernetwork.
    Enhanced with TokenEncoder, ContextEncoder, StrategyEncoder, and PerformanceEncoder.
    """

    def __init__(self, config: IntegratedMODEConfig, device):
        self.config = config
        self.device = device

        # Multiple encoders for comprehensive feature extraction
        self.binary_state_encoder = FastBinaryStateEncoder(config.lookback_window)
        self.token_encoder = TokenEncoder(config.max_seq_length)
        self.context_encoder = ContextEncoder(config.context_window, config.n_embd)
        self.strategy_encoder = StrategyEncoder(config.n_strategies)
        self.performance_encoder = PerformanceEncoder(config.lookback_window)

        # Memory hypernetwork
        self.hypernetwork = FastMemoryHypernetwork(config).to(device)

        # Strategy names
        self.strategy_names = ['uncertainty', 'loss', 'coherence', 'token_mode']

        # Training state
        self.epoch = 0
        self.total_epochs = config.epochs
        self.is_trained = False

        # History for analysis
        self.strategy_history = []

    def predict_strategy_weights(self, current_loss: float,
                               current_selection: float,
                               input_ids: torch.Tensor = None,
                               attention_mask: torch.Tensor = None,
                               hidden_states: torch.Tensor = None) -> np.ndarray:
        """
        Enhanced strategy weight prediction using multiple encoders.
        Combines binary state, token, context, and performance features.
        """
        # Encode binary state (very fast)
        binary_state = self.binary_state_encoder.encode_binary_state(
            self.epoch, self.total_epochs, current_loss, current_selection
        ).to(self.device)

        # Encode performance features
        performance_features = self.performance_encoder.encode_performance_features(
            current_loss, current_selection, self.epoch, self.total_epochs
        )

        # Enhanced state with performance features
        enhanced_state = binary_state.clone()
        
        # Add performance features to state (broadcast to sequence length if needed)
        if input_ids is not None and attention_mask is not None:
            batch_size, seq_len = input_ids.shape
            
            # Add performance features as additional dimensions
            performance_values = torch.tensor([
                performance_features['progress_ratio'],
                performance_features['loss_trend'],
                performance_features['selection_trend'],
                performance_features['loss_stability'],
                performance_features['selection_stability'],
                performance_features['loss_level'],
                performance_features['selection_level']
            ], device=self.device)
            
            # Expand performance features to match sequence length
            performance_expanded = performance_values.unsqueeze(0).expand(batch_size, -1)
            
            # Combine with binary state
            enhanced_state = torch.cat([binary_state, performance_values], dim=0)
        
        # Predict using memory hypernetwork (fast attention)
        with torch.no_grad():
            strategy_weights, _ = self.hypernetwork(enhanced_state)

        weights_np = strategy_weights.cpu().numpy()
        self.strategy_history.append(weights_np.copy())

        return weights_np

    def update_training_state(self, loss: float, selection_ratio: float,
                            strategy_weights: np.ndarray, success_signal: float,
                            input_ids: torch.Tensor = None):
        """Update state and potentially memory using multiple encoders"""
        # Update binary state encoder history
        self.binary_state_encoder.update_history(loss, selection_ratio, strategy_weights)
        
        # Update performance encoder history
        self.performance_encoder.update_performance_history(loss, selection_ratio, self.epoch)
        
        # Update strategy encoder performance
        self.strategy_encoder.update_strategy_performance(strategy_weights, success_signal)
        
        # Update token encoder frequencies
        if input_ids is not None:
            self.token_encoder.update_token_frequencies(input_ids)

        # Update memory if successful
        if success_signal > self.config.update_threshold:
            binary_state = self.binary_state_encoder.encode_binary_state(
                self.epoch, self.total_epochs, loss, selection_ratio
            ).to(self.device)
            
            # Enhanced state with performance features
            performance_features = self.performance_encoder.encode_performance_features(
                loss, selection_ratio, self.epoch, self.total_epochs
            )
            
            performance_values = torch.tensor([
                performance_features['progress_ratio'],
                performance_features['loss_trend'],
                performance_features['selection_trend'],
                performance_features['loss_stability'],
                performance_features['selection_stability'],
                performance_features['loss_level'],
                performance_features['selection_level']
            ], device=self.device)
            
            enhanced_state = torch.cat([binary_state, performance_values], dim=0)
            strategy_tensor = torch.tensor(strategy_weights, dtype=torch.float32, device=self.device)
            self.hypernetwork.update_memory(enhanced_state, strategy_tensor, success_signal)

        self.epoch += 1
        self.is_trained = True  # Always considered "trained"

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
# UPDATED TOKEN MODE MEMORY (Simplified)
# ============================================================================

class FastTokenModeMemory:
    """Simplified token MODE memory for speed"""

    def __init__(self, config: IntegratedMODEConfig, device, hidden_dim: int):
        self.config = config
        self.device = device
        self.hidden_dim = hidden_dim

        # Simplified storage (no FAISS for speed)
        self.contexts = []
        self.tokens = []
        self.values = []

    def add_context_token_pair(self, context_embedding: torch.Tensor,
                              next_token: int, value: float):
        """Store context-token pair"""
        if len(self.contexts) >= self.config.max_memory_contexts:
            # FIFO replacement
            self.contexts.pop(0)
            self.tokens.pop(0)
            self.values.pop(0)

        self.contexts.append(context_embedding.detach().cpu().numpy())
        self.tokens.append(next_token)
        self.values.append(value)

    def retrieve_mode_score(self, query_context: torch.Tensor, target_token: int) -> float:
        """Fast MODE score computation"""
        if len(self.contexts) == 0:
            return 0.0

        # Fast cosine similarity (batched)
        query_np = query_context.detach().cpu().numpy()
        contexts_array = np.array(self.contexts)

        similarities = np.dot(contexts_array, query_np) / (
            np.linalg.norm(contexts_array, axis=1) * np.linalg.norm(query_np) + 1e-8
        )

        # Get top-k
        k = min(self.config.retrieval_k, len(self.contexts))
        top_indices = np.argpartition(similarities, -k)[-k:]

        # Compute MODE score
        token_scores = defaultdict(float)
        for idx in top_indices:
            token = self.tokens[idx]
            sim = similarities[idx]
            val = self.values[idx]

            weight = sim * (1.0 / (1.0 + np.exp(-val)))
            token_scores[token] += weight

        target_score = token_scores.get(target_token, 0.0)
        total_score = sum(token_scores.values())

        if total_score > 0:
            normalized = target_score / total_score
            return normalized ** self.config.mode_sharpness

        return 0.0

    def size(self) -> int:
        return len(self.contexts)


# ============================================================================
# INTEGRATED TRAINER
# ============================================================================

class IntegratedMODETrainer:
    """
    MODE trainer with integrated binary hypernetwork.
    Much faster than original implementation.
    """

    def __init__(self, config: IntegratedMODEConfig, name: str):
        self.config = config
        self.name = name
        self.device = self._get_device()

        print(f"\n{'='*70}")
        print(f"Initializing Integrated MODE: {name}")
        print(f"Binary State Dim: {config.state_dim} | Memory Size: {config.memory_size}")
        print(f"{'='*70}")

        # Initialize models
        self._init_models()

        # Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained('gpt2')
        self.tokenizer.pad_token = self.tokenizer.eos_token

        # Fast MODE components
        actual_hidden_dim = self.model.config.n_embd
        self.token_memory = FastTokenModeMemory(config, self.device, actual_hidden_dim)
        self.mode_controller = IntegratedMODEController(config, self.device)

        # Optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )

        # Training state
        self.step = 0
        self.epoch = 0
        self.losses = []
        self.best_val_ppl = float('inf')
        self.patience_counter = 0

        # Logging
        if HAS_TB:
            log_path = Path(config.log_dir) / name
            log_path.mkdir(parents=True, exist_ok=True)
            self.writer = SummaryWriter(str(log_path))
        else:
            self.writer = None

    def _get_device(self):
        """Get optimal device"""
        if torch.cuda.is_available():
            return torch.device('cuda')
        elif torch.backends.mps.is_available():
            return torch.device('mps')
        else:
            return torch.device('cpu')

    def _init_models(self):
        """Initialize models"""
        if self.config.use_pretrained_init:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.config.pretrained_model
            ).to(self.device)
            self.ref_model = AutoModelForCausalLM.from_pretrained(
                self.config.pretrained_model
            ).to('cpu')  # Keep on CPU
        else:
            gpt_config = GPT2Config(
                vocab_size=50257,
                n_positions=self.config.max_seq_length,
                n_embd=self.config.n_embd,
                n_layer=self.config.n_layer,
                n_head=self.config.n_head
            )
            self.model = AutoModelForCausalLM.from_config(gpt_config).to(self.device)
            self.ref_model = AutoModelForCausalLM.from_config(gpt_config).to('cpu')

        # Freeze reference
        self.ref_model.eval()
        for param in self.ref_model.parameters():
            param.requires_grad = False

    def extract_fast_features(self, input_ids, attention_mask):
        """
        Enhanced feature extraction using multiple encoders.
        Combines traditional features with token, context, and strategy features.
        """
        self.model.eval()
        train_outputs = self.model(input_ids, attention_mask=attention_mask, output_hidden_states=True)
        self.model.train()

        # Reference model (CPU)
        ref_outputs = self.ref_model(input_ids.cpu(), attention_mask=attention_mask.cpu())
        ref_outputs.logits = ref_outputs.logits.to(self.device)

        logits = train_outputs.logits
        shift_logits = logits[:, :-1, :].contiguous()
        shift_targets = input_ids[:, 1:].contiguous()
        hidden_states = train_outputs.hidden_states[-1] if hasattr(train_outputs, 'hidden_states') else None

        # Traditional fast feature computation
        features = {}

        # 1. Uncertainty (entropy) - fast
        probs = F.softmax(shift_logits, dim=-1)
        entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=-1)
        features['uncertainty'] = F.pad(entropy / np.log(logits.size(-1)), (0, 1))

        # 2. Loss - fast
        loss_fn = nn.CrossEntropyLoss(reduction='none')
        train_loss = loss_fn(shift_logits.reshape(-1, shift_logits.size(-1)),
                            shift_targets.reshape(-1)).reshape(shift_targets.shape)
        features['loss'] = F.pad(train_loss, (0, 1))

        # 3. Coherence (confidence) - fast
        target_probs = probs.gather(dim=-1, index=shift_targets.unsqueeze(-1)).squeeze(-1)
        features['coherence'] = F.pad(target_probs, (0, 1))

        # 4. Token MODE - simplified
        batch_size, seq_len = input_ids.shape
        mode_scores = torch.zeros(batch_size, seq_len, device=self.device)

        if self.token_memory.size() > 0 and hidden_states is not None:
            for b in range(batch_size):
                for pos in range(seq_len - 1):
                    if attention_mask[b, pos] > 0.5:
                        # Simple context: just current hidden state
                        context = hidden_states[b, pos]
                        target_token = input_ids[b, pos + 1].item()

                        mode_score = self.token_memory.retrieve_mode_score(context, target_token)
                        mode_scores[b, pos + 1] = mode_score

        features['token_mode'] = mode_scores

        # Enhanced features using multiple encoders
        if hidden_states is not None:
            # Token-level features
            token_features = self.mode_controller.token_encoder.encode_token_features(
                input_ids, attention_mask
            )
            features.update(token_features)

            # Context features
            context_features = self.mode_controller.context_encoder.encode_context_features(
                hidden_states, attention_mask
            )
            features.update(context_features)

        return features

    def select_tokens_fast_mode(self, features: Dict[str, torch.Tensor],
                               attention_mask: torch.Tensor,
                               input_ids: torch.Tensor = None) -> Tuple[torch.Tensor, np.ndarray]:
        """Enhanced token selection using multiple encoders and binary hypernetwork"""

        # Get current training metrics
        current_loss = np.mean(self.losses[-5:]) if self.losses else 3.0
        current_selection = 0.3  # Approximate

        # Get strategy weights from hypernetwork with enhanced features
        strategy_weights = self.mode_controller.predict_strategy_weights(
            current_loss, current_selection, input_ids, attention_mask
        )

        # Combine feature scores with enhanced features
        batch_size, seq_len = attention_mask.shape
        combined_scores = torch.zeros(batch_size, seq_len, device=self.device)

        # Core strategy features
        strategy_names = ['uncertainty', 'loss', 'coherence', 'token_mode']
        for i, name in enumerate(strategy_names):
            if name in features:
                combined_scores += strategy_weights[i] * features[name]

        # Enhanced features with adaptive weighting
        enhanced_features = {
            'token_frequency': 0.1,      # Lower weight for frequency
            'token_rarity': 0.2,         # Higher weight for rare tokens
            'local_coherence': 0.15,     # Moderate weight for local coherence
            'global_coherence': 0.1,     # Lower weight for global coherence
            'context_diversity': 0.05,   # Low weight for diversity
            'semantic_distance': 0.1,    # Moderate weight for semantic distance
            'position_ratio': 0.05       # Low weight for position
        }

        for feature_name, weight in enhanced_features.items():
            if feature_name in features:
                # Normalize feature to [0, 1] range
                feature = features[feature_name]
                if feature.dim() > 2:  # Handle multi-dimensional features like position_encoding
                    feature = feature.mean(dim=-1)  # Average across last dimension
                
                # Normalize to [0, 1]
                feature_min = feature.min()
                feature_max = feature.max()
                if feature_max > feature_min:
                    feature_normalized = (feature - feature_min) / (feature_max - feature_min)
                else:
                    feature_normalized = feature
                
                combined_scores += weight * feature_normalized

        # Strategy-specific features
        strategy_features = self.mode_controller.strategy_encoder.encode_strategy_features(
            features, strategy_weights
        )
        
        # Add strategy effectiveness
        if 'strategy_effectiveness' in strategy_features:
            combined_scores += 0.2 * strategy_features['strategy_effectiveness']

        # Top-k selection with exact budget
        scores_flat = combined_scores.reshape(-1)
        mask_flat = attention_mask.reshape(-1)

        valid_scores = scores_flat[mask_flat.bool()]
        n_valid = mask_flat.sum().item()
        n_select = max(1, int(self.config.token_budget * n_valid))

        if len(valid_scores) > n_select:
            threshold = torch.topk(valid_scores, n_select).values[-1]
            selection_mask = ((combined_scores >= threshold) & attention_mask.bool()).float()

            # Exact budget enforcement
            actual_selected = selection_mask.sum().item()
            if actual_selected > n_select:
                selected_pos = torch.nonzero(selection_mask.reshape(-1)).squeeze(-1)
                excess = int(actual_selected - n_select)
                drop_indices = selected_pos[torch.randperm(len(selected_pos))[:excess]]
                selection_mask.reshape(-1)[drop_indices] = 0.0
        else:
            selection_mask = attention_mask.float()

        return selection_mask, strategy_weights

    def train_step(self, batch: Dict) -> Dict:
        """Fast training step"""
        # Tokenize
        inputs = self.tokenizer(
            batch['text'],
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=self.config.max_seq_length
        ).to(self.device)

        # Fast feature extraction
        features = self.extract_fast_features(inputs.input_ids, inputs.attention_mask)

        # Token selection
        if self.epoch < self.config.warmup_epochs:
            token_mask = inputs.attention_mask.float()
            strategy_weights = np.ones(4) / 4  # Uniform
        else:
            token_mask, strategy_weights = self.select_tokens_fast_mode(
                features, inputs.attention_mask, inputs.input_ids
            )

        # Forward pass
        outputs = self.model(inputs.input_ids, attention_mask=inputs.attention_mask, output_hidden_states=True)

        # Compute loss
        shift_logits = outputs.logits[:, :-1, :].contiguous()
        shift_labels = inputs.input_ids[:, 1:].contiguous()
        shift_mask = token_mask[:, 1:].contiguous()
        shift_attention = inputs.attention_mask[:, 1:].contiguous()

        loss_per_token = F.cross_entropy(
            shift_logits.reshape(-1, shift_logits.size(-1)),
            shift_labels.reshape(-1),
            reduction='none'
        ).reshape(shift_labels.shape)

        masked_loss = loss_per_token * shift_mask * shift_attention.float()
        n_selected = (shift_mask * shift_attention.float()).sum()

        if n_selected > 0:
            loss = masked_loss.sum() / n_selected
        else:
            loss = loss_per_token.mean()

        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
        self.optimizer.step()
        self.optimizer.zero_grad()

        # Clear MPS cache
        if self.device.type == 'mps':
            torch.mps.empty_cache()

        # Update MODE controller
        if self.epoch >= self.config.warmup_epochs:
            success_signal = np.exp(-loss.item())  # Convert loss to success
            selection_ratio = (n_selected / max(shift_attention.sum(), 1)).item()

            self.mode_controller.update_training_state(
                loss.item(), selection_ratio, strategy_weights, success_signal, inputs.input_ids
            )

            # Update token memory (simplified)
            if hasattr(outputs, 'hidden_states') and outputs.hidden_states:
                hidden = outputs.hidden_states[-1]
                for b in range(inputs.input_ids.size(0)):
                    for pos in range(inputs.input_ids.size(1) - 1):
                        if token_mask[b, pos] > 0.5:
                            context = hidden[b, pos]
                            next_token = inputs.input_ids[b, pos + 1].item()
                            self.token_memory.add_context_token_pair(
                                context, next_token, success_signal
                            )

        # Record metrics
        actual_loss = loss.item()
        self.losses.append(actual_loss)
        self.step += 1

        return {
            'loss': actual_loss,
            'selection_ratio': (n_selected / max(shift_attention.sum(), 1)).item(),
            'strategy_weights': strategy_weights
        }

    def train_epoch(self, train_loader: DataLoader) -> Dict:
        """Train one epoch"""
        self.model.train()
        epoch_losses = []
        epoch_selections = []

        progress_bar = tqdm(train_loader, desc=f"Epoch {self.epoch+1}")

        for batch in progress_bar:
            metrics = self.train_step(batch)
            epoch_losses.append(metrics['loss'])
            epoch_selections.append(metrics['selection_ratio'])

            if self.step % self.config.log_interval == 0:
                strategy_interp = self.mode_controller.get_strategy_interpretation()
                dominant = strategy_interp.get('dominant_strategy', 'unknown')

                msg = f"Loss: {metrics['loss']:.4f} | Sel: {metrics['selection_ratio']:.1%}"
                msg += f" | Dom: {dominant} | Mem: {self.token_memory.size()}"

                progress_bar.set_postfix_str(msg)

        return {
            'avg_loss': np.mean(epoch_losses),
            'avg_selection': np.mean(epoch_selections)
        }

    @torch.no_grad()
    def evaluate_perplexity(self, texts: List[str]) -> float:
        """Fast perplexity evaluation"""
        self.model.eval()

        total_nll = 0.0
        total_tokens = 0

        for text in tqdm(texts, desc="Evaluating", leave=False):
            encodings = self.tokenizer(text, return_tensors='pt', truncation=False)
            input_ids = encodings.input_ids

            if input_ids.size(1) < 2:
                continue

            # Simple sliding window
            max_len = min(self.config.eval_max_length, input_ids.size(1))
            input_chunk = input_ids[:, :max_len].to(self.device)

            outputs = self.model(input_chunk)
            shift_logits = outputs.logits[:, :-1, :].contiguous()
            shift_labels = input_chunk[:, 1:].contiguous()

            loss = F.cross_entropy(
                shift_logits.reshape(-1, shift_logits.size(-1)),
                shift_labels.reshape(-1),
                reduction='sum'
            )

            total_nll += loss.item()
            total_tokens += shift_labels.numel()

        return np.exp(total_nll / total_tokens) if total_tokens > 0 else float('inf')

    def train(self, train_loader: DataLoader, val_texts: List[str]) -> Dict:
        """Full training loop"""
        print(f"\nStarting Integrated MODE Training: {self.name}")

        for epoch in range(self.config.epochs):
            self.epoch = epoch

            print(f"\n{'='*70}")
            print(f"Epoch {epoch+1}/{self.config.epochs}")
            print(f"{'='*70}")

            # Train
            train_metrics = self.train_epoch(train_loader)

            # Evaluate
            val_ppl = self.evaluate_perplexity(val_texts)

            print(f"Train Loss: {train_metrics['avg_loss']:.4f}")
            print(f"Val PPL: {val_ppl:.2f}")
            print(f"Avg Selection: {train_metrics['avg_selection']:.1%}")

            # Strategy analysis
            strategy_interp = self.mode_controller.get_strategy_interpretation()
            if strategy_interp:
                print(f"Dominant Strategy: {strategy_interp['dominant_strategy']}")
                print(f"Strategy Weights: {strategy_interp['current_weights']}")

            # Early stopping
            if val_ppl < self.best_val_ppl:
                self.best_val_ppl = val_ppl
                self.patience_counter = 0
                print("New best perplexity!")
            else:
                self.patience_counter += 1
                if self.patience_counter >= self.config.patience:
                    print("Early stopping!")
                    break

        return {'best_val_ppl': self.best_val_ppl}


# ============================================================================
# DATA LOADING
# ============================================================================

class SimpleTextDataset(Dataset):
    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return {'text': self.samples[idx]['text']}


def load_simple_data(config: IntegratedMODEConfig):
    """Load dataset efficiently"""
    dataset = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train', streaming=True)

    samples = []
    for sample in dataset:
        text = sample['text'].strip()
        if text and len(text) >= config.min_text_length:
            samples.append(sample)

        if len(samples) >= config.train_samples + config.val_samples:
            break

    train_samples = samples[:config.train_samples]
    val_samples = samples[config.train_samples:]

    train_dataset = SimpleTextDataset(train_samples)
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=0)

    val_texts = [s['text'] for s in val_samples]

    return train_loader, val_texts


# ============================================================================
# MAIN EXPERIMENT
# ============================================================================

def run_integrated_mode_experiment():
    """Run integrated MODE with binary hypernetwork"""
    config = IntegratedMODEConfig()

    print("="*70)
    print("INTEGRATED MODE with Binary State Hypernetwork")
    print("="*70)
    print(f"Binary State Encoding: {config.state_dim}D")
    print(f"Memory Size: {config.memory_size}")
    print(f"Strategy Count: {config.n_strategies}")
    print(f"Token Budget: {config.token_budget*100:.1f}%")
    print("="*70)

    # Load data
    train_loader, val_texts = load_simple_data(config)

    # Run integrated MODE
    trainer = IntegratedMODETrainer(config, "Integrated-MODE")
    results = trainer.train(train_loader, val_texts)

    print(f"\n{'='*70}")
    print("FINAL RESULTS")
    print(f"{'='*70}")
    print(f"Best Validation Perplexity: {results['best_val_ppl']:.2f}")

    # Final analysis
    final_interp = trainer.mode_controller.get_strategy_interpretation()
    print(f"Final Strategy Analysis:")
    print(f"  Dominant Strategy: {final_interp.get('dominant_strategy', 'unknown')}")
    print(f"  Strategy Weights: {final_interp.get('current_weights', {})}")
    print(f"  Memory Size: {trainer.token_memory.size()}")
    print(f"  Hypernetwork Memory: {final_interp.get('memory_size', 0)}")

    return results


if __name__ == "__main__":
    results = run_integrated_mode_experiment()
