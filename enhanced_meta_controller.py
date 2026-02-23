import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Set, Any
from torch.utils.data import Dataset, DataLoader, Subset
from collections import Counter, defaultdict
import matplotlib.pyplot as plt
from tqdm import tqdm

class Lambda(nn.Module):
    """Helper class for custom lambda functions in nn.Sequential"""
    def __init__(self, func):
        super().__init__()
        self.func = func
        
    def forward(self, x):
        return self.func(x)

class EnhancedHypernetworkController(nn.Module):
    """Enhanced hypernetwork with cross-strategy attention"""
    def __init__(self, strategies, input_dim=24, hidden_dim=64):
        super(EnhancedHypernetworkController, self).__init__()
        
        self.strategies = strategies
        self.num_strategies = len(strategies)
        
        # Encoder for learning state
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU()
        )
        
        # Simpler architecture to avoid dimension mismatches
        # Strategy weight generator directly from encoded state
        self.weight_generator = nn.Linear(hidden_dim, self.num_strategies)
        
        # Training stage-aware temperature predictor
        self.temperature_predictor = nn.Sequential(
            nn.Linear(hidden_dim + 1, hidden_dim // 2),  # +1 for training progress
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),  # Output between 0 and 1
            Lambda(lambda x: 0.1 + 1.9 * x)  # Scale to [0.1, 2.0]
        )
        
        # Value prediction for reward estimation
        self.value_predictor = nn.Linear(hidden_dim, 1)
        
        # Initialize weights properly
        self._init_weights()
    
    def _init_weights(self):
        """Initialize with careful scaling"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, nonlinearity='relu')
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
                    
        # Special bias initialization for the weight generator
        # Give uncertainty a slight initial advantage
        with torch.no_grad():
            u_idx = self.strategies.index('S_U') if 'S_U' in self.strategies else 0
            self.weight_generator.bias.data[u_idx] = 0.5
    
    def forward(self, state, training_progress=0.0):
        """
        Simplified forward pass that avoids the complex attention mechanism
        
        Args:
            state: Tensor representing the learning state
            training_progress: Float from 0 to 1 indicating training progress
            
        Returns:
            strategy_weights, temperature, value
        """
        try:
            # Encode state
            encoded_state = self.encoder(state)
            
            # Generate strategy weights directly from encoded state
            raw_weights = self.weight_generator(encoded_state)
            
            # Predict temperature based on state and training progress
            progress_tensor = torch.tensor([[training_progress]], device=state.device)
            temp_input = torch.cat([encoded_state, progress_tensor], dim=1)
            temperature = self.temperature_predictor(temp_input)
            
            # Apply softmax with temperature
            strategy_weights = F.softmax(raw_weights / temperature, dim=-1)
            
            # Predict state value
            value = self.value_predictor(encoded_state)
            
            return strategy_weights, temperature, value
            
        except Exception as e:
            print(f"Exception in forward pass: {e}")
            # Fallback to safe defaults
            default_weights = torch.ones(1, self.num_strategies, device=state.device) / self.num_strategies
            default_temp = torch.tensor([[0.5]], device=state.device) 
            default_value = torch.tensor([[0.0]], device=state.device)
            
            return default_weights, default_temp, default_value

class LearningState:
    """
    Encapsulates the current state of the learning process
    """
    def __init__(
        self,
        accuracy: float = 0.0,
        loss: float = 0.0,
        epoch: int = 0,
        total_epochs: int = 100,
        dataset_size: int = 0,
        budget_size: int = 0,
        class_distribution: Optional[Dict[int, float]] = None,
        strategy_performances: Optional[Dict[str, float]] = None,
        accuracy_history: Optional[List[float]] = None,
        weight_history: Optional[List[Dict[str, float]]] = None,
        feature_statistics: Optional[Dict[str, float]] = None
    ):
        self.accuracy = accuracy
        self.loss = loss
        self.epoch = epoch
        self.total_epochs = total_epochs
        self.dataset_size = dataset_size
        self.budget_size = budget_size
        self.class_distribution = class_distribution or {}
        self.strategy_performances = strategy_performances or {}
        self.accuracy_history = accuracy_history or []
        self.weight_history = weight_history or []
        self.feature_statistics = feature_statistics or {}
    
    def to_tensor(self) -> torch.Tensor:
        """
        Convert learning state to tensor representation
        
        Returns:
            Tensor representation of the learning state
        """
        # Create base features
        features = [
            self.accuracy,
            self.loss,
            self.epoch / self.total_epochs,  # Normalized progress
            self.dataset_size / self.budget_size if self.budget_size > 0 else 0.0,  # Dataset fill ratio
        ]
        
        # Add class distribution features (10 for CIFAR-10)
        class_dist = [0.0] * 10
        for c, count in self.class_distribution.items():
            if 0 <= c < 10:
                class_dist[c] = count / self.dataset_size if self.dataset_size > 0 else 0.0
        features.extend(class_dist)
        
        # Add recent performance trend
        if len(self.accuracy_history) >= 3:
            recent = self.accuracy_history[-3:]
            trend = recent[-1] - recent[0]  # Improvement over last 3 epochs
            features.append(trend)
        else:
            features.append(0.0)
        
        # Add strategy performance indicators
        strategy_perf = []
        for strategy, perf in self.strategy_performances.items():
            strategy_perf.append(perf)
        
        # Ensure fixed length by padding if necessary
        while len(strategy_perf) < 6:  # Assuming max 6 strategies
            strategy_perf.append(0.0)
        
        # Truncate if too many
        strategy_perf = strategy_perf[:6]
        
        features.extend(strategy_perf)
        
        # Add feature statistics
        feature_stats = [
            self.feature_statistics.get("feature_diversity", 0.5),
            self.feature_statistics.get("feature_redundancy", 0.5),
            self.feature_statistics.get("class_separation", 0.5)
        ]
        features.extend(feature_stats)
        
        # Create tensor
        return torch.FloatTensor(features)


class ExperienceBuffer:
    """
    Buffer for storing experiences for hypernetwork training
    """
    def __init__(self, capacity: int = 1000):
        self.capacity = capacity
        self.states = []
        self.weights = []
        self.rewards = []
        self.next_states = []
        self.training_progress = []
        self.position = 0
        self.size = 0
    
    def add(self, state, weights, reward, next_state, progress):
        """Add experience to buffer"""
        if self.size < self.capacity:
            self.states.append(state)
            self.weights.append(weights)
            self.rewards.append(reward)
            self.next_states.append(next_state)
            self.training_progress.append(progress)
            self.size += 1
        else:
            self.states[self.position] = state
            self.weights[self.position] = weights
            self.rewards[self.position] = reward
            self.next_states[self.position] = next_state
            self.training_progress[self.position] = progress
            self.position = (self.position + 1) % self.capacity
    
    def sample(self, batch_size: int):
        """Sample batch of experiences"""
        if self.size < batch_size:
            indices = np.arange(self.size)
        else:
            indices = np.random.choice(self.size, batch_size, replace=False)
        
        states = torch.stack([self.states[i] for i in indices])
        weights = torch.stack([self.weights[i] for i in indices])
        rewards = torch.FloatTensor([self.rewards[i] for i in indices])
        next_states = torch.stack([self.next_states[i] for i in indices])
        progress = torch.FloatTensor([self.training_progress[i] for i in indices])
        
        return states, weights, rewards, next_states, progress
    
    def __len__(self):
        return self.size


class EnhancedHyperNetworkTrainer:
    """
    Trainer for hypernetwork using reinforcement learning
    """
    def __init__(
        self,
        hypernetwork: EnhancedHypernetworkController,
        strategies: List[str],
        lr: float = 0.001,
        gamma: float = 0.99,
        buffer_capacity: int = 1000,
        batch_size: int = 32,
        target_update_freq: int = 10,
        device=None
    ):
        self.hypernetwork = hypernetwork
        self.strategies = strategies
        self.gamma = gamma  # Discount factor
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.update_count = 0
        
        # Set device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else 
                                      "mps" if hasattr(torch, 'backends') and 
                                            hasattr(torch.backends, 'mps') and 
                                            torch.backends.mps.is_available() else 
                                      "cpu")
        else:
            self.device = device
        
        self.hypernetwork.to(self.device)
        
        # Create target network for stable learning
        self.target_network = EnhancedHypernetworkController(strategies)
        self.target_network.load_state_dict(self.hypernetwork.state_dict())
        self.target_network.to(self.device)
        self.target_network.eval()
        
        # Experience buffer
        self.buffer = ExperienceBuffer(capacity=buffer_capacity)
        
        # Optimizer
        self.optimizer = torch.optim.Adam(self.hypernetwork.parameters(), lr=lr)
        
        # Metrics
        self.train_losses = []
    
    def compute_reward(
        self,
        current_accuracy: float,
        previous_accuracy: float,
        current_loss: float,
        previous_loss: float,
        diversity_score: float,
        entropy: float,
        current_size: int,
        previous_size: int
    ) -> float:
        """
        Improved reward function with better sensitivity and normalization
        """
        # Accuracy improvement with non-linear scaling (emphasizes small improvements)
        acc_improvement = current_accuracy - previous_accuracy
        acc_reward = np.tanh(acc_improvement * 10.0)  # Tanh prevents extreme values
        
        # Loss reduction with diminishing returns
        loss_reduction = previous_loss - current_loss
        loss_reward = np.tanh(loss_reduction * 5.0)
        
        # Efficiency reward (performance per sample)
        samples_added = current_size - previous_size
        if samples_added > 0:
            efficiency = acc_improvement / (samples_added / current_size)
            efficiency_reward = np.tanh(efficiency * 3.0)
        else:
            efficiency_reward = 0.0
        
        # Exploration bonus with diminishing returns at higher entropy values
        exploration_bonus = 0.2 * min(entropy, 0.5)
        
        # Combine rewards with weighted components
        reward = 0.5 * acc_reward + 0.3 * loss_reward + 0.1 * efficiency_reward + 0.1 * exploration_bonus
        
        # Ensure reward has good range for learning
        return reward  # No need to clamp to [-1, 1] due to tanh
    
    def compute_entropy(self, weights: torch.Tensor) -> float:
        """
        Compute entropy of weight distribution as exploration measure
        
        Args:
            weights: Strategy weights tensor
            
        Returns:
            entropy: Entropy of weight distribution
        """
        # Add small epsilon to avoid log(0)
        eps = 1e-10
        entropy = -torch.sum(weights * torch.log(weights + eps))
        return entropy.item()
    
    def update(self, state_tensor, previous_weights, reward, next_state_tensor, training_progress):
        """
        Update hypernetwork using reinforcement learning
        
        Args:
            state_tensor: Current state tensor
            previous_weights: Previously predicted weights
            reward: Reward signal
            next_state_tensor: Next state tensor
            training_progress: Training progress (0-1)
            
        Returns:
            loss: Training loss
        """
        # Add experience to buffer
        self.buffer.add(
            state_tensor,
            previous_weights,
            reward,
            next_state_tensor,
            training_progress
        )
        
        # Skip if not enough experiences
        if len(self.buffer) < self.batch_size:
            return 0.0
        
        # Sample batch of experiences
        states, weights, rewards, next_states, progress = self.buffer.sample(self.batch_size)
        states = states.to(self.device)
        weights = weights.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        progress = progress.to(self.device)
        
        # Get current state value predictions
        _, _, values = self.hypernetwork(states, progress)
        values = values.squeeze(-1)
        
        # Get next state values from target network
        with torch.no_grad():
            _, _, next_values = self.target_network(next_states, progress)
            next_values = next_values.squeeze(-1)
        
        # Compute target values (TD learning)
        target_values = rewards + self.gamma * next_values
        
        # Compute value loss (MSE)
        value_loss = F.mse_loss(values, target_values)
        
        # Compute policy loss (encourage similar strategy choices that led to rewards)
        predicted_weights, _, _ = self.hypernetwork(states, progress)
        
        # Compute advantage as rewards - values
        advantages = rewards - values.detach()
        
        # Policy gradient loss: encourages higher probability for actions that led to higher advantage
        policy_loss = -torch.mean(torch.sum(torch.log(predicted_weights + 1e-10) * weights, dim=1) * advantages)
        
        # Add entropy regularization to encourage exploration
        entropy = -torch.mean(torch.sum(predicted_weights * torch.log(predicted_weights + 1e-10), dim=1))
        entropy_loss = -0.01 * entropy  # Small coefficient to balance exploration
        
        # Total loss
        loss = value_loss + policy_loss + entropy_loss
        
        # Update hypernetwork
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.hypernetwork.parameters(), 1.0)  # Prevent exploding gradients
        self.optimizer.step()
        
        # Periodically update target network
        self.update_count += 1
        if self.update_count % self.target_update_freq == 0:
            self.target_network.load_state_dict(self.hypernetwork.state_dict())
        
        # Record loss
        self.train_losses.append(loss.item())
        
        return loss.item()
    
    def predict_weights(self, state: LearningState) -> Tuple[Dict[str, float], float, float]:
        """
        Predict strategy weights and temperature based on learning state
        
        Args:
            state: Current learning state
            
        Returns:
            weights: Dictionary mapping strategy names to weights
            temperature: Temperature parameter for exploration
            value: Predicted state value
        """
        state_tensor = state.to_tensor().to(self.device)
        progress = float(state.epoch) / state.total_epochs
        
        # Forward pass through hypernetwork
        with torch.no_grad():
            strategy_weights, temperature, value = self.hypernetwork(
                state_tensor.unsqueeze(0), 
                torch.tensor([[progress]], device=self.device)
            )
        
        # Convert to dictionary
        weights = {s: w.item() for s, w in zip(self.strategies, strategy_weights[0])}
        
        return weights, temperature.item(), value.item()
    
    def plot_losses(self):
        """Plot training losses"""
        plt.figure(figsize=(10, 5))
        plt.plot(self.train_losses)
        plt.title('Hypernetwork Training Loss')
        plt.xlabel('Update Step')
        plt.ylabel('Loss')
        plt.yscale('log')
        plt.grid(True)
        return plt.gcf()


class EnhancedMetaController:
    """
    Advanced meta-controller using hypernetwork for coreset selection
    """
    def __init__(
        self,
        strategies: List[str] = ["S_U", "S_D", "S_C", "S_B", "S_G", "S_F"],
        total_epochs: int = 200,
        budget_ratio: float = 0.1,
        device=None
    ):
        """
        Initialize the enhanced MODE meta-controller with hypernetwork
        
        Args:
            strategies: List of strategy names
            total_epochs: Total number of epochs for training
            budget_ratio: Ratio of budget to full dataset size
            device: Device to use for computation
        """
        self.strategies = strategies
        self.total_epochs = total_epochs
        self.budget_ratio = budget_ratio
        
        # Set initial temperature based on budget
        if budget_ratio <= 0.1:
            self.initial_temperature = 0.4  # Lower temperature for smaller budgets
            self.min_temperature = 0.1
        elif budget_ratio <= 0.3:
            self.initial_temperature = 0.7
            self.min_temperature = 0.2
        else:
            self.initial_temperature = 0.9  # Higher temperature for larger budgets
            self.min_temperature = 0.3
        
        # Set device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else 
                                      "mps" if hasattr(torch, 'backends') and 
                                            hasattr(torch.backends, 'mps') and 
                                            torch.backends.mps.is_available() else 
                                      "cpu")
        else:
            self.device = device
        
        # Initialize hypernetwork
        self.hypernetwork = EnhancedHypernetworkController(strategies)
        
        # Initialize trainer
        self.trainer = EnhancedHyperNetworkTrainer(
            hypernetwork=self.hypernetwork,
            strategies=strategies,
            device=self.device
        )
        
        # State tracking
        self.current_state = None
        self.previous_state = None
        
        # Initialize strategy weights based on budget
        self.current_weights = self._initialize_weights_by_budget()
        
        # Performance tracking
        self.performance_history = []
        self.sample_counts = {s: 0 for s in strategies}
        self.weight_history = [self.current_weights.copy()]
        self.temperature_history = []
        self.state_value_history = []
        self.explanations = []
    
    def _initialize_weights_by_budget(self):
        """Initialize strategy weights based on budget ratio"""
        if self.budget_ratio <= 0.1:
            # For very small budgets, prioritize uncertainty and boundary
            initial_weights = {
                "S_U": 0.4,  # High uncertainty weight
                "S_D": 0.2, 
                "S_C": 0.1,
                "S_B": 0.2,  # Decent boundary weight
                "S_G": 0.05,
                "S_F": 0.05
            }
        elif self.budget_ratio <= 0.3:
            # For medium budgets, more balanced
            initial_weights = {
                "S_U": 0.3,
                "S_D": 0.2,
                "S_C": 0.2,
                "S_B": 0.15,
                "S_G": 0.1,
                "S_F": 0.05
            }
        else:
            # For large budgets, more exploration
            initial_weights = {
                "S_U": 0.25, 
                "S_D": 0.25,
                "S_C": 0.15,
                "S_B": 0.15,
                "S_G": 0.1,
                "S_F": 0.1
            }
        
        # Fill in any missing strategies with small weights
        weights = {}
        for s in self.strategies:
            weights[s] = initial_weights.get(s, 0.05)
        
        # Normalize weights
        total = sum(weights.values())
        return {s: w/total for s, w in weights.items()}
    
    def compute_temperature_schedule(self, epoch):
        """Dynamic temperature schedule based on budget and progress"""
        progress = epoch / self.total_epochs
        
        # Base temperature follows cosine annealing schedule
        base_temp = self.min_temperature + 0.5 * (self.initial_temperature - self.min_temperature) * \
                    (1 + np.cos(np.pi * progress))
        
        # Small budgets: Decrease temperature faster
        if self.budget_ratio <= 0.1:
            temp = base_temp * (0.8 - 0.4 * progress)  # More exploitation
            
        # Medium budgets: Balanced approach
        elif self.budget_ratio <= 0.3:
            temp = base_temp * (1.0 - 0.3 * progress)
            
        # Large budgets: Maintain higher exploration
        else:
            temp = base_temp * (1.2 - 0.2 * progress)  # More exploration
            
        # Apply minimum constraint
        return max(self.min_temperature, temp)
    
    def adjust_strategy_importance(self, weights, epoch):
        """Apply curriculum learning to strategy weights"""
        progress = epoch / self.total_epochs
        
        # Early stage (0-30% of training)
        if progress < 0.3:
            # Boost class balance and diversity early
            weights["S_C"] = weights.get("S_C", 0.0) * 1.5
            weights["S_D"] = weights.get("S_D", 0.0) * 1.3
            
        # Middle stage (30-70% of training)
        elif progress < 0.7:
            # Boost uncertainty and boundary
            weights["S_U"] = weights.get("S_U", 0.0) * 1.4
            weights["S_B"] = weights.get("S_B", 0.0) * 1.2
            
        # Late stage (70-100% of training)
        else:
            # Boost uncertainty and gradient
            weights["S_U"] = weights.get("S_U", 0.0) * 1.5
            weights["S_G"] = weights.get("S_G", 0.0) * 1.3
            # Reduce class balance importance
            weights["S_C"] = weights.get("S_C", 0.0) * 0.7
            
        # Normalize weights
        total = sum(weights.values())
        return {k: v/total for k, v in weights.items()}
    
    def update_state(self, 
                     epoch: int,
                     accuracy: float,
                     loss: float,
                     class_distribution: Dict[int, int],
                     dataset_size: int,
                     budget_size: int,
                     strategy_performances: Optional[Dict[str, float]] = None,
                     feature_statistics: Optional[Dict[str, float]] = None
                    ) -> LearningState:
        """
        Update the learning state with current metrics
        
        Args:
            epoch: Current epoch number
            accuracy: Current model accuracy
            loss: Current model loss
            class_distribution: Distribution of classes in current coreset
            dataset_size: Size of current coreset
            budget_size: Maximum coreset size
            strategy_performances: Optional direct performance metrics for each strategy
            feature_statistics: Optional feature space statistics
            
        Returns:
            Updated learning state
        """
        # Record performance
        performance = {
            "epoch": epoch,
            "accuracy": accuracy,
            "loss": loss,
            "dataset_size": dataset_size
        }
        self.performance_history.append(performance)
        
        # Save previous state if it exists
        if self.current_state is not None:
            self.previous_state = self.current_state
        
        # Create learning state
        self.current_state = LearningState(
            accuracy=accuracy,
            loss=loss,
            epoch=epoch,
            total_epochs=self.total_epochs,
            dataset_size=dataset_size,
            budget_size=budget_size,
            class_distribution=class_distribution,
            strategy_performances=strategy_performances or {},
            accuracy_history=[p["accuracy"] for p in self.performance_history],
            weight_history=self.weight_history,
            feature_statistics=feature_statistics or {}
        )
        
        return self.current_state
    
    def get_sample_allocation(self, n_samples) -> Dict[str, int]:
        """Enhanced sample allocation with ensemble approach"""
        # Get hypernetwork predictions if state exists
        if self.current_state is not None:
            weights, temperature, value = self.trainer.predict_weights(self.current_state)
            
            # Apply curriculum adjustments
            adjusted_weights = self.adjust_strategy_importance(
                weights, self.current_state.epoch
            )
            
            # Record metrics
            self.current_weights = adjusted_weights
            self.weight_history.append(adjusted_weights.copy())
            self.temperature_history.append(temperature)
            self.state_value_history.append(value)
            
            # Generate explanation
            explanation = self._generate_explanation(adjusted_weights, temperature, value)
            self.explanations.append(explanation)
            
            print(f"Strategy weights: {adjusted_weights}")
            print(f"Temperature: {temperature:.4f}")
            print(f"State value: {value:.4f}")
            print(f"Explanation: {explanation}")
        else:
            # Default to initialized weights if no state
            adjusted_weights = self.current_weights
            
        # Uncertainty sampling gets special treatment for small budgets
        if self.budget_ratio <= 0.2:
            # For small budgets, guarantee at least 40% to uncertainty
            if adjusted_weights.get("S_U", 0) < 0.4:
                # Boost uncertainty while preserving other relative weights
                uncertainty_boost = 0.4 - adjusted_weights.get("S_U", 0)
                scale_factor = (1.0 - 0.4) / (1.0 - adjusted_weights.get("S_U", 0))
                
                # Create a new dictionary to avoid modifying the original weights
                boosted_weights = {}
                for strategy in adjusted_weights:
                    if strategy == "S_U":
                        boosted_weights[strategy] = 0.4
                    else:
                        boosted_weights[strategy] = adjusted_weights[strategy] * scale_factor
                
                allocation_weights = boosted_weights
            else:
                allocation_weights = adjusted_weights
        else:
            allocation_weights = adjusted_weights
        
        # For very small sample counts, use more focused allocation
        if n_samples < 50:
            # Concentrate on top 2 strategies
            top_strategies = sorted(allocation_weights.items(), key=lambda x: x[1], reverse=True)[:2]
            top_sum = sum(w for _, w in top_strategies)
            focused_weights = {s: 0.0 for s in allocation_weights}
            
            for s, w in top_strategies:
                focused_weights[s] = w / top_sum
                
            allocation_weights = focused_weights
        
        # Allocate samples
        allocation = {}
        remaining = n_samples
        
        # First pass - allocate integer samples
        for strategy, weight in allocation_weights.items():
            strategy_samples = int(n_samples * weight)
            allocation[strategy] = strategy_samples
            remaining -= strategy_samples
        
        # Distribute remaining samples by priority
        strategies_sorted = sorted(allocation_weights.items(), key=lambda x: x[1], reverse=True)
        for strategy, _ in strategies_sorted:
            if remaining <= 0:
                break
            allocation[strategy] += 1
            remaining -= 1
        
        # Update sample counts
        for strategy, count in allocation.items():
            self.sample_counts[strategy] += count
        
        return allocation
    
    def update_weights(self, diversity_score=0.5):
        """Advanced hypernetwork update with historical context"""
        # Skip if not enough history
        if self.previous_state is None or self.current_state is None:
            return 0.0
        
        # Check for performance plateau
        plateaued = False
        if len(self.performance_history) >= 5:
            recent_accs = [p["accuracy"] for p in self.performance_history[-5:]]
            if max(recent_accs) - min(recent_accs) < 0.01:
                plateaued = True
        
        # Get state tensors
        prev_state_tensor = self.previous_state.to_tensor()
        curr_state_tensor = self.current_state.to_tensor()
        
        # Get previous weights as tensor
        prev_weights = torch.FloatTensor([self.weight_history[-2][s] for s in self.strategies])
        
        # Compute reward
        if len(self.performance_history) >= 2:
            # Standard reward calculation
            reward = self.trainer.compute_reward(
                current_accuracy=self.performance_history[-1]["accuracy"],
                previous_accuracy=self.performance_history[-2]["accuracy"],
                current_loss=self.performance_history[-1]["loss"],
                previous_loss=self.performance_history[-2]["loss"],
                diversity_score=diversity_score,
                entropy=self.trainer.compute_entropy(prev_weights),
                current_size=self.performance_history[-1]["dataset_size"],
                previous_size=self.performance_history[-2]["dataset_size"]
            )
            
            # Boost reward if breaking through plateau
            if plateaued and reward > 0.1:
                reward *= 2.0
            
        else:
            reward = 0.0
        
        # Add training progress to state
        training_progress = float(self.current_state.epoch) / self.current_state.total_epochs
        
        # Update hypernetwork with reward and progress information
        loss = self.trainer.update(
            prev_state_tensor,
            prev_weights,
            reward,
            curr_state_tensor,
            training_progress
        )
        
        print(f"Hypernetwork update: reward={reward:.4f}, loss={loss:.4f}, {'plateaued' if plateaued else 'normal'}")
        
        return loss
    
    def _generate_explanation(self, weights: Dict[str, float], temperature: float, value: float) -> str:
        """
        Generate explanation for strategy decision
        
        Args:
            weights: Strategy weights
            temperature: Temperature parameter
            value: Predicted state value
            
        Returns:
            explanation: Textual explanation of strategy decision
        """
        # Sort strategies by weight
        sorted_strategies = sorted(weights.items(), key=lambda x: x[1], reverse=True)
        top_strategies = sorted_strategies[:2]
        
        # Determine exploration-exploitation balance
        if temperature < 0.3:
            mode = "exploitation"
        elif temperature < 0.7:
            mode = "balanced"
        else:
            mode = "exploration"
        
        # Determine training stage
        if self.current_state is None:
            stage = "initial"
        else:
            progress = self.current_state.epoch / self.current_state.total_epochs
            if progress < 0.3:
                stage = "early"
            elif progress < 0.7:
                stage = "middle"
            else:
                stage = "late"
        
        # Generate explanation
        explanation = f"At {stage} training stage, focusing on "
        explanation += ", ".join([f"{s} ({w:.2f})" for s, w in top_strategies])
        explanation += f" with {mode} mode (temp={temperature:.2f})"
        
        # Add trend if available
        if len(self.weight_history) >= 3:
            increasing = []
            decreasing = []
            
            for strategy in self.strategies:
                if strategy in self.weight_history[-1] and strategy in self.weight_history[-3]:
                    hist = [h.get(strategy, 0.0) for h in self.weight_history[-3:]]
                    if hist[-1] > hist[0] + 0.05:
                        increasing.append(strategy)
                    elif hist[0] > hist[-1] + 0.05:
                        decreasing.append(strategy)
            
            if increasing:
                explanation += f". Increasing emphasis on {', '.join(increasing)}"
            if decreasing:
                explanation += f". Decreasing emphasis on {', '.join(decreasing)}"
        
        return explanation
    
    def select_samples_by_strategy(self, model, strategy, scores, n_samples, selected_indices):
        """Enhanced sample selection with dynamic rescoring"""
        # If selecting uncertainty samples
        if strategy == "S_U":
            # Get top 3x uncertainty candidates
            candidates = self._get_top_candidates(scores, min(n_samples * 3, len(scores) - len(selected_indices)), selected_indices)
            
            if len(candidates) > n_samples:
                # Apply additional boundary criterion to refine selection
                boundary_scores = self._boundary_scores(model, candidates)
                
                # Combine uncertainty and boundary
                combined_scores = 0.7 * self._normalize(scores[candidates]) + \
                                0.3 * self._normalize(boundary_scores)
                
                # Take final selection from candidates
                top_idx = np.argsort(combined_scores)[-n_samples:]
                return [candidates[i] for i in top_idx]
            return candidates
            
        # If selecting diversity samples    
        elif strategy == "S_D":
            # Get top 3x diversity candidates
            candidates = self._get_top_candidates(scores, min(n_samples * 3, len(scores) - len(selected_indices)), selected_indices)
            
            if len(candidates) > n_samples:
                # Apply class balance to ensure diverse selection
                class_scores = self._class_balance_scores(selected_indices, candidates)
                
                # Combine diversity and class balance
                combined_scores = 0.8 * self._normalize(scores[candidates]) + \
                                0.2 * self._normalize(class_scores)
                
                top_idx = np.argsort(combined_scores)[-n_samples:]
                return [candidates[i] for i in top_idx]
            return candidates
        
        # For other strategies, use standard selection
        return self._get_top_candidates(scores, n_samples, selected_indices)
    
    def _get_top_candidates(self, scores, n_samples, exclude_indices):
        """Get top-scoring samples excluding already selected indices"""
        available = np.ones(len(scores), dtype=bool)
        available[exclude_indices] = False
        
        if not np.any(available):
            return np.array([])
            
        avail_indices = np.where(available)[0]
        avail_scores = scores[available]
        
        if len(avail_indices) <= n_samples:
            return avail_indices
        
        top_idx = np.argsort(avail_scores)[-n_samples:]
        return avail_indices[top_idx]
    
    def _normalize(self, scores):
        """Normalize scores to [0, 1] range"""
        if len(scores) == 0:
            return scores
        min_val = np.min(scores)
        max_val = np.max(scores)
        if max_val > min_val:
            return (scores - min_val) / (max_val - min_val)
        return np.ones_like(scores) * 0.5
    
    def _boundary_scores(self, model, indices):
        """Compute boundary proximity scores for selected indices"""
        # Placeholder implementation - should be replaced with actual boundary scoring logic
        # For example, this could evaluate model confidence on these samples
        return np.random.rand(len(indices))
    
    def _class_balance_scores(self, selected_indices, candidate_indices):
        """Score candidates based on how they improve class balance"""
        # Placeholder implementation - should be replaced with actual class balance computation
        # This would evaluate how each candidate improves overall class distribution
        return np.random.rand(len(candidate_indices))
    
    def get_current_weights(self) -> Dict[str, float]:
        """Get current strategy weights"""
        return self.current_weights
    
    def get_weight_history(self) -> List[Dict[str, float]]:
        """Get history of strategy weights"""
        return self.weight_history
    
    def get_temperature_history(self) -> List[float]:
        """Get history of temperature values"""
        return self.temperature_history
    
    def get_sample_counts(self) -> Dict[str, int]:
        """Get total samples selected by each strategy"""
        return self.sample_counts
    
    def get_latest_explanation(self) -> str:
        """Get the latest decision explanation"""
        return self.explanations[-1] if self.explanations else ""
    
    def get_all_explanations(self) -> List[str]:
        """Get all decision explanations"""
        return self.explanations
    
    def save_model(self, path: str):
        """Save hypernetwork model to file"""
        torch.save({
            'hypernetwork_state_dict': self.hypernetwork.state_dict(),
            'target_network_state_dict': self.trainer.target_network.state_dict(),
            'optimizer_state_dict': self.trainer.optimizer.state_dict(),
            'weight_history': self.weight_history,
            'temperature_history': self.temperature_history,
            'state_value_history': self.state_value_history,
            'performance_history': self.performance_history,
            'sample_counts': self.sample_counts,
            'explanations': self.explanations
        }, path)
        print(f"Model saved to {path}")
    
    def load_model(self, path: str):
        """Load hypernetwork model from file"""
        checkpoint = torch.load(path, map_location=self.device)
        self.hypernetwork.load_state_dict(checkpoint['hypernetwork_state_dict'])
        self.trainer.target_network.load_state_dict(checkpoint['target_network_state_dict'])
        self.trainer.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.weight_history = checkpoint['weight_history']
        self.temperature_history = checkpoint['temperature_history']
        self.state_value_history = checkpoint['state_value_history']
        self.performance_history = checkpoint['performance_history']
        self.sample_counts = checkpoint['sample_counts']
        self.explanations = checkpoint['explanations']
        print(f"Model loaded from {path}")
        
    def plot_weight_history(self):
        """Plot strategy weight history"""
        if not self.weight_history:
            return None
            
        plt.figure(figsize=(12, 6))
        epochs = list(range(len(self.weight_history)))
        
        for strategy in self.strategies:
            weights = [h.get(strategy, 0.0) for h in self.weight_history]
            plt.plot(epochs, weights, label=strategy)
        
        plt.title('Strategy Weight Evolution')
        plt.xlabel('Training Step')
        plt.ylabel('Weight')
        plt.legend()
        plt.grid(True)
        
        return plt.gcf()
    
    def plot_temperature_history(self):
        """Plot temperature history"""
        if not self.temperature_history:
            return None
            
        plt.figure(figsize=(12, 6))
        epochs = list(range(len(self.temperature_history)))
        
        plt.plot(epochs, self.temperature_history)
        plt.title('Temperature Evolution (Exploration-Exploitation Tradeoff)')
        plt.xlabel('Training Step')
        plt.ylabel('Temperature')
        plt.grid(True)
        
        return plt.gcf()

# Example usage
if __name__ == "__main__":
    # Available strategies
    strategies = ["S_U", "S_D", "S_C", "S_B", "S_G", "S_F"]
    
    # Create meta-controller with budget ratio
    controller = EnhancedMetaController(
        strategies=strategies,
        budget_ratio=0.1  # 10% budget
    )
    
    # Simulate a training loop
    for epoch in range(100):
        # Simulate metrics
        accuracy = 0.5 + 0.3 * (1 - np.exp(-0.05 * epoch))
        loss = 1.0 - 0.6 * (1 - np.exp(-0.05 * epoch))
        class_distribution = {i: 100 + i*10 for i in range(10)}
        dataset_size = 1000 + 200 * min(epoch, 20)
        budget_size = 10000
        
        # Update state
        state = controller.update_state(
            epoch=epoch,
            accuracy=accuracy,
            loss=loss,
            class_distribution=class_distribution,
            dataset_size=dataset_size,
            budget_size=budget_size,
            strategy_performances={s: np.random.random() * 0.1 for s in strategies}
        )
        
        # Get sample allocation
        n_samples = 100 if epoch < 50 else 50
        allocation = controller.get_sample_allocation(n_samples)
        
        # Update hypernetwork with reward
        controller.update_weights(diversity_score=0.5)
        
        # Print progress every 10 epochs
        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Accuracy: {accuracy:.4f}, Loss: {loss:.4f}")
            print(f"Strategy weights: {controller.get_current_weights()}")
            print(f"Sample allocation: {allocation}")
            print(f"Explanation: {controller.get_latest_explanation()}")
            print()
    
    # Plot results
    controller.plot_weight_history()
    controller.plot_temperature_history()
    controller.trainer.plot_losses()