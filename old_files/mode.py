"""
Dual Contextualized MODE vs Rho-1: Complete Comparison
Integrates 5-strategy MODE with token-level memory and Contextualized.ml meta-controller

FIXES APPLIED:
1. Selection logic: Proper top-k with exact budget enforcement
2. Pretrained init: Now defaults to GPT-2 (use_pretrained_init=True) for reasonable starting perplexity
3. Warmup phase: First epoch uses full training to stabilize before selective training
4. Perplexity calc: Fixed sliding window with proper token counting
5. Memory updates: Only during non-warmup epochs

EXPECTED RESULTS:
- With pretrained GPT-2: Starting PPL ~25-35, should improve to ~20-30
- Random init: Starting PPL ~1000+, extremely slow convergence (not recommended)
- Selection ratio: Should be exactly 30% (or 100% during warmup epoch)
"""

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
from collections import defaultdict

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

try:
    from contextualized.easy import ContextualizedRegressor
    HAS_CONTEXTUALIZED = True
except:
    HAS_CONTEXTUALIZED = False
    print("WARNING: Contextualized.ml not available, using fallback MLP controller")


# ============================================================================
# CONFIG
# ============================================================================

@dataclass
class Config:
    # Model Architecture (reduced for MPS memory constraints)
    n_layer: int = 4          # Reduced from 8
    n_embd: int = 256         # Reduced from 512
    n_head: int = 4           # Reduced from 8
    dropout: float = 0.1
    max_seq_length: int = 256  # Reduced from 512
    
    # Training
    batch_size: int = 2  # Minimal batch size to avoid MPS OOM
    grad_accumulation: int = 1  # No accumulation for faster iteration
    epochs: int = 3  # Fewer epochs for quick testing
    learning_rate: float = 5e-5
    warmup_steps: int = 100
    warmup_epochs: int = 1  # Number of warmup epochs with full training
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    patience: int = 4
    
    # Token Selection Budget
    token_budget: float = 0.30  # 30% for all methods
    
    # MODE Configuration
    n_strategies: int = 5  # uncertainty, loss, coherence, diversity, token_mode
    
    # Token MODE Memory
    max_memory_contexts: int = 50000  # Reduced for efficiency
    retrieval_k: int = 50
    mode_sharpness: float = 2.0
    context_window: int = 5
    
    # Contextualized Meta-Controller
    num_archetypes: int = 10
    contextualized_alpha: float = 0.1
    contextualized_epochs: int = 20
    contextualized_train_threshold: int = 500
    
    # Fallback MLP Controller (if contextualized unavailable)
    controller_hidden: int = 32   # Reduced from 64
    controller_lr: float = 5e-4

    # Data
    train_samples: int = 200  # Reduced further for faster testing
    val_samples: int = 50   # Reduced for faster evaluation
    min_text_length: int = 100

    # Evaluation (reduced for lower memory usage)
    eval_stride: int = 128      # Reduced from 256
    eval_max_length: int = 256  # Reduced from 512
    
    # Logging
    log_interval: int = 50
    log_dir: str = './runs_dual_mode_comparison'
    
    # Initialization
    use_pretrained_init: bool = True  # Use pretrained GPT-2 for reasonable starting perplexity
    pretrained_model: str = 'distilgpt2'  # Smaller, faster model (82M params vs 124M)


# ============================================================================
# DEVICE SETUP
# ============================================================================

def get_device():
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"Using CUDA: {torch.cuda.get_device_name(0)}")
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
        print(f"Using MPS (Apple Silicon)")
    else:
        device = torch.device('cpu')
        print(f"Using CPU")
    return device


# ============================================================================
# TOKEN MODE MEMORY STORE
# ============================================================================

class TokenModeMemoryStore:
    """
    Stores (context_embedding, next_token, value) for retrieval-based MODE.
    Supports both FAISS and fallback implementations.
    """

    def __init__(self, config: Config, device, actual_hidden_dim: int = None):
        self.config = config
        self.device = device
        # Use actual model hidden dim if provided, otherwise use config
        self.hidden_dim = actual_hidden_dim if actual_hidden_dim is not None else config.n_embd

        # Storage
        self.context_embeddings = []
        self.next_tokens = []
        self.values = []

        # FAISS index - delay creation until first add to infer dimension
        # Disable FAISS for now to avoid dimension mismatch issues
        self.use_faiss = False  # HAS_FAISS and config.max_memory_contexts > 1000
        self.index = None
        print("Using fallback brute-force retrieval")
    
    def add_context_token_pair(self, context_embedding: torch.Tensor,
                               next_token: int, value: float):
        """Store a (context, token, value) tuple"""
        context_np = context_embedding.cpu().numpy().astype('float32')

        # Initialize FAISS index on first add (infer dimension from data)
        if self.use_faiss and self.index is None and len(self.context_embeddings) == 0:
            self.hidden_dim = context_np.shape[0]
            self.index = faiss.IndexFlatIP(self.hidden_dim)

        # FIFO replacement if full
        if len(self.context_embeddings) >= self.config.max_memory_contexts:
            self.context_embeddings.pop(0)
            self.next_tokens.pop(0)
            self.values.pop(0)

            # Rebuild FAISS index
            if self.index is not None:
                self.index = faiss.IndexFlatIP(self.hidden_dim)
                if len(self.context_embeddings) > 0:
                    all_contexts = np.array(self.context_embeddings).astype('float32')
                    self.index.add(all_contexts)

        self.context_embeddings.append(context_np)
        self.next_tokens.append(next_token)
        self.values.append(value)

        # Add to FAISS
        if self.index is not None:
            self.index.add(context_np.reshape(1, -1))
    
    def retrieve_mode_score(self, query_context: torch.Tensor, 
                           target_token: int) -> float:
        """
        Core MODE computation: Given context, what's the MODE token?
        Returns score for target_token based on frequency in similar contexts.
        """
        if len(self.context_embeddings) == 0:
            return 0.0
        
        query_np = query_context.cpu().numpy().astype('float32').reshape(1, -1)
        k = min(self.config.retrieval_k, len(self.context_embeddings))
        
        # Retrieve k most similar contexts
        if self.index is not None:
            similarities, indices = self.index.search(query_np, k)
            neighbor_tokens = [self.next_tokens[i] for i in indices[0]]
            neighbor_sims = similarities[0]
            neighbor_values = [self.values[i] for i in indices[0]]
        else:
            # Fallback: brute force cosine similarity
            contexts_tensor = torch.tensor(
                np.array(self.context_embeddings), 
                device=self.device
            )
            query_tensor = query_context.to(self.device)
            
            sims = F.cosine_similarity(
                query_tensor.unsqueeze(0),
                contexts_tensor,
                dim=1
            )
            
            neighbor_sims, indices = torch.topk(sims, k)
            neighbor_tokens = [self.next_tokens[i] for i in indices.tolist()]
            neighbor_values = [self.values[i] for i in indices.tolist()]
            neighbor_sims = neighbor_sims.cpu().numpy()
        
        # Compute value-weighted MODE
        token_votes = defaultdict(float)
        for token, sim, val in zip(neighbor_tokens, neighbor_sims, neighbor_values):
            weight = sim * (1.0 / (1.0 + np.exp(-val)))  # Sigmoid(value)
            token_votes[token] += weight
        
        if len(token_votes) == 0:
            return 0.0
        
        # Score for target token
        target_score = token_votes.get(target_token, 0.0)
        total_votes = sum(token_votes.values())
        
        if total_votes > 0:
            normalized_score = target_score / total_votes
            sharpened_score = normalized_score ** self.config.mode_sharpness
            return sharpened_score
        
        return 0.0
    
    def size(self) -> int:
        return len(self.context_embeddings)


# ============================================================================
# CONTEXTUALIZED META-CONTROLLER
# ============================================================================

class ContextualizedMetaController:
    """
    Meta-controller using Contextualized.ml or fallback MLP.
    Maps context features → strategy weights.
    """

    def __init__(self, config: Config, device, actual_hidden_dim: int = None):
        self.config = config
        self.device = device
        self.actual_hidden_dim = actual_hidden_dim if actual_hidden_dim is not None else config.n_embd

        # Try to use Contextualized.ml
        if HAS_CONTEXTUALIZED:
            self.contextualized = ContextualizedRegressor(
                encoder_type='mlp',
                num_archetypes=config.num_archetypes,
                alpha=config.contextualized_alpha,
                mu_ratio=0.5
            )
            self.use_contextualized = True
            print("Using Contextualized.ml meta-controller")
        else:
            # Fallback: simple MLP
            self.fallback_mlp = nn.Sequential(
                nn.Linear(self.actual_hidden_dim, config.controller_hidden),
                nn.Tanh(),
                nn.Dropout(config.dropout),
                nn.Linear(config.controller_hidden, config.n_strategies),
                nn.Softmax(dim=-1)
            ).to(device)
            self.use_contextualized = False
            print("Using fallback MLP meta-controller")
        
        # Training data collection
        self.context_history = []
        self.weight_history = []
        self.is_trained = False
    
    def collect_training_pair(self, context_features: np.ndarray,
                             strategy_scores: Dict[str, float],
                             performance_signal: float):
        """Collect training data: (context, strategy_weights)"""
        # Convert scores to weights
        score_array = np.array(list(strategy_scores.values()))
        
        if score_array.max() > score_array.min():
            normalized = (score_array - score_array.min()) / \
                        (score_array.max() - score_array.min())
        else:
            normalized = np.ones_like(score_array) / len(score_array)
        
        # Weight by performance signal
        weighted = normalized * performance_signal
        
        self.context_history.append(context_features)
        self.weight_history.append(weighted)
    
    def train(self):
        """Train the meta-controller"""
        if len(self.context_history) < self.config.contextualized_train_threshold:
            return
        
        print(f"\nTraining meta-controller with {len(self.context_history)} samples...")
        
        C = np.array(self.context_history)
        Y = np.array(self.weight_history)
        
        if self.use_contextualized:
            X = np.zeros((len(C), 1))  # Dummy
            self.contextualized.fit(
                X, Y, C,
                max_epochs=self.config.contextualized_epochs,
                verbose=False
            )
        else:
            # Train fallback MLP
            optimizer = torch.optim.Adam(
                self.fallback_mlp.parameters(), 
                lr=self.config.controller_lr
            )
            
            C_tensor = torch.tensor(C, dtype=torch.float32, device=self.device)
            Y_tensor = torch.tensor(Y, dtype=torch.float32, device=self.device)
            
            for epoch in range(20):
                optimizer.zero_grad()
                pred = self.fallback_mlp(C_tensor)
                loss = F.mse_loss(pred, Y_tensor)
                loss.backward()
                optimizer.step()
        
        self.is_trained = True
        print("Meta-controller trained!")
    
    def predict_weights(self, context_features: np.ndarray) -> np.ndarray:
        """Predict strategy weights for given context"""
        if not self.is_trained:
            # Uniform weights before training
            return np.ones(self.config.n_strategies) / self.config.n_strategies
        
        if self.use_contextualized:
            X_dummy = np.zeros((1, 1))
            C_query = context_features.reshape(1, -1)
            predicted = self.contextualized.predict(X_dummy, C_query)[0]
            
            # Normalize to positive and sum to 1
            weights = np.abs(predicted)
            weights = weights / (weights.sum() + 1e-10)
        else:
            # Fallback MLP
            with torch.no_grad():
                C_tensor = torch.tensor(
                    context_features.reshape(1, -1), 
                    dtype=torch.float32, 
                    device=self.device
                )
                weights = self.fallback_mlp(C_tensor).cpu().numpy()[0]
        
        return weights


# ============================================================================
# FEATURE EXTRACTOR
# ============================================================================

class FeatureExtractor:
    """
    Extract token-level features for all strategies.
    Supports both Rho-1 (excess_loss only) and MODE (all 5 strategies).
    """
    
    def __init__(self, model, reference_model, config, device):
        self.model = model
        self.reference_model = reference_model
        self.config = config
        self.device = device

        # Get token embeddings for diversity calculation
        self.token_embeddings = model.transformer.wte.weight

        # Get actual hidden dimension from model
        self.actual_hidden_dim = model.config.n_embd

    def extract_context_bow(self, hidden_states: torch.Tensor,
                           position: int) -> torch.Tensor:
        """Extract bag-of-words context embedding"""
        start = max(0, position - self.config.context_window)
        context_window = hidden_states[start:position]

        if len(context_window) == 0:
            return torch.zeros(self.actual_hidden_dim, device=self.device)
        
        # BoW = mean pooling + L2 normalization
        context_bow = context_window.mean(dim=0)
        context_bow = F.normalize(context_bow, p=2, dim=0)
        
        return context_bow
    
    @torch.no_grad()
    def extract_all_features(self, input_ids, attention_mask, 
                            token_mode_memory: Optional[TokenModeMemoryStore] = None):
        """
        Extract all 5 strategy features:
        1. uncertainty (entropy)
        2. loss (cross-entropy)
        3. coherence (confidence)
        4. diversity (embedding distance)
        5. token_mode (retrieval-based)
        """
        batch_size, seq_len = input_ids.shape
        
        # Get training model predictions
        self.model.eval()
        train_outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True
        )
        self.model.train()

        # Get reference model predictions (on CPU to save MPS memory)
        input_ids_cpu = input_ids.cpu()
        attention_mask_cpu = attention_mask.cpu()
        ref_outputs = self.reference_model(input_ids_cpu, attention_mask=attention_mask_cpu)
        # Move back to device
        ref_outputs.logits = ref_outputs.logits.to(self.device)
        
        logits = train_outputs.logits
        hidden_states = train_outputs.hidden_states[-1]
        
        # Shift for next-token prediction
        shift_logits = logits[:, :-1, :].contiguous()
        shift_targets = input_ids[:, 1:].contiguous()
        shift_hidden = hidden_states[:, :-1, :].contiguous()
        
        vocab_size = shift_logits.size(-1)
        
        features = {}
        context_bows = []
        
        # Compute per-token losses
        def compute_token_losses(logits, labels):
            loss_fct = nn.CrossEntropyLoss(reduction='none')
            losses = loss_fct(
                logits.reshape(-1, logits.size(-1)),
                labels.reshape(-1)
            ).reshape(batch_size, seq_len - 1)
            return losses
        
        train_loss = compute_token_losses(shift_logits, shift_targets)
        ref_logits_shift = ref_outputs.logits[:, :-1, :].contiguous()
        ref_loss = compute_token_losses(ref_logits_shift, shift_targets)
        
        # Strategy 1: Uncertainty (entropy)
        probs = F.softmax(shift_logits, dim=-1)
        entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=-1)
        features['uncertainty'] = entropy / np.log(vocab_size)
        
        # Strategy 2: Loss (cross-entropy) + excess_loss for Rho-1
        features['loss'] = train_loss
        features['excess_loss'] = train_loss - ref_loss
        
        # Strategy 3: Coherence (model confidence)
        target_probs = probs.gather(dim=-1, index=shift_targets.unsqueeze(-1)).squeeze(-1)
        features['coherence'] = target_probs
        
        # Strategy 4: Diversity (simplified - use random scores for speed)
        # Note: Full diversity calculation is slow. Using placeholder for faster iteration.
        diversity_scores = torch.rand(batch_size, seq_len - 1, device=self.device)
        features['diversity'] = diversity_scores
        
        # Strategy 5: Token MODE (retrieval-based)
        if token_mode_memory is not None:
            mode_scores = torch.zeros(batch_size, seq_len - 1, device=self.device)
            
            for b in range(batch_size):
                for pos in range(seq_len - 1):
                    context_bow = self.extract_context_bow(shift_hidden[b], pos)
                    context_bows.append(context_bow)
                    
                    target_token = shift_targets[b, pos].item()
                    mode_score = token_mode_memory.retrieve_mode_score(
                        context_bow,
                        target_token
                    )
                    mode_scores[b, pos] = mode_score
            
            features['token_mode'] = mode_scores
        else:
            features['token_mode'] = torch.zeros(batch_size, seq_len - 1, device=self.device)
            context_bows = []
        
        # Pad all features to original seq_len
        for key in features:
            features[key] = F.pad(features[key], (0, 1), value=0.0)
        
        return features, context_bows


# ============================================================================
# DATASET
# ============================================================================

class TextDataset(Dataset):
    def __init__(self, samples):
        self.samples = samples
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        return {'text': self.samples[idx]['text']}


# ============================================================================
# TRAINER
# ============================================================================

class Trainer:
    """
    Unified trainer supporting:
    - Full training (baseline)
    - Rho-1 (excess_loss selection)
    - Dual MODE (5 strategies + meta-controller + memory)
    """
    
    def __init__(self, config: Config, name: str, method: str):
        self.config = config
        self.name = name
        self.method = method  # 'full', 'rho1', 'mode'
        self.device = get_device()
        
        print(f"\n{'='*70}")
        print(f"Initializing: {name}")
        print(f"Method: {method.upper()} | Budget: {config.token_budget*100:.1f}%")
        print(f"{'='*70}")
        
        # Setup logging
        if HAS_TB:
            log_path = Path(config.log_dir) / name
            log_path.mkdir(parents=True, exist_ok=True)
            self.writer = SummaryWriter(str(log_path))
        else:
            self.writer = None
        
        # Initialize models
        self._init_models()
        
        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained('gpt2')
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Feature extractor
        self.feature_extractor = FeatureExtractor(
            self.model,
            self.ref_model,
            config,
            self.device
        )

        # MODE-specific components
        if self.method == 'mode':
            # Get actual hidden dimension from model
            actual_hidden_dim = self.model.config.n_embd
            self.token_mode_memory = TokenModeMemoryStore(config, self.device, actual_hidden_dim)
            self.meta_controller = ContextualizedMetaController(config, self.device, actual_hidden_dim)
            self.strategy_names = ['uncertainty', 'loss', 'coherence',
                                  'diversity', 'token_mode']
        
        # Model optimizer
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
        
        # Statistics
        self.strategy_history = []
        self.memory_stats = []
    
    def _init_models(self):
        """Initialize training and reference models"""
        if self.config.use_pretrained_init:
            print(f"Loading pretrained: {self.config.pretrained_model}")
            self.model = AutoModelForCausalLM.from_pretrained(
                self.config.pretrained_model
            ).to(self.device)
            # Keep ref_model on CPU to save MPS memory
            self.ref_model = AutoModelForCausalLM.from_pretrained(
                self.config.pretrained_model
            ).to('cpu')
        else:
            print(f"Initializing random model")
            gpt_config = GPT2Config(
                vocab_size=50257,
                n_positions=self.config.max_seq_length,
                n_embd=self.config.n_embd,
                n_layer=self.config.n_layer,
                n_head=self.config.n_head,
                resid_pdrop=self.config.dropout,
                embd_pdrop=self.config.dropout,
                attn_pdrop=self.config.dropout
            )
            
            self.model = AutoModelForCausalLM.from_config(gpt_config).to(self.device)
            self.ref_model = AutoModelForCausalLM.from_config(gpt_config).to(self.device)
        
        # Freeze reference model
        self.ref_model.eval()
        for param in self.ref_model.parameters():
            param.requires_grad = False
        
        n_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad) / 1e6
        print(f"Model: {n_params:.2f}M trainable parameters")
    
    def select_tokens_rho1(self, features: Dict[str, torch.Tensor], 
                          attention_mask: torch.Tensor) -> torch.Tensor:
        """Rho-1: Select based on excess_loss only"""
        batch_size, seq_len = attention_mask.shape
        scores = features['excess_loss']
        
        # Flatten scores and mask
        scores_flat = scores.reshape(-1)
        mask_flat = attention_mask.reshape(-1)
        
        # Only consider valid tokens
        valid_scores = scores_flat[mask_flat.bool()]
        n_valid = mask_flat.sum().item()
        n_select = max(1, int(self.config.token_budget * n_valid))
        
        if len(valid_scores) > n_select:
            # Get threshold value
            threshold_value = torch.topk(valid_scores, n_select).values[-1]
            
            # Create selection mask
            selection_mask = ((scores >= threshold_value) & (attention_mask.bool())).float()
            
            # Ensure exact budget by random dropping if needed
            actual_selected = selection_mask.sum().item()
            if actual_selected > n_select:
                # Find selected positions
                selected_positions = torch.nonzero(selection_mask.reshape(-1), as_tuple=False).squeeze(-1)
                # Randomly drop excess
                excess = int(actual_selected - n_select)
                drop_indices = selected_positions[torch.randperm(len(selected_positions))[:excess]]
                selection_mask.reshape(-1)[drop_indices] = 0.0
        else:
            selection_mask = attention_mask.float()
        
        return selection_mask
    
    def select_tokens_mode(self, features: Dict[str, torch.Tensor],
                          context_bows: List[torch.Tensor],
                          attention_mask: torch.Tensor,
                          performance_signal: float) -> torch.Tensor:
        """MODE: Select using meta-controller weighted combination"""
        batch_size, seq_len = attention_mask.shape
        combined_scores = torch.zeros(batch_size, seq_len, device=self.device)
        
        context_idx = 0
        
        for b in range(batch_size):
            for pos in range(seq_len - 1):  # We have scores for seq_len-1 positions
                if attention_mask[b, pos] > 0.5:
                    context_bow = context_bows[context_idx]
                    context_features = context_bow.cpu().numpy()
                    
                    # Predict strategy weights
                    strategy_weights = self.meta_controller.predict_weights(context_features)
                    self.strategy_history.append(strategy_weights)
                    
                    # Combine scores
                    position_score = sum(
                        strategy_weights[i] * features[self.strategy_names[i]][b, pos].item()
                        for i in range(len(self.strategy_names))
                    )
                    combined_scores[b, pos] = position_score
                    
                    # Collect training data for meta-controller
                    if not self.meta_controller.is_trained:
                        score_dict = {
                            name: features[name][b, pos].item()
                            for name in self.strategy_names
                        }
                        self.meta_controller.collect_training_pair(
                            context_features,
                            score_dict,
                            performance_signal
                        )
                    
                    context_idx += 1
        
        # Select top-k tokens using same logic as Rho-1
        scores_flat = combined_scores.reshape(-1)
        mask_flat = attention_mask.reshape(-1)
        
        valid_scores = scores_flat[mask_flat.bool()]
        n_valid = mask_flat.sum().item()
        n_select = max(1, int(self.config.token_budget * n_valid))
        
        if len(valid_scores) > n_select:
            threshold_value = torch.topk(valid_scores, n_select).values[-1]
            selection_mask = ((combined_scores >= threshold_value) & (attention_mask.bool())).float()
            
            # Ensure exact budget
            actual_selected = selection_mask.sum().item()
            if actual_selected > n_select:
                selected_positions = torch.nonzero(selection_mask.reshape(-1), as_tuple=False).squeeze(-1)
                excess = int(actual_selected - n_select)
                drop_indices = selected_positions[torch.randperm(len(selected_positions))[:excess]]
                selection_mask.reshape(-1)[drop_indices] = 0.0
        else:
            selection_mask = attention_mask.float()
        
        return selection_mask
    
    def update_token_mode_memory(self, input_ids: torch.Tensor,
                                hidden_states: torch.Tensor,
                                mask: torch.Tensor,
                                value: float):
        """Update token MODE memory with selected tokens"""
        batch_size, seq_len = input_ids.shape
        
        outputs = self.model(input_ids, output_hidden_states=True)
        hidden = outputs.hidden_states[-1]
        
        for b in range(batch_size):
            for pos in range(seq_len - 1):
                if mask[b, pos] > 0.5:
                    context_bow = self.feature_extractor.extract_context_bow(
                        hidden[b], pos
                    )
                    next_token = input_ids[b, pos + 1].item()
                    
                    self.token_mode_memory.add_context_token_pair(
                        context_bow,
                        next_token,
                        value
                    )
    
    def train_step(self, batch: Dict) -> Dict:
        """Single training step"""
        # Tokenize
        inputs = self.tokenizer(
            batch['text'],
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=self.config.max_seq_length
        ).to(self.device)
        
        # Extract features
        token_mode_memory = self.token_mode_memory if self.method == 'mode' else None
        features, context_bows = self.feature_extractor.extract_all_features(
            inputs.input_ids,
            inputs.attention_mask,
            token_mode_memory
        )
        
        # Select tokens based on method
        # Warmup phase: use full training for first epoch
        if self.epoch < self.config.warmup_epochs:
            token_mask = inputs.attention_mask.float()
        elif self.method == 'full':
            token_mask = inputs.attention_mask.float()
        elif self.method == 'rho1':
            token_mask = self.select_tokens_rho1(features, inputs.attention_mask)
        else:  # mode
            performance_signal = 0.5 if len(self.losses) == 0 else \
                                np.exp(-np.mean(self.losses[-10:]))
            token_mask = self.select_tokens_mode(
                features, context_bows, inputs.attention_mask, performance_signal
            )
        
        # Forward pass
        outputs = self.model(inputs.input_ids, attention_mask=inputs.attention_mask)
        
        # Compute masked loss
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
        
        # Backward
        loss = loss / self.config.grad_accumulation
        loss.backward()
        
        # Optimizer step
        if (self.step + 1) % self.config.grad_accumulation == 0:
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config.max_grad_norm
            )
            self.optimizer.step()
            self.optimizer.zero_grad()

            # Clear MPS cache to avoid OOM
            if self.device.type == 'mps':
                torch.mps.empty_cache()
        
        # Update MODE memory
        if self.method == 'mode' and self.epoch >= self.config.warmup_epochs:
            self.update_token_mode_memory(
                inputs.input_ids,
                outputs.hidden_states,
                token_mask,
                -loss.item()  # Negative loss as value
            )
        
        # Record
        actual_loss = loss.item() * self.config.grad_accumulation
        self.losses.append(actual_loss)
        self.step += 1
        
        # Train meta-controller
        if (self.method == 'mode' and 
            self.step % 100 == 0 and 
            not self.meta_controller.is_trained and
            len(self.meta_controller.context_history) >= 
            self.config.contextualized_train_threshold):
            self.meta_controller.train()
        
        return {
            'loss': actual_loss,
            'selection_ratio': (n_selected / max(shift_attention.sum(), 1)).item()
        }
    
    def train_epoch(self, train_loader: DataLoader) -> Dict:
        """Train for one epoch"""
        self.model.train()
        epoch_losses = []
        epoch_selections = []
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {self.epoch+1}")
        
        for batch in progress_bar:
            metrics = self.train_step(batch)
            epoch_losses.append(metrics['loss'])
            epoch_selections.append(metrics['selection_ratio'])
            
            if self.step % self.config.log_interval == 0:
                msg = f"Loss: {metrics['loss']:.4f} | Sel: {metrics['selection_ratio']:.1%}"
                
                if self.method == 'mode' and self.strategy_history:
                    s = self.strategy_history[-1]
                    msg += f" | Strat: [{s[0]:.2f},{s[1]:.2f},{s[2]:.2f},{s[3]:.2f},{s[4]:.2f}]"
                    msg += f" | Mem: {self.token_mode_memory.size()}"
                
                progress_bar.set_postfix_str(msg)
                
                if self.writer:
                    self.writer.add_scalar('train/loss', metrics['loss'], self.step)
                    self.writer.add_scalar('train/selection', metrics['selection_ratio'], self.step)
        
        return {
            'avg_loss': np.mean(epoch_losses),
            'avg_selection': np.mean(epoch_selections)
        }
    
    @torch.no_grad()
    def evaluate_perplexity(self, texts: List[str]) -> float:
        """
        Correct perplexity with sliding window - FIXED VERSION
        Key fixes:
        1. Accumulate raw logits loss (reduction='sum')
        2. Track exact token coverage with no overlaps
        3. Verify all tokens counted exactly once
        """
        self.model.eval()

        max_length = min(self.config.eval_max_length, self.model.config.n_positions)
        stride = self.config.eval_stride

        total_nll = 0.0
        total_tokens = 0

        for text in tqdm(texts, desc="Evaluating", leave=False):
            encodings = self.tokenizer(
                text,
                return_tensors='pt',
                truncation=False,
                add_special_tokens=True
            )

            input_ids = encodings.input_ids
            seq_len = input_ids.size(1)

            # Need at least 2 tokens for next-token prediction
            if seq_len < 2:
                continue

            # Track which tokens we've evaluated
            evaluated_positions = set()

            for begin_loc in range(0, seq_len, stride):
                end_loc = min(begin_loc + max_length, seq_len)

                # Can't evaluate if window is too small
                if end_loc - begin_loc < 2:
                    break

                input_chunk = input_ids[:, begin_loc:end_loc].to(self.device)

                # Get model outputs
                outputs = self.model(input_chunk)
                logits = outputs.logits

                # Shift for next-token prediction
                shift_logits = logits[:, :-1, :].contiguous()
                shift_labels = input_chunk[:, 1:].contiguous()

                # Compute loss with NO reduction (get per-token losses)
                loss_fct = nn.CrossEntropyLoss(reduction='none')
                token_losses = loss_fct(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1)
                )

                # Determine which positions to count (avoid double-counting overlaps)
                # For first window: count all
                # For subsequent windows: only count tokens beyond previous window's coverage
                if begin_loc == 0:
                    # First window: evaluate all positions
                    positions_to_count = list(range(shift_labels.size(1)))
                else:
                    # Subsequent windows: skip overlap, only count new tokens
                    overlap = max_length - stride
                    positions_to_count = list(range(overlap, shift_labels.size(1)))

                # Accumulate losses for non-overlapping positions
                for local_pos in positions_to_count:
                    global_pos = begin_loc + local_pos

                    # Verify no double-counting
                    if global_pos in evaluated_positions:
                        raise RuntimeError(f"Double-counting detected at position {global_pos}")

                    evaluated_positions.add(global_pos)
                    total_nll += token_losses[local_pos].item()
                    total_tokens += 1

                # Stop if we've covered the sequence
                if end_loc >= seq_len:
                    break

            # VERIFICATION: Did we evaluate all predictable tokens?
            expected_tokens = seq_len - 1  # All positions except first
            if len(evaluated_positions) != expected_tokens:
                print(f"WARNING: Expected {expected_tokens} tokens, evaluated {len(evaluated_positions)}")

        if total_tokens == 0:
            return float('inf')

        avg_nll = total_nll / total_tokens
        perplexity = np.exp(avg_nll)

        print(f"  → Evaluated {total_tokens} tokens total")

        return perplexity
    
    def train(self, train_loader: DataLoader, val_texts: List[str]) -> Dict:
        """Full training loop"""
        print(f"\nStarting training: {self.name}")
        
        for epoch in range(self.config.epochs):
            self.epoch = epoch
            
            print(f"\n{'='*70}")
            print(f"Epoch {epoch+1}/{self.config.epochs}")
            print(f"{'='*70}")
            
            # Train
            train_metrics = self.train_epoch(train_loader)
            
            # Evaluate
            val_ppl = self.evaluate_perplexity(val_texts)
            
            print(f"\nEpoch Summary:")
            print(f"  Train Loss: {train_metrics['avg_loss']:.4f}")
            print(f"  Val Perplexity: {val_ppl:.2f}")
            print(f"  Avg Selection: {train_metrics['avg_selection']:.1%}")
            
            if self.method == 'mode':
                print(f"  Memory Size: {self.token_mode_memory.size()}")
                print(f"  Meta-Controller Trained: {self.meta_controller.is_trained}")
                
                if self.strategy_history:
                    avg_weights = np.mean(self.strategy_history[-100:], axis=0)
                    print(f"  Avg Strategy Weights: {np.round(avg_weights, 3)}")
            
            # Log to tensorboard
            if self.writer:
                self.writer.add_scalar('epoch/train_loss', train_metrics['avg_loss'], epoch)
                self.writer.add_scalar('epoch/val_ppl', val_ppl, epoch)
                
                if self.method == 'mode' and self.strategy_history:
                    for i, name in enumerate(self.strategy_names):
                        self.writer.add_scalar(
                            f'strategy/{name}',
                            avg_weights[i],
                            epoch
                        )
            
            # Early stopping
            if val_ppl < self.best_val_ppl:
                self.best_val_ppl = val_ppl
                self.patience_counter = 0
                print(f"  New best perplexity!")
            else:
                self.patience_counter += 1
                print(f"  Patience: {self.patience_counter}/{self.config.patience}")
                
                if self.patience_counter >= self.config.patience:
                    print(f"\nEarly stopping triggered")
                    break
        
        if self.writer:
            self.writer.close()
        
        return {
            'best_val_ppl': self.best_val_ppl,
            'final_train_loss': train_metrics['avg_loss']
        }


# ============================================================================
# DATA LOADING
# ============================================================================

def load_data(config: Config):
    """Load WikiText-2 dataset"""
    print(f"\n{'='*70}")
    print("Loading WikiText-2 dataset...")
    print(f"{'='*70}")
    
    dataset = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train', streaming=True)
    
    samples = []
    for sample in dataset:
        text = sample['text'].strip()
        if text and len(text) >= config.min_text_length:
            samples.append(sample)
        
        if len(samples) >= config.train_samples + config.val_samples:
            break
    
    print(f"Collected {len(samples)} valid samples")
    
    train_samples = samples[:config.train_samples]
    val_samples = samples[config.train_samples:]
    
    print(f"Train: {len(train_samples)} | Val: {len(val_samples)}")
    
    train_dataset = TextDataset(train_samples)
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=0
    )
    
    val_texts = [s['text'] for s in val_samples]
    
    return train_loader, val_texts


# ============================================================================
# MAIN EXPERIMENT
# ============================================================================

def run_comparison():
    """Run complete comparison: Full vs Rho-1 vs Dual MODE"""
    config = Config()
    
    print("\n" + "="*70)
    print("Dual Contextualized MODE vs Rho-1 Comparison")
    print("="*70)
    print(f"\nConfiguration:")
    print(f"  Model: {config.n_layer}L / {config.n_embd}D / {config.n_head}H")
    print(f"  Training: {config.train_samples} samples, {config.epochs} epochs")
    print(f"  Budget: {config.token_budget*100:.1f}%")
    print(f"  5 Strategies: uncertainty, loss, coherence, diversity, token_mode")
    print("="*70)
    
    # Load data
    train_loader, val_texts = load_data(config)
    
    # Run experiments
    results = {}
    
    experiments = [
        ('Full-Training', 'full'),
        ('Rho-1', 'rho1'),
        ('Dual-MODE', 'mode'),
    ]
    
    for name, method in experiments:
        print(f"\n{'='*70}")
        print(f"Running: {name}")
        print(f"{'='*70}")
        
        trainer = Trainer(config, name, method)
        result = trainer.train(train_loader, val_texts)
        results[name] = result
        
        print(f"\n{name} complete")
        print(f"  Best Val Perplexity: {result['best_val_ppl']:.2f}")
    
    # Final comparison
    print(f"\n\n{'='*70}")
    print("FINAL RESULTS")
    print(f"{'='*70}")
    
    for name in ['Full-Training', 'Rho-1', 'Dual-MODE']:
        ppl = results[name]['best_val_ppl']
        print(f"  {name:20s}: {ppl:7.2f} PPL")
    
    # Compute improvements
    full_ppl = results['Full-Training']['best_val_ppl']
    rho1_ppl = results['Rho-1']['best_val_ppl']
    mode_ppl = results['Dual-MODE']['best_val_ppl']
    
    rho1_improvement = (full_ppl - rho1_ppl) / full_ppl * 100
    mode_improvement = (full_ppl - mode_ppl) / full_ppl * 100
    mode_vs_rho1 = (rho1_ppl - mode_ppl) / rho1_ppl * 100
    
    print(f"\n{'='*70}")
    print("Improvements vs Full Training:")
    print(f"  Rho-1:      {rho1_improvement:+.2f}%")
    print(f"  Dual MODE:  {mode_improvement:+.2f}%")
    print(f"\nDual MODE vs Rho-1:")
    print(f"  Improvement: {mode_vs_rho1:+.2f}%")
    print(f"{'='*70}\n")
    
    if mode_vs_rho1 > 2:
        print("SUCCESS: Dual MODE significantly outperforms Rho-1!")
    elif mode_vs_rho1 > -2:
        print("COMPARABLE: Dual MODE and Rho-1 are similar")
    else:
        print("Dual MODE underperforming - may need tuning")
    
    return results


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    results = run_comparison()
    print("\nExperiment complete!")