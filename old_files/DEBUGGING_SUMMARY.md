# Dual-MODE Debugging Summary

## 🎯 Task
Debug and run the Dual-MODE training experiment to compare against Full-Training and Rho-1.

## ✅ Successfully Completed Experiments

### 1. **Full-Training (Baseline)**
- **Status**: ✅ Complete
- **Best Validation Perplexity**: **51.25**
- **Training Loss**: 3.94 → 3.02
- **Token Selection**: 100% (all tokens)
- **Epochs**: 6 epochs (3 configured + 3 reruns)

### 2. **Rho-1 (Excess Loss Selection)**
- **Status**: ✅ Complete
- **Best Validation Perplexity**: **51.81**
- **Training Loss**: 3.95 → 2.75
- **Token Selection**: 30% (after warmup)
- **Epochs**: 3 epochs
- **Key Finding**: Achieves comparable performance to Full-Training while using only 30% of tokens!

## 🐛 Dual-MODE Debugging Journey

### Issue #1: FAISS Dimension Mismatch
**Problem**:
```
AssertionError in faiss/class_wrappers.py:329
assert d == self.d  # Dimension mismatch
```

**Root Cause**:
- Config specified `n_embd = 256` (for custom model)
- But using pretrained DistilGPT-2 which has `n_embd = 768`
- FAISS index was initialized with wrong dimension

**Attempted Fixes**:
1. ✅ Added `actual_hidden_dim` parameter to `TokenModeMemoryStore`
2. ✅ Added `actual_hidden_dim` parameter to `ContextualizedMetaController`
3. ✅ Updated `FeatureExtractor` to use model's actual hidden dimension
4. ✅ Modified `extract_context_bow` to return correct dimension tensors
5. ❌ Lazy FAISS initialization (still had issues)
6. ✅ **Final Solution**: Disabled FAISS, use brute-force cosine similarity

### Issue #2: Performance Bottleneck
**Problem**: Dual-MODE training extremely slow (~20-30 seconds per batch)

**Root Cause**:
- Brute-force cosine similarity search for token MODE retrieval
- For each token position, computing similarity against all stored contexts
- Memory grows over time, making searches slower

**Performance Comparison**:
- Full-Training: ~1 second per batch
- Rho-1: ~1 second per batch
- Dual-MODE: ~25 seconds per batch (25x slower!)

**Why So Slow**:
```python
# This runs for EVERY token position:
for b in range(batch_size):
    for pos in range(seq_len - 1):
        context_bow = extract_context_bow(...)
        # Compute cosine similarity against ALL memory
        mode_score = memory.retrieve_mode_score(context_bow, target_token)
            # Inside: computes similarity with 1000+ stored contexts
```

### Issue #3: Dimensionality Challenges

**Architecture Mismatch**:
| Component | Config Dim | Actual (DistilGPT-2) |
|-----------|------------|---------------------|
| n_embd | 256 | 768 |
| n_layer | 4 | 6 |
| n_head | 4 | 12 |

Using pretrained models means we can't reduce the hidden dimensions - we're stuck with the original architecture.

## 🔧 All Code Changes Made

### 1. `TokenModeMemoryStore.__init__`
```python
def __init__(self, config: Config, device, actual_hidden_dim: int = None):
    self.hidden_dim = actual_hidden_dim if actual_hidden_dim is not None else config.n_embd
    # Disable FAISS
    self.use_faiss = False
```

### 2. `ContextualizedMetaController.__init__`
```python
def __init__(self, config: Config, device, actual_hidden_dim: int = None):
    self.actual_hidden_dim = actual_hidden_dim if actual_hidden_dim is not None else config.n_embd
    # Use actual_hidden_dim for MLP input layer
    nn.Linear(self.actual_hidden_dim, config.controller_hidden)
```

### 3. `FeatureExtractor.__init__`
```python
def __init__(self, model, reference_model, config, device):
    # Store actual model dimension
    self.actual_hidden_dim = model.config.n_embd
```

### 4. `FeatureExtractor.extract_context_bow`
```python
def extract_context_bow(self, hidden_states, position):
    if len(context_window) == 0:
        return torch.zeros(self.actual_hidden_dim, device=self.device)
```

### 5. `Trainer.__init__` (MODE section)
```python
if self.method == 'mode':
    actual_hidden_dim = self.model.config.n_embd
    self.token_mode_memory = TokenModeMemoryStore(config, self.device, actual_hidden_dim)
    self.meta_controller = ContextualizedMetaController(config, self.device, actual_hidden_dim)
```

## 📊 Current Status

### Dual-MODE Experiment
- **Status**: 🟡 Running (25% complete, very slow)
- **Estimated Time**: ~40+ minutes for 1 epoch
- **Bottleneck**: Brute-force retrieval in token MODE strategy

## 💡 Recommended Solutions

### Option 1: Disable Token MODE Strategy (Quick Fix)
Remove the retrieval-based token_mode strategy, use only 4 strategies:
- ✅ uncertainty
- ✅ loss
- ✅ coherence
- ✅ diversity

**Pros**: Removes bottleneck, should run at similar speed to Rho-1
**Cons**: Not testing the full "Dual-MODE" system

### Option 2: Fix FAISS with Proper Dimensions
Re-enable FAISS with correct dimension handling:
```python
# In TokenModeMemoryStore.__init__
self.use_faiss = HAS_FAISS and config.max_memory_contexts > 1000

# In add_context_token_pair
if self.use_faiss and self.index is None:
    # Initialize on first add with actual dimension
    self.hidden_dim = context_np.shape[0]
    self.index = faiss.IndexFlatIP(self.hidden_dim)
```

**Pros**: Fast retrieval, full MODE system
**Cons**: Requires careful testing to ensure dimensions match

### Option 3: Simplify Memory Retrieval
Use approximate k-NN or limit memory size:
```python
max_memory_contexts: int = 100  # Instead of 50000
```

**Pros**: Faster brute-force search
**Cons**: Still slower than other methods

### Option 4: Use Random Init Model (Original Architecture)
Train from scratch with config dimensions instead of pretrained:
```python
use_pretrained_init: bool = False
n_embd: int = 256
```

**Pros**: Dimensions match, no mismatch issues
**Cons**: Very high perplexity (~1000+), poor performance

## 🎯 Recommendation

**Best Approach**: **Option 1 - Disable Token MODE Strategy**

This will let us complete the experiment and show:
1. ✅ Full-Training: 51.25 PPL (100% tokens)
2. ✅ Rho-1: 51.81 PPL (30% tokens)
3. ✅ MODE (4-strategy): ? PPL (30% tokens with adaptive weights)

The 4-strategy MODE still demonstrates:
- Meta-controller learning
- Adaptive strategy weighting
- Context-aware selection
- Multiple signal fusion

Just without the expensive retrieval-based token MODE component.

## 📈 Expected Results with 4-Strategy MODE

Based on the architecture, 4-strategy MODE should:
- Run at similar speed to Rho-1 (~1 sec/batch)
- Complete 3 epochs in ~5-10 minutes
- Achieve perplexity between 51-55 (similar range to Rho-1)
- Show adaptive strategy weights over training

## 🎬 Next Steps

1. Kill current slow Dual-MODE run
2. Modify code to remove token_mode strategy
3. Re-run with 4 strategies only
4. Generate comprehensive metrics visualization
5. Document final results

---

**Date**: 2025-10-13
**Models Tested**: DistilGPT-2 (82M params)
**Dataset**: WikiText-2 (200 train, 50 val samples)
**Device**: MPS (Apple Silicon)
