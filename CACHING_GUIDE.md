# Caching Implementation Guide

## Overview

The caching system eliminates redundant computation of reference model losses during selective training. Since the reference model is **frozen** (never updated), its outputs for each sample remain constant across all epochs.

## Problem Statement

**Without caching:**
```
For each epoch:
    For each batch:
        1. Forward pass through training model
        2. Forward pass through reference model  ← REPEATED EVERY EPOCH!
        3. Compute excess loss
        4. Select samples
        5. Backprop on training model
```

**Result:** Reference model processes each sample N_epochs times (wasteful!)

## Solution: Intelligent Caching

**With caching:**
```
# One-time precomputation
For each sample in dataset:
    Forward pass through reference model
    Store loss in cache

# Training (reuses cached values)
For each epoch:
    For each batch:
        1. Forward pass through training model
        2. Lookup reference loss from cache  ← INSTANT!
        3. Compute excess loss
        4. Select samples
        5. Backprop on training model
```

**Result:** Reference model processes each sample only ONCE!

## Implementation Details

### 1. Cache Modes

Three modes available via `config.CACHE_MODE`:

```python
# No caching - recompute every time
config.CACHE_MODE = 'none'

# Memory caching - store in RAM (fastest)
config.CACHE_MODE = 'memory'

# Disk caching - store on disk (memory efficient)
config.CACHE_MODE = 'disk'
```

### 2. IndexedDataset Wrapper

To enable caching, we need to track which sample indices are in each batch:

```python
class IndexedDataset(Dataset):
    """Wrapper that returns (data, target, index)"""

    def __init__(self, base_dataset):
        self.base_dataset = base_dataset

    def __getitem__(self, idx):
        data, target = self.base_dataset[idx]
        return data, target, idx  # Include index!
```

### 3. Precomputation Phase

Before training starts, compute all reference losses:

```python
def precompute_all_losses(self, dataloader):
    """One-time computation of all reference losses"""
    self.model.eval()
    sample_idx = 0

    with torch.no_grad():
        for inputs, targets in dataloader:
            outputs = self.model(inputs)
            losses = CrossEntropyLoss(reduction='none')(outputs, targets)

            # Cache each sample's loss
            for i, loss in enumerate(losses):
                self.loss_cache[sample_idx + i] = loss.item()

            sample_idx += len(targets)
```

### 4. Cache Lookup During Training

During training, look up cached values instead of recomputing:

```python
def compute_reference_loss(self, inputs, targets, sample_indices=None):
    """Compute or retrieve reference losses"""

    # Use cache if available
    if sample_indices is not None and len(self.loss_cache) > 0:
        return self.get_cached_losses(sample_indices)

    # Fallback: compute on-the-fly
    outputs = self.model(inputs)
    return CrossEntropyLoss(reduction='none')(outputs, targets)
```

## Configuration

```python
class SelectiveClassificationConfig:
    # ... other settings ...

    # Caching settings
    CACHE_MODE = 'memory'  # 'none', 'memory', or 'disk'
    PRECOMPUTE_REFERENCE_LOSSES = True  # Precompute before training
```

## Usage Example

```python
# 1. Create reference model with caching
reference_model = SelectiveReferenceModel(cache_mode='memory')

# 2. Wrap dataset with indices
train_ds = IndexedDataset(original_dataset)
train_loader = DataLoader(train_ds, ...)

# 3. Precompute all reference losses (one-time cost)
if config.PRECOMPUTE_REFERENCE_LOSSES:
    reference_model.precompute_all_losses(train_loader)
    # Now cache contains all 50,000 losses for CIFAR-10

# 4. Training loop uses cached values
for epoch in range(num_epochs):
    for inputs, targets, indices in train_loader:
        # Reference losses retrieved from cache (instant!)
        ref_losses = reference_model.compute_reference_loss(
            inputs, targets, sample_indices=indices
        )
```

## Performance Impact

### Computational Savings

| Dataset | Samples | Epochs | Without Cache | With Cache | Speedup |
|---------|---------|--------|---------------|------------|---------|
| CIFAR-10 | 50K | 200 | 10M passes | 50K passes | 200× |
| CIFAR-100 | 50K | 200 | 10M passes | 50K passes | 200× |
| ImageNet | 1.3M | 100 | 130M passes | 1.3M passes | 100× |

### Memory Overhead

Each cached loss is a single float32 (4 bytes):

- **CIFAR-10**: 50,000 samples × 4 bytes = **200 KB** (negligible!)
- **CIFAR-100**: 50,000 samples × 4 bytes = **200 KB**
- **ImageNet**: 1,300,000 samples × 4 bytes = **5.2 MB**

Actual memory usage slightly higher due to dictionary overhead, but still very small.

### Disk Caching

For large datasets or limited RAM, use disk caching:

```python
config.CACHE_MODE = 'disk'
```

- Cache stored in: `{OUTPUT_DIR}/reference_loss_cache.pt`
- Automatically saved after precomputation
- Automatically loaded if file exists
- Persists across runs (reuse cache!)

## Benefits

1. **Massive speedup**: Up to 200× faster for selective training
2. **Low memory**: Only 4 bytes per sample
3. **Disk persistence**: Cache survives restarts
4. **Automatic**: Works transparently once configured
5. **Optional**: Can disable for debugging

## When NOT to Use Caching

1. **Reference model is being updated**: Cache becomes stale
2. **Data augmentation affects reference**: Cached values won't match
3. **Debugging**: Easier to verify correctness without cache

For these cases, set `config.CACHE_MODE = 'none'`

## Code Locations

- **Cache implementation**: `SelectiveReferenceModel` class (main.py:140-280)
- **IndexedDataset wrapper**: `IndexedDataset` class (main.py:85-96)
- **Precomputation**: `precompute_all_losses()` method (main.py:190-225)
- **Cache lookup**: `compute_reference_loss()` method (main.py:250-280)
- **Configuration**: `SelectiveClassificationConfig` (main.py:57-59)

## Advanced: Custom Cache Strategies

You can extend the caching system:

### 1. Feature Caching

Cache intermediate features instead of losses:

```python
def precompute_features(self, dataloader):
    """Cache intermediate features for flexibility"""
    for inputs, targets in dataloader:
        features = self.model.forward_features(inputs)  # Before classifier
        self.feature_cache[sample_idx] = features
```

### 2. Lazy Loading

Load cache entries on-demand for huge datasets:

```python
def get_cached_loss(self, idx):
    """Load from disk only when needed"""
    if idx not in self.memory_cache:
        self.memory_cache[idx] = self.disk_cache[idx]
    return self.memory_cache[idx]
```

### 3. Periodic Updates

Update cache every N epochs if reference model fine-tunes:

```python
if epoch % config.CACHE_UPDATE_FREQUENCY == 0:
    reference_model.precompute_all_losses(train_loader)
```

## Conclusion

The caching system provides **huge performance improvements** with **minimal overhead**. For typical selective training workloads, it's a **must-have optimization** that saves hours of compute time.

**Recommendation**: Always use `CACHE_MODE='memory'` and `PRECOMPUTE_REFERENCE_LOSSES=True` unless you have a specific reason not to.
