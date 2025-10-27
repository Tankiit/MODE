# Caching System Visualization

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                     SELECTIVE TRAINING PIPELINE                  │
└─────────────────────────────────────────────────────────────────┘

WITHOUT CACHING (Slow):
═══════════════════════════════════════════════════════════════════

Epoch 1:
  Sample 1 → [Reference Model] → Loss: 2.3 ✗ (computed)
  Sample 2 → [Reference Model] → Loss: 1.8 ✗ (computed)
  Sample 3 → [Reference Model] → Loss: 2.1 ✗ (computed)
  ...

Epoch 2:
  Sample 1 → [Reference Model] → Loss: 2.3 ✗ (RECOMPUTED!)
  Sample 2 → [Reference Model] → Loss: 1.8 ✗ (RECOMPUTED!)
  Sample 3 → [Reference Model] → Loss: 2.1 ✗ (RECOMPUTED!)
  ...

Epoch N:
  Sample 1 → [Reference Model] → Loss: 2.3 ✗ (RECOMPUTED!)
  ...

Total: N_samples × N_epochs forward passes


WITH CACHING (Fast):
═══════════════════════════════════════════════════════════════════

PRECOMPUTATION PHASE (Once):
  Sample 1 → [Reference Model] → Loss: 2.3 → [Cache] ✓
  Sample 2 → [Reference Model] → Loss: 1.8 → [Cache] ✓
  Sample 3 → [Reference Model] → Loss: 2.1 → [Cache] ✓
  ...

TRAINING PHASE (Every Epoch):
  Sample 1 → [Cache Lookup] → Loss: 2.3 (instant!)
  Sample 2 → [Cache Lookup] → Loss: 1.8 (instant!)
  Sample 3 → [Cache Lookup] → Loss: 2.1 (instant!)
  ...

Total: N_samples forward passes (200× faster!)
```

## Data Flow

```
┌──────────────────────────────────────────────────────────────┐
│                    TRAINING ITERATION                         │
└──────────────────────────────────────────────────────────────┘

Input Batch: (images, labels, indices)
     │
     ├──────────────────┬────────────────────┐
     │                  │                    │
     ▼                  ▼                    ▼
┌─────────┐      ┌──────────┐      ┌──────────────┐
│ Training│      │Reference │      │    Cache     │
│  Model  │      │  Model   │      │   Lookup     │
└─────────┘      └──────────┘      └──────────────┘
     │                  │                    │
     │                  │                    │
     │                  │    ┌───────────────┘
     │                  │    │
     │                  │    │ (if not cached)
     │                  │    │
     ▼                  ▼    ▼
┌──────────┐      ┌──────────────┐
│ Training │      │  Reference   │
│  Loss    │      │    Loss      │
└──────────┘      └──────────────┘
     │                  │
     └─────────┬────────┘
               │
               ▼
        ┌─────────────┐
        │Excess Loss  │
        │(L_t - L_r)  │
        └─────────────┘
               │
               ▼
        ┌─────────────┐
        │  Selection  │
        │    Mask     │
        └─────────────┘
               │
               ▼
        ┌─────────────┐
        │ Backprop on │
        │   Selected  │
        └─────────────┘
```

## Cache Structure

```
┌─────────────────────────────────────────────────────────────┐
│                    LOSS CACHE (Dictionary)                   │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Sample Index  →  Reference Loss                            │
│  ─────────────────────────────────                          │
│       0        →     2.3045                                  │
│       1        →     1.8234                                  │
│       2        →     2.1098                                  │
│       3        →     1.5432                                  │
│      ...       →      ...                                    │
│    49,999      →     2.0123                                  │
│                                                              │
│  Total: 50,000 entries × 4 bytes = 200 KB                   │
└─────────────────────────────────────────────────────────────┘
```

## Memory vs Disk Caching

```
MEMORY CACHING:
═══════════════════════════════════════════════════════════
┌──────────────┐
│     RAM      │
├──────────────┤
│ Loss Cache:  │
│ {0: 2.3,     │ ← Ultra-fast lookup (nanoseconds)
│  1: 1.8,     │
│  2: 2.1,     │
│  ...}        │
└──────────────┘

Pros: Fastest access
Cons: Uses RAM (but minimal - only 200KB for CIFAR-10)


DISK CACHING:
═══════════════════════════════════════════════════════════
┌──────────────┐
│     RAM      │ ← Small LRU cache (optional)
└──────────────┘
       ↕
┌──────────────┐
│     DISK     │
├──────────────┤
│  cache.pt:   │ ← Fast file I/O (microseconds)
│  {0: 2.3,    │
│   1: 1.8,    │
│   ...}       │
└──────────────┘

Pros: No RAM usage, persists across runs
Cons: Slightly slower than memory (but still 100× faster than recompute)
```

## Timeline Comparison

```
CIFAR-10 Training: 50,000 samples, 200 epochs, ResNet-18 reference

WITHOUT CACHING:
═══════════════════════════════════════════════════════════
Time:     0s ────────────────────────────────────────► 200h
          │                                             │
          │  ████████████████████████████████████████  │
          │  Reference model forward passes            │
          │  (10 million passes!)                      │


WITH MEMORY CACHING:
═══════════════════════════════════════════════════════════
Time:     0s ───────────► 1h
          │              │
          │  ██          │ (Precomputation: 50K passes)
          │              │
          └──────────────┘ Then training uses cache!

          Training: 1h (vs 200h without cache)
          Speedup: 200×
```

## Cache Hit/Miss Flow

```
┌─────────────────────────────────────────────────────────┐
│          Cache Lookup Decision Tree                     │
└─────────────────────────────────────────────────────────┘

Sample needs reference loss
        │
        ▼
   ┌─────────┐
   │ Cache   │
   │ enabled?│
   └─────────┘
    │        │
   Yes      No
    │        │
    ▼        └────────────────┐
┌─────────┐                   │
│ Sample  │                   │
│  in     │                   │
│ cache?  │                   │
└─────────┘                   │
 │        │                   │
Yes      No                   │
 │        │                   │
 ▼        ▼                   ▼
┌──────┐ ┌─────────────────────┐
│Cache │ │Compute reference    │
│ Hit! │ │loss on-the-fly      │
│     │ │(forward pass)       │
└──────┘ └─────────────────────┘
   │              │
   └──────┬───────┘
          ▼
   Reference loss ready
```

## Storage Requirements

```
Dataset Size vs Cache Storage:
═══════════════════════════════════════════════════════════

CIFAR-10 (50K samples):
  ├─ Cache size: 200 KB
  ├─ Overhead: Negligible
  └─ Recommendation: Use memory caching

CIFAR-100 (50K samples):
  ├─ Cache size: 200 KB
  ├─ Overhead: Negligible
  └─ Recommendation: Use memory caching

ImageNet (1.3M samples):
  ├─ Cache size: 5.2 MB
  ├─ Overhead: Minimal
  └─ Recommendation: Use memory or disk caching

Custom Large Dataset (10M samples):
  ├─ Cache size: 40 MB
  ├─ Overhead: Minimal
  └─ Recommendation: Use disk caching if RAM limited
```

## Example: Cache Utilization Over Time

```
Training Progress with Caching:
═══════════════════════════════════════════════════════════

                    Cache Hits
                        ↓
Epoch 1:  ████████████████████████████████ (50K lookups)
Epoch 2:  ████████████████████████████████ (50K lookups)
Epoch 3:  ████████████████████████████████ (50K lookups)
...
Epoch 200:████████████████████████████████ (50K lookups)

Total cache hits: 10 million
Total reference forward passes: 50 thousand
Computation saved: 99.5%


Without caching:
═══════════════════════════════════════════════════════════

Epoch 1:  ████████████████████████████████ (50K computes)
Epoch 2:  ████████████████████████████████ (50K computes)
Epoch 3:  ████████████████████████████████ (50K computes)
...
Epoch 200:████████████████████████████████ (50K computes)

Total reference forward passes: 10 million
```

## Key Takeaways

1. **Precomputation is one-time cost**: Happens once before training
2. **Cache lookups are instant**: Dictionary lookup vs forward pass
3. **Memory overhead is tiny**: 4 bytes per sample
4. **Speedup is massive**: Up to 200× for typical training
5. **Works transparently**: Just enable in config

```python
# Enable caching (recommended!)
config.CACHE_MODE = 'memory'
config.PRECOMPUTE_REFERENCE_LOSSES = True
```
