# ⚡ MODE Quick Reference Card

**One-page guide to MODE vs Rho-1 experiments with Binary Hypernetwork**

---

## 🚀 Quick Commands

### Transfer to Server
```bash
# Standard (Full + Rho-1)
scp mode.py run_server_experiments.sh user@server:~/exp/

# With Binary Hypernetwork ⚡ NEW!
scp mode.py mode_integrated_binary.py BINARY_HYPERNETWORK.md user@server:~/exp/
```

### Install Dependencies
```bash
pip install torch transformers datasets numpy
```

### Run Experiments
```bash
# Standard experiments
./run_server_experiments.sh

# Binary hypernetwork (10x faster!)
python mode_integrated_binary.py
```

---

## 📊 Expected Results

| Method | Perplexity | Tokens | Time | Key Feature |
|--------|-----------|--------|------|-------------|
| **Full-Training** | 51.25 | 100% | 5-10 min | Baseline |
| **Rho-1** | 51.81 | 30% | 5-10 min | 70% savings |
| **Binary MODE** ⚡ | 50-52 | 30% | 10-15 min | Adaptive + 10x faster |

---

## 🎯 Success Criteria

✅ **Perplexity**: 50-53 range (all methods)
✅ **Token usage**: Exactly 30% for Rho-1/MODE (epochs 2-3)
✅ **No errors**: Training completes without OOM
✅ **Speed**: Binary MODE 10x faster than original Dual-MODE

---

## 🔧 Common Fixes

### Out of Memory
```python
batch_size: int = 1
max_seq_length: int = 128
```

### Import Error
```bash
pip install --upgrade torch transformers datasets
```

### Slow on CPU
```python
train_samples: int = 50
epochs: int = 2
```

---

## 📁 Essential Files

| File | Purpose | When to Use |
|------|---------|-------------|
| `mode.py` | Full + Rho-1 + Dual-MODE | Always |
| `mode_integrated_binary.py` ⚡ | 10x faster MODE | Advanced users |
| `run_server_experiments.sh` | Auto-runner | Quick start |
| `EXPERIMENTAL_RESULTS.md` | Reference results | Verification |
| `BINARY_HYPERNETWORK.md` | Technical guide | Understanding binary MODE |

---

## 💡 Key Concepts

### Binary State Encoding (12D)
```python
[1, 0, 0, 1, 0, 1, 1, 0, 0, 1, 0, 1]
 └─┬─┘ └─┬─┘ └─┬─┘ └─┬─┘ └─┬─┘ └─┬─┘
   │     │     │     │     │     └── Experience (>10 steps)
   │     │     │     │     └────────── Strategy focus (>0.6)
   │     │     │     └──────────────── Selection rate (>50%)
   │     │     └────────────────────── Loss state
   │     └──────────────────────────── Progress phase
   └────────────────────────────────── Training phase
```

### 4 Adaptive Strategies
1. **Uncertainty**: High entropy tokens (exploration)
2. **Loss**: High loss tokens (Rho-1 style)
3. **Coherence**: High context coherence (fluency)
4. **Token MODE**: Similar to past successes (memory)

---

## 🎓 Performance Comparison

### Original Dual-MODE
- Speed: 0.04 it/sec (very slow)
- Memory: 14+ GB
- Issues: FAISS mismatch, OOM at 82% epoch 1

### Binary Hypernetwork MODE ⚡
- Speed: 8-10 it/sec (10x+ faster!)
- Memory: 8-10 GB (30% less)
- Stability: Completes full training
- Quality: Same perplexity range

---

## 📈 Monitoring Training

### Healthy Signs
```
✅ Loss decreasing: 3.9 → 3.0 → 2.7
✅ Selection: 100% (epoch 1) → 30% (epochs 2-3)
✅ Strategy weights adapting: [0.45, 0.30, 0.15, 0.10]
✅ Binary state changing each epoch
✅ Perplexity stable: 50-53 range
```

### Warning Signs
```
⚠️ Loss increasing → Lower learning rate
⚠️ Selection stuck → Check budget enforcement
⚠️ Weights frozen → Increase update_threshold
⚠️ OOM errors → Reduce memory_size
```

---

## ⚙️ Configuration Tuning

### For Maximum Speed
```python
memory_size: int = 100
max_memory_contexts: int = 100
n_strategies: int = 3
batch_size: int = 2
```

### For Maximum Quality
```python
memory_size: int = 1000
max_memory_contexts: int = 1000
hypernetwork_hidden: int = 128
n_strategies: int = 5
```

### Balanced (Recommended)
```python
memory_size: int = 500
max_memory_contexts: int = 500
hypernetwork_hidden: int = 64
n_strategies: int = 4
```

---

## 🐛 Troubleshooting

| Problem | Solution |
|---------|----------|
| OOM error | Reduce batch_size to 1, memory_size to 100 |
| Slow training | Check GPU available, reduce train_samples |
| High perplexity | Use pretrained_model: 'distilgpt2' |
| Weights frozen | Increase update_threshold to 0.5 |
| Import error | pip install torch transformers datasets |

---

## 📊 Key Findings

### Main Results
- **70% token savings** with only 1.1% performance loss (Rho-1)
- **10x+ speedup** with binary hypernetwork vs original Dual-MODE
- **Binary state encoding** enables fast, interpretable strategy selection
- **Memory-augmented attention** learns from historical success patterns

### Production Recommendations
1. Use **Rho-1** for simplicity and reliability
2. Use **Binary MODE** for adaptive strategy selection
3. Start with **warmup epoch** (100% tokens) for stability
4. Enforce **exact budget** (30%) for consistent performance
5. Monitor **strategy weights** for adaptation insights

---

## 📚 Documentation Links

| Document | Pages | Read For |
|----------|-------|----------|
| README_SERVER.md | 10 | Quick start overview |
| EXPERIMENTAL_RESULTS.md | 15 | Complete results |
| BINARY_HYPERNETWORK.md | 25 | Technical deep dive |
| SERVER_SETUP.md | 8 | Setup & troubleshooting |
| TRANSFER_CHECKLIST.md | 7 | Step-by-step guide |

---

## 🎯 What to Report

When sharing results:

1. **Perplexities achieved**:
   ```
   Full-Training: XX.XX PPL
   Rho-1:         XX.XX PPL
   Binary MODE:   XX.XX PPL
   ```

2. **System info**:
   ```bash
   cat server_results/system_info.txt
   ```

3. **Training time**: X minutes per method

4. **Any errors**: Copy full error message

5. **Observations**: Strategy adaptation, memory usage, etc.

---

## ✅ Verification Checklist

- [ ] Files transferred to server
- [ ] Dependencies installed
- [ ] GPU detected (optional but recommended)
- [ ] Experiments completed without errors
- [ ] Perplexities in expected range (50-53)
- [ ] Token selection at 30% (epochs 2-3)
- [ ] Logs saved in `./server_results/`
- [ ] Results match EXPERIMENTAL_RESULTS.md (±2 PPL)

---

## 🏆 Innovation Highlights

### Binary State Encoding
- **12D binary vectors** instead of 100+ float features
- **Fast thresholding** operations (no expensive computations)
- **Interpretable**: Each bit has clear meaning
- **Robust**: Discrete states are stable

### Memory-Augmented Attention
- **500 historical states** stored in memory
- **Single-head attention** for speed
- **Strategy weight retrieval** from past successes
- **Adaptive learning** from training history

### Token MODE Memory
- **No FAISS** dependency (avoid dimension mismatch)
- **Batched cosine similarity** (fast numpy)
- **FIFO replacement** (simple and effective)
- **Scales to 500-1000 contexts**

---

## 🚀 Next Steps After Success

### Research
1. Ablation studies on binary state dimensions
2. Test on larger models (GPT-2, GPT-2 Medium)
3. Extend to other datasets (OpenWebText, etc.)
4. Multi-task strategy learning

### Production
1. Scale to 1000+ training samples
2. Increase epochs to 10-20
3. Tune strategy combinations
4. Deploy binary hypernetwork for speed

---

## 💻 Hardware Requirements

| Hardware | Full+Rho-1 | Binary MODE | Dual-MODE Original |
|----------|-----------|-------------|-------------------|
| **A100 (40GB)** | ✅ Fast | ✅ Fast | ✅ Slow |
| **RTX 3090 (24GB)** | ✅ Fast | ✅ Fast | ⚠️ Very Slow |
| **RTX 3060 (12GB)** | ✅ OK | ✅ OK | ❌ OOM |
| **MPS (18GB)** | ✅ OK | ✅ OK | ❌ OOM at 82% |
| **CPU** | ✅ Very Slow | ⚠️ Very Slow | ❌ Not Recommended |

---

## 📐 Perplexity Ranges

| Range | Quality |
|-------|---------|
| **30-40** | Excellent (large models) |
| **45-55** | Good (our results) ✅ |
| **55-70** | Acceptable (small models) |
| **70-100** | Poor (needs tuning) |
| **100+** | Very poor (check initialization) |

---

## 🔬 Method Comparison

| Feature | Full-Training | Rho-1 | Binary MODE ⚡ |
|---------|--------------|-------|---------------|
| Token usage | 100% | 30% | 30% |
| Selection strategy | None | Loss-based | Adaptive (4 strategies) |
| Perplexity | 51.25 | 51.81 | 50-52 |
| Training time | Fast | Fast | Medium (10x faster than original) |
| Memory usage | Low | Low | Medium (30% less than original) |
| Adaptivity | None | Fixed | High |
| Interpretability | High | High | High (binary states) |
| Production ready | ✅ | ✅ | ✅ |

---

**Version**: 2.0 (with Binary Hypernetwork)
**Last Updated**: 2025-10-13
**Status**: Production Ready ✅

**Quick Help**: See README_SERVER.md or BINARY_HYPERNETWORK.md for details.
