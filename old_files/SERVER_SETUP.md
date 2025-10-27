# Server Setup and Verification Guide

## 📦 Quick Start

### 1. Transfer Files to Server

Copy these files to your server:
```bash
scp mode.py user@server:~/experiment/
scp cachelib_strategies.py user@server:~/experiment/
scp mode_hypernetwork.py user@server:~/experiment/
scp run_server_experiments.sh user@server:~/experiment/
scp summarize_metrics.py user@server:~/experiment/
scp EXPERIMENTAL_RESULTS.md user@server:~/experiment/
```

Or use rsync:
```bash
rsync -avz *.py *.sh *.md user@server:~/experiment/
```

### 2. Install Dependencies

On the server:
```bash
pip install torch transformers datasets numpy matplotlib tensorboard
```

For FAISS (optional, for faster retrieval):
```bash
# CPU version
pip install faiss-cpu

# GPU version (if CUDA available)
pip install faiss-gpu
```

### 3. Run Experiments

```bash
cd ~/experiment
chmod +x run_server_experiments.sh
./run_server_experiments.sh
```

The script will:
1. Check Python environment
2. Verify GPU availability
3. Let you choose which experiments to run
4. Save all results to `./server_results/`

---

## 🎯 Expected Results

### Option 1: Full-Training + Rho-1 (Recommended)
**Time**: 15-20 minutes
**Memory**: 8-12GB VRAM
**Output**:
```
Full-Training: ~51.25 PPL
Rho-1:         ~51.81 PPL
Improvement:   -1.1% (70% token savings)
```

### Option 2: All Three Methods
**Time**: 30-40 minutes (if Dual-MODE completes)
**Memory**: 16GB+ VRAM recommended
**Output**: All three methods with complete metrics

---

## 📊 Verification Checklist

After running, verify these results:

### ✅ Full-Training
- [ ] Best validation perplexity: 50-52 range
- [ ] Training loss decreases: ~3.9 → ~3.0
- [ ] Token selection: 100% throughout
- [ ] 3-6 epochs completed
- [ ] No errors or crashes

### ✅ Rho-1
- [ ] Best validation perplexity: 51-53 range
- [ ] Training loss decreases: ~3.9 → ~2.7
- [ ] Token selection: 100% (epoch 1), 30% (epochs 2-3)
- [ ] Perplexity within 2% of Full-Training
- [ ] 3 epochs completed

### ✅ Dual-MODE (if running)
- [ ] Training progresses without errors
- [ ] Memory usage stable
- [ ] Meta-controller trains successfully
- [ ] Strategy weights adapt over time
- [ ] 3 epochs completed

---

## 🐛 Troubleshooting

### Issue: Out of Memory

**GPU OOM**:
```bash
# Reduce batch size in mode.py
batch_size: int = 1  # Or even smaller

# Reduce sequence length
max_seq_length: int = 128  # From 256

# Use gradient accumulation
grad_accumulation: int = 4  # Simulate larger batches
```

**For Dual-MODE specifically**:
```bash
# In run_dual_mode_fast.py
config.max_memory_contexts = 100  # From 500
config.retrieval_k = 5  # From 10
```

### Issue: Slow Training

**Check device**:
```python
python -c "import torch; print(torch.cuda.is_available())"
```

If False, model is running on CPU (very slow).

**Solutions**:
1. Reduce dataset size:
   ```python
   train_samples: int = 100  # From 200
   val_samples: int = 25     # From 50
   ```

2. Reduce model size (not using pretrained):
   ```python
   use_pretrained_init: bool = False
   n_layer: int = 2
   n_embd: int = 128
   ```

### Issue: Import Errors

**Missing packages**:
```bash
pip install torch transformers datasets numpy matplotlib tensorboard
```

**FAISS errors (optional)**:
```python
# In mode.py, FAISS is optional
# The code will fall back to brute-force search
# You can safely ignore FAISS warnings
```

### Issue: Dataset Download Fails

**Solution**:
```python
# Pre-download dataset
from datasets import load_dataset
dataset = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train')
# This will cache it for future runs
```

---

## 📈 Understanding the Output

### Training Progress
```
Epoch 1:  50%|█████     | 100/200 [01:30<01:30, Loss: 3.62 | Sel: 100.0%]
```
- `50%`: Progress through epoch
- `100/200`: Current batch / total batches
- `Loss: 3.62`: Current training loss
- `Sel: 100.0%`: Percentage of tokens selected

### Epoch Summary
```
Epoch Summary:
  Train Loss: 3.94
  Val Perplexity: 51.25
  Avg Selection: 100.0%
  New best perplexity!
```
- `Train Loss`: Average loss over all batches
- `Val Perplexity`: Model performance on validation set (lower is better)
- `Avg Selection`: Average % of tokens used
- `New best`: This epoch achieved best validation score

### Final Results
```
FINAL RESULTS
======================================================================
  Full-Training       :   51.25 PPL
  Rho-1               :   51.81 PPL

Improvements vs Full Training:
  Rho-1:      -1.10%
```
- Negative % means slightly worse performance
- Rho-1 uses 70% fewer tokens for only 1.1% worse performance

---

## 🔬 Detailed Metrics Files

After running, check these files:

### `./server_results/metrics_summary.log`
Complete metrics with:
- Per-epoch training losses
- Validation perplexities
- Token selection rates
- Strategy weights (Dual-MODE)

### `./runs_dual_mode_comparison/`
TensorBoard logs for visualization:
```bash
tensorboard --logdir=./runs_dual_mode_comparison --port=6006
# Then visit http://server:6006 in browser
```

### `./server_results/system_info.txt`
Hardware information:
- GPU model and memory
- CUDA version
- PyTorch version

---

## 📊 Expected Performance by Hardware

### High-End Server (A100, 40GB)
- All experiments: ✅
- Dual-MODE: ✅ Complete
- Time: 30-40 minutes

### Mid-Range GPU (RTX 3090, 24GB)
- Full + Rho-1: ✅
- Dual-MODE: ✅ Likely complete
- Time: 40-50 minutes

### Consumer GPU (RTX 3060, 12GB)
- Full + Rho-1: ✅
- Dual-MODE: ⚠️ May OOM (reduce batch size to 1)
- Time: 50-60 minutes

### CPU Only
- Full + Rho-1: ✅ (very slow)
- Time: 3-4 hours
- Not recommended for Dual-MODE

---

## ✅ Success Criteria

Your experiments are successful if:

1. **Both methods complete** without errors
2. **Perplexities are reasonable** (45-55 range)
3. **Rho-1 within 5% of Full-Training** performance
4. **Token selection is 30%** for Rho-1 (after warmup)
5. **Logs are generated** in `./server_results/`

---

## 📧 Reporting Results

When sharing results, include:

1. **Command used**:
   ```bash
   ./run_server_experiments.sh
   # Option chosen: 1 or 2
   ```

2. **Final perplexities**:
   ```
   Full-Training: XX.XX PPL
   Rho-1:         XX.XX PPL
   ```

3. **System info**:
   ```bash
   cat ./server_results/system_info.txt
   ```

4. **Any errors or warnings**

5. **Runtime**:
   ```
   Full-Training: XX minutes
   Rho-1: XX minutes
   ```

---

## 🚀 Advanced: Custom Configurations

### Reduce Memory Usage
Edit `mode.py`:
```python
# Smaller sequences
max_seq_length: int = 128

# Smaller batch
batch_size: int = 1

# More accumulation steps
grad_accumulation: int = 8
```

### Faster Experiments
```python
# Fewer samples
train_samples: int = 100
val_samples: int = 25

# Fewer epochs
epochs: int = 2
```

### Higher Quality Results
```python
# More data
train_samples: int = 1000
val_samples: int = 200

# More epochs
epochs: int = 5

# Larger model
pretrained_model: str = 'gpt2'  # Instead of 'distilgpt2'
```

---

## 📚 Additional Resources

- **Full methodology**: `EXPERIMENTAL_RESULTS.md`
- **Debugging guide**: `DEBUGGING_SUMMARY.md`
- **Code**: `mode.py`, `mode_hypernetwork.py`
- **Visualization**: `analyze_mode_metrics.py`

---

## ⚡ Quick Commands

```bash
# Run experiments
./run_server_experiments.sh

# View results
cat ./server_results/metrics_summary.log

# Check TensorBoard logs
ls -la ./runs_dual_mode_comparison/

# View system info
cat ./server_results/system_info.txt

# Re-run just analysis
python summarize_metrics.py
```

---

**Questions?** Check `EXPERIMENTAL_RESULTS.md` for complete documentation.
