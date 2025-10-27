# Server Transfer and Verification Checklist

## ✅ Pre-Transfer Checklist

- [ ] Read `EXPERIMENTAL_RESULTS.md` to understand expected results
- [ ] Read `SERVER_SETUP.md` for setup instructions  
- [ ] Verify you have server access: `ssh user@server`
- [ ] Check server has adequate disk space: ~500MB required

## 📦 Files to Transfer

### Essential (Minimum to run experiments)
- [ ] `mode.py` - Main implementation
- [ ] `run_server_experiments.sh` - Execution script

### Recommended (For full verification)
- [ ] `summarize_metrics.py` - Metrics analysis
- [ ] `EXPERIMENTAL_RESULTS.md` - Results reference
- [ ] `SERVER_SETUP.md` - Setup guide

### Optional (Additional context)
- [ ] `DEBUGGING_SUMMARY.md` - Debugging details
- [ ] `cachelib_strategies.py` - CacheLib implementation
- [ ] `mode_hypernetwork.py` - Hypernetwork code
- [ ] `visualize_mode_strategies.py` - Visualization tool

## 🚀 Transfer Commands

Choose one method:

**Method 1: SCP (Simple)**
```bash
cd /Users/tanmoy/research/Dataset_Distillation/Coreset/LLM_stuff
scp mode.py run_server_experiments.sh summarize_metrics.py \
    EXPERIMENTAL_RESULTS.md SERVER_SETUP.md \
    user@yourserver:~/experiment/
```

**Method 2: Rsync (Recommended)**
```bash
rsync -avz --progress \
    mode.py run_server_experiments.sh summarize_metrics.py \
    EXPERIMENTAL_RESULTS.md SERVER_SETUP.md \
    user@yourserver:~/experiment/
```

- [ ] Files transferred successfully
- [ ] Verify files on server: `ssh user@server "ls -lh ~/experiment/"`

## 🔧 Server Setup Checklist

### 1. Environment Setup
```bash
ssh user@server
cd ~/experiment
```

- [ ] Python 3.8+ installed: `python --version`
- [ ] Pip available: `pip --version`

### 2. Install Dependencies
```bash
pip install torch transformers datasets numpy matplotlib tensorboard
```

- [ ] PyTorch installed: `python -c "import torch; print(torch.__version__)"`
- [ ] Transformers installed: `python -c "import transformers; print(transformers.__version__)"`
- [ ] Datasets installed: `python -c "import datasets; print(datasets.__version__)"`

### 3. Check GPU (if available)
```bash
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}'); \
           print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"
```

- [ ] GPU detected (optional but recommended)
- [ ] CUDA available: Yes / No
- [ ] GPU memory: ______ GB

## ▶️ Execution Checklist

### 1. Make Script Executable
```bash
chmod +x run_server_experiments.sh
```

- [ ] Script is executable: `ls -la run_server_experiments.sh`

### 2. Run Experiments
```bash
./run_server_experiments.sh
```

**Choose option when prompted:**
- [ ] Option 1: All three methods (30-40 min, needs 16GB+ GPU)
- [ ] Option 2: Full + Rho-1 only (15-20 min, recommended)

### 3. Monitor Progress
Watch for these milestones:
- [ ] Dataset downloading (first run only)
- [ ] Model downloading (first run only)  
- [ ] Full-Training epoch 1 starts
- [ ] Full-Training completes (~5-10 min)
- [ ] Rho-1 epoch 1 starts
- [ ] Rho-1 completes (~5-10 min)

## 📊 Verification Checklist

### 1. Check Completion
```bash
ls -la ./server_results/
cat ./server_results/metrics_summary.log
```

- [ ] Experiments completed without errors
- [ ] Log files created
- [ ] TensorBoard logs exist: `ls ./runs_dual_mode_comparison/`

### 2. Verify Results

**Expected Results** (your results should be within ±2 PPL):
```
Full-Training:  50-52 PPL
Rho-1:          51-53 PPL  
```

**Your Results**:
```
Full-Training:  _____ PPL
Rho-1:          _____ PPL
```

- [ ] Full-Training PPL in expected range (50-52)
- [ ] Rho-1 PPL in expected range (51-53)
- [ ] Rho-1 within 5% of Full-Training
- [ ] Rho-1 uses 30% of tokens (after warmup)

### 3. Check Key Metrics

```bash
grep "Best Val Perplexity" ./server_results/*.log
grep "Token usage" ./server_results/*.log
```

**Full-Training**:
- [ ] Training loss decreases over epochs
- [ ] Token selection: 100% throughout
- [ ] Best PPL achieved in epoch 3-5

**Rho-1**:
- [ ] Training loss decreases over epochs  
- [ ] Epoch 1 selection: 100% (warmup)
- [ ] Epochs 2-3 selection: 30% (budget enforced)
- [ ] Best PPL typically in epoch 1

## 🎯 Success Criteria

Your experiments are successful if:

- [ ] ✅ Both methods complete without crashes
- [ ] ✅ Perplexities are reasonable (45-55 range)
- [ ] ✅ Rho-1 within 5% of Full-Training performance
- [ ] ✅ Rho-1 uses exactly 30% of tokens after warmup
- [ ] ✅ Logs generated in `./server_results/`
- [ ] ✅ No CUDA out of memory errors
- [ ] ✅ Results reproducible (can run again)

## 📋 Results Collection

### 1. Save Logs
```bash
cat ./server_results/metrics_summary.log > my_results.txt
cat ./server_results/system_info.txt >> my_results.txt
```

- [ ] Metrics summary saved
- [ ] System info saved

### 2. Package Results (Optional)
```bash
tar -czf results_$(date +%Y%m%d).tar.gz \
    server_results/ runs_dual_mode_comparison/
```

- [ ] Results packaged
- [ ] Ready to transfer back: `scp server:~/experiment/results_*.tar.gz .`

## 🐛 Troubleshooting

If issues occur, check:

### Out of Memory
- [ ] Reduce batch size to 1 in `mode.py`
- [ ] Reduce sequence length to 128
- [ ] Use CPU if GPU unavailable (slow)

### Slow Performance  
- [ ] Verify GPU is being used
- [ ] Reduce dataset size (train_samples: 100)
- [ ] Check CPU usage: `top` or `htop`

### Import Errors
- [ ] Re-install dependencies
- [ ] Check Python version (needs 3.8+)
- [ ] Virtual environment activated?

### Dataset Download Fails
- [ ] Check internet connection
- [ ] Try manual download first:
  ```python
  from datasets import load_dataset
  load_dataset('wikitext', 'wikitext-2-raw-v1')
  ```

## 📝 Notes Section

**Your Server Details**:
- Server name: _______________
- GPU type: _______________
- GPU memory: _______________
- Python version: _______________
- PyTorch version: _______________

**Execution Details**:
- Start time: _______________
- End time: _______________
- Total duration: _______________
- Option chosen: _______________

**Observations**:
- _________________________________
- _________________________________
- _________________________________

**Issues Encountered**:
- _________________________________
- _________________________________

**Solutions Applied**:
- _________________________________
- _________________________________

## ✉️ Reporting Back

Share these items:
1. [ ] Final perplexity values
2. [ ] `server_results/metrics_summary.log`
3. [ ] `server_results/system_info.txt`
4. [ ] Screenshots of key outputs (optional)
5. [ ] Any errors or warnings
6. [ ] Runtime duration

## 🎓 Reference Documents

- `EXPERIMENTAL_RESULTS.md` - Complete results and methodology
- `SERVER_SETUP.md` - Detailed setup guide
- `DEBUGGING_SUMMARY.md` - Known issues and fixes
- `FILES_FOR_SERVER.txt` - File descriptions

---

**Last Updated**: 2025-10-13
**Version**: 1.0
