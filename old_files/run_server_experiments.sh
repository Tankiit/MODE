#!/bin/bash
# Server Experiment Runner for MODE vs Rho-1 Comparison
# Run this on a server with adequate GPU memory (16GB+ recommended)

set -e  # Exit on error

echo "============================================"
echo "MODE vs Rho-1 Comparison - Server Runner"
echo "============================================"
echo ""

# Check Python and dependencies
echo "Checking Python environment..."
python --version
pip list | grep -E "torch|transformers|datasets" || {
    echo "ERROR: Missing required packages"
    echo "Install with: pip install torch transformers datasets numpy matplotlib tensorboard"
    exit 1
}

# Check GPU availability
echo ""
echo "Checking GPU availability..."
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"CPU\"}')"

# Create output directory
mkdir -p ./server_results
mkdir -p ./runs_dual_mode_comparison

echo ""
echo "============================================"
echo "Starting Experiments"
echo "============================================"
echo ""

# Option 1: Run all three experiments (Full, Rho-1, Dual-MODE)
echo "Option 1: Run all experiments (Full-Training, Rho-1, Dual-MODE)"
echo "Option 2: Run only Full-Training and Rho-1 (faster, proven)"
echo ""
read -p "Choose option (1 or 2): " option

case $option in
    1)
        echo ""
        echo "Running ALL experiments (may take 30-40 minutes)..."
        python mode.py 2>&1 | tee ./server_results/all_experiments.log
        ;;
    2)
        echo ""
        echo "Running Full-Training and Rho-1 only (15-20 minutes)..."
        # Create modified script that skips Dual-MODE
        python -c "
from mode import Config, Trainer, load_data
import sys

config = Config()
print('\\n' + '='*70)
print('MODE Comparison - Server Run (Full + Rho-1 Only)')
print('='*70)

# Load data
train_loader, val_texts = load_data(config)

results = {}

for name, method in [('Full-Training', 'full'), ('Rho-1', 'rho1')]:
    print(f'\\n{\"=\"*70}')
    print(f'Running: {name}')
    print(f'{\"=\"*70}')

    trainer = Trainer(config, name, method)
    result = trainer.train(train_loader, val_texts)
    results[name] = result

    print(f'\\n{name} complete')
    print(f'  Best Val Perplexity: {result[\"best_val_ppl\"]:.2f}')

# Final comparison
print(f'\\n\\n{\"=\"*70}')
print('FINAL RESULTS')
print(f'{\"=\"*70}')

for name in ['Full-Training', 'Rho-1']:
    ppl = results[name]['best_val_ppl']
    print(f'  {name:20s}: {ppl:7.2f} PPL')

full_ppl = results['Full-Training']['best_val_ppl']
rho1_ppl = results['Rho-1']['best_val_ppl']
improvement = (full_ppl - rho1_ppl) / full_ppl * 100

print(f'\\n{\"=\"*70}')
print('Analysis:')
print(f'  Rho-1 vs Full: {improvement:+.2f}%')
print(f'  Token savings: 70% (using only 30% of tokens)')
print(f'{\"=\"*70}\\n')
" 2>&1 | tee ./server_results/full_rho1_only.log
        ;;
    *)
        echo "Invalid option. Exiting."
        exit 1
        ;;
esac

# Generate summary
echo ""
echo "============================================"
echo "Generating Results Summary"
echo "============================================"
echo ""

if [ -f summarize_metrics.py ]; then
    python summarize_metrics.py 2>&1 | tee ./server_results/metrics_summary.log
else
    echo "summarize_metrics.py not found, skipping visualization"
fi

# Save system info
echo ""
echo "Saving system information..."
python -c "
import torch
import sys
import platform

with open('./server_results/system_info.txt', 'w') as f:
    f.write('System Information\\n')
    f.write('='*50 + '\\n')
    f.write(f'Python: {sys.version}\\n')
    f.write(f'Platform: {platform.platform()}\\n')
    f.write(f'PyTorch: {torch.__version__}\\n')
    f.write(f'CUDA Available: {torch.cuda.is_available()}\\n')
    if torch.cuda.is_available():
        f.write(f'CUDA Version: {torch.version.cuda}\\n')
        f.write(f'GPU: {torch.cuda.get_device_name(0)}\\n')
        f.write(f'GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB\\n')
"

echo ""
echo "============================================"
echo "Experiments Complete!"
echo "============================================"
echo ""
echo "Results saved in:"
echo "  - ./server_results/          (logs and summaries)"
echo "  - ./runs_dual_mode_comparison/  (tensorboard logs)"
echo ""
echo "To view detailed metrics:"
echo "  cat ./server_results/metrics_summary.log"
echo ""
echo "To view TensorBoard:"
echo "  tensorboard --logdir=./runs_dual_mode_comparison"
echo ""
