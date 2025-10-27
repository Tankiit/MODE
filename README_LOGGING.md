# Selective Multi-Classifier Coreset Selection with TensorBoard & WandB

> **Real-time experiment tracking for coreset selection with deep learning**

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![TensorBoard](https://img.shields.io/badge/TensorBoard-enabled-orange.svg)](https://www.tensorflow.org/tensorboard)
[![WandB](https://img.shields.io/badge/WandB-integrated-yellow.svg)](https://wandb.ai/)

## Overview

This project implements **selective multi-classifier coreset selection** with comprehensive experiment tracking using **TensorBoard** and **Weights & Biases**. Track every aspect of your training in real-time!

### Key Features

- **4 Scoring Functions**: Uncertainty, Diversity, Boundary, Excess Loss
- **Adaptive Gating**: Learn which scorer to trust
- **Confidence-Based Rejection**: Filter unreliable predictions
- **Sample Waste Tracking**: Identify inefficient training samples
- **Multi-Ratio Experiments**: Test different selection ratios automatically
- **Multi-Model Support**: Compare architectures (ResNet, EfficientNet, etc.)
- **TensorBoard Integration**: Local visualization and analysis
- **WandB Integration**: Cloud-based tracking and collaboration

## Quick Start

### 1. Installation

```bash
# Clone or navigate to the repository
cd /path/to/LLM_stuff

# Install dependencies
bash setup_logging.sh

# Or manually:
pip install -r requirements_logging.txt
```

### 2. Setup WandB (Optional)

```bash
# Login to WandB
wandb login

# Or skip if you only want TensorBoard
# Set USE_WANDB = False in run_with_logging.py
```

### 3. Run Your First Experiment

```bash
# Quick test (10 epochs, 3 ratios)
python run_with_logging.py
```

### 4. View Results

**TensorBoard** (open in another terminal):
```bash
tensorboard --logdir=./runs
# Visit: http://localhost:6006
```

**Weights & Biases**:
- Check the URL printed in console
- Or visit: https://wandb.ai/your-username/cifar10-selective-coreset

## What You Get

### Real-Time Metrics

| Category | Metrics | Visualization |
|----------|---------|---------------|
| **Training** | Loss, Accuracy, LR, Gradient Norms | Line plots, moving averages |
| **Validation** | Accuracy, Loss, Best Accuracy | Comparison across experiments |
| **Selection** | Ratio, Gate Weights, Rejection Rates | Time series, distributions |
| **Sample Waste** | Mastered Early, Stagnant, Harmful | Stacked area charts |
| **Model** | Architecture, Weights, Gradients | Histograms, graphs |

### Example TensorBoard View

```
Scalars Tab:
├── exp_0/train/
│   ├── batch_loss
│   ├── epoch_acc
│   └── learning_rate
├── exp_0/val/
│   ├── acc
│   └── best_acc
├── exp_0/selection/
│   ├── num_selected
│   └── ratio
└── exp_0/gate/
    ├── scorer_0_weight  (Uncertainty)
    ├── scorer_1_weight  (Diversity)
    ├── scorer_2_weight  (Boundary)
    └── scorer_3_weight  (Excess Loss)

Histograms Tab:
├── exp_0/weights/
│   ├── layer1.0.conv1.weight
│   ├── layer2.0.conv1.weight
│   └── fc.weight
```

### Example WandB Dashboard

- **Overview**: All experiments in one place
- **Comparison**: Side-by-side metric comparison
- **System**: GPU/CPU usage, memory
- **Charts**: Interactive plots with zooming
- **Tables**: Sample-level analysis
- **Reports**: Shareable experiment summaries

## 📁 Project Structure

```
.
├── main_multiexp_with_logging.py    # Enhanced training with logging
├── run_with_logging.py              # Quick start script
├── LOGGING_GUIDE.md                 # Detailed logging documentation
├── README_LOGGING.md                # This file
├── requirements_logging.txt         # Dependencies
├── setup_logging.sh                 # Installation script
│
├── runs/                            # TensorBoard logs
│   ├── exp_0_resnet18_ratio_0.30/
│   ├── exp_1_resnet18_ratio_0.50/
│   └── exp_2_resnet18_ratio_0.70/
│
├── experiment_results/              # JSON/CSV results
│   └── cifar10_logged_*/
│       ├── results.json
│       ├── results_summary.csv
│       └── plots/
│
├── checkpoints/                     # Model checkpoints
│   └── cifar10_logged_*/
│
└── wandb/                          # WandB logs
```

## 🎓 Usage Examples

### Basic Usage

```python
# Edit run_with_logging.py
config.NUM_EPOCHS = 10
config.RATIOS_TO_TEST = [0.3, 0.5, 0.7]
config.USE_TENSORBOARD = True
config.USE_WANDB = True

# Run
python run_with_logging.py
```

### Multi-Model Comparison

```python
config.ENABLE_MULTI_MODEL = True
config.MODELS_TO_TEST = [
    ("resnet18", "resnet18"),
    ("resnet34", "resnet18"),
    ("resnet50", "resnet34"),
]
config.RATIOS_TO_TEST = [0.3, 0.5, 0.7]

# This will run 3 models × 3 ratios = 9 experiments
```

### Custom Logging

```python
# In your training loop
logger_obj.log_metrics({
    'custom/my_metric': my_value,
    'custom/another_metric': another_value
}, step=global_step)

# Log images
logger_obj.log_image('predictions/batch_0', image_tensor, step)

# Log histograms
logger_obj.log_histogram('activations/layer1', activations, step)
```

## Monitoring During Training

### Terminal Output
```
2025-10-26 19:45:23 - INFO - EXPERIMENT 0
2025-10-26 19:45:23 - INFO - Model: resnet18, Reference: resnet18
2025-10-26 19:45:23 - INFO - Selection Ratio: 0.3
2025-10-26 19:45:25 - INFO - TensorBoard logging to: ./runs/exp_0...
2025-10-26 19:45:26 - INFO - WandB logging initialized: https://wandb.ai/...

Epoch 1/10: 100%|████████| 469/469 [02:15<00:00]
  loss: 0.4532, acc: 86.24%

2025-10-26 19:47:41 - INFO - Epoch 1: Train Loss=0.4532, Train Acc=0.8624, Val Acc=0.8521
```

### TensorBoard Commands

```bash
# Basic usage
tensorboard --logdir=./runs

# Custom port
tensorboard --logdir=./runs --port=6007

# Compare specific experiments
tensorboard --logdir_spec=\
  baseline:./runs/exp_0,\
  high_ratio:./runs/exp_2

# Reload interval (for live updates)
tensorboard --logdir=./runs --reload_interval=5
```

### WandB Features

```python
# View in browser (automatically opens)
# Or visit dashboard manually

# Key features:
- Real-time metric updates
- Experiment comparison
- Hyperparameter importance
- System monitoring (GPU/CPU)
- Custom plots and tables
- Collaborative annotations
- Export to PDF/PNG
```

## Configuration

### Logging Settings

```python
# In run_with_logging.py

# Enable/disable backends
config.USE_TENSORBOARD = True
config.USE_WANDB = True

# WandB project settings
config.WANDB_PROJECT = "your-project-name"
config.WANDB_ENTITY = "your-username"

# Logging frequency
config.LOG_INTERVAL = 10        # Log every 10 batches
config.EVAL_FREQUENCY = 5       # Validate every 5 epochs

# What to log
config.LOG_IMAGES = True        # Sample images
config.LOG_HISTOGRAMS = True    # Weight distributions
config.LOG_GRADIENTS = True     # Gradient norms
```

### Training Settings

```python
# Quick test
config.NUM_EPOCHS = 10
config.BATCH_SIZE = 32
config.RATIOS_TO_TEST = [0.3, 0.5, 0.7]

# Full training
config.NUM_EPOCHS = 200
config.BATCH_SIZE = 128
config.RATIOS_TO_TEST = [0.1, 0.2, 0.3, 0.5, 0.7, 0.9]
```

## Example Results

After running experiments, you'll see:

### Console Output
```
======================================================================
EXPERIMENT SUMMARY
======================================================================

   Exp ID     Model  Ratio  Final Acc  Best Val Acc
0       0  resnet18    0.3     0.8521        0.8563
1       1  resnet18    0.5     0.8734        0.8801
2       2  resnet18    0.7     0.8856        0.8923

======================================================================
BEST CONFIGURATION
======================================================================
Model: resnet18
Selection Ratio: 0.7
Final Accuracy: 0.8856
```

### TensorBoard Charts
- Training/validation curves
- Gate weight evolution
- Sample waste tracking
- Model weight distributions

### WandB Dashboard
- Interactive comparison plots
- Hyperparameter importance
- Run comparison table
- System resource usage

## 🐛 Troubleshooting

### TensorBoard not showing data?
```bash
# Clear cache
rm -rf /tmp/.tensorboard-info/
tensorboard --logdir=./runs --reload_interval=5
```

### WandB connection issues?
```bash
# Check status
wandb status

# Re-authenticate
wandb login --relogin

# Use offline mode
export WANDB_MODE=offline
# Sync later: wandb sync wandb/run-*/
```

### Out of memory?
```python
# Reduce batch size
config.BATCH_SIZE = 32

# Disable heavy logging
config.LOG_IMAGES = False
config.LOG_HISTOGRAMS = False

# Reduce logging frequency
config.LOG_INTERVAL = 100
```

## 📚 Documentation

- **[LOGGING_GUIDE.md](LOGGING_GUIDE.md)**: Comprehensive logging guide
- **[TensorBoard Docs](https://www.tensorflow.org/tensorboard)**: Official TensorBoard documentation
- **[WandB Docs](https://docs.wandb.ai/)**: Official Weights & Biases documentation

## Tips & Best Practices

1. **Start Small**: Run quick tests (10 epochs) before full training
2. **Use Tags**: Organize experiments with wandb tags
3. **Monitor Early**: Open TensorBoard before starting training
4. **Compare Systematically**: Run baseline first, then variants
5. **Save Checkpoints**: Enable `SAVE_CHECKPOINTS = True`
6. **Document Everything**: WandB auto-logs git commit hash

## Advanced Features

### Hyperparameter Sweeps (WandB)

```python
import wandb

sweep_config = {
    'method': 'bayes',
    'metric': {'name': 'val/acc', 'goal': 'maximize'},
    'parameters': {
        'selection_ratio': {'values': [0.1, 0.3, 0.5, 0.7, 0.9]},
        'learning_rate': {'min': 0.001, 'max': 0.1},
        'batch_size': {'values': [32, 64, 128]}
    }
}

sweep_id = wandb.sweep(sweep_config, project="cifar10-selective-coreset")
wandb.agent(sweep_id, function=train)
```

### Custom Callbacks

```python
class CustomLogger(ExperimentLogger):
    def on_epoch_end(self, epoch, metrics):
        # Custom logging logic
        super().log_metrics(metrics, step=epoch)
        # Send email, Slack notification, etc.
```

## 📞 Support

- **Issues**: Report bugs or request features
- **Discussions**: Ask questions, share results
- **Email**: [Your contact]

## 📄 License

[Your License]

## 🙏 Acknowledgments

- PyTorch team for the framework
- timm library for pretrained models
- TensorBoard for visualization
- Weights & Biases for experiment tracking

---

**Happy Experimenting!**

View your experiments in real-time and make data-driven decisions!
