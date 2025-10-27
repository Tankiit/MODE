# TensorBoard & Weights & Biases Logging Guide

## 📊 Overview

This guide shows you how to use TensorBoard and Weights & Biases (wandb) for tracking your selective coreset experiments.

## 🚀 Quick Start

### 1. Install Dependencies

```bash
# Core dependencies (if not already installed)
pip install torch torchvision timm pandas matplotlib seaborn scipy tqdm

# TensorBoard (included with PyTorch)
pip install tensorboard

# Weights & Biases (optional but recommended)
pip install wandb
```

### 2. Setup WandB (Optional)

If you want to use Weights & Biases for cloud-based tracking:

```bash
# Login to wandb (one-time setup)
wandb login

# Or set API key
export WANDB_API_KEY=your_api_key_here
```

**Note**: If you don't want to use wandb, set `config.USE_WANDB = False` in the script.

### 3. Run Experiment with Logging

```bash
# Quick test (10 epochs, 3 ratios)
python run_with_logging.py

# Full training (edit the script first)
# Set: config.NUM_EPOCHS = 200
# Set: config.RATIOS_TO_TEST = [0.1, 0.2, 0.3, 0.5, 0.7, 0.9]
python run_with_logging.py
```

### 4. View Results

**TensorBoard**:
```bash
# Open in another terminal while training
tensorboard --logdir=./runs

# Then visit: http://localhost:6006
```

**Weights & Biases**:
- Visit your wandb dashboard (link printed in console)
- Or go to: https://wandb.ai/your-username/cifar10-selective-coreset

## 📈 What Gets Logged

### Training Metrics (Every Batch/Epoch)

| Metric | Description | Logged To |
|--------|-------------|-----------|
| `train/batch_loss` | Loss per batch | TB, WandB |
| `train/batch_acc` | Accuracy per batch | TB, WandB |
| `train/epoch_loss` | Average epoch loss | TB, WandB |
| `train/epoch_acc` | Epoch accuracy | TB, WandB |
| `train/learning_rate` | Current LR | TB, WandB |
| `train/grad_norm` | Gradient L2 norm | TB, WandB |

### Validation Metrics

| Metric | Description | Logged To |
|--------|-------------|-----------|
| `val/acc` | Validation accuracy | TB, WandB |
| `val/loss` | Validation loss | TB, WandB |
| `val/best_acc` | Best validation accuracy | TB, WandB |

### Selection Metrics

| Metric | Description | Logged To |
|--------|-------------|-----------|
| `selection/num_selected` | Number of samples selected | TB, WandB |
| `selection/ratio` | Selection ratio used | TB, WandB |
| `gate/scorer_X_weight` | Gating weight for scorer X | TB, WandB |
| `rejection/scorer_X_rate` | Rejection rate for scorer X | TB, WandB |

### Sample Waste Metrics

| Metric | Description | Logged To |
|--------|-------------|-----------|
| `waste/mastered_early` | Samples mastered too early | TB, WandB |
| `waste/stagnant` | Samples not improving | TB, WandB |
| `waste/harmful` | Potentially harmful samples | TB, WandB |
| `waste/useful` | Useful samples | TB, WandB |

### Model Information

| Type | Description | Logged To |
|------|-------------|-----------|
| Model Graph | Network architecture | TB |
| Weight Histograms | Parameter distributions | TB |
| Gradient Flow | Gradient statistics | TB |
| Model Watching | Real-time parameter tracking | WandB |

## 🎯 TensorBoard Features

### 1. Scalars View
Track metrics over time:
- Training loss and accuracy curves
- Validation performance
- Selection ratios and gate weights
- Learning rate schedule

### 2. Histograms View
Visualize distributions:
- Weight distributions per layer
- Gradient distributions
- Activation patterns

### 3. Graphs View
Explore model architecture:
- Network structure
- Layer connections
- Parameter counts

### 4. HParams View
Compare experiments:
- Different selection ratios
- Different models
- Hyperparameter effects

### 5. Time Series
Analyze training dynamics:
- Convergence patterns
- Overfitting detection
- Sample waste evolution

## 🌐 Weights & Biases Features

### 1. Real-Time Tracking
- Live metric updates during training
- Email/Slack notifications on completion
- Mobile app for monitoring

### 2. Experiment Comparison
```python
# Compare multiple runs automatically
# WandB shows all experiments in interactive plots
```

### 3. Hyperparameter Sweeps
```python
# Define sweep configuration
sweep_config = {
    'method': 'bayes',
    'parameters': {
        'selection_ratio': {'values': [0.1, 0.3, 0.5, 0.7, 0.9]},
        'learning_rate': {'min': 0.001, 'max': 0.1}
    }
}
```

### 4. Artifacts & Model Registry
- Save and version models
- Track dataset versions
- Share reproducible results

### 5. Reports
- Create shareable experiment reports
- Embed interactive plots
- Collaborative annotations

## 📊 Example Visualizations

### Accuracy vs Selection Ratio
```python
# WandB automatically creates comparison plots
# TensorBoard HParams tab shows parallel coordinates
```

### Gate Weight Evolution
```python
# Track how gating network adapts over training
# See which scorers are trusted at different stages
```

### Sample Waste Analysis
```python
# Visualize wasted training samples
# Identify inefficiencies in selection
```

## 🔧 Configuration Options

### In `run_with_logging.py`:

```python
# Enable/disable logging backends
config.USE_TENSORBOARD = True
config.USE_WANDB = True

# WandB settings
config.WANDB_PROJECT = "your-project-name"
config.WANDB_ENTITY = "your-username-or-team"

# Logging frequency
config.LOG_INTERVAL = 10  # Log every 10 batches
config.EVAL_FREQUENCY = 5  # Validate every 5 epochs

# What to log
config.LOG_IMAGES = True  # Log sample images
config.LOG_HISTOGRAMS = True  # Log weight distributions
config.LOG_GRADIENTS = True  # Log gradient norms

# Directories
config.TENSORBOARD_DIR = "./runs"
config.RESULTS_DIR = "./experiment_results"
```

## 📁 Output Structure

```
.
├── runs/                          # TensorBoard logs
│   ├── exp_0_resnet18_ratio_0.30/
│   │   └── events.out.tfevents.*
│   ├── exp_1_resnet18_ratio_0.50/
│   └── exp_2_resnet18_ratio_0.70/
│
├── experiment_results/            # JSON/CSV results
│   └── cifar10_logged_20250126.../
│       ├── results.json
│       ├── results_summary.csv
│       └── plots/
│
├── checkpoints/                   # Model checkpoints
│   └── cifar10_logged_20250126.../
│       ├── exp_0_model_resnet18_ratio_0.30.pth
│       ├── exp_1_model_resnet18_ratio_0.50.pth
│       └── exp_2_model_resnet18_ratio_0.70.pth
│
└── wandb/                         # WandB logs (if enabled)
    └── run-20250126.../
```

## 🎓 Advanced Usage

### 1. Multi-Model Comparison

```python
config.ENABLE_MULTI_MODEL = True
config.MODELS_TO_TEST = [
    ("resnet18", "resnet18"),
    ("resnet34", "resnet18"),
    ("resnet50", "resnet34"),
]
```

### 2. Custom Metrics

```python
# In training loop, add custom logging:
logger_obj.log_metrics({
    'custom/my_metric': my_value
}, step=global_step, prefix=f'exp_{exp_id}')
```

### 3. Compare with Baseline

```python
# Log baseline results
logger_obj.log_metrics({
    'baseline/accuracy': baseline_acc,
    'improvement': current_acc - baseline_acc
})
```

### 4. Offline Mode (WandB)

```bash
# Run without internet
export WANDB_MODE=offline

# Sync later
wandb sync wandb/run-*/
```

## 🐛 Troubleshooting

### TensorBoard not updating?
```bash
# Kill existing TensorBoard instances
pkill -f tensorboard

# Clear cache and restart
rm -rf /tmp/.tensorboard-info/
tensorboard --logdir=./runs --reload_interval=5
```

### WandB login issues?
```bash
# Check login status
wandb status

# Re-login
wandb login --relogin

# Use API key directly
export WANDB_API_KEY=your_key
```

### Out of memory with logging?
```python
# Reduce logging frequency
config.LOG_INTERVAL = 100  # Log less often
config.LOG_HISTOGRAMS = False  # Disable heavy logging
config.LOG_IMAGES = False
```

## 📚 Best Practices

1. **Consistent Naming**: Use descriptive experiment names
   ```python
   config.EXPERIMENT_NAME = "cifar10_resnet18_ablation_20250126"
   ```

2. **Tag Experiments**: Add tags for organization
   ```python
   wandb.init(tags=["ablation", "baseline", "paper-results"])
   ```

3. **Document Config**: Log all hyperparameters
   ```python
   # Automatically logged in ExperimentLogger.__init__
   ```

4. **Regular Checkpoints**: Save models frequently
   ```python
   config.SAVE_FREQUENCY = 10  # Every 10 epochs
   ```

5. **Compare Systematically**: Run baselines first
   ```python
   # Run standard training, then selective training
   # Both logged to same project for comparison
   ```

## 🎯 Quick Tips

- **Parallel coordinates in TensorBoard**: Great for hyperparameter analysis
- **WandB Tables**: Perfect for sample-level analysis
- **TensorBoard projector**: Visualize feature embeddings
- **WandB Sweeps**: Automated hyperparameter tuning
- **Git integration**: WandB auto-logs git commit hash

## 📞 Support

- TensorBoard docs: https://www.tensorflow.org/tensorboard
- WandB docs: https://docs.wandb.ai/
- Issues: https://github.com/your-repo/issues

## 🎉 Example Commands

```bash
# Quick test with TensorBoard only
python run_with_logging.py --no-wandb

# Full training with both
python run_with_logging.py

# View TensorBoard
tensorboard --logdir=./runs --port=6006

# Compare multiple experiments
tensorboard --logdir_spec=\
  exp1:./runs/exp_0,\
  exp2:./runs/exp_1,\
  exp3:./runs/exp_2

# Export WandB plots
wandb artifact get your-project/results:latest
```

Happy experimenting! 🚀
