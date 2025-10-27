# Selective Classification for Coreset/Subset Selection with Deep Models

This implementation provides selective training for image classification using deep neural networks. It adapts the Rho-1 selective training approach for classification tasks to enable efficient coreset and subset selection.

## Features

### Supported Models
- **ResNet**: resnet18, resnet34, resnet50
- **VGG**: vgg16, vgg19
- **DenseNet**: densenet121
- **MobileNet**: mobilenet_v2

All models can be easily swapped via command-line arguments.

### Supported Datasets
- **CIFAR-10**: 10 classes, 32x32 images
- **CIFAR-100**: 100 classes, 32x32 images
- **ImageNet**: 1000 classes, 224x224 images
- **SVHN**: 10 classes, 32x32 images
- **STL-10**: 10 classes, 96x96 images

### Training Modes

#### 1. Baseline Training
Standard training using all samples without selection.

```bash
python main.py --mode baseline --dataset CIFAR10 --model resnet18 --epochs 100
```

#### 2. Online Selective Training
Performs sample selection dynamically during training based on excess loss.
- Computes excess loss for each batch
- Selects top-k% samples with highest excess loss
- Trains only on selected samples

```bash
python main.py --mode online --dataset CIFAR10 --model resnet18 --selection-ratio 0.3 --epochs 100
```

#### 3. Coreset (Offline) Training
Precomputes a coreset before training and trains only on the selected subset.
- Evaluates excess loss for entire dataset
- Selects top-k% samples with highest excess loss
- Creates coreset and trains exclusively on it

```bash
python main.py --mode coreset --dataset CIFAR10 --model resnet18 --selection-ratio 0.3 --epochs 100
```

## Key Components

### 1. Reference Model with Caching
- Used to compute baseline loss for excess loss calculation
- Can be same or different architecture from training model
- Helps identify "hard" or "informative" samples
- **NEW: Intelligent Caching System**
  - **Memory caching**: Stores reference losses in RAM for instant lookup
  - **Disk caching**: Persists losses to disk for memory efficiency
  - **Precomputation**: Computes all reference losses once before training
  - **Huge speedup**: Avoids recomputing reference model forward passes every epoch!

### 2. Sample Selector
- Computes excess loss: L_train - L_ref
- Selects samples with highest excess loss
- Tracks selection statistics
- Uses cached reference losses for efficiency

### 3. Selective Training
- **Online**: Batch-level selection during training
- **Coreset**: Dataset-level selection before training

## Caching System

### Why Caching?

The reference model is **fixed** during training (no gradient updates), so its outputs don't change. Without caching, we recompute reference losses **every epoch**:

- **Without cache**: N_samples × N_epochs forward passes through reference model
- **With cache**: N_samples forward passes (computed once, reused N_epochs times)

For CIFAR-10 with 50,000 samples and 200 epochs:
- **Without cache**: 10,000,000 forward passes
- **With cache**: 50,000 forward passes (200× speedup!)

### Cache Modes

Configure via `config.CACHE_MODE`:

1. **`'none'`**: No caching, recompute every time (slow but no memory overhead)
2. **`'memory'`**: Cache in RAM (fast, ~200MB for CIFAR-10)
3. **`'disk'`**: Cache on disk (slower than memory but no RAM usage)

### Precomputation

Enable via `config.PRECOMPUTE_REFERENCE_LOSSES = True`:

- Computes all reference losses **once** before training starts
- Stores them in cache for instant lookup during training
- Dramatically speeds up selective training

```python
# Configuration example
config.CACHE_MODE = 'memory'  # Use memory caching
config.PRECOMPUTE_REFERENCE_LOSSES = True  # Precompute all losses
```

## Configuration

Key hyperparameters in `SelectiveClassificationConfig`:

```python
MODEL_NAME = "resnet18"          # Deep model architecture
DATASET = "CIFAR10"              # Vision dataset
SELECTION_RATIO = 0.3            # Fraction of samples to select
NUM_EPOCHS = 100                 # Training epochs
BATCH_SIZE = 128                 # Batch size
LEARNING_RATE = 0.1              # Initial learning rate
MOMENTUM = 0.9                   # SGD momentum
WEIGHT_DECAY = 5e-4              # L2 regularization
LR_SCHEDULE = "cosine"           # Learning rate schedule
```

## Command-Line Arguments

- `--mode`: Training mode (online, coreset, baseline)
- `--dataset`: Dataset choice (CIFAR10, CIFAR100, ImageNet, SVHN, STL10)
- `--model`: Model architecture (resnet18, resnet34, resnet50, vgg16, vgg19, densenet121, mobilenet_v2)
- `--selection-ratio`: Fraction of samples to select (0.0-1.0)
- `--epochs`: Number of training epochs
- `--batch-size`: Batch size

## Example Usage

### Quick Start with CIFAR-10
```bash
# Baseline
python main.py --mode baseline --dataset CIFAR10 --model resnet18 --epochs 100

# Online selection (30% of samples)
python main.py --mode online --dataset CIFAR10 --model resnet18 --selection-ratio 0.3 --epochs 100

# Coreset selection (30% of samples)
python main.py --mode coreset --dataset CIFAR10 --model resnet18 --selection-ratio 0.3 --epochs 100
```

### Experiment with Different Models
```bash
# ResNet-50 on CIFAR-100
python main.py --mode online --dataset CIFAR100 --model resnet50 --selection-ratio 0.3 --epochs 100

# VGG-16 on SVHN
python main.py --mode coreset --dataset SVHN --model vgg16 --selection-ratio 0.4 --epochs 50
```

### Varying Selection Ratios
```bash
# 10% selection
python main.py --mode online --dataset CIFAR10 --model resnet18 --selection-ratio 0.1 --epochs 100

# 50% selection
python main.py --mode online --dataset CIFAR10 --model resnet18 --selection-ratio 0.5 --epochs 100
```

## Output

The script saves:
- `best_model.pth`: Model with best test accuracy
- `final_model.pth`: Final model after all epochs
- `checkpoint_epoch_*.pth`: Periodic checkpoints

Training logs include:
- Training/test loss and accuracy per epoch
- Selection statistics (for selective modes)
- Best accuracy achieved
- Final model performance

## Dependencies

```bash
pip install torch torchvision numpy tqdm
```

## Methodology

### Excess Loss Computation

The excess loss for a sample is defined as:
```
excess_loss = L_train(x, y) - L_ref(x, y)
```

Where:
- `L_train`: Loss from the training model
- `L_ref`: Loss from the reference model

Samples with high excess loss are considered more informative and are prioritized for training.

### Selection Strategy

**Online Mode**:
- Each batch: compute excess loss → select top-k% → train on selected

**Coreset Mode**:
- Before training: compute excess loss for all samples → select top-k% → create coreset → train only on coreset

## Benefits

1. **Reduced Training Time**: Train on fewer samples while maintaining performance
2. **Improved Efficiency**: Focus on informative/hard samples
3. **Dataset Distillation**: Create compact, representative subsets
4. **Flexibility**: Easy to swap models and datasets

## Performance Comparison: With vs Without Caching

### Training Time Comparison (CIFAR-10, 200 epochs)

| Configuration | Reference Forward Passes | Speedup |
|---------------|-------------------------|---------|
| **No caching** | 50,000 × 200 = 10M | 1× (baseline) |
| **Memory caching** | 50,000 × 1 = 50K | **~200×** |
| **Disk caching** | 50,000 × 1 = 50K | **~150×** |

### Memory Usage

| Cache Mode | RAM Usage (CIFAR-10) | RAM Usage (ImageNet) |
|------------|---------------------|---------------------|
| `'none'` | 0 MB | 0 MB |
| `'memory'` | ~200 MB | ~5 GB |
| `'disk'` | ~0 MB (stored on disk) | ~0 MB |

### When to Use Each Mode

- **`'memory'`**: Best for small-medium datasets (CIFAR-10/100) - fastest
- **`'disk'`**: Best for large datasets (ImageNet) - memory efficient
- **`'none'`**: Only for debugging or when reference model changes

## Notes

- For ImageNet, ensure you have the dataset downloaded and specify the correct path
- GPU training is recommended for larger models and datasets
- Adjust batch size based on available GPU memory
- Selection ratio can be tuned based on dataset size and complexity
- **Use memory caching for maximum speed** on datasets that fit in RAM
- Disk caching automatically saves/loads cache between runs
