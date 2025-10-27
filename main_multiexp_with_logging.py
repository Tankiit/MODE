#!/usr/bin/env python3
"""
Selective Multi-Classifier Coreset Selection with TensorBoard & WandB Logging
==============================================================================

Extended version with comprehensive experiment tracking:
- TensorBoard for local visualization
- Weights & Biases for cloud-based tracking and collaboration
- Real-time metric logging
- Experiment comparison
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Subset, Dataset
from torch.utils.tensorboard import SummaryWriter
import torchvision
import torchvision.transforms as transforms
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
import logging
from tqdm import tqdm
import timm
from collections import defaultdict
from scipy import optimize
from scipy.special import comb
import json
import os
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Try to import wandb
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("Warning: wandb not installed. Run: pip install wandb")

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ==============================================================================
# CONFIGURATION (Extended for Logging)
# ==============================================================================

class SelectiveMultiClassifierConfig:
    """Configuration for selective multi-classifier coreset selection"""

    # Model settings
    MODEL_NAME = "resnet34"
    REFERENCE_MODEL_NAME = "resnet18"
    PRETRAINED = True

    # Data settings
    DATASET = "CIFAR10"
    DATA_PATH = "./data"
    IMAGE_SIZE = 32
    NUM_CLASSES = 10

    # Multi-Classifier Settings
    NUM_SCORERS = 4
    USE_GATING = True
    USE_CONFIDENCE_REJECTION = True
    CONFIDENCE_THRESHOLD_INIT = 0.5

    # Gating network settings
    GATE_STATE_DIM = 6
    GATE_HIDDEN_DIM = 64
    TEMPERATURE_INIT = 1.0

    # Selection settings
    SELECTION_RATIO = 0.5
    SELECTION_FREQUENCY = 5
    ADAPTIVE_BUDGET = False

    # Risk-Coverage Guarantee Settings
    DELTA = 0.001
    TARGET_RISK = 0.05
    COMPUTE_RISK_BOUNDS = True

    # Training settings
    OUTPUT_DIR = "./selective_mc_output"
    NUM_EPOCHS = 200
    BATCH_SIZE = 128
    LEARNING_RATE = 0.1
    MOMENTUM = 0.9
    WEIGHT_DECAY = 1e-4
    LR_SCHEDULE = "cosine"

    # Multi-Experiment Settings
    ENABLE_MULTI_RATIO = True
    ENABLE_MULTI_MODEL = True
    RATIOS_TO_TEST = [0.1, 0.2, 0.3, 0.5, 0.7, 0.9]
    MODELS_TO_TEST = [
        ("resnet18", "resnet18"),
        ("resnet34", "resnet18"),
        ("resnet50", "resnet34"),
    ]
    SINGLE_MODEL_MULTI_RATIO = True

    # Experiment tracking
    EXPERIMENT_NAME = f"multi_exp_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    RESULTS_DIR = "./experiment_results"
    SAVE_CHECKPOINTS = True
    CHECKPOINT_DIR = "./checkpoints"

    # Logging settings (NEW)
    USE_TENSORBOARD = True
    USE_WANDB = True
    WANDB_PROJECT = "selective-coreset"
    WANDB_ENTITY = None  # Set to your wandb username/team
    TENSORBOARD_DIR = "./runs"
    LOG_INTERVAL = 10  # Log every N batches
    LOG_IMAGES = True  # Log sample images
    LOG_HISTOGRAMS = True  # Log weight histograms
    LOG_GRADIENTS = True  # Log gradient statistics

    # Logging and analysis
    LOGGING_STEPS = 100
    EVAL_FREQUENCY = 5
    SAVE_FREQUENCY = 20
    TRACK_SAMPLE_WASTE = True
    TRACK_CONFIDENCE_EVOLUTION = True

    # Visualization
    GENERATE_PLOTS = True
    PLOT_FORMAT = "png"

    # Device
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

config = SelectiveMultiClassifierConfig()

# ==============================================================================
# EXPERIMENT LOGGER (NEW)
# ==============================================================================

class ExperimentLogger:
    """Unified logger for TensorBoard and Weights & Biases"""

    def __init__(self, exp_name: str, exp_config: Dict, use_tensorboard: bool = True,
                 use_wandb: bool = True):
        self.exp_name = exp_name
        self.use_tensorboard = use_tensorboard and config.USE_TENSORBOARD
        self.use_wandb = use_wandb and config.USE_WANDB and WANDB_AVAILABLE

        # TensorBoard setup
        self.tb_writer = None
        if self.use_tensorboard:
            log_dir = os.path.join(config.TENSORBOARD_DIR, exp_name)
            self.tb_writer = SummaryWriter(log_dir=log_dir)
            logger.info(f"TensorBoard logging to: {log_dir}")

        # WandB setup
        self.wandb_run = None
        if self.use_wandb:
            try:
                self.wandb_run = wandb.init(
                    project=config.WANDB_PROJECT,
                    entity=config.WANDB_ENTITY,
                    name=exp_name,
                    config=exp_config,
                    reinit=True
                )
                logger.info(f"WandB logging initialized: {self.wandb_run.url}")
            except Exception as e:
                logger.warning(f"Failed to initialize wandb: {e}")
                self.use_wandb = False

        self.step = 0

    def log_metrics(self, metrics: Dict, step: Optional[int] = None, prefix: str = ""):
        """Log metrics to both TensorBoard and WandB"""
        if step is None:
            step = self.step
            self.step += 1

        # Add prefix to metrics
        if prefix:
            metrics = {f"{prefix}/{k}": v for k, v in metrics.items()}

        # TensorBoard
        if self.use_tensorboard and self.tb_writer:
            for key, value in metrics.items():
                if isinstance(value, (int, float)):
                    self.tb_writer.add_scalar(key, value, step)

        # WandB
        if self.use_wandb and self.wandb_run:
            wandb.log(metrics, step=step)

    def log_histogram(self, tag: str, values, step: Optional[int] = None):
        """Log histogram to TensorBoard"""
        if step is None:
            step = self.step

        if self.use_tensorboard and self.tb_writer:
            if isinstance(values, torch.Tensor):
                values = values.detach().cpu().numpy()
            self.tb_writer.add_histogram(tag, values, step)

    def log_image(self, tag: str, image, step: Optional[int] = None):
        """Log image to both backends"""
        if step is None:
            step = self.step

        if self.use_tensorboard and self.tb_writer:
            self.tb_writer.add_image(tag, image, step)

        if self.use_wandb and self.wandb_run:
            if isinstance(image, torch.Tensor):
                image = image.detach().cpu().numpy()
            wandb.log({tag: wandb.Image(image)}, step=step)

    def log_figure(self, tag: str, figure, step: Optional[int] = None):
        """Log matplotlib figure"""
        if step is None:
            step = self.step

        if self.use_tensorboard and self.tb_writer:
            self.tb_writer.add_figure(tag, figure, step)

        if self.use_wandb and self.wandb_run:
            wandb.log({tag: wandb.Image(figure)}, step=step)

        plt.close(figure)

    def log_text(self, tag: str, text: str, step: Optional[int] = None):
        """Log text to both backends"""
        if step is None:
            step = self.step

        if self.use_tensorboard and self.tb_writer:
            self.tb_writer.add_text(tag, text, step)

        if self.use_wandb and self.wandb_run:
            wandb.log({tag: wandb.Html(text)}, step=step)

    def log_model_graph(self, model: nn.Module, input_shape: Tuple):
        """Log model graph to TensorBoard"""
        if self.use_tensorboard and self.tb_writer:
            dummy_input = torch.randn(input_shape).to(config.DEVICE)
            self.tb_writer.add_graph(model, dummy_input)

    def watch_model(self, model: nn.Module):
        """Watch model with WandB"""
        if self.use_wandb and self.wandb_run:
            wandb.watch(model, log='all', log_freq=config.LOG_INTERVAL)

    def log_hyperparameters(self, hparams: Dict, metrics: Dict):
        """Log hyperparameters with associated metrics"""
        if self.use_tensorboard and self.tb_writer:
            self.tb_writer.add_hparams(hparams, metrics)

    def close(self):
        """Close both logging backends"""
        if self.tb_writer:
            self.tb_writer.close()

        if self.use_wandb and self.wandb_run:
            self.wandb_run.finish()

# Import components from main_multiexp (we'll reuse most of the code)
# For brevity, I'll import the key classes and modify the training loop

# Simplified imports - in practice, you'd import from main_multiexp or duplicate
from main_multiexp import (
    IndexedDataset, load_data,
    ConfidenceEstimator, EntropyConfidence, MarginConfidence,
    StabilityConfidence, DistanceConfidence,
    ScoringFunction, UncertaintyScorer, DiversityScorer,
    BoundaryScorer, ExcessLossScorer,
    AdaptiveGatingNetwork, TrainingStateComputer,
    SelectiveMultiClassifierSelector
)

# ==============================================================================
# ENHANCED TRAINING LOOP WITH LOGGING
# ==============================================================================

def train_with_logging(
    model: nn.Module,
    train_dataset: Dataset,
    val_dataset: Dataset,
    selector: SelectiveMultiClassifierSelector,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    logger_obj: ExperimentLogger,
    scheduler: Optional[optim.lr_scheduler._LRScheduler] = None,
    selection_ratio: float = 0.5,
    exp_id: int = 0
):
    """Enhanced training loop with comprehensive logging"""

    train_loader_full = DataLoader(train_dataset, batch_size=config.BATCH_SIZE,
                                   shuffle=False, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE,
                           shuffle=False, num_workers=2)

    best_val_acc = 0.0
    global_step = 0

    # Log model graph
    if exp_id == 0:  # Only log once
        logger_obj.log_model_graph(model, (1, 3, 32, 32))
        if config.USE_WANDB:
            logger_obj.watch_model(model)

    for epoch in range(config.NUM_EPOCHS):
        epoch_start_metrics = {}

        # Select coreset periodically
        if epoch % config.SELECTION_FREQUENCY == 0:
            logger.info(f"Selecting coreset with budget={selection_ratio:.2f}...")
            selected_indices, selection_info = selector.select_coreset(
                train_dataset, epoch, selection_ratio
            )

            # Log selection statistics
            logger_obj.log_metrics({
                'selection/num_selected': len(selected_indices),
                'selection/ratio': selection_ratio,
            }, step=global_step, prefix=f'exp_{exp_id}')

            # Log gate weights
            for i, weight in enumerate(selection_info['gate_weights']):
                logger_obj.log_metrics({
                    f'gate/scorer_{i}_weight': weight
                }, step=global_step, prefix=f'exp_{exp_id}')

            # Log rejection rates
            for i, rate in enumerate(selection_info['rejection_rates']):
                logger_obj.log_metrics({
                    f'rejection/scorer_{i}_rate': rate
                }, step=global_step, prefix=f'exp_{exp_id}')

            train_subset = Subset(train_dataset, selected_indices.tolist())
            train_loader = DataLoader(train_subset, batch_size=config.BATCH_SIZE,
                                     shuffle=True, num_workers=2)

        # Training epoch
        model.train()
        epoch_loss = 0.0
        epoch_correct = 0
        epoch_total = 0
        batch_count = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.NUM_EPOCHS}", leave=False)
        for batch_idx, batch_data in enumerate(pbar):
            if len(batch_data) == 3:
                inputs, targets, _ = batch_data
            else:
                inputs, targets = batch_data

            inputs, targets = inputs.to(config.DEVICE), targets.to(config.DEVICE)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()

            # Log gradients
            if config.LOG_GRADIENTS and batch_idx % config.LOG_INTERVAL == 0:
                total_norm = 0
                for p in model.parameters():
                    if p.grad is not None:
                        param_norm = p.grad.data.norm(2)
                        total_norm += param_norm.item() ** 2
                total_norm = total_norm ** 0.5

                logger_obj.log_metrics({
                    'train/grad_norm': total_norm
                }, step=global_step, prefix=f'exp_{exp_id}')

            optimizer.step()

            # Statistics
            epoch_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            batch_correct = (predicted == targets).sum().item()
            epoch_correct += batch_correct
            epoch_total += targets.size(0)
            batch_count += 1

            # Log batch metrics
            if batch_idx % config.LOG_INTERVAL == 0:
                batch_acc = batch_correct / targets.size(0)
                logger_obj.log_metrics({
                    'train/batch_loss': loss.item(),
                    'train/batch_acc': batch_acc,
                    'train/learning_rate': optimizer.param_groups[0]['lr']
                }, step=global_step, prefix=f'exp_{exp_id}')

                pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'acc': f'{100.*epoch_correct/epoch_total:.2f}%'
                })

            global_step += 1

        if scheduler:
            scheduler.step()

        # Epoch metrics
        epoch_loss /= batch_count
        train_acc = epoch_correct / epoch_total

        logger_obj.log_metrics({
            'train/epoch_loss': epoch_loss,
            'train/epoch_acc': train_acc,
            'train/epoch': epoch
        }, step=global_step, prefix=f'exp_{exp_id}')

        # Log weight histograms
        if config.LOG_HISTOGRAMS and epoch % 5 == 0:
            for name, param in model.named_parameters():
                if param.requires_grad:
                    logger_obj.log_histogram(
                        f'exp_{exp_id}/weights/{name}',
                        param.data,
                        step=epoch
                    )

        # Validation
        if epoch % config.EVAL_FREQUENCY == 0:
            val_acc, val_loss = evaluate_model_with_logging(
                model, val_loader, criterion
            )

            logger_obj.log_metrics({
                'val/acc': val_acc,
                'val/loss': val_loss,
            }, step=global_step, prefix=f'exp_{exp_id}')

            logger.info(f"Epoch {epoch+1}: Train Loss={epoch_loss:.4f}, "
                       f"Train Acc={train_acc:.4f}, Val Acc={val_acc:.4f}")

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                logger_obj.log_metrics({
                    'val/best_acc': best_val_acc
                }, step=global_step, prefix=f'exp_{exp_id}')

        # Sample waste analysis
        if config.TRACK_SAMPLE_WASTE and epoch % 10 == 0 and epoch > 0:
            waste_stats = selector.compute_sample_waste(epoch)
            logger_obj.log_metrics({
                'waste/mastered_early': waste_stats['mastered_early'],
                'waste/stagnant': waste_stats['stagnant'],
                'waste/harmful': waste_stats['potentially_harmful'],
                'waste/useful': waste_stats['useful'],
            }, step=global_step, prefix=f'exp_{exp_id}')

    training_history = {
        'final_train_acc': train_acc,
        'best_val_acc': best_val_acc
    }

    return model, training_history, best_val_acc

def evaluate_model_with_logging(model: nn.Module, val_loader: DataLoader,
                                criterion: nn.Module) -> Tuple[float, float]:
    """Evaluate model and return accuracy and loss"""
    model.eval()
    correct = 0
    total = 0
    total_loss = 0.0
    batch_count = 0

    with torch.no_grad():
        for batch_data in val_loader:
            if len(batch_data) == 3:
                inputs, targets, _ = batch_data
            else:
                inputs, targets = batch_data

            inputs, targets = inputs.to(config.DEVICE), targets.to(config.DEVICE)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            total_loss += loss.item()
            batch_count += 1

            _, predicted = torch.max(outputs, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()

    model.train()
    return correct / total, total_loss / batch_count

# ==============================================================================
# ENHANCED EXPERIMENT RUNNER WITH LOGGING
# ==============================================================================

class ExperimentRunnerWithLogging:
    """Runs experiments with TensorBoard and WandB logging"""

    def __init__(self):
        self.results = []
        self.results_dir = os.path.join(config.RESULTS_DIR, config.EXPERIMENT_NAME)
        os.makedirs(self.results_dir, exist_ok=True)

        if config.SAVE_CHECKPOINTS:
            self.checkpoint_dir = os.path.join(config.CHECKPOINT_DIR, config.EXPERIMENT_NAME)
            os.makedirs(self.checkpoint_dir, exist_ok=True)

        # Create master logger for experiment suite
        self.master_logger = ExperimentLogger(
            f"{config.EXPERIMENT_NAME}_master",
            self._get_config_dict(),
            use_tensorboard=config.USE_TENSORBOARD,
            use_wandb=config.USE_WANDB
        )

    def _get_config_dict(self) -> Dict:
        """Convert config to dictionary"""
        return {k: v for k, v in config.__dict__.items()
                if not k.startswith('_') and not callable(v)}

    def run_single_experiment(self, model_name: str, reference_model_name: str,
                            selection_ratio: float, train_dataset: Dataset,
                            val_dataset: Dataset, exp_id: int) -> Dict:
        """Run single experiment with logging"""

        logger.info("="*70)
        logger.info(f"EXPERIMENT {exp_id}")
        logger.info(f"Model: {model_name}, Reference: {reference_model_name}")
        logger.info(f"Selection Ratio: {selection_ratio}")
        logger.info("="*70)

        # Create experiment-specific logger
        exp_name = f"exp_{exp_id}_{model_name}_ratio_{selection_ratio:.2f}"
        exp_config = {
            'exp_id': exp_id,
            'model_name': model_name,
            'reference_model_name': reference_model_name,
            'selection_ratio': selection_ratio,
            **self._get_config_dict()
        }

        exp_logger = ExperimentLogger(
            exp_name,
            exp_config,
            use_tensorboard=config.USE_TENSORBOARD,
            use_wandb=config.USE_WANDB
        )

        # Initialize models
        model = timm.create_model(model_name, pretrained=config.PRETRAINED,
                                 num_classes=config.NUM_CLASSES)
        model = model.to(config.DEVICE)

        reference_model = None
        if reference_model_name:
            reference_model = timm.create_model(reference_model_name,
                                               pretrained=config.PRETRAINED,
                                               num_classes=config.NUM_CLASSES)
            reference_model = reference_model.to(config.DEVICE)
            reference_model.eval()

        # Initialize selector
        selector = SelectiveMultiClassifierSelector(model, reference_model)

        # Setup training
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=config.LEARNING_RATE,
                             momentum=config.MOMENTUM, weight_decay=config.WEIGHT_DECAY)

        if config.LR_SCHEDULE == "cosine":
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.NUM_EPOCHS)
        else:
            scheduler = None

        # Train with logging
        model, training_history, best_val_acc = train_with_logging(
            model, train_dataset, val_dataset, selector,
            criterion, optimizer, exp_logger, scheduler,
            selection_ratio, exp_id
        )

        # Final evaluation
        val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE,
                               shuffle=False, num_workers=2)
        final_acc, final_loss = evaluate_model_with_logging(model, val_loader, criterion)

        # Log final results
        exp_logger.log_metrics({
            'final/accuracy': final_acc,
            'final/best_accuracy': best_val_acc,
        })

        # Log to master
        self.master_logger.log_metrics({
            f'experiments/exp_{exp_id}_final_acc': final_acc,
            f'experiments/exp_{exp_id}_best_acc': best_val_acc,
        })

        # Close experiment logger
        exp_logger.close()

        # Package results
        result = {
            'exp_id': exp_id,
            'model_name': model_name,
            'reference_model_name': reference_model_name,
            'selection_ratio': selection_ratio,
            'final_acc': final_acc,
            'best_val_acc': best_val_acc,
            'training_history': training_history,
            'timestamp': datetime.now().isoformat()
        }

        # Save checkpoint
        if config.SAVE_CHECKPOINTS:
            checkpoint_path = os.path.join(
                self.checkpoint_dir,
                f"exp_{exp_id}_model_{model_name}_ratio_{selection_ratio:.2f}.pth"
            )
            torch.save({
                'model_state_dict': model.state_dict(),
                'result': result
            }, checkpoint_path)

        logger.info(f"Experiment {exp_id} completed: Final Acc = {final_acc:.4f}")

        return result

    def run_all_experiments(self, train_dataset: Dataset, val_dataset: Dataset):
        """Run complete experiment suite with logging"""

        # Run multi-ratio experiments
        if config.SINGLE_MODEL_MULTI_RATIO:
            model_name = config.MODEL_NAME
            reference_model_name = config.REFERENCE_MODEL_NAME

            exp_id = 0
            for ratio in config.RATIOS_TO_TEST:
                result = self.run_single_experiment(
                    model_name, reference_model_name, ratio,
                    train_dataset, val_dataset, exp_id
                )
                self.results.append(result)
                exp_id += 1
                self._save_results()

        # Run multi-model experiments
        if config.ENABLE_MULTI_MODEL:
            exp_id = len(self.results)
            for model_name, reference_model_name in config.MODELS_TO_TEST:
                for ratio in config.RATIOS_TO_TEST:
                    result = self.run_single_experiment(
                        model_name, reference_model_name, ratio,
                        train_dataset, val_dataset, exp_id
                    )
                    self.results.append(result)
                    exp_id += 1
                    self._save_results()

        # Close master logger
        self.master_logger.close()

        # Generate report
        self._generate_report()

    def _save_results(self):
        """Save results to JSON"""
        results_file = os.path.join(self.results_dir, "results.json")
        with open(results_file, 'w') as f:
            json.dump(self.results, f, indent=2)

    def _generate_report(self):
        """Generate final report"""
        logger.info("\n" + "="*70)
        logger.info("EXPERIMENT SUMMARY")
        logger.info("="*70)

        df_data = []
        for result in self.results:
            df_data.append({
                'Exp ID': result['exp_id'],
                'Model': result['model_name'],
                'Ratio': result['selection_ratio'],
                'Final Acc': result['final_acc'],
                'Best Val Acc': result['best_val_acc']
            })

        df = pd.DataFrame(df_data)
        logger.info("\n" + str(df))

        csv_path = os.path.join(self.results_dir, "results_summary.csv")
        df.to_csv(csv_path, index=False)
        logger.info(f"\nSummary saved to {csv_path}")

        # Best configuration
        best_result = df.loc[df['Final Acc'].idxmax()]
        logger.info("\n" + "="*70)
        logger.info("BEST CONFIGURATION")
        logger.info("="*70)
        logger.info(f"Model: {best_result['Model']}")
        logger.info(f"Selection Ratio: {best_result['Ratio']}")
        logger.info(f"Final Accuracy: {best_result['Final Acc']:.4f}")

# ==============================================================================
# MAIN EXECUTION
# ==============================================================================

def main():
    """Main execution with logging"""

    logger.info("="*70)
    logger.info("SELECTIVE MULTI-CLASSIFIER WITH TENSORBOARD & WANDB")
    logger.info("="*70)
    logger.info(f"TensorBoard: {config.USE_TENSORBOARD}")
    logger.info(f"WandB: {config.USE_WANDB and WANDB_AVAILABLE}")
    logger.info(f"Dataset: {config.DATASET}")
    logger.info(f"Device: {config.DEVICE}")
    logger.info("="*70)

    # Load data
    logger.info("\nLoading data...")
    train_dataset, val_dataset = load_data()

    # Initialize runner
    runner = ExperimentRunnerWithLogging()

    # Run experiments
    logger.info("\nStarting experiments...")
    runner.run_all_experiments(train_dataset, val_dataset)

    logger.info("\n" + "="*70)
    logger.info("ALL EXPERIMENTS COMPLETED!")
    logger.info(f"Results: {runner.results_dir}")
    if config.USE_TENSORBOARD:
        logger.info(f"TensorBoard: tensorboard --logdir={config.TENSORBOARD_DIR}")
    if config.USE_WANDB and WANDB_AVAILABLE:
        logger.info(f"WandB: Check your dashboard at wandb.ai")
    logger.info("="*70)

if __name__ == "__main__":
    main()
