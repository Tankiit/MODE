#!/usr/bin/env python3
"""
Run Dual-MODE with optimizations for speed.
Reduces memory size and batch processing for faster execution.
"""

from mode import Config, Trainer, load_data
from dataclasses import replace

if __name__ == "__main__":
    config = Config()

    # Optimize for speed
    config.max_memory_contexts = 500  # Much smaller memory (was 50000)
    config.retrieval_k = 10  # Fewer neighbors (was 50)
    config.batch_size = 1  # Smaller batches for faster iteration

    print("\n" + "="*70)
    print("Running: Dual-MODE (Optimized for Speed)")
    print("="*70)
    print(f"\nConfiguration:")
    print(f"  Model: distilgpt2 (pretrained)")
    print(f"  Training: {config.train_samples} samples, {config.epochs} epochs")
    print(f"  Budget: {config.token_budget*100:.1f}%")
    print(f"  Memory: {config.max_memory_contexts} contexts (reduced)")
    print(f"  Retrieval k: {config.retrieval_k} neighbors (reduced)")
    print(f"  Batch size: {config.batch_size} (reduced)")
    print(f"  5 Strategies: uncertainty, loss, coherence, diversity, token_mode")
    print("="*70)

    # Load data
    train_loader, val_texts = load_data(config)

    # Run Dual-MODE
    print(f"\n{'='*70}")
    print(f"Initializing Dual-MODE Trainer (Fast Version)...")
    print(f"{'='*70}")

    trainer = Trainer(config, 'Dual-MODE-Fast', 'mode')
    result = trainer.train(train_loader, val_texts)

    print(f"\n{'='*70}")
    print("DUAL-MODE COMPLETE")
    print(f"{'='*70}")
    print(f"  Best Val Perplexity: {result['best_val_ppl']:.2f}")
    print(f"  Final Train Loss: {result['final_train_loss']:.4f}")

    if hasattr(trainer, 'token_mode_memory'):
        print(f"  Final Memory Size: {trainer.token_mode_memory.size()}")

    if hasattr(trainer, 'meta_controller') and trainer.meta_controller.is_trained:
        print(f"  Meta-Controller: Trained")

    if hasattr(trainer, 'strategy_history') and trainer.strategy_history:
        import numpy as np
        avg_weights = np.mean(trainer.strategy_history[-100:], axis=0)
        strategy_names = ['Uncertainty', 'Loss', 'Coherence', 'Diversity', 'Token MODE']
        print(f"\n  Final Strategy Weights:")
        for name, weight in zip(strategy_names, avg_weights):
            print(f"    {name:15s}: {weight:.4f}")

    print(f"{'='*70}\n")
