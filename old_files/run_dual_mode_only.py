#!/usr/bin/env python3
"""
Run only the Dual-MODE experiment.
"""

from mode import Config, Trainer, load_data

if __name__ == "__main__":
    config = Config()

    print("\n" + "="*70)
    print("Running: Dual-MODE Only")
    print("="*70)
    print(f"\nConfiguration:")
    print(f"  Model: distilgpt2 (pretrained)")
    print(f"  Training: {config.train_samples} samples, {config.epochs} epochs")
    print(f"  Budget: {config.token_budget*100:.1f}%")
    print(f"  5 Strategies: uncertainty, loss, coherence, diversity, token_mode")
    print("="*70)

    # Load data
    train_loader, val_texts = load_data(config)

    # Run Dual-MODE
    print(f"\n{'='*70}")
    print(f"Initializing Dual-MODE Trainer...")
    print(f"{'='*70}")

    trainer = Trainer(config, 'Dual-MODE', 'mode')
    result = trainer.train(train_loader, val_texts)

    print(f"\n{'='*70}")
    print("DUAL-MODE COMPLETE")
    print(f"{'='*70}")
    print(f"  Best Val Perplexity: {result['best_val_ppl']:.2f}")
    print(f"  Final Train Loss: {result['final_train_loss']:.4f}")
    print(f"{'='*70}\n")
