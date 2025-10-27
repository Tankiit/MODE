#!/usr/bin/env python3
"""
Simple metrics summary from TensorBoard logs.
"""

from tensorboard.backend.event_processing import event_accumulator
from pathlib import Path
import numpy as np

def load_experiment(log_dir):
    """Load metrics from a single experiment"""
    event_files = list(Path(log_dir).glob('events.out.tfevents.*'))
    if not event_files:
        return None

    # Get the most recent event file
    event_file = sorted(event_files, key=lambda x: x.stat().st_mtime)[-1]

    ea = event_accumulator.EventAccumulator(str(event_file.parent))
    ea.Reload()

    metrics = {}

    # Extract available metrics
    for tag in ea.Tags()['scalars']:
        events = ea.Scalars(tag)
        metrics[tag] = {
            'steps': [e.step for e in events],
            'values': [e.value for e in events]
        }

    return metrics

def main():
    base_dir = Path('./runs_dual_mode_comparison')

    experiments = ['Full-Training', 'Rho-1', 'Dual-MODE']

    print("\n" + "="*80)
    print("MODE COMPARISON - COMPREHENSIVE METRICS")
    print("="*80)

    results = {}

    for exp_name in experiments:
        exp_dir = base_dir / exp_name
        print(f"\n{exp_name}:")
        print("-" * 40)

        metrics = load_experiment(exp_dir)
        if not metrics:
            print("  No data found")
            continue

        results[exp_name] = {}

        # Training loss
        if 'train/loss' in metrics:
            losses = metrics['train/loss']['values']
            print(f"  Training Loss:")
            print(f"    Initial: {losses[0]:.4f}")
            print(f"    Final:   {losses[-1]:.4f}")
            print(f"    Min:     {min(losses):.4f}")
            print(f"    Avg:     {np.mean(losses):.4f}")
            results[exp_name]['train_loss'] = losses

        # Epoch training loss
        if 'epoch/train_loss' in metrics:
            epoch_losses = metrics['epoch/train_loss']['values']
            print(f"  Epoch Training Loss:")
            for i, loss in enumerate(epoch_losses):
                print(f"    Epoch {i+1}: {loss:.4f}")

        # Validation perplexity
        if 'epoch/val_ppl' in metrics:
            ppls = metrics['epoch/val_ppl']['values']
            print(f"  Validation Perplexity:")
            for i, ppl in enumerate(ppls):
                print(f"    Epoch {i+1}: {ppl:.2f}")
            print(f"    Best:    {min(ppls):.2f}")
            results[exp_name]['val_ppl'] = ppls

        # Token selection
        if 'train/selection' in metrics:
            selections = metrics['train/selection']['values']
            print(f"  Token Selection Rate:")
            print(f"    Average:  {np.mean(selections)*100:.1f}%")
            print(f"    Final 50: {np.mean(selections[-50:])*100:.1f}%")
            results[exp_name]['selection'] = selections

        # MODE strategies
        if exp_name == 'Dual-MODE':
            strategies = ['uncertainty', 'loss', 'coherence', 'diversity', 'token_mode']
            has_strategies = False
            for strategy in strategies:
                key = f'strategy/{strategy}'
                if key in metrics:
                    if not has_strategies:
                        print(f"  Strategy Weights (Final):")
                        has_strategies = True
                    weights = metrics[key]['values']
                    if weights:
                        print(f"    {strategy.title():15s}: {weights[-1]:.4f}")

    # Comparative analysis
    print("\n" + "="*80)
    print("COMPARATIVE ANALYSIS")
    print("="*80)

    if all(exp in results for exp in ['Full-Training', 'Rho-1']):
        full_ppl = min(results['Full-Training']['val_ppl'])
        rho1_ppl = min(results['Rho-1']['val_ppl'])

        print(f"\nBest Validation Perplexity:")
        print(f"  Full-Training:  {full_ppl:.2f}")
        print(f"  Rho-1:          {rho1_ppl:.2f}  ({(full_ppl-rho1_ppl)/full_ppl*100:+.2f}% vs Full)")

        if 'Dual-MODE' in results and 'val_ppl' in results['Dual-MODE']:
            mode_ppl = min(results['Dual-MODE']['val_ppl'])
            print(f"  Dual-MODE:      {mode_ppl:.2f}  ({(full_ppl-mode_ppl)/full_ppl*100:+.2f}% vs Full)")
            print(f"\nDual-MODE vs Rho-1:  {(rho1_ppl-mode_ppl)/rho1_ppl*100:+.2f}%")

        print(f"\nTraining Efficiency:")
        full_sel = np.mean(results['Full-Training']['selection'])
        rho1_sel = np.mean(results['Rho-1']['selection'][-100:])  # After warmup
        print(f"  Full-Training uses:  {full_sel*100:.0f}% of tokens")
        print(f"  Rho-1 uses:          {rho1_sel*100:.0f}% of tokens")

        if 'Dual-MODE' in results and 'selection' in results['Dual-MODE']:
            mode_sel = np.mean(results['Dual-MODE']['selection'][-100:])
            print(f"  Dual-MODE uses:      {mode_sel*100:.0f}% of tokens")

    print("\n" + "="*80)
    print("\nKEY FINDINGS:")
    print("-" * 40)

    if all(exp in results for exp in ['Full-Training', 'Rho-1']):
        full_ppl = min(results['Full-Training']['val_ppl'])
        rho1_ppl = min(results['Rho-1']['val_ppl'])

        print(f"Full-Training achieved {full_ppl:.2f} PPL using 100% of tokens")
        print(f"Rho-1 achieved {rho1_ppl:.2f} PPL using only ~30% of tokens")

        if rho1_ppl <= full_ppl * 1.05:
            print(f"Rho-1 matches Full-Training performance with 70% fewer tokens!")

        if 'Dual-MODE' in results and 'val_ppl' in results['Dual-MODE']:
            mode_ppl = min(results['Dual-MODE']['val_ppl'])
            print(f"Dual-MODE achieved {mode_ppl:.2f} PPL with adaptive strategy selection")

            if mode_ppl < rho1_ppl:
                print(f"Dual-MODE outperforms Rho-1 by {(rho1_ppl-mode_ppl):.2f} PPL!")

    print("\n" + "="*80)

if __name__ == "__main__":
    main()
