#!/usr/bin/env python3
"""
Analyze and visualize metrics from MODE comparison experiments.
Extracts data from TensorBoard logs and creates comprehensive visualizations.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from tensorboard.backend.event_processing import event_accumulator
import pandas as pd

def extract_tensorboard_data(log_dir):
    """Extract metrics from TensorBoard logs"""
    ea = event_accumulator.EventAccumulator(str(log_dir))
    ea.Reload()

    data = {}

    # Get all available tags
    tags = ea.Tags()

    # Extract scalar data
    for tag in tags['scalars']:
        events = ea.Scalars(tag)
        steps = [e.step for e in events]
        values = [e.value for e in events]
        data[tag] = {'steps': steps, 'values': values}

    return data

def analyze_experiments(base_dir='./runs_dual_mode_comparison'):
    """Analyze all three experiments"""
    base_path = Path(base_dir)

    experiments = {
        'Full-Training': None,
        'Rho-1': None,
        'Dual-MODE': None
    }

    # Load data for each experiment
    for exp_name in experiments.keys():
        exp_path = base_path / exp_name
        if exp_path.exists():
            # Find event file
            event_files = list(exp_path.glob('**/events.out.tfevents.*'))
            if event_files:
                print(f"Loading {exp_name}...")
                experiments[exp_name] = extract_tensorboard_data(event_files[0].parent)

    return experiments

def create_comprehensive_visualization(experiments):
    """Create comprehensive multi-panel visualization"""

    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

    colors = {
        'Full-Training': '#2E86AB',
        'Rho-1': '#A23B72',
        'Dual-MODE': '#F18F01'
    }

    # Panel 1: Training Loss over Steps
    ax1 = fig.add_subplot(gs[0, :2])
    for exp_name, data in experiments.items():
        if data and 'train/loss' in data:
            steps = data['train/loss']['steps']
            values = data['train/loss']['values']
            ax1.plot(steps, values, label=exp_name, color=colors[exp_name],
                    linewidth=2, alpha=0.8)
    ax1.set_xlabel('Training Step', fontsize=12)
    ax1.set_ylabel('Training Loss', fontsize=12)
    ax1.set_title('Training Loss Evolution', fontsize=14, fontweight='bold')
    ax1.legend(loc='upper right', fontsize=10)
    ax1.grid(True, alpha=0.3)

    # Panel 2: Validation Perplexity per Epoch
    ax2 = fig.add_subplot(gs[0, 2])
    epoch_ppls = {}
    for exp_name, data in experiments.items():
        if data and 'epoch/val_ppl' in data:
            epochs = data['epoch/val_ppl']['steps']
            values = data['epoch/val_ppl']['values']
            ax2.plot(epochs, values, marker='o', label=exp_name,
                    color=colors[exp_name], linewidth=2, markersize=8, alpha=0.8)
            epoch_ppls[exp_name] = values
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Perplexity', fontsize=12)
    ax2.set_title('Validation Perplexity', fontsize=14, fontweight='bold')
    ax2.legend(loc='best', fontsize=9)
    ax2.grid(True, alpha=0.3)

    # Panel 3: Token Selection Rate
    ax3 = fig.add_subplot(gs[1, 0])
    for exp_name, data in experiments.items():
        if data and 'train/selection' in data:
            steps = data['train/selection']['steps']
            values = [v * 100 for v in data['train/selection']['values']]  # Convert to percentage
            ax3.plot(steps, values, label=exp_name, color=colors[exp_name],
                    linewidth=2, alpha=0.8)
    ax3.set_xlabel('Training Step', fontsize=12)
    ax3.set_ylabel('Selection Rate (%)', fontsize=12)
    ax3.set_title('Token Selection Rate', fontsize=14, fontweight='bold')
    ax3.legend(loc='best', fontsize=10)
    ax3.grid(True, alpha=0.3)
    ax3.axhline(y=30, color='red', linestyle='--', alpha=0.5, label='Budget (30%)')

    # Panel 4: MODE Strategy Weights (if available)
    ax4 = fig.add_subplot(gs[1, 1:])
    mode_data = experiments.get('Dual-MODE')
    if mode_data:
        strategy_names = ['uncertainty', 'loss', 'coherence', 'diversity', 'token_mode']
        for strategy in strategy_names:
            key = f'strategy/{strategy}'
            if key in mode_data:
                epochs = mode_data[key]['steps']
                values = mode_data[key]['values']
                ax4.plot(epochs, values, marker='o', label=strategy.replace('_', ' ').title(),
                        linewidth=2, markersize=6, alpha=0.8)
        ax4.set_xlabel('Epoch', fontsize=12)
        ax4.set_ylabel('Strategy Weight', fontsize=12)
        ax4.set_title('MODE Strategy Weights Evolution', fontsize=14, fontweight='bold')
        ax4.legend(loc='best', fontsize=9)
        ax4.grid(True, alpha=0.3)
    else:
        ax4.text(0.5, 0.5, 'MODE Strategy Data Not Available',
                ha='center', va='center', fontsize=12)
        ax4.axis('off')

    # Panel 5: Final Comparison Table
    ax5 = fig.add_subplot(gs[2, :])
    ax5.axis('off')

    # Create comparison table
    table_data = []
    headers = ['Method', 'Best Val PPL', 'Final Train Loss', 'Avg Selection %', 'Improvement']

    full_ppl = None
    for exp_name in ['Full-Training', 'Rho-1', 'Dual-MODE']:
        data = experiments.get(exp_name)
        if data:
            # Best validation perplexity
            if 'epoch/val_ppl' in data:
                best_ppl = min(data['epoch/val_ppl']['values'])
                if exp_name == 'Full-Training':
                    full_ppl = best_ppl
                    improvement = '0.00%'
                else:
                    improvement = f"{((full_ppl - best_ppl) / full_ppl * 100):+.2f}%"
            else:
                best_ppl = 'N/A'
                improvement = 'N/A'

            # Final training loss
            if 'epoch/train_loss' in data:
                final_loss = data['epoch/train_loss']['values'][-1]
            else:
                final_loss = 'N/A'

            # Average selection rate
            if 'train/selection' in data:
                avg_sel = np.mean(data['train/selection']['values'][-100:]) * 100
            else:
                avg_sel = 'N/A'

            if isinstance(best_ppl, float):
                best_ppl = f"{best_ppl:.2f}"
            if isinstance(final_loss, float):
                final_loss = f"{final_loss:.4f}"
            if isinstance(avg_sel, float):
                avg_sel = f"{avg_sel:.1f}%"

            table_data.append([exp_name, best_ppl, final_loss, avg_sel, improvement])

    table = ax5.table(cellText=table_data, colLabels=headers,
                     cellLoc='center', loc='center',
                     bbox=[0.1, 0.3, 0.8, 0.5])
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 2)

    # Style header
    for i in range(len(headers)):
        table[(0, i)].set_facecolor('#4472C4')
        table[(0, i)].set_text_props(weight='bold', color='white')

    # Style rows
    for i in range(1, len(table_data) + 1):
        for j in range(len(headers)):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#E7E6E6')

    ax5.set_title('Final Comparison Summary', fontsize=14, fontweight='bold', pad=20)

    # Main title
    fig.suptitle('MODE vs Rho-1 vs Full-Training: Comprehensive Analysis',
                fontsize=16, fontweight='bold', y=0.995)

    return fig

def print_detailed_summary(experiments):
    """Print detailed text summary"""
    print("\n" + "="*80)
    print("DETAILED METRICS SUMMARY")
    print("="*80)

    for exp_name, data in experiments.items():
        if not data:
            continue

        print(f"\n{exp_name}")
        print("-" * 40)

        # Training metrics
        if 'train/loss' in data:
            losses = data['train/loss']['values']
            print(f"  Training Loss:")
            print(f"    Initial: {losses[0]:.4f}")
            print(f"    Final:   {losses[-1]:.4f}")
            print(f"    Min:     {min(losses):.4f}")

        # Validation perplexity
        if 'epoch/val_ppl' in data:
            ppls = data['epoch/val_ppl']['values']
            print(f"  Validation Perplexity:")
            print(f"    Epoch 1: {ppls[0]:.2f}")
            if len(ppls) > 1:
                print(f"    Epoch 2: {ppls[1]:.2f}")
            if len(ppls) > 2:
                print(f"    Epoch 3: {ppls[2]:.2f}")
            print(f"    Best:    {min(ppls):.2f}")

        # Selection rate
        if 'train/selection' in data:
            sel_rates = data['train/selection']['values']
            print(f"  Token Selection Rate:")
            print(f"    Average: {np.mean(sel_rates)*100:.1f}%")
            print(f"    Final:   {np.mean(sel_rates[-50:])*100:.1f}%")

        # MODE-specific metrics
        if exp_name == 'Dual-MODE':
            strategy_names = ['uncertainty', 'loss', 'coherence', 'diversity', 'token_mode']
            print(f"  Strategy Weights (Final Epoch):")
            for strategy in strategy_names:
                key = f'strategy/{strategy}'
                if key in data:
                    final_weight = data[key]['values'][-1]
                    print(f"    {strategy.replace('_', ' ').title():15s}: {final_weight:.4f}")

    # Comparative analysis
    print(f"\n" + "="*80)
    print("COMPARATIVE ANALYSIS")
    print("="*80)

    full_ppl = None
    rho1_ppl = None
    mode_ppl = None

    if experiments.get('Full-Training') and 'epoch/val_ppl' in experiments['Full-Training']:
        full_ppl = min(experiments['Full-Training']['epoch/val_ppl']['values'])
    if experiments.get('Rho-1') and 'epoch/val_ppl' in experiments['Rho-1']:
        rho1_ppl = min(experiments['Rho-1']['epoch/val_ppl']['values'])
    if experiments.get('Dual-MODE') and 'epoch/val_ppl' in experiments['Dual-MODE']:
        mode_ppl = min(experiments['Dual-MODE']['epoch/val_ppl']['values'])

    if full_ppl and rho1_ppl and mode_ppl:
        print(f"\nBest Validation Perplexity:")
        print(f"  Full-Training: {full_ppl:.2f}")
        print(f"  Rho-1:         {rho1_ppl:.2f} ({((full_ppl - rho1_ppl) / full_ppl * 100):+.2f}% vs Full)")
        print(f"  Dual-MODE:     {mode_ppl:.2f} ({((full_ppl - mode_ppl) / full_ppl * 100):+.2f}% vs Full)")
        print(f"\nDual-MODE vs Rho-1:")
        print(f"  Improvement: {((rho1_ppl - mode_ppl) / rho1_ppl * 100):+.2f}%")

        if mode_ppl < rho1_ppl:
            print("\n✓ SUCCESS: Dual-MODE outperforms Rho-1!")
        elif abs(mode_ppl - rho1_ppl) < 1:
            print("\n≈ COMPARABLE: Dual-MODE and Rho-1 are similar")
        else:
            print("\n✗ Rho-1 currently performs better")

    print("\n" + "="*80)

if __name__ == "__main__":
    print("Analyzing MODE comparison experiments...")

    # Extract data
    experiments = analyze_experiments()

    # Print detailed summary
    print_detailed_summary(experiments)

    # Create visualization
    print("\nCreating comprehensive visualization...")
    fig = create_comprehensive_visualization(experiments)

    # Save figure
    output_file = Path('./mode_comparison_metrics.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\nVisualization saved to: {output_file}")

    plt.show()

    print("\nAnalysis complete!")
