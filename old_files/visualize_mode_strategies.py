#!/usr/bin/env python3
"""
Visualize MODE Strategy Evolution
Shows how all 4 strategies (alignment, diversity, text_quality, balanced) adapt over training.
"""

import numpy as np
import matplotlib.pyplot as plt
from mode_hypernetwork import AdaptiveMODETrainer, HypernetworkConfig
from pathlib import Path


def run_and_visualize_strategies(output_dir: str = "./hypernetwork_results"):
    """Run simulation and visualize strategy evolution."""

    # Create config
    config = HypernetworkConfig(
        data_dir="/Users/tanmoy/research/data",
        output_dir=output_dir
    )

    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("MODE Strategy Evolution Visualization")
    print("=" * 70)

    # Initialize trainer
    trainer = AdaptiveMODETrainer(config=config)

    # Store all strategy weights over time
    epochs = []
    alignment_weights = []
    diversity_weights = []
    text_quality_weights = []
    balanced_weights = []

    performances = []
    losses = []
    grad_norms = []

    # Simulate training
    print("\nSimulating training...")
    for epoch in range(100):
        # Simulate realistic training metrics
        progress = epoch / 100

        # Performance improves over time
        performance = 0.1 + 0.8 * (1 - np.exp(-2 * progress)) + np.random.normal(0, 0.02)
        performance = np.clip(performance, 0, 1)

        # Loss decreases
        loss = 3.0 * np.exp(-1.5 * progress) + 0.1 + np.random.normal(0, 0.05)
        loss = np.clip(loss, 0.05, 5.0)

        # Gradient norm stabilizes
        grad_norm = 2.0 * np.exp(-progress) + 0.2 + np.random.normal(0, 0.05)
        grad_norm = np.clip(grad_norm, 0.1, 5.0)

        # Get strategy prediction
        predicted_strategies = trainer.training_step(performance, loss, grad_norm)

        # Store data
        epochs.append(epoch)
        alignment_weights.append(predicted_strategies['alignment'])
        diversity_weights.append(predicted_strategies['diversity'])
        text_quality_weights.append(predicted_strategies['text_quality'])
        balanced_weights.append(predicted_strategies['balanced'])

        performances.append(performance)
        losses.append(loss)
        grad_norms.append(grad_norm)

        if epoch % 10 == 0:
            print(f"Epoch {epoch:3d} | "
                  f"Perf: {performance:.3f} | "
                  f"Loss: {loss:.3f} | "
                  f"Alignment: {predicted_strategies['alignment']:.3f} | "
                  f"Diversity: {predicted_strategies['diversity']:.3f} | "
                  f"Quality: {predicted_strategies['text_quality']:.3f} | "
                  f"Balanced: {predicted_strategies['balanced']:.3f}")

    print("\nTraining simulation complete!")

    # Create comprehensive visualization
    print("\nCreating visualizations...")

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('MODE Strategy Evolution During Training', fontsize=16, fontweight='bold')

    # Plot 1: All strategies over time
    ax1 = axes[0, 0]
    ax1.plot(epochs, alignment_weights, 'r-', linewidth=2, label='Alignment', alpha=0.8)
    ax1.plot(epochs, diversity_weights, 'b-', linewidth=2, label='Diversity', alpha=0.8)
    ax1.plot(epochs, text_quality_weights, 'g-', linewidth=2, label='Text Quality', alpha=0.8)
    ax1.plot(epochs, balanced_weights, 'm-', linewidth=2, label='Balanced', alpha=0.8)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Strategy Weight', fontsize=12)
    ax1.set_title('Strategy Weights Over Time', fontsize=14, fontweight='bold')
    ax1.legend(loc='best', fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([0, 0.5])

    # Add training phase annotations
    ax1.axvspan(0, 30, alpha=0.1, color='red', label='Early Phase')
    ax1.axvspan(30, 70, alpha=0.1, color='yellow', label='Middle Phase')
    ax1.axvspan(70, 100, alpha=0.1, color='green', label='Late Phase')

    # Plot 2: Training metrics
    ax2 = axes[0, 1]
    ax2_twin = ax2.twinx()

    line1 = ax2.plot(epochs, performances, 'g-', linewidth=2, label='Performance', alpha=0.8)
    line2 = ax2_twin.plot(epochs, losses, 'r-', linewidth=2, label='Loss', alpha=0.8)

    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Performance', fontsize=12, color='g')
    ax2_twin.set_ylabel('Loss', fontsize=12, color='r')
    ax2.set_title('Training Metrics', fontsize=14, fontweight='bold')
    ax2.tick_params(axis='y', labelcolor='g')
    ax2_twin.tick_params(axis='y', labelcolor='r')
    ax2.grid(True, alpha=0.3)

    # Combined legend
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax2.legend(lines, labels, loc='center right', fontsize=10)

    # Plot 3: Strategy composition (stacked area)
    ax3 = axes[1, 0]
    ax3.stackplot(epochs,
                  alignment_weights,
                  diversity_weights,
                  text_quality_weights,
                  balanced_weights,
                  labels=['Alignment', 'Diversity', 'Text Quality', 'Balanced'],
                  colors=['red', 'blue', 'green', 'magenta'],
                  alpha=0.7)
    ax3.set_xlabel('Epoch', fontsize=12)
    ax3.set_ylabel('Strategy Proportion', fontsize=12)
    ax3.set_title('Strategy Composition (Stacked)', fontsize=14, fontweight='bold')
    ax3.legend(loc='upper right', fontsize=10)
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim([0, 1])

    # Plot 4: Gradient norms
    ax4 = axes[1, 1]
    ax4.plot(epochs, grad_norms, 'purple', linewidth=2, alpha=0.8)
    ax4.set_xlabel('Epoch', fontsize=12)
    ax4.set_ylabel('Gradient Norm', fontsize=12)
    ax4.set_title('Gradient Norm Evolution', fontsize=14, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    ax4.axhline(y=1.0, color='r', linestyle='--', alpha=0.5, label='Target Stability')
    ax4.legend(loc='best', fontsize=10)

    plt.tight_layout()

    # Save figure
    output_file = Path(output_dir) / 'mode_strategy_evolution.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved visualization to: {output_file}")

    plt.show()

    # Print final analysis
    print("\n" + "=" * 70)
    print("FINAL ANALYSIS")
    print("=" * 70)

    final_interpretation = trainer.get_current_state_interpretation()

    print(f"\nTraining Phase: {final_interpretation['training_phase']}")
    print(f"\nFinal Strategy Weights:")
    for strategy, weight in final_interpretation['predicted_strategies'].items():
        print(f"  {strategy:15s}: {weight:.4f}")

    print(f"\nStrategy Evolution Summary:")
    print(f"  Alignment:    {alignment_weights[0]:.3f} → {alignment_weights[-1]:.3f} "
          f"(Δ={alignment_weights[-1]-alignment_weights[0]:+.3f})")
    print(f"  Diversity:    {diversity_weights[0]:.3f} → {diversity_weights[-1]:.3f} "
          f"(Δ={diversity_weights[-1]-diversity_weights[0]:+.3f})")
    print(f"  Text Quality: {text_quality_weights[0]:.3f} → {text_quality_weights[-1]:.3f} "
          f"(Δ={text_quality_weights[-1]-text_quality_weights[0]:+.3f})")
    print(f"  Balanced:     {balanced_weights[0]:.3f} → {balanced_weights[-1]:.3f} "
          f"(Δ={balanced_weights[-1]-balanced_weights[0]:+.3f})")

    print("\nTop 3 Most Relevant Memory Patterns:")
    for i, pattern in enumerate(final_interpretation['relevant_memory_patterns'][:3]):
        print(f"\n  Pattern {i+1}:")
        print(f"    Attention Weight: {pattern['attention_weight']:.4f}")
        print(f"    Success Score:    {pattern['success_value']:.4f}")
        print(f"    Strategy Mix:")
        for name, weight in pattern['strategy_pattern'].items():
            print(f"      {name:15s}: {weight:.4f}")

    print("\n" + "=" * 70)

    return trainer, {
        'epochs': epochs,
        'alignment': alignment_weights,
        'diversity': diversity_weights,
        'text_quality': text_quality_weights,
        'balanced': balanced_weights,
        'performance': performances,
        'loss': losses,
        'grad_norms': grad_norms
    }


if __name__ == "__main__":
    trainer, results = run_and_visualize_strategies()

    print("\nVisualization complete!")
    print("Check ./hypernetwork_results/mode_strategy_evolution.png for the plot")
