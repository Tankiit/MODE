import matplotlib.pyplot as plt
import numpy as np

def create_curriculum_visualization():
    """
    Create a visualization showing MODE's curriculum learning behavior
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Data from your report
    epochs = ['Early\n(1-15)', 'Middle\n(16-35)', 'Late\n(36-50)']
    strategies = ['S_D: Diversity', 'S_C: Class Balance', 'S_U: Uncertainty', 'S_B: Boundary']
    
    # Strategy weights by stage (from your data)
    weights = np.array([
        [0.200, 0.187, 0.162, 0.156],  # Early
        [0.180, 0.160, 0.185, 0.172],  # Middle  
        [0.120, 0.105, 0.247, 0.233]   # Late
    ]).T
    
    # Plot 1: Stacked area chart showing evolution
    ax1.stackplot(range(3), weights, labels=strategies, alpha=0.8)
    ax1.set_xticks(range(3))
    ax1.set_xticklabels(epochs)
    ax1.set_ylabel('Strategy Weight')
    ax1.set_title('Strategy Weight Evolution Across Training Stages')
    ax1.legend(loc='upper left', bbox_to_anchor=(1, 1))
    ax1.set_ylim(0, 1)
    
    # Plot 2: Strategy dominance across budgets
    budgets = ['10%', '30%', '50%']
    dominance_uncertainty = [48, 46, 50]  # from your data
    dominance_diversity = [44, 42, 38]
    
    x = np.arange(len(budgets))
    width = 0.35
    
    ax2.bar(x - width/2, dominance_uncertainty, width, label='S_U: Uncertainty', color='#1f77b4')
    ax2.bar(x + width/2, dominance_diversity, width, label='S_D: Diversity', color='#ff7f0e')
    
    ax2.set_xlabel('Budget')
    ax2.set_ylabel('Dominance (% of epochs)')
    ax2.set_title('Strategy Dominance by Budget')
    ax2.set_xticks(x)
    ax2.set_xticklabels(budgets)
    ax2.legend()
    
    plt.tight_layout()
    
    # Save the figure
    plt.savefig('curriculum_visualization.png', dpi=300, bbox_inches='tight')
    plt.savefig('curriculum_visualization.pdf', bbox_inches='tight')
    print("Images saved as 'curriculum_visualization.png' and 'curriculum_visualization.pdf'")
    
    return fig

def main():
    """Main function to create and save the curriculum visualization"""
    fig = create_curriculum_visualization()
    plt.show()  # Display the plot
    return fig

if __name__ == "__main__":
    main()