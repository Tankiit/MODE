import numpy as np
import matplotlib.pyplot as plt

# ---- Simple moving average for smoother curves ----
def smooth_curve(y, k=3):
    if k <= 1: 
        return y
    kernel = np.ones(k)/k
    y_pad = np.pad(y, (k//2, k-1-k//2), mode='edge')
    return np.convolve(y_pad, kernel, mode='valid')

def plot_weights(epochs, weights_dict, dataset_name, save_path,
                 phase_bounds=(10, 20), smooth_k=3):
    """
    epochs: array-like of epoch numbers
    weights_dict: dict with keys S_U, S_C, S_B, S_D -> list of weights per epoch
    dataset_name: str for title
    save_path: str path to save the figure
    phase_bounds: tuple of (end_early, end_middle)
    smooth_k: smoothing window size
    """
    order = ["S_U", "S_C", "S_B", "S_D"]
    colors = {
        "S_U": "#1f77b4",  # blue
        "S_C": "#ff7f0e",  # orange
        "S_B": "#2ca02c",  # green
        "S_D": "#d62728",  # red
    }
    labels = {
        "S_U": "S_U (Uncertainty)",
        "S_C": "S_C (Class balance)",
        "S_B": "S_B (Boundary)",
        "S_D": "S_D (Diversity)"
    }
    
    # Smooth weights
    weights_smooth = {k: smooth_curve(weights_dict[k], smooth_k) for k in order}

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    
    # --- Line plot ---
    for k in order:
        ax1.plot(epochs, weights_smooth[k], label=labels[k],
                 color=colors[k], marker='o', markersize=3, linewidth=1.8)
    
    # Add phase lines
    for bound in phase_bounds:
        ax1.axvline(bound, linestyle='--', color='k', alpha=0.7)
    ax1.axvspan(epochs[0], phase_bounds[0], alpha=0.05, color='gray')
    ax1.axvspan(phase_bounds[0], phase_bounds[1], alpha=0.05, color='gray')
    ax1.axvspan(phase_bounds[1], epochs[-1], alpha=0.05, color='gray')

    ax1.set_ylabel("Weight")
    ax1.set_title(f"Strategy Weights over Epochs — {dataset_name}")
    ax1.legend(ncol=2, frameon=False)
    ax1.set_ylim(0, 1.0)

    # --- Stacked area plot ---
    stack_data = np.vstack([weights_smooth[k] for k in order])
    ax2.stackplot(epochs, stack_data, labels=[labels[k] for k in order],
                  colors=[colors[k] for k in order], alpha=0.9)
    for bound in phase_bounds:
        ax2.axvline(bound, linestyle='--', color='k', alpha=0.7)
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Weight")
    ax2.set_title(f"Strategy Weight Distribution over Epochs — {dataset_name}")
    ax2.set_ylim(0, 1.0)

    plt.tight_layout()
    plt.savefig(save_path, dpi=160, bbox_inches="tight")
    plt.close()
    print(f"Saved figure: {save_path}")

# ---- Example: using your existing weights but over 30 epochs ----
epochs = np.arange(1, 31)

# Replace these with the actual weights from your two datasets
cifar10_weights = {
    "S_U": np.random.rand(30),
    "S_C": np.random.rand(30),
    "S_B": np.random.rand(30),
    "S_D": np.random.rand(30)
}
cifar100_weights = {
    "S_U": np.random.rand(30),
    "S_C": np.random.rand(30),
    "S_B": np.random.rand(30),
    "S_D": np.random.rand(30)
}

plot_weights(epochs, cifar10_weights, "CIFAR-10", "cifar10_mode_weights.png")
plot_weights(epochs, cifar100_weights, "CIFAR-100", "cifar100_mode_weights.png")