"""
Example script showing how to use the SimpleTrainLoader.

This demonstrates basic usage patterns for the simplified training data loader.
"""

import torch
from data.simple_train_loader import SimpleTrainLoader, create_simple_loader


def example_basic_usage():
    """Basic usage example."""
    print("=" * 60)
    print("Example 1: Basic Usage")
    print("=" * 60)

    # Create a simple loader
    loader = SimpleTrainLoader(
        filename_pattern="data/*.bin",
        num_tokens=512,  # tokens per batch
        max_seq_len=512,
        align_to_bos=True,
    )

    # Get a few batches
    for i, (inputs, targets) in enumerate(loader):
        if i >= 3:  # Just show 3 batches
            break

        print(f"\nBatch {i + 1}:")
        print(f"  Inputs shape: {inputs.shape}")
        print(f"  Targets shape: {targets.shape}")
        print(f"  Inputs dtype: {inputs.dtype}")
        print(f"  Targets dtype: {targets.dtype}")
        print(f"  Inputs device: {inputs.device}")
        print(f"  Sample inputs (first 10): {inputs[:10].tolist()}")
        print(f"  Sample targets (first 10): {targets[:10].tolist()}")


def example_convenience_function():
    """Example using the convenience function."""
    print("\n" + "=" * 60)
    print("Example 2: Using Convenience Function")
    print("=" * 60)

    # Use the convenience function
    loader = create_simple_loader(
        filename_pattern="data/*.bin",
        batch_size_tokens=256,
        seq_len=256,
    )

    inputs, targets = next(loader)
    print(f"Single batch:")
    print(f"  Inputs: {inputs.shape}, {inputs.dtype}, {inputs.device}")
    print(f"  Targets: {targets.shape}, {targets.dtype}, {targets.device}")


def example_training_loop():
    """Example of using in a simple training loop."""
    print("\n" + "=" * 60)
    print("Example 3: Simple Training Loop")
    print("=" * 60)

    # Create loader
    loader = SimpleTrainLoader(
        filename_pattern="data/*.bin",
        num_tokens=128,
        max_seq_len=128,
        align_to_bos=True,
    )

    # Simulate a simple training loop
    num_steps = 5
    total_loss = 0.0

    for step in range(num_steps):
        inputs, targets = next(loader)

        # Simulate forward pass (replace with actual model)
        # loss = model(inputs, targets)
        # For demo, just use a random loss
        loss = torch.randn(1).item()

        total_loss += loss

        print(f"Step {step + 1}/{num_steps}: loss={loss:.4f}")

    avg_loss = total_loss / num_steps
    print(f"\nAverage loss over {num_steps} steps: {avg_loss:.4f}")


def example_device_specification():
    """Example showing device specification."""
    print("\n" + "=" * 60)
    print("Example 4: Device Specification")
    print("=" * 60)

    # Load to CPU explicitly
    device = torch.device("cpu")
    loader = SimpleTrainLoader(
        filename_pattern="data/*.bin",
        num_tokens=64,
        max_seq_len=64,
        align_to_bos=True,
        device=device,
    )

    inputs, targets = next(loader)
    print(f"Data loaded to: {inputs.device}")


def example_sequential_vs_bos_aligned():
    """Example comparing sequential vs BOS-aligned batching."""
    print("\n" + "=" * 60)
    print("Example 5: Sequential vs BOS-Aligned Batching")
    print("=" * 60)

    # BOS-aligned (respects document boundaries)
    print("\nBOS-aligned batching:")
    loader_bos = SimpleTrainLoader(
        filename_pattern="data/*.bin",
        num_tokens=128,
        max_seq_len=128,
        align_to_bos=True,
    )
    inputs, targets = next(loader_bos)
    print(f"  Batch shape: {inputs.shape}")
    print(f"  First 20 tokens: {inputs[:20].tolist()}")

    # Sequential (simple sliding window)
    print("\nSequential batching:")
    loader_seq = SimpleTrainLoader(
        filename_pattern="data/*.bin",
        num_tokens=128,
        max_seq_len=128,
        align_to_bos=False,
    )
    inputs, targets = next(loader_seq)
    print(f"  Batch shape: {inputs.shape}")
    print(f"  First 20 tokens: {inputs[:20].tolist()}")


if __name__ == "__main__":
    try:
        example_basic_usage()
        example_convenience_function()
        example_training_loop()
        example_device_specification()
        example_sequential_vs_bos_aligned()

        print("\n" + "=" * 60)
        print("All examples completed successfully!")
        print("=" * 60)

    except FileNotFoundError as e:
        print(f"\nError: {e}")
        print("\nNote: Make sure you have data files available.")
        print("You can generate small test datasets using:")
        print("  python data/small_datasets.py -d tinyshakespeare")
        print("  python data/small_datasets.py -d wikitext2")

    except Exception as e:
        print(f"\nUnexpected error: {e}")
        import traceback

        traceback.print_exc()
