"""
SIEVE training script with simplified data loading.

This script demonstrates how to use the SIEVE token selection method
with the simple training data loader for efficient training.
"""

import argparse
import os
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import Tensor

from data.simple_train_loader import SimpleTrainLoader
from model import GPT, set_flex_attention_kernel_options
from optimizers import create_optimizer


def sieve_masked_loss(model, inputs: Tensor, targets: Tensor,
                      mask: Tensor, sliding_window_num_blocks: Tensor,
                      scale: bool = True) -> Tensor:
    """Compute loss only on SIEVE-selected tokens.

    Flow:
      1. Hook lm_head to capture logits during the normal forward pass.
      2. Run full forward (all tokens — required for causal context).
      3. Apply softcap to raw logits (replicating model.forward).
      4. Compute cross-entropy ONLY on selected tokens (selective logit indexing).
      5. Scale gradient magnitude to match full-batch training.

    The key optimization is step 4: we index *before* cross_entropy so that
    autograd never builds the softmax backward graph for masked tokens.
    """
    captured = {}

    def hook_fn(module, input, output):
        captured["logits"] = output

    handle = model.lm_head.register_forward_hook(hook_fn)

    try:
        # Full forward — we discard this loss and recompute with the mask
        full_loss = model(inputs, targets, sliding_window_num_blocks)
    finally:
        handle.remove()

    logits = captured.get("logits")
    if logits is None:
        # Hook failed (e.g. compiled model); fall back to full loss
        return full_loss

    # Replicate the softcap that model.forward applies after lm_head
    logits = model._logits_softcap_scale * torch.sigmoid(
        (logits + model._logits_softcap_shift) / model._logits_softcap_divisor
    )

    logits_flat = logits.view(-1, logits.size(-1))
    if not model.training:
        logits_flat = logits_flat.float()

    # Selective logit indexing - index selected tokens BEFORE CE
    # This reduces computation and memory for masked tokens
    selected_logits = logits_flat[mask]       # [num_selected, vocab_size]
    selected_targets = targets[mask]           # [num_selected]
    loss = F.cross_entropy(selected_logits, selected_targets, reduction="sum")

    if scale:
        # Scale so gradient magnitude matches full-batch training
        num_selected = mask.sum().clamp(min=1).float()
        total = float(targets.size(0))
        loss = loss * (total / num_selected)

    return loss


def create_selector(method: str, ratio: float = 0.7, device: str = "cuda"):
    """
    Create a token selector based on the specified method.

    For now, implements simple random selection as a placeholder.
    In production, this would integrate with the full SIEVE system.
    """
    if method == "full":
        return None  # No selection, use all tokens

    class SimpleSelector:
        """Simple random token selector for demonstration."""
        def __init__(self, ratio: float, device: str):
            self.ratio = ratio
            self.device = device

        def get_token_mask(self, inputs: Tensor, **kwargs) -> Tensor:
            """Generate a random token mask."""
            num_tokens = inputs.size(0)
            num_selected = int(num_tokens * self.ratio)

            # Random selection
            indices = torch.randperm(num_tokens, device=self.device)[:num_selected]
            mask = torch.zeros(num_tokens, dtype=torch.bool, device=self.device)
            mask[indices] = True
            return mask

    return SimpleSelector(ratio=ratio, device=device)


def get_window_size_blocks(window_size: int, block_size: int = 128, device=None) -> Tensor:
    """Get window size in blocks for FlexAttention."""
    if device is None:
        device = torch.device(
            "cuda" if torch.cuda.is_available() else
            "mps" if (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()) else "cpu"
        )
    return torch.tensor(window_size, dtype=torch.int32, device=device)


def run_sieve_training(
    config_file: str,
    data_pattern: str,
    sieve_method: str = "full",
    sieve_ratio: float = 0.7,
    num_tokens: int = 512,
    max_seq_len: int = 512,
    num_steps: int = 100,
    device: str = None,
):
    """
    Run training with SIEVE token selection.

    Args:
        config_file: Path to model configuration file
        data_pattern: Glob pattern for training data files
        sieve_method: Token selection method ("full", "random", etc.)
        sieve_ratio: Fraction of tokens to keep when using SIEVE
        num_tokens: Number of tokens per batch
        max_seq_len: Maximum sequence length
        num_steps: Number of training steps
        device: Device to train on
    """
    # Setup device
    if device is None:
        device = torch.device(
            "cuda" if torch.cuda.is_available() else
            "mps" if (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()) else "cpu"
        )
    else:
        device = torch.device(device)

    print(f"Using device: {device}")

    # Load configuration
    import importlib.util
    spec = importlib.util.spec_from_file_location("config", config_file)
    config_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config_module)
    config = config_module

    # Configure model backend
    detected_gpu = {"architecture": "unknown"}
    if device.type == "cuda":
        detected_gpu["architecture"] = torch.cuda.get_device_name(0)
    set_flex_attention_kernel_options(detected_gpu.get("architecture"))

    # Create model
    model_config = config.model_config
    attention_config = getattr(config, "attention_config", {})
    lambda_config = getattr(config, "lambda_config", None)
    lr_multipliers = getattr(config, "optimizer_config", {}).get("lr_multipliers", {})
    wd_multipliers = getattr(config, "optimizer_config", {}).get("wd_multipliers", {})
    rope_config = getattr(config, "rope_config", None)
    embed_config = getattr(config, "embed_config", None)
    gating_config = getattr(config, "gating_config", None)
    skip_config = getattr(config, "skip_config", None)
    residual_connection_config = getattr(config, "residual_connection_config", None)
    low_rank_config = getattr(config, "low_rank_config", None)
    attention_pattern_config = getattr(config, "attention_pattern_config", None)

    model = GPT(
        model_config=model_config,
        attention_config=attention_config,
        lambda_config=lambda_config,
        lr_multipliers=lr_multipliers,
        max_seq_len=max_seq_len,
        attention_pattern_config=attention_pattern_config,
        gating_config=gating_config,
        skip_config=skip_config,
        rope_config=rope_config,
        embed_config=embed_config,
        residual_connection_config=residual_connection_config,
        wd_multipliers=wd_multipliers,
        low_rank_config=low_rank_config,
    ).to(device)

    # Convert to bfloat16 on CUDA
    if device.type == "cuda":
        for m in model.modules():
            if isinstance(m, (torch.nn.Embedding, torch.nn.Linear)):
                m.weight.data = m.weight.data.bfloat16()

    # Create optimizers
    optimizer_config = config.optimizer_config
    optimizer_state = create_optimizer(model, optimizer_config, print_fn=print)
    optimizers = optimizer_state["optimizers"]

    # Initialize learning rates
    for opt in optimizers:
        for group in opt.param_groups:
            group["initial_lr"] = group["lr"]

    # Create data loader
    print(f"Loading data from: {data_pattern}")
    train_loader = SimpleTrainLoader(
        filename_pattern=data_pattern,
        num_tokens=num_tokens,
        max_seq_len=max_seq_len,
        align_to_bos=True,
        device=device,
    )

    # Create SIEVE selector
    use_sieve = sieve_method != "full"
    if use_sieve:
        selector = create_selector(sieve_method, ratio=sieve_ratio, device=str(device))
        print(f"SIEVE enabled: method={sieve_method}, ratio={sieve_ratio}")
    else:
        selector = None
        print("SIEVE disabled: using full batch training")

    # Training loop
    print(f"\nStarting training for {num_steps} steps...")
    print("=" * 80)

    window_size = 3  # Default window size
    window_blocks = get_window_size_blocks(window_size, device=device)

    model.train()
    total_loss = 0.0
    start_time = time.perf_counter()

    for step in range(num_steps):
        inputs, targets = next(train_loader)

        if use_sieve and selector is not None:
            # SIEVE path: select tokens and compute masked loss
            mask = selector.get_token_mask(inputs)
            loss = sieve_masked_loss(
                model, inputs, targets, mask, window_blocks,
                scale=True,
            )
        else:
            # Standard path: compute loss on all tokens
            loss = model(inputs, targets, window_blocks)

        # Backward pass
        loss.backward()

        # Step optimizers
        for opt in optimizers:
            opt.step()
            opt.zero_grad()

        # Logging
        total_loss += loss.detach().item()
        avg_loss = total_loss / (step + 1)

        if (step + 1) % 10 == 0 or step == 0:
            elapsed = time.perf_counter() - start_time
            print(f"Step {step+1:4d}/{num_steps} | "
                  f"loss: {loss.item():.4f} | "
                  f"avg_loss: {avg_loss:.4f} | "
                  f"time: {elapsed:.2f}s")

    # Final stats
    total_time = time.perf_counter() - start_time
    final_avg_loss = total_loss / num_steps

    print("=" * 80)
    print(f"Training complete!")
    print(f"Total time: {total_time:.2f}s")
    print(f"Average loss: {final_avg_loss:.4f}")
    print(f"Steps per second: {num_steps/total_time:.2f}")


def main():
    parser = argparse.ArgumentParser(
        description="SIEVE training with simplified data loading"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/base.py",
        help="Path to model configuration file",
    )
    parser.add_argument(
        "--data_pattern",
        type=str,
        default="data/*.bin",
        help="Glob pattern for training data files",
    )
    parser.add_argument(
        "--method",
        type=str,
        default="full",
        choices=["full", "random", "sieve", "sieve_online", "sieve_offline"],
        help="Token selection method (default: full = standard training)",
    )
    parser.add_argument(
        "--select_ratio",
        type=float,
        default=0.7,
        help="Fraction of tokens to keep when using selection (default: 0.7)",
    )
    parser.add_argument(
        "--num_tokens",
        type=int,
        default=512,
        help="Number of tokens per batch (default: 512)",
    )
    parser.add_argument(
        "--max_seq_len",
        type=int,
        default=512,
        help="Maximum sequence length (default: 512)",
    )
    parser.add_argument(
        "--num_steps",
        type=int,
        default=100,
        help="Number of training steps (default: 100)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to train on (default: auto-detect)",
    )

    args = parser.parse_args()

    # Validate data pattern exists
    import glob
    matching_files = glob.glob(args.data_pattern)
    if not matching_files:
        print(f"Error: No data files found matching pattern: {args.data_pattern}")
        print("\nHint: You can generate small test datasets using:")
        print("  python data/small_datasets.py -d tinyshakespeare")
        print("  python data/small_datasets.py -d wikitext2")
        return 1

    print(f"Found {len(matching_files)} data file(s)")

    try:
        run_sieve_training(
            config_file=args.config,
            data_pattern=args.data_pattern,
            sieve_method=args.method,
            sieve_ratio=args.select_ratio,
            num_tokens=args.num_tokens,
            max_seq_len=args.max_seq_len,
            num_steps=args.num_steps,
            device=args.device,
        )
        return 0
    except Exception as e:
        print(f"\nError during training: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
