"""
SIEVE masked loss implementation.

This module provides the SIEVE token selection loss function that computes
cross-entropy only on selected tokens while maintaining causal attention
over all tokens.
"""

import torch
import torch.nn.functional as F
from torch import Tensor


def sieve_masked_loss(model, inputs: Tensor, targets: Tensor,
                      mask: Tensor, sliding_window_num_blocks: Tensor,
                      scale: bool = True) -> Tensor:
    """Compute loss only on SIEVE-selected tokens.

    This function implements the SIEVE (Selective Loss Improvement via
    Value-based Evaluation) method for efficient training by computing
    loss only on a subset of tokens while maintaining full causal context.

    Flow:
      1. Hook lm_head to capture logits during the normal forward pass.
      2. Run full forward (all tokens — required for causal context).
      3. Apply softcap to raw logits (replicating model.forward).
      4. Compute cross-entropy ONLY on selected tokens (selective logit indexing).
      5. Scale gradient magnitude to match full-batch training.

    The key optimization is step 4: we index *before* cross_entropy so that
    autograd never builds the softmax backward graph for masked tokens.
    This provides ~30% reduction in CE backward cost and reduces gradient
    memory usage.

    Args:
        model: The GPT model
        inputs: Input token IDs [T]
        targets: Target token IDs [T]
        mask: Boolean mask indicating which tokens to select [T]
        sliding_window_num_blocks: Window size in blocks for FlexAttention
        scale: If True, scale loss to match full-batch gradient magnitude

    Returns:
        Loss scalar (optionally scaled to match full batch)
    """
    captured = {}

    def hook_fn(module, input, output):
        captured["logits"] = output

    handle = model.lm_head.register_forward_hook(hook_fn)

    try:
        # Full forward — we discard this loss and recompute with the mask
        # We need the full forward pass for causal attention context
        full_loss = model(inputs, targets, sliding_window_num_blocks)
    finally:
        handle.remove()

    logits = captured.get("logits")
    if logits is None:
        # Hook failed (e.g. compiled model); fall back to full loss
        # This can happen with torch.compile
        return full_loss

    # Replicate the softcap that model.forward applies after lm_head
    # This ensures consistency with the standard forward pass
    logits = model._logits_softcap_scale * torch.sigmoid(
        (logits + model._logits_softcap_shift) / model._logits_softcap_divisor
    )

    # Flatten logits: [T, vocab_size]
    logits_flat = logits.view(-1, logits.size(-1))

    # Use float32 for logits in eval mode for numerical stability
    if not model.training:
        logits_flat = logits_flat.float()

    # ---- Selective logit indexing (key optimization) ----
    # OLD approach (wasteful):
    #   per_token = F.cross_entropy(logits_flat, targets, reduction="none")
    #   loss = (per_token * mask.float()).sum()
    #
    # This computes CE for ALL tokens, then zeros out masked ones.
    # The backward pass still computes gradients for masked tokens.
    #
    # NEW approach (efficient):
    #   Index selected tokens BEFORE CE computation
    #   This ensures autograd only runs softmax backward for selected tokens
    #
    # Benefits at 70% selection ratio:
    #   - ~30% reduction in CE backward cost
    #   - Reduces [T, V] gradient tensor from 3.3 GB to 2.3 GB
    #   - Less memory bandwidth usage
    selected_logits = logits_flat[mask]       # [num_selected, vocab_size]
    selected_targets = targets[mask]           # [num_selected]

    # Compute cross-entropy only on selected tokens
    loss = F.cross_entropy(selected_logits, selected_targets, reduction="sum")

    if scale:
        # Scale so gradient magnitude matches full-batch training
        # Without this, effective LR drops proportionally to selection ratio
        num_selected = mask.sum().clamp(min=1).float()
        total = float(targets.size(0))
        loss = loss * (total / num_selected)

    return loss


def sieve_masked_loss_simple(model, inputs: Tensor, targets: Tensor,
                             mask: Tensor, sliding_window_num_blocks: Tensor,
                             scale: bool = True) -> Tensor:
    """Simplified SIEVE loss using per-token loss computation.

    This version computes per-token loss for all tokens and then masks,
    which is simpler but less efficient. Use this for debugging or if
    the selective indexing approach has issues.

    Args:
        model: The GPT model
        inputs: Input token IDs [T]
        targets: Target token IDs [T]
        mask: Boolean mask indicating which tokens to select [T]
        sliding_window_num_blocks: Window size in blocks for FlexAttention
        scale: If True, scale loss to match full-batch gradient magnitude

    Returns:
        Loss scalar (optionally scaled to match full batch)
    """
    captured = {}

    def hook_fn(module, input, output):
        captured["logits"] = output

    handle = model.lm_head.register_forward_hook(hook_fn)

    try:
        full_loss = model(inputs, targets, sliding_window_num_blocks)
    finally:
        handle.remove()

    logits = captured.get("logits")
    if logits is None:
        return full_loss

    logits = model._logits_softcap_scale * torch.sigmoid(
        (logits + model._logits_softcap_shift) / model._logits_softcap_divisor
    )

    logits_flat = logits.view(-1, logits.size(-1))
    if not model.training:
        logits_flat = logits_flat.float()

    # Compute per-token loss for all tokens, then mask
    per_token = F.cross_entropy(logits_flat, targets, reduction="none")
    masked = per_token * mask.float()
    loss = masked.sum()

    if scale:
        num_selected = mask.sum().clamp(min=1).float()
        total = float(targets.size(0))
        loss = loss * (total / num_selected)

    return loss


class SieveLossTracker:
    """Track statistics about SIEVE training."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.total_tokens = 0
        self.selected_tokens = 0
        self.total_loss = 0.0
        self.num_steps = 0

    def update(self, mask: Tensor, loss: Tensor):
        """Update statistics with a new step."""
        self.total_tokens += mask.numel()
        self.selected_tokens += mask.sum().item()
        self.total_loss += loss.detach().item()
        self.num_steps += 1

    def get_selection_ratio(self) -> float:
        """Get the actual selection ratio so far."""
        if self.total_tokens == 0:
            return 0.0
        return self.selected_tokens / self.total_tokens

    def get_avg_loss(self) -> float:
        """Get the average loss so far."""
        if self.num_steps == 0:
            return 0.0
        return self.total_loss / self.num_steps

    def get_stats(self) -> dict:
        """Get all statistics as a dictionary."""
        return {
            "total_tokens": self.total_tokens,
            "selected_tokens": self.selected_tokens,
            "selection_ratio": self.get_selection_ratio(),
            "avg_loss": self.get_avg_loss(),
            "num_steps": self.num_steps,
        }

    def __str__(self) -> str:
        """String representation of statistics."""
        stats = self.get_stats()
        return (
            f"SieveLossTracker(steps={stats['num_steps']}, "
            f"selection_ratio={stats['selection_ratio']:.2%}, "
            f"avg_loss={stats['avg_loss']:.4f})"
        )
