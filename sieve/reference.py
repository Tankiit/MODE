"""
Reference-model losses for the excess-loss scorer S_E.

S_E(y_t) = ℓ_θ(y_t | x_<t) − ℓ_ref(y_t | x_<t) (paper §3). Both terms are
computed on the same batch with the same context, so the frozen reference
model is run online at every step on the current input_ids. (A loss table
precomputed on fixed corpus windows would give the reference a different
context than the randomly offset training sequences.)
"""
from __future__ import annotations
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM


def load_reference(name: str, device: torch.device) -> torch.nn.Module:
    """Frozen reference model (float32 weights, eval mode, no gradients)."""
    model = AutoModelForCausalLM.from_pretrained(
        name, dtype=torch.float32, attn_implementation="sdpa"
    ).to(device).eval()
    model.requires_grad_(False)
    return model


@torch.no_grad()
def reference_nll(ref_model: torch.nn.Module, input_ids: torch.Tensor,
                  autocast_ctx) -> torch.Tensor:
    """
    Per-token NLL under the reference model, [B, T] float32.
    Column t is the loss of token t given tokens < t; column 0 is 0.
    """
    with autocast_ctx:
        logits = ref_model(input_ids=input_ids, use_cache=False).logits
    B, T, V = logits.shape
    nll = F.cross_entropy(
        logits[:, :-1].float().reshape(-1, V),
        input_ids[:, 1:].reshape(-1),
        reduction="none",
    ).reshape(B, T - 1)
    return F.pad(nll, (1, 0), value=0.0)
