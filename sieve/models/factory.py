"""Trained model, frozen reference model, and autocast helper."""
from __future__ import annotations

import contextlib

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, GPT2Config, GPT2LMHeadModel

GPT2_VOCAB = 50257
GPT2_EOS = 50256


def build_model(name: str, seq_len: int, device: torch.device,
                tiny_layers: int = 2, tiny_dim: int = 64, tiny_heads: int = 2) -> torch.nn.Module:
    """
    float32 weights (the forward runs under bf16 autocast on CUDA).
    `name` is a Hugging Face id (pretrained checkpoint) or "tiny", a small
    randomly initialised GPT-2 used by the tests.
    """
    if name == "tiny":
        cfg = GPT2Config(vocab_size=GPT2_VOCAB, n_positions=seq_len, n_embd=tiny_dim,
                         n_layer=tiny_layers, n_head=tiny_heads, resid_pdrop=0.0,
                         embd_pdrop=0.0, attn_pdrop=0.0, bos_token_id=GPT2_EOS,
                         eos_token_id=GPT2_EOS, pad_token_id=GPT2_EOS)
        return GPT2LMHeadModel(cfg).to(device=device, dtype=torch.float32)
    return AutoModelForCausalLM.from_pretrained(
        name, dtype=torch.float32, attn_implementation="sdpa").to(device)


def load_reference(name: str, device: torch.device) -> torch.nn.Module:
    """Frozen reference model for excess loss (eval mode, no gradients)."""
    model = build_model(name, 1024, device).eval()
    model.requires_grad_(False)
    return model


def autocast(device: torch.device, dtype_name: str = "bfloat16"):
    """bf16 autocast on CUDA; a no-op elsewhere."""
    if device.type == "cuda":
        return torch.autocast("cuda", dtype=getattr(torch, dtype_name))
    return contextlib.nullcontext()


@torch.no_grad()
def reference_nll(ref_model: torch.nn.Module, input_ids: torch.Tensor,
                  autocast_ctx) -> torch.Tensor:
    """
    Per-token NLL under the reference model on the same batch, [B, T] float32.
    Column t is the loss of token t given tokens < t; column 0 is 0.
    """
    with autocast_ctx:
        logits = ref_model(input_ids=input_ids, use_cache=False).logits
    B, T, V = logits.shape
    nll = F.cross_entropy(logits[:, :-1].float().reshape(-1, V),
                          input_ids[:, 1:].reshape(-1), reduction="none").reshape(B, T - 1)
    return F.pad(nll, (1, 0), value=0.0)
