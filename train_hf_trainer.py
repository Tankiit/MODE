"""
Train ramenGPT model with Hugging Face Trainer (optional DeepSpeed).

This script wraps the existing `model.GPT` into an HF Trainer workflow.

Key features:
- Uses Hugging Face `datasets` + `transformers` for data and training loop.
- Optional token masking via a simple SIEVE-like selector (random) to demo integration.
- Optional DeepSpeed via `--deepspeed path/to/ds_config.json`.

Notes:
- The ramenGPT `GPT.forward` expects 1D input and target tensors and a `sliding_window_num_blocks` tensor.
  We run with `per_device_train_batch_size=1` and perform the shift in `compute_loss`.
- For Colab T4, prefer `fp16=True` and `bf16=False`.
- This script downloads datasets/models if needed; run it in an environment with network access.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Dict, Any, Optional

import torch
from torch import Tensor


def _detect_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _load_config_module(path: str):
    import importlib.util

    spec = importlib.util.spec_from_file_location("config", path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)
    return module


@dataclass
class SelectorArgs:
    method: str = "full"  # "full" or "random"
    ratio: float = 0.7


class RandomSelector:
    def __init__(self, ratio: float, device: torch.device):
        self.ratio = ratio
        self.device = device

    def get_token_mask(self, inputs: Tensor) -> Tensor:
        # inputs is 1D: [T]
        T = inputs.size(0)
        k = max(1, int(T * self.ratio))
        idx = torch.randperm(T, device=self.device)[:k]
        mask = torch.zeros(T, dtype=torch.bool, device=self.device)
        mask[idx] = True
        return mask


def _get_window_blocks(window_size: int, device: torch.device) -> Tensor:
    return torch.tensor(window_size, dtype=torch.int32, device=device)


def build_model_from_config(config_module, max_seq_len: int, device: torch.device):
    # Lazy import to avoid overhead if script is only parsed
    from model import GPT, set_flex_attention_kernel_options

    # Configure flex attention kernels (best-effort)
    arch = torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu"
    set_flex_attention_kernel_options(arch)

    model_config = config_module.model_config
    attention_config = getattr(config_module, "attention_config", {})
    lambda_config = getattr(config_module, "lambda_config", None)
    lr_multipliers = getattr(config_module, "optimizer_config", {}).get("lr_multipliers", {})
    wd_multipliers = getattr(config_module, "optimizer_config", {}).get("wd_multipliers", {})
    rope_config = getattr(config_module, "rope_config", None)
    embed_config = getattr(config_module, "embed_config", None)
    gating_config = getattr(config_module, "gating_config", None)
    skip_config = getattr(config_module, "skip_config", None)
    residual_connection_config = getattr(config_module, "residual_connection_config", None)
    low_rank_config = getattr(config_module, "low_rank_config", None)
    attention_pattern_config = getattr(config_module, "attention_pattern_config", None)

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

    return model


def sieve_masked_loss(model, inputs: Tensor, targets: Tensor, mask: Tensor) -> Tensor:
    """Selective loss on masked tokens using logits captured from lm_head.

    Mirrors the optimization in sieve_run.sieve_masked_loss but minimal.
    """
    import torch.nn.functional as F

    captured: Dict[str, Tensor] = {}

    def hook_fn(module, _in, out):
        captured["logits"] = out

    handle = model.lm_head.register_forward_hook(hook_fn)
    try:
        _ = model(inputs, targets, torch.tensor(3, dtype=torch.int32, device=inputs.device))
    finally:
        handle.remove()

    logits = captured.get("logits")
    if logits is None:
        # Fallback: full loss already computed during forward (mean/sum). Use it.
        return _

    logits = model._logits_softcap_scale * torch.sigmoid(
        (logits + model._logits_softcap_shift) / model._logits_softcap_divisor
    )

    logits_flat = logits.view(-1, logits.size(-1))
    if not model.training:
        logits_flat = logits_flat.float()

    selected_logits = logits_flat[mask]
    selected_targets = targets[mask]
    loss = F.cross_entropy(selected_logits, selected_targets, reduction="sum")

    # Scale gradient magnitude to match full tokens
    num_selected = mask.sum().clamp(min=1).float()
    total = float(targets.size(0))
    loss = loss * (total / num_selected)
    return loss


def main():
    parser = argparse.ArgumentParser(description="Train ramenGPT with HF Trainer")
    parser.add_argument("--config", type=str, required=True, help="Path to config .py (e.g., config/base.py)")
    parser.add_argument("--dataset", type=str, default="wikitext", help="HF dataset name (default: wikitext)")
    parser.add_argument("--dataset_config", type=str, default="wikitext-2-raw-v1", help="HF dataset config")
    parser.add_argument("--text_column", type=str, default="text", help="Text column name in dataset")
    parser.add_argument("--model_name_or_path", type=str, default="gpt2", help="Tokenizer to use for encoding")
    parser.add_argument("--block_size", type=int, default=512, help="Sequence length")
    parser.add_argument("--max_steps", type=int, default=1000)
    parser.add_argument("--output_dir", type=str, default="./hf_trainer_out")
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--warmup_ratio", type=float, default=0.01)
    parser.add_argument("--logging_steps", type=int, default=20)
    parser.add_argument("--save_steps", type=int, default=200)
    parser.add_argument("--eval_steps", type=int, default=0, help="0 disables eval during training")
    parser.add_argument("--deepspeed", type=str, default=None, help="Path to DeepSpeed config JSON")
    parser.add_argument("--fp16", action="store_true", help="Enable FP16 training")
    parser.add_argument("--bf16", action="store_true", help="Enable BF16 training")
    parser.add_argument("--window_size_blocks", type=int, default=3, help="Sliding window size in blocks")

    # Optional SIEVE-like random masking
    parser.add_argument("--sieve_method", type=str, default="full", choices=["full", "random"], help="Token selection method")
    parser.add_argument("--sieve_ratio", type=float, default=0.7, help="If using random, fraction of tokens to keep")

    args = parser.parse_args()

    device = _detect_device()
    print(f"Device: {device}")

    config_module = _load_config_module(args.config)
    model = build_model_from_config(config_module, max_seq_len=args.block_size, device=device)

    # Build dataset + tokenizer
    from datasets import load_dataset
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    if tokenizer.pad_token is None:
        # For GPT2, pad as eos
        tokenizer.pad_token = tokenizer.eos_token

    raw_ds = load_dataset(args.dataset, args.dataset_config)
    column = args.text_column

    def tokenize_fn(examples):
        return tokenizer(examples[column], return_attention_mask=False)

    tokenized = raw_ds.map(tokenize_fn, batched=True, remove_columns=raw_ds["train"].column_names)

    # Group into fixed-size blocks
    def group_texts(examples):
        concatenated = {k: sum(examples[k], []) for k in examples.keys()}
        total_length = (len(concatenated["input_ids"]) // args.block_size) * args.block_size
        result = {
            k: [t[i : i + args.block_size] for i in range(0, total_length, args.block_size)]
            for k, t in concatenated.items()
        }
        return result

    lm_ds = tokenized.map(group_texts, batched=True)

    # HF Trainer setup
    from transformers import Trainer, TrainingArguments

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        evaluation_strategy=("steps" if args.eval_steps and args.eval_steps > 0 else "no"),
        eval_steps=(args.eval_steps if args.eval_steps and args.eval_steps > 0 else None),
        deepspeed=args.deepspeed,
        fp16=args.fp16,
        bf16=args.bf16,
        dataloader_drop_last=True,
        report_to=["none"],
    )

    window_blocks = _get_window_blocks(args.window_size_blocks, device=device)

    selector: Optional[RandomSelector] = None
    if args.sieve_method == "random":
        selector = RandomSelector(ratio=args.sieve_ratio, device=device)

    class RamenTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False):  # type: ignore[override]
            # inputs["input_ids"]: shape [B, T]
            input_ids = inputs["input_ids"]
            # We enforce B==1 to match model.forward signature; but in case user sets B>1, flatten leading dim.
            if input_ids.ndim == 2 and input_ids.size(0) > 1:
                input_ids = input_ids.view(-1)
            else:
                input_ids = input_ids.squeeze(0)

            # Shift for targets
            src = input_ids[:-1].to(device)
            tgt = input_ids[1:].to(device)

            model = model.to(device)
            model.train()

            if selector is not None:
                mask = selector.get_token_mask(src)
                loss = sieve_masked_loss(model, src, tgt, mask)
                outputs = None
            else:
                loss = model(src, tgt, window_blocks)
                outputs = None

            return (loss, outputs) if return_outputs else loss

    trainer = RamenTrainer(
        model=model,
        args=training_args,
        train_dataset=lm_ds["train"],
        eval_dataset=(lm_ds.get("validation") if args.eval_steps and args.eval_steps > 0 else None),
        tokenizer=tokenizer,
    )

    trainer.train()


if __name__ == "__main__":
    main()

