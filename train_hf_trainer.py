"""
Train with Hugging Face Trainer (optional DeepSpeed).

Supports two model backends:
- **ramenGPT** (default when --config is provided): custom GPT from model.py
- **HF native** (when --config is omitted): any AutoModelForCausalLM (e.g. TinyLlama)

Key features:
- Uses Hugging Face `datasets` + `transformers` for data and training loop.
- Optional token masking via a simple SIEVE-like selector (random) to demo integration.
- Optional DeepSpeed via `--deepspeed path/to/ds_config.json`.

Examples:
    # ramenGPT mode (original)
    python train_hf_trainer.py --config config/base.py --model_name_or_path gpt2

    # HF native mode (e.g. TinyLlama)
    python train_hf_trainer.py --model_name_or_path TinyLlama/TinyLlama-1.1B-Chat-v1.0

Notes:
- ramenGPT mode: GPT.forward expects 1D tensors; batch_size forced to 1.
- HF native mode: standard batched training, no softcap, no sliding window.
- For Colab T4, prefer `fp16=True` and `bf16=False`.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from typing import Dict, Any, Optional

import torch
import torch.nn.functional as F
from torch import Tensor
from transformers import SchedulerType


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

    def get_token_mask(self, inputs) -> Tensor:
        # inputs can be a Tensor (ramenGPT) or an int (HF native)
        T = inputs if isinstance(inputs, int) else inputs.size(0)
        k = max(1, int(T * self.ratio))
        idx = torch.randperm(T, device=self.device)[:k]
        mask = torch.zeros(T, dtype=torch.bool, device=self.device)
        mask[idx] = True
        return mask


def _get_window_blocks(window_size: int, device: torch.device) -> Tensor:
    return torch.tensor(window_size, dtype=torch.int32, device=device)


def build_model_from_config(config_module, max_seq_len: int, device: torch.device):
    """Build a ramenGPT model from a config module."""
    from model import GPT, set_flex_attention_kernel_options

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


def build_model_hf_native(model_name_or_path: str, bf16: bool = False, fp16: bool = False):
    """Build a native HuggingFace causal LM (e.g. TinyLlama, GPT-2, LLaMA)."""
    from transformers import AutoModelForCausalLM

    dtype = torch.bfloat16 if bf16 else (torch.float16 if fp16 else torch.float32)
    model = AutoModelForCausalLM.from_pretrained(model_name_or_path, torch_dtype=dtype)
    print(f"Loaded HF model: {model_name_or_path} ({sum(p.numel() for p in model.parameters())/1e6:.1f}M params)")
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


def sieve_masked_loss_hf(logits: Tensor, labels: Tensor, mask: Tensor,
                         scale: bool = True) -> Tensor:
    """SIEVE masked loss for native HF models.

    Takes logits directly from model output (no hook, no softcap).
    Shift is done here (predict next token).
    """
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()

    flat_logits = shift_logits.view(-1, shift_logits.size(-1))
    flat_labels = shift_labels.view(-1)

    selected_logits = flat_logits[mask]
    selected_labels = flat_labels[mask]

    loss = F.cross_entropy(selected_logits, selected_labels, reduction="sum")

    if scale:
        num_selected = mask.sum().clamp(min=1).float()
        total = float(flat_labels.size(0))
        loss = loss * (total / num_selected)

    loss = loss / float(flat_labels.size(0))
    return loss


def main():
    parser = argparse.ArgumentParser(description="Train with HF Trainer (ramenGPT or native HF model)")
    parser.add_argument("--config", type=str, default=None,
                        help="Path to ramenGPT config .py. If omitted, uses --model_name_or_path as a native HF model")
    parser.add_argument("--dataset", type=str, default="wikitext", help="HF dataset name (default: wikitext)")
    parser.add_argument("--dataset_config", type=str, default="wikitext-2-raw-v1", help="HF dataset config")
    parser.add_argument("--text_column", type=str, default="text", help="Text column name in dataset")
    parser.add_argument("--model_name_or_path", type=str, default="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
                        help="HF model/tokenizer name (used as model when --config is omitted)")
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
    parser.add_argument("--save_total_limit", type=int, default=2, help="Max checkpoints to keep")
    parser.add_argument(
        "--lr_scheduler_type",
        type=str,
        default="cosine",
        choices=[s.value for s in SchedulerType],
        help="LR scheduler type",
    )
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--gradient_checkpointing", action="store_true", help="Enable gradient checkpointing (if supported)")

    # SIEVE token selection
    parser.add_argument("--sieve_method", type=str, default="full",
                        choices=["full", "random", "sieve"],
                        help="Token selection method: full (all tokens), random (baseline), sieve (adaptive multi-strategy)")
    parser.add_argument("--sieve_ratio", type=float, default=0.7, help="Fraction of tokens to keep when using selection")
    parser.add_argument("--sieve_rescore_every", type=int, default=50, help="SIEVE: re-score tokens every N steps")

    args = parser.parse_args()

    device = _detect_device()
    print(f"Device: {device}")

    # --- Model loading ---
    use_ramen = args.config is not None
    if use_ramen:
        config_module = _load_config_module(args.config)
        model = build_model_from_config(config_module, max_seq_len=args.block_size, device=device)
        print("Backend: ramenGPT")
    else:
        model = build_model_hf_native(args.model_name_or_path, bf16=args.bf16, fp16=args.fp16)
        print("Backend: HF native")

    # Build dataset + tokenizer
    from datasets import load_dataset
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    if tokenizer.pad_token is None:
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
        save_total_limit=args.save_total_limit,
        eval_strategy=("steps" if args.eval_steps and args.eval_steps > 0 else "no"),
        eval_steps=(args.eval_steps if args.eval_steps and args.eval_steps > 0 else None),
        deepspeed=args.deepspeed,
        fp16=args.fp16,
        bf16=args.bf16,
        lr_scheduler_type=args.lr_scheduler_type,
        max_grad_norm=args.max_grad_norm,
        gradient_checkpointing=args.gradient_checkpointing,
        dataloader_drop_last=True,
        dataloader_pin_memory=False,  # Disable for MPS (Apple Silicon)
        report_to=["none"],
        remove_unused_columns=False,  # we use input_ids in compute_loss; model has custom forward
    )

    selector = None
    sieve_selector = None  # real SIEVE selector (separate from simple RandomSelector)
    if args.sieve_method == "random":
        selector = RandomSelector(ratio=args.sieve_ratio, device=device)
        print(f"SIEVE enabled: method=random, ratio={args.sieve_ratio}")
    elif args.sieve_method == "sieve":
        from sieve_config import SieveConfig
        from sieve_models import SieveSelector
        sieve_cfg = SieveConfig(
            select_ratio=args.sieve_ratio,
            rescore_every=args.sieve_rescore_every,
        )
        sieve_selector = SieveSelector(sieve_cfg, device=str(device))
        print(f"SIEVE enabled: method=sieve (adaptive), ratio={args.sieve_ratio}, rescore_every={args.sieve_rescore_every}")

    # --- Trainer selection ---
    if use_ramen:
        window_blocks = _get_window_blocks(args.window_size_blocks, device=device)

        class RamenTrainer(Trainer):
            def compute_loss(self, model, inputs, return_outputs=False, *args, **kwargs):  # type: ignore[override]
                # inputs["input_ids"]: shape [B, T]; *args/**kwargs for Trainer API (e.g. num_items_in_batch)
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

            def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix: str = "eval"):
                metrics = super().evaluate(eval_dataset=eval_dataset, ignore_keys=ignore_keys, metric_key_prefix=metric_key_prefix)
                key = f"{metric_key_prefix}_loss"
                if key in metrics and math.isfinite(metrics[key]):
                    try:
                        metrics[f"{metric_key_prefix}_perplexity"] = float(math.exp(metrics[key]))
                    except OverflowError:
                        metrics[f"{metric_key_prefix}_perplexity"] = float("inf")
                return metrics

        TrainerClass = RamenTrainer
    else:
        _hf_step = [0]  # mutable counter for SIEVE step tracking

        class HFNativeTrainer(Trainer):
            def compute_loss(self, model, inputs, return_outputs=False, *extra_args, **kwargs):
                input_ids = inputs["input_ids"].to(device)
                labels = input_ids.clone()

                # Shift targets for scoring (predict next token)
                shift_labels = labels[..., 1:].contiguous().view(-1)
                seq_len = shift_labels.size(0)

                if sieve_selector is not None:
                    # Real SIEVE: adaptive multi-strategy selection
                    # Get logits via forward pass
                    outputs = model(input_ids=input_ids)
                    logits = outputs.logits
                    shift_logits = logits[..., :-1, :].contiguous()
                    flat_logits = shift_logits.view(-1, shift_logits.size(-1))

                    # SIEVE scores tokens and returns a mask
                    # We pass the shifted logits/targets directly
                    mask = sieve_selector.get_token_mask(
                        step=_hf_step[0],
                        total_steps=args.max_steps,
                        model=model,
                        inputs=input_ids.view(-1)[:-1],  # src tokens
                        targets=shift_labels,
                        sliding_window_num_blocks=None,  # HF native, no sliding window
                        train_loss=getattr(self, '_last_loss', 10.0),
                    )
                    _hf_step[0] += 1

                    selected_logits = flat_logits[mask]
                    selected_labels = shift_labels[mask]
                    loss = F.cross_entropy(selected_logits, selected_labels, reduction="sum")
                    num_selected = mask.sum().clamp(min=1).float()
                    loss = loss * (float(seq_len) / num_selected) / float(seq_len)
                    self._last_loss = loss.detach().item()

                elif selector is not None:
                    # Random baseline
                    outputs = model(input_ids=input_ids)
                    logits = outputs.logits
                    mask = selector.get_token_mask(seq_len)
                    loss = sieve_masked_loss_hf(logits, labels, mask)

                else:
                    # Full training
                    outputs = model(input_ids=input_ids)
                    logits = outputs.logits
                    shift_logits = logits[..., :-1, :].contiguous()
                    loss = F.cross_entropy(
                        shift_logits.view(-1, shift_logits.size(-1)),
                        shift_labels,
                    )

                return (loss, outputs if 'outputs' in dir() else None) if return_outputs else loss

            def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
                metrics = super().evaluate(eval_dataset=eval_dataset, ignore_keys=ignore_keys, metric_key_prefix=metric_key_prefix)
                key = f"{metric_key_prefix}_loss"
                if key in metrics and math.isfinite(metrics[key]):
                    try:
                        metrics[f"{metric_key_prefix}_perplexity"] = float(math.exp(metrics[key]))
                    except OverflowError:
                        metrics[f"{metric_key_prefix}_perplexity"] = float("inf")
                return metrics

        TrainerClass = HFNativeTrainer

    trainer = TrainerClass(
        model=model,
        args=training_args,
        train_dataset=lm_ds["train"],
        eval_dataset=(lm_ds.get("validation") if args.eval_steps and args.eval_steps > 0 else None),
        tokenizer=tokenizer,
    )

    trainer.train()


if __name__ == "__main__":
    main()
