"""LoRA continued language-model training through Tinker.

Run --dry-run first. API calls occur only in main(), after local validation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

from sieve.config import SieveConfig
from sieve.data import MemmapDataset, tokenizer_fingerprint
from sieve.tinker_backend import MODES, Selector, evaluate, make_data, train_step


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base-model", default="Qwen/Qwen3-8B")
    p.add_argument("--data-dir", required=True)
    p.add_argument("--run-dir", default="runs/tinker-pilot")
    p.add_argument("--mode", choices=MODES, default="sieve_nd")
    p.add_argument("--rank", type=int, default=32)
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--seq-len", type=int, default=512)
    p.add_argument("--alpha", type=float, default=0.7)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--eval-interval", type=int, default=50)
    p.add_argument("--eval-batches", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--data-fraction", type=float, default=1.0)
    p.add_argument("--train-unembed", action="store_true")
    p.add_argument("--resume", help="Local checkpoint JSON; use a NEW run directory")
    p.add_argument("--dry-run", action="store_true", help="Local data/schema checks only; no service calls")
    return p


def atomic_json(path: Path, value):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def signature(args):
    # A resume preserves the original total-step schedule and validation set.
    excluded = {"run_dir", "resume", "dry_run", "data_dir"}
    return {k: v for k, v in vars(args).items() if k not in excluded}


def data_digest(data_dir: str):
    result = {}
    for name in ("meta.json", "train.bin", "val.bin", "freq.pt"):
        path = Path(data_dir) / name
        if not path.exists():
            continue
        h = hashlib.sha256()
        with path.open("rb") as f:
            for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
                h.update(chunk)
        result[name] = h.hexdigest()
    return result


def load_data(args):
    positive = ("rank", "steps", "batch_size", "seq_len", "eval_interval", "eval_batches")
    if any(getattr(args, k) <= 0 for k in positive) or args.seq_len < 2:
        raise ValueError("Counts must be positive and seq_len >= 2")
    if not 0 < args.alpha <= 1 or not 0 < args.data_fraction <= 1:
        raise ValueError("alpha and data_fraction must be in (0, 1]")
    if not math.isfinite(args.lr) or args.lr <= 0:
        raise ValueError("lr must be finite and positive")
    if any(not math.isfinite(x) or x < 0 for x in (args.weight_decay, args.grad_clip)):
        raise ValueError("weight_decay and grad_clip must be finite and nonnegative")
    cfg = SieveConfig(data_dir=args.data_dir, seq_len=args.seq_len,
                      data_fraction=args.data_fraction, seed=args.seed)
    train = MemmapDataset(args.data_dir, "train", cfg)
    val = MemmapDataset(args.data_dir, "val", cfg)
    if train.meta.get("tokenizer") != args.base_model:
        raise ValueError("Retokenize the dataset with --tokenizer matching --base-model")
    if "tokenizer_vocab_sha256" not in train.meta:
        raise ValueError("Reprepare data to record the tokenizer vocabulary fingerprint")
    vocab_size = int(train.meta["vocab_size"])
    if vocab_size > np.iinfo(train.arr.dtype).max + 1:
        raise ValueError("Token dtype cannot represent this vocabulary; reprepare as uint32")
    for ds in (train, val):
        if int(ds.arr[:ds.N].max()) >= vocab_size:
            raise ValueError("Dataset contains token IDs outside the recorded vocabulary")
    frequency = train.freq_table
    if frequency is None:
        if args.mode in ("sieve_nd", "rarity"):
            raise ValueError(f"{args.mode} requires freq.pt from the training corpus")
        frequency = torch.zeros(vocab_size)
    if frequency.ndim != 1 or frequency.numel() != vocab_size or not torch.isfinite(frequency).all():
        raise ValueError("Invalid frequency table")
    return train, val, frequency.float()


def checkpoint(client, directory, completed, args, train, selector, digest, counters):
    name = f"step-{completed:07d}"
    # Finish both remote writes before publishing a resumable local record.
    state_path = client.save_state(name).result().path
    sampler_path = client.save_weights_for_sampler(name).result().path
    payload = {
        "format": 1, "completed_steps": completed,
        "state_path": state_path, "sampler_path": sampler_path,
        "config": signature(args), "data_sha256": digest,
        "dataset_rng": train.rng.bit_generator.state,
        "selector": selector.state_dict(), "counters": counters.copy(),
    }
    path = directory / f"{name}.json"
    atomic_json(path, payload)
    atomic_json(directory / "latest.json", payload)
    return path


def run_training(args, client, train, val, frequency, digest, resume=None):
    """Separate from client construction so the full loop can be tested offline."""
    import tinker
    directory = Path(args.run_dir)
    if directory.exists() and any(directory.iterdir()):
        raise ValueError("Use an empty run directory (including when resuming)")
    directory.mkdir(parents=True, exist_ok=True)
    atomic_json(directory / "config.json", {
        "config": vars(args), "data_sha256": digest, "tinker_version": tinker.__version__,
        "objective": "sum(selected NLL) / all eligible targets",
        "scorers": ["current_nll", "corpus_rarity"] if args.mode == "sieve_nd" else [args.mode],
    })
    selector = Selector(args.mode, args.alpha, args.seed)
    counters = {"train_input_tokens": 0, "selected_targets": 0,
                "extra_scoring_forward_tokens": 0, "validation_input_tokens": 0}
    completed = 0
    if resume:
        completed = resume["completed_steps"]
        train.rng.bit_generator.state = resume["dataset_rng"]
        selector.load_state_dict(resume["selector"])
        counters = resume["counters"].copy()
        if selector.active:
            raise ValueError("Expected a checkpoint at an evaluation boundary")

    def log(record):
        with (directory / "metrics.jsonl").open("a") as f:
            f.write(json.dumps(record, allow_nan=False) + "\n")
        print(json.dumps(record, allow_nan=False), flush=True)

    def assess(step):
        nll, n_tokens = evaluate(client, val, args.batch_size, args.eval_batches)
        if not math.isfinite(nll):
            raise ValueError("Non-finite validation NLL")
        counters["validation_input_tokens"] += n_tokens
        reward = selector.observe(nll)
        ppl = math.exp(nll) if nll < 700 else None
        log({"event": "eval", "step": step, "val_nll": nll, "val_ppl": ppl,
             "reward": reward, "credited_weights": selector.weights.tolist(),
             "posterior_weights": selector.bandit.expected_weights(selector.bandit._current_bin).tolist(),
             **counters})

    params = tinker.AdamParams(learning_rate=args.lr, beta1=0.9, beta2=0.95,
                               eps=1e-8, weight_decay=args.weight_decay,
                               grad_clip_norm=args.grad_clip)
    if resume is None:
        assess(0)
        checkpoint(client, directory, 0, args, train, selector, digest, counters)
    for step in range(completed, args.steps):
        if not selector.active:
            selector.begin_interval(step, args.steps)
        ids = train.sample_batch(args.batch_size, torch.device("cpu"))["input_ids"]
        started = time.perf_counter()
        metrics = train_step(client, ids, selector, frequency, params)
        elapsed = time.perf_counter() - started
        n_targets = ids[:, 1:].numel()
        counters["train_input_tokens"] += n_targets
        counters["selected_targets"] += round(metrics["selected_fraction"] * n_targets)
        counters["extra_scoring_forward_tokens"] += metrics["extra_scoring_forward"] * n_targets
        completed = step + 1
        log({"event": "train", "step": completed, "step_seconds": elapsed,
             "sampled_weights": selector.weights.tolist(), **metrics})
        if completed % args.eval_interval == 0 or completed == args.steps:
            assess(completed)
            checkpoint(client, directory, completed, args, train, selector, digest, counters)
    return directory / "latest.json"


def validate_resume(args, digest):
    if not args.resume:
        return None
    state = json.loads(Path(args.resume).read_text())
    if state.get("format") != 1 or state["config"] != signature(args):
        raise ValueError("Resume configuration differs; preserve original flags and total steps")
    if state["data_sha256"] != digest:
        raise ValueError("Resume dataset differs from the checkpoint")
    if not 0 <= state["completed_steps"] < args.steps:
        raise ValueError("Checkpoint is already complete or has an invalid step")
    return state


def main():
    args = parser().parse_args()
    train, val, frequency = load_data(args)
    digest = data_digest(args.data_dir)
    resume = validate_resume(args, digest)
    if args.dry_run:
        # Exercise the installed SDK's schema without creating a service client.
        probe = train.sample_batch(args.batch_size, torch.device("cpu"))["input_ids"]
        datums = make_data(probe)
        print(json.dumps({"status": "local checks passed; no API calls",
                          "mode": args.mode, "datum_count": len(datums),
                          "tokens_per_training_step": args.batch_size * (args.seq_len - 1),
                          "extra_scoring_forward": args.mode in ("nll", "sieve_nd"),
                          "dtype": str(train.arr.dtype), "data_sha256": digest}, indent=2))
        return
    directory = Path(args.run_dir)
    if directory.exists() and any(directory.iterdir()):
        raise ValueError("Use an empty run directory")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    if tokenizer_fingerprint(tokenizer) != train.meta["tokenizer_vocab_sha256"]:
        raise ValueError("Tokenizer vocabulary differs from the prepared dataset")
    import tinker
    service = tinker.ServiceClient()
    capabilities = service.get_server_capabilities()
    supported = {m.model_name for m in capabilities.supported_models
                 if m.model_name is not None and m.trainable is not False}
    if args.base_model not in supported:
        raise ValueError(f"Model unavailable to this account: {args.base_model}; available: {sorted(supported)}")
    info = next(m for m in capabilities.supported_models if m.model_name == args.base_model)
    if info.max_context_length and args.seq_len - 1 > info.max_context_length:
        raise ValueError("seq_len exceeds this model's context limit")
    if resume:
        client = service.create_training_client_from_state_with_optimizer(resume["state_path"])
    else:
        client = service.create_lora_training_client(
            base_model=args.base_model, rank=args.rank, seed=args.seed,
            train_mlp=True, train_attn=True, train_unembed=args.train_unembed,
        )
    if tokenizer_fingerprint(client.get_tokenizer()) != train.meta["tokenizer_vocab_sha256"]:
        raise ValueError("Tinker tokenizer differs from the prepared dataset")
    path = run_training(args, client, train, val, frequency, digest, resume)
    print(f"Finished. Resume/export metadata: {path}")


if __name__ == "__main__":
    main()
