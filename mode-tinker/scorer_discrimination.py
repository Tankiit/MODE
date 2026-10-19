"""Paired continuations from one Tinker model AND optimizer checkpoint.

No bandit. Seeds vary data order and random selection, not the starting model.
The held-out validation suffix is never passed to a selector or controller.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from sieve.tinker_backend import Selector, make_data, top_fraction, train_step
from train_sieve_tinker import atomic_json, data_digest, load_data

ARMS = ("random", "nll", "rarity", "clm")


def overlap(a, b):
    """Directional overlap with exact independent fixed-size mask baseline.

    Chance correction is (observed - chance) / (1 - chance). Jaccard's
    baseline below is the ratio of expected counts, not exact E[Jaccard].
    """
    if a.shape != b.shape or a.ndim != 2:
        raise ValueError("Masks must have identical [batch, targets] shapes")
    a, b = a.bool(), b.bool()
    na, nb = a.sum(1).double(), b.sum(1).double()
    if not bool((na > 0).all() and (nb > 0).all()):
        raise ValueError("Overlap requires nonempty masks")
    shared = float((a & b).sum())
    expected_shared = float((na * nb / a.shape[1]).sum())
    observed = shared / float(na.sum())
    chance = expected_shared / float(na.sum())
    return {
        "overlap_fraction_of_first": observed,
        "random_overlap_baseline": chance,
        "chance_corrected_overlap": ((observed-chance)/(1-chance)
                                     if chance < 1 else None),
        "jaccard": shared / float((a | b).sum()),
        "random_jaccard_ratio_of_expectations": expected_shared /
            (float(na.sum()+nb.sum())-expected_shared),
    }


def evaluation_batches(val, batch_size, skip, count):
    batches = [b["input_ids"] for b in itertools.islice(
        val.val_batches(batch_size, torch.device("cpu")), skip, skip+count)]
    if len(batches) != count:
        raise ValueError("Not enough disjoint validation batches")
    return batches


def evaluate_batches(client, batches):
    total, count = 0.0, 0
    for ids in batches:
        result = client.forward(make_data(ids), "cross_entropy").result()
        for out in result.loss_fn_outputs:
            lp = out["logprobs"].to_torch().double()
            total -= float(lp.sum())
            count += lp.numel()
    value = total/count
    if not math.isfinite(value):
        raise ValueError("Non-finite independent evaluation loss")
    return value


def common_checkpoint_masks(client, batches, frequency, alpha, seed):
    """Compare scorer choices on identical model states, contexts and targets."""
    random = Selector("random", alpha, seed)
    masks = {arm: [] for arm in ARMS}
    for ids in batches:
        result = client.forward(make_data(ids), "cross_entropy").result()
        nll = -torch.stack([x["logprobs"].to_torch() for x in result.loss_fn_outputs])
        targets = ids[:, 1:]
        masks["nll"].append(top_fraction(nll, alpha))
        masks["rarity"].append(top_fraction(frequency[targets], alpha))
        masks["random"].append(random.static_mask(targets))
        masks["clm"].append(torch.ones_like(targets, dtype=torch.bool))
    masks = {k: torch.cat(v) for k, v in masks.items()}
    return {f"{a}__{b}": overlap(masks[a], masks[b])
            for a, b in itertools.combinations(ARMS, 2)}


def continuation(client, batches, heldout, frequency, arm, alpha, seed, params,
                 log=lambda record: None):
    initial = evaluate_batches(client, heldout)
    selector = Selector(arm, alpha, seed)
    started = time.perf_counter()
    for step, ids in enumerate(batches, 1):
        metrics = train_step(client, ids, selector, frequency, params)
        if any(not math.isfinite(v) for v in metrics.values()):
            raise ValueError("Non-finite training metric")
        log({"event": "train", "step": step, **metrics})
    final = evaluate_batches(client, heldout)
    return {"initial_nll": initial, "final_nll": final,
            "improvement": initial-final, "final_ppl": math.exp(final),
            "seconds": time.perf_counter()-started}


def summarize(records):
    result = {}
    for arm in ARMS:
        rows = [r for r in records if r["arm"] == arm]
        if not rows:
            continue
        random = {r["seed"]: r for r in records if r["arm"] == "random"}
        paired = [r["improvement"]-random[r["seed"]]["improvement"]
                  for r in rows if r["seed"] in random]
        result[arm] = {
            "seeds": [r["seed"] for r in rows],
            "mean_improvement": float(np.mean([r["improvement"] for r in rows])),
            "paired_improvement_over_random": paired,
            "paired_mean": float(np.mean(paired)) if paired else None,
            "paired_sample_sd": float(np.std(paired, ddof=1)) if len(paired)>1 else None,
        }
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True, help="Local Tinker checkpoint JSON")
    p.add_argument("--data-dir", required=True)
    p.add_argument("--run-dir", required=True)
    p.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    p.add_argument("--steps", type=int, default=10)
    p.add_argument("--alpha", type=float, default=0.7)
    p.add_argument("--eval-batches", type=int, default=8)
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()
    if args.steps <= 0 or args.eval_batches <= 0 or len(set(args.seeds)) != len(args.seeds):
        p.error("Use positive counts and unique seeds")
    if not 0 < args.alpha <= 1:
        p.error("alpha must be in (0,1]")
    state = json.loads(Path(args.checkpoint).read_text())
    cfg = dict(state["config"])
    cfg.update(data_dir=args.data_dir, mode="rarity", alpha=args.alpha)
    train, val, frequency = load_data(SimpleNamespace(**cfg))
    if state["data_sha256"] != data_digest(args.data_dir):
        raise ValueError("Checkpoint data hashes do not match")
    # Keep the original batch geometry, and exclude every evaluation-prefix token.
    heldout = evaluation_batches(val, cfg["batch_size"], cfg["eval_batches"], args.eval_batches)
    plans = {}
    for seed in args.seeds:
        train.rng = np.random.default_rng(seed)
        plans[seed] = [train.sample_batch(cfg["batch_size"], torch.device("cpu"))["input_ids"]
                       for _ in range(args.steps)]
    manifest = {"arguments": vars(args), "source_state": state["state_path"],
        "source_completed_steps": state["completed_steps"], "config": cfg,
        "data_sha256": state["data_sha256"], "arms": ARMS,
        "eval_token_start": cfg["eval_batches"]*cfg["batch_size"]*cfg["seq_len"],
        "eval_token_end": cfg["eval_batches"]*cfg["batch_size"]*cfg["seq_len"]+
            sum(x.numel() for x in heldout),
        "batch_sha256": {str(seed): hashlib.sha256(b"".join(x.numpy().tobytes() for x in batches)).hexdigest()
                         for seed, batches in plans.items()},
        "seed_scope": "paired data order and random masks; identical model and optimizer",
        "mask_scope": "common starting checkpoint on every planned training batch",
    }
    if args.dry_run:
        print(json.dumps({"status": "local checks passed; no service calls", **manifest}, indent=2))
        return
    directory = Path(args.run_dir)
    if directory.exists() and any(directory.iterdir()):
        raise ValueError("Use a new empty experiment directory")
    import tinker
    service = tinker.ServiceClient()
    directory.mkdir(parents=True, exist_ok=True)
    atomic_json(directory/"manifest.json", manifest)
    params = tinker.AdamParams(learning_rate=cfg["lr"], beta1=0.9, beta2=0.95,
        eps=1e-8, weight_decay=cfg["weight_decay"], grad_clip_norm=cfg["grad_clip"])
    records = []
    initial_nll = None
    for seed, batches in plans.items():
        for arm in ARMS:
            client = service.create_training_client_from_state_with_optimizer(state["state_path"])
            from sieve.data import tokenizer_fingerprint
            if tokenizer_fingerprint(client.get_tokenizer()) != train.meta["tokenizer_vocab_sha256"]:
                raise ValueError("Remote tokenizer differs from prepared data")
            if arm == "random":
                atomic_json(directory/f"overlap-seed{seed}.json",
                    common_checkpoint_masks(client, batches, frequency, args.alpha, seed))
            def log(record):
                record = {"seed": seed, "arm": arm, **record}
                with (directory/"metrics.jsonl").open("a") as f:
                    f.write(json.dumps(record, allow_nan=False)+"\n")
                print(json.dumps(record), flush=True)
            result = continuation(client, batches, heldout, frequency, arm, args.alpha, seed, params, log)
            if initial_nll is None:
                initial_nll = result["initial_nll"]
            elif not math.isclose(initial_nll, result["initial_nll"], abs_tol=1e-5, rel_tol=0):
                raise ValueError("Restored arms do not share the same initial evaluation loss")
            name = f"{arm}-seed{seed}"
            result.update(seed=seed, arm=arm,
                state_path=client.save_state(name).result().path,
                sampler_path=client.save_weights_for_sampler(name).result().path)
            records.append(result)
            log({"event": "result", **result})
            atomic_json(directory/"results.json", records)
            atomic_json(directory/"summary.json", summarize(records))
    print(f"Finished paired discrimination experiment: {directory}", flush=True)


if __name__ == "__main__":
    main()
