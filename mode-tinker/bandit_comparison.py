"""Paired bandit comparisons with validation-selected constant mixtures.

Tune one global fixed NLL weight on development loss, lock it, then evaluate
predeclared comparators on a disjoint reporting slice. No remote calls in dry run.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from scorer_discrimination import evaluate_batches, evaluation_batches
from sieve.data import tokenizer_fingerprint
from sieve.tinker_backend import Selector, train_step
from train_sieve_tinker import atomic_json, data_digest, load_data


def schedule_bins(steps, interval):
    if steps <= 0 or interval <= 0:
        raise ValueError("steps and eval_interval must be positive")
    bandit = Selector("sieve_nd", 0.7, 42).bandit
    bins = [bandit.encode_context(s, steps, 0., 0.) for s in range(0, steps, interval)]
    if any(n < 2 for n in Counter(bins).values()):
        raise ValueError("Each visited phase bin needs at least two decisions; reduce eval_interval")
    return bins


def select_fixed(records, grid, seeds):
    """Only development NLL may enter selection; choose one weight across seeds."""
    means = {}
    for weight in sorted(grid):
        rows = [r for r in records if r.get("fixed_nll_weight") == weight
                and r.get("kind") == "fixed"]
        if sorted(r["seed"] for r in rows) != sorted(seeds):
            raise ValueError("Fixed-mixture selection needs exactly one result per weight and seed")
        values = [r["final_dev_nll"] for r in rows]
        if not all(math.isfinite(v) for v in values):
            raise ValueError("Non-finite tuning loss")
        means[weight] = float(np.mean(values))
    # Deterministic tie break: lowest NLL weight. No reporting loss inspected.
    winner = min(means, key=lambda w: (means[w], w))
    return {"fixed_nll_weight": winner, "mean_dev_nll_by_weight": means,
            "selection_metric": "mean final development NLL across paired seeds",
            "tie_break": "smallest NLL weight"}


def run_arm(client, batches, dev, frequency, alpha, seed, params, interval,
            kind, fixed_nll_weight=None, expected_initial=None, log=lambda r: None):
    initial = evaluate_batches(client, dev)
    if expected_initial is not None and not math.isclose(initial, expected_initial, abs_tol=1e-5, rel_tol=0):
        raise ValueError("Restored arms do not share the same initial development loss")
    if kind == "fixed":
        if fixed_nll_weight is None or not 0 <= fixed_nll_weight <= 1:
            raise ValueError("Fixed mixture requires an NLL weight in [0,1]")
        mode = "rarity" if fixed_nll_weight == 0 else "nll" if fixed_nll_weight == 1 else "sieve_nd"
    elif kind in ("bandit", "random"):
        mode = "sieve_nd" if kind == "bandit" else "random"
    else:
        raise ValueError("Unknown comparison arm")
    selector = Selector(mode, alpha, seed)
    if kind == "fixed":
        selector.weights = torch.tensor([fixed_nll_weight, 1-fixed_nll_weight])
    if kind == "bandit":
        schedule_bins(len(batches), interval)
        selector.observe(initial)
    counters = {"train_input_tokens": 0, "extra_scoring_forward_tokens": 0,
                "dev_evaluation_input_tokens": sum(x[:, 1:].numel() for x in dev)}
    final = initial
    for step, ids in enumerate(batches):
        if kind == "bandit" and not selector.active:
            selector.begin_interval(step, len(batches))
        metrics = train_step(client, ids, selector, frequency, params)
        if any(not math.isfinite(v) for v in metrics.values()):
            raise ValueError("Non-finite training metric")
        n = ids[:, 1:].numel()
        counters["train_input_tokens"] += n
        counters["extra_scoring_forward_tokens"] += n*metrics["extra_scoring_forward"]
        log({"event": "train", "step": step+1,
             "weights": selector.weights.tolist() if mode == "sieve_nd" else None, **metrics})
        if (step+1) % interval == 0 or step+1 == len(batches):
            final = evaluate_batches(client, dev)
            counters["dev_evaluation_input_tokens"] += sum(x[:, 1:].numel() for x in dev)
            reward = selector.observe(final) if kind == "bandit" else None
            log({"event": "dev_eval", "step": step+1, "dev_nll": final,
                 "reward": reward, "credited_weights": selector.weights.tolist() if kind == "bandit" else None,
                 "context_bin": selector.bandit._current_bin if kind == "bandit" else None})
    return {"initial_dev_nll": initial, "final_dev_nll": final, **counters,
            "selector": selector.state_dict() if kind == "bandit" else None}


def paired_summary(rows):
    adaptive = {r["seed"]: r for r in rows if r["arm"] == "bandit"}
    result = {}
    for arm in ("nll", "rarity", "fixed_50_50", "best_fixed", "random"):
        baselines = {r["seed"]: r for r in rows if r["arm"] == arm}
        seeds = sorted(adaptive.keys() & baselines.keys())
        values = [baselines[s]["report_nll"]-adaptive[s]["report_nll"] for s in seeds]
        result[arm] = {"seeds": seeds, "bandit_nll_gain": values,
            "mean_gain": float(np.mean(values)) if values else None,
            "sample_sd": float(np.std(values, ddof=1)) if len(values)>1 else None}
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--data-dir", required=True)
    p.add_argument("--run-dir", required=True)
    p.add_argument("--seeds", type=int, nargs="+", default=[42,43,44])
    p.add_argument("--steps", type=int, default=100)
    p.add_argument("--eval-interval", type=int, default=5)
    p.add_argument("--alpha", type=float, default=0.7)
    p.add_argument("--fixed-grid", type=float, nargs="+", default=[0.,0.25,0.5,0.75,1.])
    p.add_argument("--skip-val-batches", type=int, default=16,
                   help="Exclude original controller prefix AND previous diagnostic slice")
    p.add_argument("--dev-batches", type=int, default=8)
    p.add_argument("--report-batches", type=int, default=8)
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()
    bins = schedule_bins(args.steps, args.eval_interval)
    grid = sorted(args.fixed_grid)
    if (len(set(grid)) != len(grid) or not {0.,0.5,1.}.issubset(grid)
            or any(not 0 <= w <= 1 for w in grid)):
        p.error("Use unique fixed weights in [0,1], including 0, 0.5, and 1")
    if len(set(args.seeds)) != len(args.seeds) or min(args.dev_batches,args.report_batches) <= 0:
        p.error("Use unique seeds and positive evaluation batch counts")
    source = json.loads(Path(args.checkpoint).read_text())
    cfg = dict(source["config"])
    cfg.update(data_dir=args.data_dir, mode="rarity", alpha=args.alpha)
    train, val, frequency = load_data(SimpleNamespace(**cfg))
    if source["data_sha256"] != data_digest(args.data_dir):
        raise ValueError("Checkpoint dataset hashes differ")
    if args.skip_val_batches < cfg["eval_batches"]:
        p.error("The development slice must exclude the original controller prefix")
    dev = evaluation_batches(val,cfg["batch_size"],args.skip_val_batches,args.dev_batches)
    report = evaluation_batches(val,cfg["batch_size"],args.skip_val_batches+args.dev_batches,args.report_batches)
    plans = {}
    for seed in args.seeds:
        train.rng = np.random.default_rng(seed)
        plans[seed] = [train.sample_batch(cfg["batch_size"],torch.device("cpu"))["input_ids"] for _ in range(args.steps)]
    batch_tokens = cfg["batch_size"]*cfg["seq_len"]
    manifest = {"arguments": vars(args), "source_state": source["state_path"],
        "source_config": source["config"], "data_sha256": source["data_sha256"],
        "context_bins": bins, "visits_per_bin": dict(Counter(bins)),
        "development_token_range": [args.skip_val_batches*batch_tokens,(args.skip_val_batches+args.dev_batches)*batch_tokens],
        "reporting_token_range": [(args.skip_val_batches+args.dev_batches)*batch_tokens,(args.skip_val_batches+args.dev_batches+args.report_batches)*batch_tokens],
        "continuations": (len(grid)+2)*len(args.seeds),
        "total_optimizer_steps": (len(grid)+2)*len(args.seeds)*args.steps,
        "batch_sha256": {str(seed): hashlib.sha256(b"".join(x.numpy().tobytes() for x in batches)).hexdigest() for seed,batches in plans.items()},
        "seed_scope": "paired data order and selector RNG, identical starting model/optimizer",
        "selection_rule": "one fixed weight minimizing mean final development NLL; locked before report evaluation"}
    if args.dry_run:
        print(json.dumps({"status":"local checks passed; no API calls",**manifest},indent=2))
        return
    directory = Path(args.run_dir)
    if directory.exists() and any(directory.iterdir()):
        raise ValueError("Use an empty experiment directory")
    import tinker
    service = tinker.ServiceClient()
    directory.mkdir(parents=True,exist_ok=True)
    atomic_json(directory/"manifest.json",manifest)
    def restore(path):
        client = service.create_training_client_from_state_with_optimizer(path)
        if tokenizer_fingerprint(client.get_tokenizer()) != train.meta["tokenizer_vocab_sha256"]:
            raise ValueError("Remote tokenizer differs from prepared data")
        return client
    params = tinker.AdamParams(learning_rate=cfg["lr"], beta1=0.9,beta2=0.95,
        eps=1e-8,weight_decay=cfg["weight_decay"],grad_clip_norm=cfg["grad_clip"])
    records, initial = [], None
    # Complete tuning and adaptive training without ever evaluating reporting data.
    for kind, weight in [("fixed",w) for w in grid]+[("bandit",None),("random",None)]:
        for seed,batches in plans.items():
            name = f"fixed-{weight:g}-seed{seed}" if kind == "fixed" else f"{kind}-seed{seed}"
            def log(record):
                record = {"run":name,"seed":seed,**record}
                with (directory/"metrics.jsonl").open("a") as f:
                    f.write(json.dumps(record,allow_nan=False)+"\n")
                print(json.dumps(record),flush=True)
            client = restore(source["state_path"])
            result = run_arm(client,batches,dev,frequency,args.alpha,seed,params,
                             args.eval_interval,kind,weight,initial,log)
            initial = result["initial_dev_nll"] if initial is None else initial
            result.update(kind=kind,fixed_nll_weight=weight,seed=seed,
                          state_path=client.save_state(name).result().path)
            records.append(result)
            atomic_json(directory/"development_results.json",records)
    selected = select_fixed(records,grid,args.seeds)
    atomic_json(directory/"selected_fixed.json",selected)
    initial_report = evaluate_batches(restore(source["state_path"]),report)
    report_rows, cache = [], {}
    # best_fixed can alias an endpoint or 50/50: do not retrain or count it twice.
    comparisons = [("bandit","bandit",None),("nll","fixed",1.),
        ("rarity","fixed",0.),("fixed_50_50","fixed",0.5),
        ("best_fixed","fixed",selected["fixed_nll_weight"]),("random","random",None)]
    for arm,kind,weight in comparisons:
        for seed in args.seeds:
            row = next(r for r in records if r["kind"]==kind and r["fixed_nll_weight"]==weight and r["seed"]==seed)
            path = row["state_path"]
            if path not in cache:
                cache[path] = evaluate_batches(restore(path),report)
            report_rows.append({"arm":arm,"seed":seed,"fixed_nll_weight":weight,
                "initial_report_nll":initial_report,"report_nll":cache[path],
                "report_ppl":math.exp(cache[path]),"improvement":initial_report-cache[path],
                "state_path":path})
            atomic_json(directory/"report_results.json",report_rows)
            atomic_json(directory/"paired_comparisons.json",paired_summary(report_rows))
    print(f"Finished comparison matrix: {directory}",flush=True)


if __name__ == "__main__":
    main()
