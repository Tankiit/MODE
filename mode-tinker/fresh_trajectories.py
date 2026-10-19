"""Fresh, paired LoRA adaptation trajectories; test data is endpoint-only.

One untrained adapter snapshot per seed ensures exact paired initialization.
Each method loads weights only (fresh optimizer); crash recovery alone restores
optimizer, data RNG, and selector state. Resume requires the same manifest.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import math
from pathlib import Path
import time
import traceback
import uuid

import numpy as np
import torch

from bandit_comparison import schedule_bins, select_fixed
from scorer_discrimination import evaluate_batches, evaluation_batches
from sieve.config import SieveConfig
from sieve.data import MemmapDataset, tokenizer_fingerprint
from sieve.tinker_backend import Selector, make_data, train_step
from train_sieve_tinker import atomic_json, data_digest, load_data

METHODS = ("nll", "random", "clm", "rarity", "fixed_50", "bandit", "fixed_25", "fixed_75")
FIXED = {"rarity": 0., "fixed_25": .25, "fixed_50": .5, "fixed_75": .75, "nll": 1.}


def now():
    return datetime.now(timezone.utc).isoformat()


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(8*1024*1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def random_streams(seed):
    # Independent streams for adapter initialization, data order and selection.
    data_rng = np.random.default_rng(np.random.SeedSequence([seed, 101]))
    selector_seed = int(np.random.SeedSequence([seed, 202]).generate_state(1)[0])
    return data_rng, selector_seed


def selector_for(method, alpha, seed):
    _, selection_seed = random_streams(seed)
    mode = "sieve_nd" if method == "bandit" or method.startswith("fixed_") else method
    selector = Selector(mode, alpha, selection_seed)
    if method in FIXED:
        w = FIXED[method]
        selector.weights = torch.tensor([w, 1-w])
    return selector


def per_example_eval(client, batches, split, seq_len, start_token=0):
    """One example is one fixed nonoverlapping token window, not an article."""
    examples, total, count, index = [], 0., 0, 0
    for ids in batches:
        outputs = client.forward(make_data(ids), "cross_entropy").result().loss_fn_outputs
        if len(outputs) != len(ids):
            raise ValueError("Unexpected evaluation output count")
        for row, output in zip(ids, outputs, strict=True):
            losses = -output["logprobs"].to_torch().double()
            if losses.numel() != row.numel()-1 or not torch.isfinite(losses).all():
                raise ValueError("Invalid per-example evaluation losses")
            nll_sum, n = float(losses.sum()), losses.numel()
            token_start = start_token+index*seq_len
            examples.append({"example_id": f"{split}:{token_start}:{token_start+row.numel()}",
                "token_start": token_start, "token_end": token_start+row.numel(),
                "target_count": n, "nll_sum": nll_sum, "nll": nll_sum/n,
                "input_sha256": hashlib.sha256(row.cpu().numpy().tobytes()).hexdigest()})
            total += nll_sum
            count += n
            index += 1
    if not count:
        raise ValueError("Empty evaluation split")
    return {"split": split, "nll": total/count, "ppl": math.exp(total/count),
            "target_count": count, "example_count": len(examples), "examples": examples}


def verify_initial(actual, expected):
    if actual["example_count"] != expected["example_count"]:
        raise ValueError("Initial evaluation example count differs")
    for a, b in zip(actual["examples"], expected["examples"], strict=True):
        if (a["input_sha256"] != b["input_sha256"] or
                not math.isclose(a["nll"], b["nll"], abs_tol=1e-5, rel_tol=0)):
            raise ValueError("Method does not match its seed's untrained initialization")


def write_evaluation(path, result, completed, tokens):
    atomic_json(path, {"completed_steps": completed, "train_input_tokens": tokens,
                      **result})


def train_trajectory(client, train, reward_batches, dev_batches, frequency, args,
                     method, seed, directory, initialization, manifest_id, resume=None):
    """No test dataset or test callback is accepted by the training loop."""
    import tinker
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    attempt = uuid.uuid4().hex[:12]
    selector = selector_for(method, args.alpha, seed)
    train.rng, _ = random_streams(seed)
    completed = 0
    counters = {"train_input_tokens": 0, "selected_targets": 0,
                "extra_scoring_forward_tokens": 0, "reward_input_tokens": 0,
                "development_input_tokens": 0, "training_step_seconds": 0.}
    last_dev = None
    if resume:
        if resume["manifest_id"] != manifest_id or resume["method"] != method or resume["seed"] != seed:
            raise ValueError("Resume identity mismatch")
        completed = resume["completed_steps"]
        train.rng.bit_generator.state = resume["dataset_rng"]
        selector.load_state_dict(resume["selector"])
        counters = resume["counters"].copy()
        last_dev = resume["dev_nll"]
        if selector.active:
            raise ValueError("Expected an evaluation-boundary checkpoint")
    params = tinker.AdamParams(learning_rate=args.lr,beta1=.9,beta2=.95,eps=1e-8,
                              weight_decay=args.weight_decay,grad_clip_norm=args.grad_clip)
    dev_start = args.reward_batches*args.batch_size*args.seq_len

    def log(record):
        record = {"time": now(), "attempt": attempt, "method": method, "seed": seed, **record}
        with (directory/"metrics.jsonl").open("a") as f:
            f.write(json.dumps(record,allow_nan=False)+"\n")
        print(json.dumps(record),flush=True)

    def assess_dev(step):
        result = per_example_eval(client,dev_batches,"validation-development",args.seq_len,dev_start)
        counters["development_input_tokens"] += result["target_count"]
        write_evaluation(directory/f"dev-{step:07d}.json",result,step,counters["train_input_tokens"])
        log({"event":"development","step":step,"dev_nll":result["nll"],
             "dev_ppl":result["ppl"],"train_input_tokens":counters["train_input_tokens"]})
        return result

    def checkpoint(step, dev_nll, initial=False):
        name = f"step-{step:07d}"
        state_path = initialization["state_path"] if initial else client.save_state(name).result().path
        sampler_path = initialization["sampler_path"] if initial else client.save_weights_for_sampler(name).result().path
        record = {"manifest_id":manifest_id,"method":method,"seed":seed,
            "completed_steps":step,"state_path":state_path,"sampler_path":sampler_path,
            "dataset_rng":train.rng.bit_generator.state,"selector":selector.state_dict(),
            "counters":counters.copy(),"dev_nll":dev_nll,"initialization":initialization["state_path"]}
        atomic_json(directory/f"{name}.json",record)
        atomic_json(directory/"latest.json",record)
        return record

    if resume is None:
        # Verify paired predictions before any optimizer updates.
        initial_dev = assess_dev(0)
        verify_initial(initial_dev,initialization["development"])
        last_dev = initial_dev["nll"]
        initial_reward = evaluate_batches(client,reward_batches)
        counters["reward_input_tokens"] += sum(x[:,1:].numel() for x in reward_batches)
        if method == "bandit":
            selector.observe(initial_reward)
        log({"event":"initial","step":0,"reward_nll":initial_reward,
             "fresh_optimizer":True,"initialization":initialization["state_path"]})
        checkpoint(0,last_dev,initial=True)
    else:
        log({"event":"resume","step":completed,"state_path":resume["state_path"]})

    for step in range(completed,args.steps):
        if method == "bandit" and not selector.active:
            selector.begin_interval(step,args.steps)
        ids = train.sample_batch(args.batch_size,torch.device("cpu"))["input_ids"]
        started = time.perf_counter()
        metrics = train_step(client,ids,selector,frequency,params)
        elapsed = time.perf_counter()-started
        if any(not math.isfinite(v) for v in metrics.values()):
            raise ValueError("Non-finite training metrics")
        n = ids[:,1:].numel()
        counters["train_input_tokens"] += n
        counters["selected_targets"] += round(metrics["selected_fraction"]*n)
        counters["extra_scoring_forward_tokens"] += metrics["extra_scoring_forward"]*n
        counters["training_step_seconds"] += elapsed
        completed = step+1
        log({"event":"train","step":completed,"step_seconds":elapsed,
             "batch_sha256":hashlib.sha256(ids.numpy().tobytes()).hexdigest(),
             "weights":selector.weights.tolist() if method == "bandit" or method in FIXED else None,
             **metrics, **counters})
        if completed % args.reward_interval == 0:
            reward_nll = evaluate_batches(client,reward_batches)
            counters["reward_input_tokens"] += sum(x[:,1:].numel() for x in reward_batches)
            weights = selector.weights.tolist()
            reward = selector.observe(reward_nll) if method == "bandit" else None
            log({"event":"reward_eval","step":completed,"reward_nll":reward_nll,
                 "reward":reward,"credited_weights":weights if method == "bandit" else None,
                 "context_bin":selector.bandit._current_bin if method == "bandit" else None})
        if completed % args.checkpoint_interval == 0 or completed == args.steps:
            last_dev = assess_dev(completed)["nll"]
            checkpoint(completed,last_dev)
        if completed % 10 == 0:
            atomic_json(directory/"status.json",{"status":"running","step":completed,
                "total_steps":args.steps,"time":now(),"attempt":attempt})
    result = {"method":method,"seed":seed,"completed_steps":completed,
        "final_dev_nll":last_dev,"kind":"fixed" if method in FIXED else method,
        "fixed_nll_weight":FIXED.get(method),**counters,
        "state_path":json.loads((directory/"latest.json").read_text())["state_path"]}
    atomic_json(directory/"result.json",result)
    atomic_json(directory/"status.json",{"status":"training_complete","step":completed,"time":now()})
    return result


def initialize_seed(service,args,seed,dev_batches,directory,manifest_id,fingerprint):
    path = directory/f"initialization-seed{seed}.json"
    if path.exists():
        result = json.loads(path.read_text())
        if result["manifest_id"] != manifest_id:
            raise ValueError("Initialization manifest mismatch")
        return result
    client = service.create_lora_training_client(base_model=args.base_model,rank=args.rank,seed=seed,
        train_mlp=True,train_attn=True,train_unembed=False)
    if tokenizer_fingerprint(client.get_tokenizer()) != fingerprint:
        raise ValueError("Initialization tokenizer mismatch")
    result = {"manifest_id":manifest_id,"seed":seed,"base_model":args.base_model,
        "optimizer_steps":0,"state_path":client.save_state(f"untrained-seed{seed}").result().path,
        "sampler_path":client.save_weights_for_sampler(f"untrained-seed{seed}").result().path,
        "development":per_example_eval(client,dev_batches,"validation-development",args.seq_len,
            args.reward_batches*args.batch_size*args.seq_len)}
    atomic_json(path,result)
    return result


def restore_method(service,initialization,resume):
    # A zero-step checkpoint contains no training history. New methods always
    # use the weights-only API, which explicitly resets optimizer state.
    if resume and resume["completed_steps"] > 0:
        return service.create_training_client_from_state_with_optimizer(resume["state_path"])
    return service.create_training_client_from_state(initialization["state_path"])


def load_run_data(args,seed):
    cfg = SieveConfig(data_dir=args.data_dir,seq_len=args.seq_len,seed=seed)
    train = MemmapDataset(args.data_dir,"train",cfg)
    val = MemmapDataset(args.data_dir,"val",cfg)
    reward = evaluation_batches(val,args.batch_size,0,args.reward_batches)
    dev = evaluation_batches(val,args.batch_size,args.reward_batches,args.dev_batches)
    return train,reward,dev,train.freq_table.float()


def run_job(args,method,seed,initialization,manifest_id,directory):
    import tinker
    run_dir = directory/f"{method}-seed{seed}"
    run_dir.mkdir(exist_ok=True)
    if (run_dir/"result.json").exists():
        return json.loads((run_dir/"result.json").read_text())
    for attempt in range(args.max_retries+1):
        service = tinker.ServiceClient()
        try:
            train,reward,dev,frequency = load_run_data(args,seed)
            latest = run_dir/"latest.json"
            resume = json.loads(latest.read_text()) if latest.exists() else None
            client = restore_method(service,initialization,resume)
            result = train_trajectory(client,train,reward,dev,frequency,args,method,seed,
                                      run_dir,initialization,manifest_id,resume)
            service.close("success").result()
            return result
        except Exception as exc:
            atomic_json(run_dir/"status.json",{"status":"retrying" if attempt<args.max_retries else "failed",
                "attempt":attempt,"error":str(exc),"traceback":traceback.format_exc(),"time":now()})
            try:
                service.close("errored",detail=str(exc)[:200]).result()
            except Exception:
                pass
            if attempt == args.max_retries:
                raise
            time.sleep(min(2**attempt,10))


def summarize_development(directory,args):
    """Development-only curves and grid preferences at common token budgets."""
    curves, stages = [], []
    for step in range(0,args.steps+1,args.checkpoint_interval):
        fixed = {}
        for method in METHODS:
            points = []
            for seed in args.seeds:
                path = directory/f"{method}-seed{seed}"/f"dev-{step:07d}.json"
                if path.exists():
                    points.append((seed,json.loads(path.read_text())["nll"]))
            if points:
                values = [x[1] for x in points]
                curves.append({"method":method,"step":step,
                    "train_input_tokens":step*args.batch_size*(args.seq_len-1),
                    "seeds":[x[0] for x in points],"per_seed_nll":values,
                    "mean_nll":float(np.mean(values)),
                    "sample_sd_nll":float(np.std(values,ddof=1)) if len(values)>1 else None})
                if method in FIXED and len(points)==len(args.seeds):
                    fixed[FIXED[method]] = float(np.mean(values))
        if len(fixed)==len(FIXED):
            stages.append({"step":step,"best_grid_nll_weight":min(fixed,key=lambda w:(fixed[w],w)),
                "mean_development_nll_by_weight":fixed,
                "interpretation":"descriptive development-only grid preference; not a test result"})
    atomic_json(directory/"development_curves.json",curves)
    atomic_json(directory/"development_stage_mixtures.json",stages)


def final_test(directory,args,records,manifest_id):
    """Gate every test call behind complete training and a locked dev selection."""
    expected = {(m,s) for m in METHODS for s in args.seeds}
    if len(records) != len(expected) or {(r["method"],r["seed"]) for r in records} != expected or any(r["completed_steps"] != args.steps for r in records):
        raise ValueError("Test evaluation requires all methods at the predefined final endpoint")
    selected = select_fixed(records,sorted(FIXED.values()),args.seeds)
    selected.update(manifest_id=manifest_id,endpoint_step=args.steps)
    selection_path = directory/"selected_fixed.json"
    if selection_path.exists() and json.loads(selection_path.read_text()) != json.loads(json.dumps(selected)):
        raise ValueError("Locked fixed-mixture selection differs")
    atomic_json(selection_path,selected)
    import tinker
    service = tinker.ServiceClient()
    try:
        cfg = SieveConfig(data_dir=args.data_dir,seq_len=args.seq_len)
        test = MemmapDataset(args.data_dir,"test",cfg)
        results = []
        for record in records:
            method,seed = record["method"],record["seed"]
            path = directory/f"{method}-seed{seed}"/f"test-{args.steps:07d}.json"
            if path.exists():
                result = json.loads(path.read_text())
                if result["manifest_id"] != manifest_id or result["completed_steps"] != args.steps:
                    raise ValueError("Existing endpoint test result does not match the experiment")
            else:
                client = service.create_training_client_from_state(record["state_path"])
                batches = (b["input_ids"] for b in test.val_batches(args.batch_size,torch.device("cpu")))
                result = per_example_eval(client,batches,"test",args.seq_len)
                result.update(method=method,seed=seed,completed_steps=args.steps,
                    train_input_tokens=record["train_input_tokens"],manifest_id=manifest_id)
                atomic_json(path,result)
            results.append({k:v for k,v in result.items() if k != "examples"})
            atomic_json(directory/"test_results.json",results)
        best_method = next(m for m,w in FIXED.items() if w == selected["fixed_nll_weight"])
        for row in list(results):
            if row["method"] == best_method:
                results.append({**row,"method":"best_fixed","selected_method":best_method})
        adaptive = {r["seed"]:r for r in results if r["method"] == "bandit"}
        summary = {}
        for method in (*METHODS,"best_fixed"):
            rows = sorted([r for r in results if r["method"]==method],key=lambda r:r["seed"])
            nlls = [r["nll"] for r in rows]
            diffs = [r["nll"]-adaptive[r["seed"]]["nll"] for r in rows]
            summary[method] = {"seeds":[r["seed"] for r in rows],"mean_test_nll":float(np.mean(nlls)),
                "sample_sd_test_nll":float(np.std(nlls,ddof=1)) if len(nlls)>1 else None,
                "paired_bandit_nll_gain":diffs,"mean_bandit_gain":float(np.mean(diffs)),
                "sample_sd_bandit_gain":float(np.std(diffs,ddof=1)) if len(diffs)>1 else None}
        atomic_json(directory/"summary.json",summary)
        service.close("success").result()
    except Exception:
        try:
            service.close("errored").result()
        except Exception:
            pass
        raise


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base-model",default="Qwen/Qwen3-8B")
    p.add_argument("--data-dir",required=True)
    p.add_argument("--run-dir",required=True)
    p.add_argument("--seeds",type=int,nargs="+",default=[42,43,44])
    p.add_argument("--steps",type=int,default=1000)
    p.add_argument("--rank",type=int,default=32)
    p.add_argument("--alpha",type=float,default=.7)
    p.add_argument("--lr",type=float,default=1e-4)
    p.add_argument("--weight-decay",type=float,default=0.)
    p.add_argument("--grad-clip",type=float,default=1.)
    p.add_argument("--seq-len",type=int,default=512)
    p.add_argument("--batch-size",type=int,default=4)
    p.add_argument("--reward-interval",type=int,default=50)
    p.add_argument("--checkpoint-interval",type=int,default=100)
    p.add_argument("--reward-batches",type=int,default=8)
    p.add_argument("--dev-batches",type=int,default=16)
    p.add_argument("--workers",type=int,default=3)
    p.add_argument("--max-retries",type=int,default=2)
    p.add_argument("--resume",action="store_true")
    p.add_argument("--dry-run",action="store_true")
    return p


def main():
    args = parser().parse_args()
    bins = schedule_bins(args.steps,args.reward_interval)
    positive = (args.checkpoint_interval,args.reward_batches,args.dev_batches,args.workers)
    if (min(positive)<=0 or args.max_retries<0 or len(set(args.seeds))!=len(args.seeds)
            or min(args.seeds)<0 or args.steps%args.checkpoint_interval
            or args.checkpoint_interval%args.reward_interval):
        raise ValueError("Use positive counts, unique nonnegative seeds, and aligned evaluation/checkpoint intervals")
    from types import SimpleNamespace
    validation = dict(vars(args),mode="rarity",eval_interval=args.reward_interval,
        eval_batches=args.dev_batches,data_fraction=1.,seed=args.seeds[0])
    train,val,frequency = load_data(SimpleNamespace(**validation))
    test = MemmapDataset(args.data_dir,"test",SieveConfig(seq_len=args.seq_len))
    if int(test.arr.max()) >= train.meta["vocab_size"]:
        raise ValueError("Test token IDs exceed tokenizer vocabulary")
    reward = evaluation_batches(val,args.batch_size,0,args.reward_batches)
    dev = evaluation_batches(val,args.batch_size,args.reward_batches,args.dev_batches)
    if any(len(b)!=args.batch_size for b in reward+dev):
        raise ValueError("Reward/development slices must contain full batches")
    data_hashes = {**data_digest(args.data_dir),"test.bin":file_hash(Path(args.data_dir)/"test.bin")}
    source_files = [Path(__file__),Path(__file__).parent/"bandit_comparison.py",
        Path(__file__).parent/"scorer_discrimination.py",Path(__file__).parent/"train_sieve_tinker.py"]
    source_files += [Path(__file__).parent/"sieve"/name for name in ("tinker_backend.py","bandit.py","data.py","config.py")]
    code_hashes = {str(p.relative_to(Path(__file__).parent)):file_hash(p) for p in source_files}
    config = {k:v for k,v in vars(args).items() if k not in ("run_dir","dry_run","resume","workers","max_retries")}
    token_budget = args.steps*args.batch_size*(args.seq_len-1)
    data_plans = {}
    for seed in args.seeds:
        train.rng,_ = random_streams(seed)
        digest = hashlib.sha256()
        for _ in range(args.steps):
            digest.update(train.sample_batch(args.batch_size,torch.device("cpu"))["input_ids"].numpy().tobytes())
        data_plans[str(seed)] = digest.hexdigest()
    manifest = {"config":config,"methods":METHODS,"data_sha256":data_hashes,"code_sha256":code_hashes,
        "batch_plan_sha256":data_plans,"context_bins":bins,"runs":len(METHODS)*len(args.seeds),
        "eligible_training_targets_per_run":token_budget,"total_optimizer_steps":args.steps*len(METHODS)*len(args.seeds),
        "reward_validation_tokens":[0,args.reward_batches*args.batch_size*args.seq_len],
        "development_validation_tokens":[args.reward_batches*args.batch_size*args.seq_len,
            (args.reward_batches+args.dev_batches)*args.batch_size*args.seq_len],
        "test_endpoint_steps":[args.steps],"test_split":"official WikiText test; every complete seq_len window",
        "initialization":"per-seed untrained adapter snapshot; independent weights-only loads with fresh optimizers",
        "token_budget_scope":"eligible targets matched; selected targets differ for full-token baseline",
        "evaluation_example":"fixed nonoverlapping token window; not original WikiText document",
        "cost_scope":"submitted tokens and wall time, not invoices; resumed counters exclude discarded work"}
    manifest = json.loads(json.dumps(manifest))
    manifest_id = hashlib.sha256(json.dumps(manifest,sort_keys=True).encode()).hexdigest()
    manifest["manifest_id"] = manifest_id
    if args.dry_run:
        print(json.dumps({"status":"local checks passed; no API calls",**manifest},indent=2))
        return
    directory = Path(args.run_dir)
    if directory.exists() and any(directory.iterdir()) and not args.resume:
        raise ValueError("Use an empty directory or --resume")
    directory.mkdir(parents=True,exist_ok=True)
    lock = (directory/"process.lock").open("a")
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if (directory/"manifest.json").exists():
        if json.loads((directory/"manifest.json").read_text()) != manifest:
            raise ValueError("Resume manifest differs (code, data or configuration)")
    else:
        atomic_json(directory/"manifest.json",manifest)
    import tinker
    initializations = {}
    atomic_json(directory/"status.json",{"status":"initializing","time":now(),"runs":manifest["runs"]})
    for seed in args.seeds:
        service = tinker.ServiceClient()
        try:
            caps = service.get_server_capabilities()
            model = next((m for m in caps.supported_models if m.model_name==args.base_model and m.trainable is not False),None)
            if model is None or (model.max_context_length and args.seq_len-1>model.max_context_length):
                raise ValueError("Requested trainable model/context unavailable")
            initializations[seed] = initialize_seed(service,args,seed,dev,directory,manifest_id,
                                                   train.meta["tokenizer_vocab_sha256"])
            service.close("success").result()
        except Exception:
            try:
                service.close("errored").result()
            except Exception:
                pass
            raise
        print(f"Saved untrained adapter initialization for seed {seed}",flush=True)
    atomic_json(directory/"status.json",{"status":"training","time":now(),"runs":manifest["runs"]})
    records, failures = [], []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(run_job,args,m,s,initializations[s],manifest_id,directory):(m,s)
                   for m in METHODS for s in args.seeds}
        for future in as_completed(futures):
            method,seed = futures[future]
            try:
                records.append(future.result())
                atomic_json(directory/"development_results.json",records)
                summarize_development(directory,args)
            except Exception as exc:
                failures.append({"method":method,"seed":seed,"error":str(exc)})
            atomic_json(directory/"status.json",{"status":"training","time":now(),
                "completed_runs":len(records),"total_runs":manifest["runs"],"failures":failures})
    if failures:
        atomic_json(directory/"status.json",{"status":"failed","failures":failures,"time":now()})
        raise RuntimeError("Some trajectories failed; test gate remains closed. Resume to recover.")
    atomic_json(directory/"status.json",{"status":"final_test","time":now()})
    final_test(directory,args,records,manifest_id)
    atomic_json(directory/"status.json",{"status":"complete","time":now(),"completed_runs":len(records)})
    print(f"Finished fresh trajectories: {directory}",flush=True)


if __name__ == "__main__":
    main()
