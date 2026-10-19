from __future__ import annotations

import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import modal

REPO = Path(__file__).resolve().parent

app = modal.App("sieve-drop-se-lm-eval")

volume_results = modal.Volume.from_name("sieve-results", create_if_missing=True)
volume_hf = modal.Volume.from_name("sieve-hf-cache", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .pip_install(
        "torch==2.3.0",
        "transformers==4.41.0",
        "accelerate==0.30.0",
        "datasets==2.19.0",
        "tokenizers==0.19.1",
        "sentencepiece",
        "protobuf",
        "numpy==1.26.4",
        "scipy==1.13.0",
        "sacrebleu",
        "rouge-score",
        "evaluate",
        "lm-eval==0.4.3",
        "huggingface-hub==0.23.5",
    )
    .add_local_dir(
        REPO,
        remote_path="/repo",
        ignore=["**/.git/**", "**/__pycache__/**", "**/*.pyc"],
    )
)

RUNS = [
    "sieve_no_se_wikitext103_gpt2l_a70_s42_st25000",
    "su_only_wikitext103_gpt2l_a70_s42_st25000",
    "sl_only_wikitext103_gpt2l_a70_s42_st25000",
    "sd_only_wikitext103_gpt2l_a70_s42_st25000",
]


def _prepare_hf_dir(run_name: str, step: int = 25000) -> Path:
    ckpt_root = Path("/results") / run_name / "checkpoints"
    step_dir = ckpt_root / f"ckpt-step{step:06d}"
    if not step_dir.exists():
        raise FileNotFoundError(f"missing checkpoint: {step_dir}")
    if not (step_dir / "pytorch_model.bin").exists():
        raise FileNotFoundError(f"missing weights: {step_dir / 'pytorch_model.bin'}")

    out = Path("/tmp/eval_models") / run_name / f"step{step:06d}"
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)

    # Do not copy tokenizer files from these checkpoints. Some were saved with
    # a newer fast-tokenizer JSON than the pinned lm-eval stack can parse.
    # lm-eval is told explicitly to use the base gpt2-large tokenizer instead.
    for name in ["config.json", "generation_config.json"]:
        src = ckpt_root / name
        if src.exists():
            shutil.copy2(src, out / name)

    shutil.copy2(step_dir / "pytorch_model.bin", out / "pytorch_model.bin")
    return out


def _extract_metrics(result_json: Path) -> dict:
    raw = json.loads(result_json.read_text())
    out = {"raw_file": str(result_json)}
    for task, values in raw.get("results", {}).items():
        for key, value in values.items():
            if isinstance(value, (int, float)):
                out[f"{task}/{key}"] = value
    return out


@app.function(
    image=image,
    gpu="L4",
    timeout=10 * 3600,
    volumes={"/results": volume_results, "/hf_cache": volume_hf},
    memory=32768,
    cpu=4.0,
    max_containers=4,
)
def eval_one(
    run_name: str,
    step: int = 25000,
    batch_size: str = "8",
    limit: int = 0,
    tasks: str = "gsm8k,mmlu",
) -> dict:
    import os

    os.environ["HF_HOME"] = "/hf_cache"
    os.environ["HF_DATASETS_CACHE"] = "/tmp/hf_datasets_cache"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    volume_results.reload()
    model_dir = _prepare_hf_dir(run_name, step=step)
    out_dir = Path("/results") / "downstream_eval" / run_name / f"step{step:06d}"
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    requested = [t.strip() for t in tasks.split(",") if t.strip()]
    all_task_specs = {
        "gsm8k": ("gsm8k", "5"),
        "mmlu": ("mmlu", "5"),
    }
    task_specs = [all_task_specs.get(t, (t, "5")) for t in requested]

    summary: dict = {
        "run_name": run_name,
        "step": step,
        "model_dir": str(model_dir),
        "status": "ok",
        "status_note": "running",
        "tasks": {},
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    volume_results.commit()

    for task, fewshot in task_specs:
        task_out = out_dir / task
        task_out.mkdir(parents=True, exist_ok=True)
        cmd = [
            sys.executable,
            "-m",
            "lm_eval",
            "--model",
            "hf",
            "--model_args",
            f"pretrained={model_dir},tokenizer=gpt2-large,dtype=float16",
            "--tasks",
            task,
            "--num_fewshot",
            fewshot,
            "--batch_size",
            batch_size,
            "--output_path",
            str(task_out),
        ]
        if limit > 0:
            cmd.extend(["--limit", str(limit)])
        print(f"[lm-eval] {run_name} task={task}: {' '.join(cmd)}")
        t0 = time.time()
        proc = subprocess.run(
            cmd,
            cwd="/repo",
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
        wall = time.time() - t0
        (task_out / "lm_eval.log").write_text(proc.stdout)

        result_files = sorted(task_out.rglob("*.json"))
        task_summary = {
            "returncode": proc.returncode,
            "wall_seconds": wall,
            "output_dir": str(task_out),
        }
        if proc.returncode != 0:
            summary["status"] = "failed"
            task_summary["error_tail"] = proc.stdout[-4000:]
        elif result_files:
            task_summary.update(_extract_metrics(result_files[-1]))
        else:
            summary["status"] = "failed"
            task_summary["error_tail"] = "lm-eval returned 0 but produced no JSON result"
        summary["tasks"][task] = task_summary

    summary["status_note"] = "completed"
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    volume_results.commit()
    return summary


@app.local_entrypoint()
def run_drop_se_downstream(
    step: int = 25000,
    batch_size: str = "8",
    limit: int = 0,
    tasks: str = "gsm8k,mmlu",
    dry_run: bool = False,
) -> None:
    print("[downstream] final-checkpoint lm-eval")
    for run_name in RUNS:
        print(f"  - {run_name}/checkpoints/ckpt-step{step:06d}")
    if dry_run:
        return

    calls = [
        eval_one.spawn(run_name, step=step, batch_size=batch_size, limit=limit, tasks=tasks)
        for run_name in RUNS
    ]
    print(f"[downstream] submitted {len(calls)} eval jobs")
    for run_name, call in zip(RUNS, calls):
        print(f"  {run_name}: call={call.object_id}")


@app.local_entrypoint()
def run_one_downstream(
    run_name: str = "sieve_no_se_wikitext103_gpt2l_a70_s42_st25000",
    step: int = 25000,
    batch_size: str = "8",
    limit: int = 0,
    tasks: str = "gsm8k,mmlu",
    detach: bool = True,
) -> None:
    print("[downstream-one] final-checkpoint lm-eval")
    print(f"  - {run_name}/checkpoints/ckpt-step{step:06d}")
    if limit > 0:
        print(f"  - lm-eval limit={limit}")
    print(f"  - tasks={tasks}")
    if detach:
        call = eval_one.spawn(
            run_name, step=step, batch_size=batch_size, limit=limit, tasks=tasks
        )
        print(f"[downstream-one] submitted {run_name} call={call.object_id}")
    else:
        result = eval_one.remote(
            run_name, step=step, batch_size=batch_size, limit=limit, tasks=tasks
        )
        print(json.dumps(result, indent=2))
