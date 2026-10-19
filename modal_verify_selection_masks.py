from __future__ import annotations

import json
from pathlib import Path

import modal


app = modal.App("sieve-selection-mask-audit")
REPO = Path(__file__).resolve().parent
volume_data = modal.Volume.from_name("sieve-data", create_if_missing=True)
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch",
        "transformers>=4.36",
        "numpy",
        "huggingface-hub",
        "accelerate",
        "liger-kernel",
        "kernels",
    )
    .add_local_dir(
        REPO,
        remote_path="/repo",
        ignore=["**/.git/**", "**/__pycache__/**", "**/*.pyc"],
    )
)


def jaccard(mask_a, mask_b) -> float:
    inter = ((mask_a > 0) & (mask_b > 0)).sum().item()
    union = ((mask_a > 0) | (mask_b > 0)).sum().item()
    return float(inter / max(union, 1))


@app.function(
    image=image,
    gpu="A100-40GB",
    timeout=20 * 60,
    volumes={"/data": volume_data},
)
def audit_selection_masks(
    dataset: str = "wikitext103",
    model_key: str = "gpt2l",
    alpha: float = 0.7,
    seed: int = 42,
    batch_size: int = 4,
    seq_len: int = 1024,
) -> dict:
    import os
    import random
    import sys
    from pathlib import Path

    import numpy as np
    import torch
    from transformers import AutoModelForCausalLM

    sys.path.insert(0, "/repo")

    from train_sieve import MODEL_NAMES
    from sieve.config import SieveConfig
    from sieve.data import MemmapDataset
    from sieve.state import SieveState
    from sieve.loss import selective_loss

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    Path("/repo/data").mkdir(exist_ok=True)
    for ds in ("wikitext2", "wikitext103", "openwebmath_1b"):
        src = Path(f"/data/{ds}")
        dst = Path(f"/repo/data/{ds}")
        if src.exists() and not dst.exists():
            dst.symlink_to(src)

    dataset_dir = "openwebmath_1b" if dataset == "owm" else dataset
    data_dir = f"/repo/data/{dataset_dir}"
    device = torch.device("cuda")

    cfg = SieveConfig(
        model_name=MODEL_NAMES[model_key],
        data_dir=data_dir,
        selection_ratio=alpha,
        batch_size=batch_size,
        seq_len=seq_len,
        seed=seed,
    )
    freq_path = Path(data_dir) / "freq.pt"
    if freq_path.exists():
        cfg.freq_table_path = str(freq_path)

    ds = MemmapDataset(data_dir, "train", cfg)
    batch = ds.sample_batch(batch_size, device)
    input_ids = batch["input_ids"]
    ref_losses = batch.get("ref_losses")

    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_name,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
    ).to(device).eval()

    with torch.no_grad():
        logits = model(input_ids=input_ids, labels=None, use_cache=False).logits.float()

    variants = {
        "sieve_no_se": ["S_U", "S_L", "S_D"],
        "su_only": ["S_U"],
        "sl_only": ["S_L"],
        "sd_only": ["S_D"],
        "rho1_se_only": ["S_E"],
    }

    masks = {}
    variant_reports = {}
    B, T = input_ids.shape
    full_loss = selective_loss(logits, input_ids, None).item()
    for name, scorers in variants.items():
        cfg_i = SieveConfig(
            model_name=cfg.model_name,
            data_dir=data_dir,
            selection_ratio=alpha,
            batch_size=batch_size,
            seq_len=seq_len,
            seed=seed,
            scorer_names=scorers,
            n_strategies=len(scorers),
            freq_table_path=cfg.freq_table_path,
        )
        state = SieveState(cfg_i, device)
        state.step_begin(0, 25000, 0.0, 0.0)
        if len(scorers) == 1:
            state.weights = torch.ones(1)
        mask = state.score_and_mask(logits, input_ids, ref_losses)
        masks[name] = mask.detach().cpu()
        selected_loss = selective_loss(logits, input_ids, mask).item()
        mask_fraction_all = float(mask.float().mean().item())
        mask_fraction_supervised = float(mask[:, 1:].float().mean().item())
        variant_reports[name] = {
            "scorers": scorers,
            "n_selected": int(mask.sum().item()),
            "tokens_total": int(B * T),
            "supervised_tokens_total": int(B * (T - 1)),
            "mask_fraction_all_positions": mask_fraction_all,
            "mask_fraction_supervised_positions": mask_fraction_supervised,
            "matches_alpha_supervised": abs(mask_fraction_supervised - alpha) < 0.01,
            "full_loss_gradient_scale": full_loss,
            "selective_loss_gradient_scale": selected_loss,
        }

    pairwise = []
    names = list(masks)
    expected_random_jaccard = alpha / (2 - alpha)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            jac = jaccard(masks[a], masks[b])
            pairwise.append(
                {
                    "compare": f"{a} vs {b}",
                    "jaccard": jac,
                    "expected_random_jaccard": expected_random_jaccard,
                    "identical_bug_signal": jac > 0.97,
                }
            )

    return {
        "dataset": dataset,
        "model": model_key,
        "alpha": alpha,
        "seed": seed,
        "batch_shape": [int(B), int(T)],
        "variants": variant_reports,
        "pairwise": pairwise,
    }


@app.local_entrypoint()
def main(out: str = "selection_mask_audit.json"):
    data = audit_selection_masks.remote()
    Path(out).write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
    print(json.dumps(data, indent=2, sort_keys=True))
    print(f"wrote {Path(out).resolve()}")
