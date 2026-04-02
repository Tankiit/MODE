"""
Framework-free SIEVE training loop.
Works on toy model, GPT-2 124M, TinyLlama.
Three hooks are explicit and labelled.
"""
from __future__ import annotations
import math, time
import torch
import torch.nn.functional as F
from typing import Optional

from .config import SieveConfig
from .device import DeviceCfg
from .data import MemmapDataset
from .state import SieveState
from .loss import selective_loss


def evaluate(
    model:    torch.nn.Module,
    dataset:  MemmapDataset,
    device:   torch.device,
    max_batches: int = 50,
) -> float:
    """Fast val PPL — capped at max_batches during training."""
    model.eval()
    total_nll, total_tok, n = 0.0, 0, 0
    with torch.no_grad():
        for batch in dataset.val_batches(8, device):
            ids    = batch["input_ids"]            # [B, T]
            B, T   = ids.shape
            out    = model(input_ids=ids, labels=None, use_cache=False)
            V      = out.logits.size(-1)
            nll    = F.cross_entropy(
                out.logits[:, :-1].reshape(-1, V),
                ids[:, 1:].reshape(-1),
                reduction="sum",
            )
            total_nll += nll.item()
            total_tok += B * (T - 1)
            n += 1
            if max_batches and n >= max_batches:
                break
    model.train()
    return math.exp(total_nll / max(total_tok, 1))


def train(
    model:      torch.nn.Module,
    train_ds:   MemmapDataset,
    val_ds:     MemmapDataset,
    cfg:        SieveConfig,
    dev_cfg:    DeviceCfg,
    sieve:      Optional[SieveState] = None,  # None = full CLM baseline
) -> dict:
    """
    Main training loop.
    Pass sieve=None to run standard CLM (full baseline).
    Pass sieve=SieveState(...) to run SIEVE.
    """
    device = dev_cfg.device
    optim  = torch.optim.AdamW(
        model.parameters(), lr=cfg.lr, weight_decay=0.1
    )

    history = {"loss": [], "val_ppl": [], "weights": []}
    prev_loss, prev_grad = 1.0, 0.0

    # Optional wandb
    log_fn = _make_logger(cfg)

    t0 = time.time()
    for step in range(cfg.max_steps):

        # ── Hook 1: sample bandit weights if rescore due ──────────────
        if sieve is not None:
            sieve.step_begin(step, cfg.max_steps, prev_loss, prev_grad)

        # ── Data ──────────────────────────────────────────────────────
        batch     = train_ds.sample_batch(cfg.batch_size, device)
        input_ids = batch["input_ids"]          # [B, T]
        ref_losses = batch.get("ref_losses")    # [B, T] or None

        # ── Forward ───────────────────────────────────────────────────
        optim.zero_grad()
        with dev_cfg.autocast:
            out    = model(input_ids=input_ids, labels=None, use_cache=False)
        logits = out.logits.float()             # always fp32 for scoring

        # ── Hook 2: score tokens, get mask ────────────────────────────
        if sieve is not None:
            mask = sieve.score_and_mask(logits, input_ids, ref_losses)
        else:
            mask = None                         # full CLM: no masking

        # ── Loss ──────────────────────────────────────────────────────
        loss = selective_loss(logits, input_ids, mask)

        # ── Backward ──────────────────────────────────────────────────
        loss.backward()
        prev_grad = torch.nn.utils.clip_grad_norm_(
            model.parameters(), cfg.grad_clip
        ).item()
        optim.step()
        if device.type == "mps":
            torch.mps.synchronize()

        prev_loss = loss.item()

        # ── Logging ───────────────────────────────────────────────────
        if step % cfg.log_interval == 0:
            elapsed = time.time() - t0
            log = {"step": step, "loss": prev_loss,
                   "grad": prev_grad, "elapsed": elapsed}
            if sieve is not None:
                log.update(sieve.log_dict())
            log_fn(log)
            history["loss"].append(prev_loss)
            if sieve:
                history["weights"].append(
                    sieve.bandit.expected_weights(
                        sieve.bandit._current_bin
                    ).tolist()
                )

        # ── Evaluation + Hook 3 ───────────────────────────────────────
        if step > 0 and step % cfg.eval_interval == 0:
            val_ppl = evaluate(model, val_ds, device, max_batches=50)
            history["val_ppl"].append(val_ppl)
            print(f"  step {step}  val_ppl={val_ppl:.2f}")

            if sieve is not None:
                sieve.on_eval(math.log(val_ppl))   # pass log-PPL as loss

            log_fn({"step": step, "val/ppl": val_ppl})

    return history


def _make_logger(cfg: SieveConfig):
    """Returns a logging function — wandb if configured, else print."""
    try:
        if cfg.wandb_project:
            import wandb
            wandb.init(project=cfg.wandb_project, name=cfg.run_name,
                       config=cfg.__dict__)
            def log(d): wandb.log(d)
        else:
            raise ImportError
    except Exception:
        def log(d):
            step = d.get("step", "")
            loss = d.get("loss", "")
            ppl  = d.get("val/ppl", "")
            we   = d.get("sieve/w_S_E", "")
            if loss:
                print(f"step {step:5d}  loss={loss:.4f}  grad={d.get('grad',0):.3f}"
                      + (f"  S_E={we:.2f}" if we != "" else ""))
            if ppl:
                print(f"          val_ppl={ppl:.2f}")
    return log
