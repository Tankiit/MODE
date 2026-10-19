"""Local training with fresh masks and evaluation-block reward attribution."""
from __future__ import annotations
import json as _json
import math, time
from pathlib import Path
from time import perf_counter
from collections import defaultdict
import torch
import torch.nn.functional as F
from typing import Optional

from .config import SieveConfig
from .device import DeviceCfg
from .data import MemmapDataset
from .state import SieveState
from .loss import selective_loss
from .patience import PatienceMonitor


def _dirichlet_entropy(bandit, bin_idx: int) -> Optional[float]:
    """
    Exact H(Dir(α)) for the current context bin when scipy is available.
    Returns None on failure — caller may fall back to ``posterior_entropy``.
    """
    try:
        if not hasattr(bandit, "alpha"):
            return None
        import numpy as np
        from scipy.special import digamma, gammaln

        a = np.asarray(bandit.alpha[bin_idx], dtype=np.float64).ravel()
        a0 = float(a.sum())
        k = int(a.size)
        if not math.isfinite(a0) or a0 <= 0 or k < 1:
            return None
        log_b = float(gammaln(a).sum() - gammaln(a0))
        term1 = (a0 - k) * float(digamma(a0))
        term2 = float(((a - 1.0) * digamma(a)).sum())
        return float(log_b + term1 - term2)
    except Exception:
        return None


def _has_bandit(sieve: Optional[SieveState]) -> bool:
    """
    True iff the selector exposes a bandit interface (SIEVE, scalarization,
    su_only, sd_only). False for Rho-1 (single-actor) and CLM (no selector).

    This is the duck-type check to use anywhere we touch sieve.bandit.
    """
    return (
        sieve is not None
        and hasattr(sieve, "bandit")
        and getattr(sieve, "bandit", None) is not None
    )


def _collect_bandit_state(
    sieve: Optional[SieveState], history: dict
) -> Optional[dict]:
    if not _has_bandit(sieve):
        return None
    return {
        "alpha": getattr(sieve.bandit, "alpha", None),
        "history": history.get("weights", []),
    }


def _bandit_variance_snapshot(sieve: Optional[SieveState]) -> Optional[dict]:
    """Current-bin variance diagnostics, kept JSON-serialisable for logs."""
    if not _has_bandit(sieve):
        return None
    b = sieve.bandit._current_bin
    ew = sieve.bandit.expected_weights(b)
    var = sieve.bandit.posterior_variance(b)
    names = list(getattr(sieve, "scorer_names", ["S_E", "S_U", "S_L", "S_D"]))
    var_by_name = {name: float(var[i]) for i, name in enumerate(names)}
    order = sorted((float(x) for x in ew), reverse=True)
    return {
        "context_bin": int(b),
        "variance_sum": float(var.sum()),
        "variance_S_E": var_by_name.get("S_E", 0.0),
        "variance_S_U": var_by_name.get("S_U", 0.0),
        "variance_S_L": var_by_name.get("S_L", 0.0),
        "variance_S_D": var_by_name.get("S_D", 0.0),
        "weight_variance": float(ew.var()),
        "weight_gap": float(order[0] - order[1]) if len(order) > 1 else 0.0,
    }


def evaluate(
    model:       torch.nn.Module,
    dataset:     MemmapDataset,
    device:      torch.device,
    max_batches: int = 50,
) -> float:
    """Fast val PPL — capped at max_batches during training."""
    model.eval()
    total_nll, total_tok, n = 0.0, 0, 0
    with torch.no_grad():
        for batch in dataset.val_batches(8, device):
            ids  = batch["input_ids"]
            B, T = ids.shape
            # Autocast where available; cast logits to float32 for CE
            with torch.autocast(device.type, enabled=(device.type == 'cuda')):
                out = model(input_ids=ids, labels=None, use_cache=False)
            V    = out.logits.size(-1)
            nll  = F.cross_entropy(
                out.logits.float()[:, :-1].reshape(-1, V),
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


def _save_checkpoint(
    model: torch.nn.Module,
    step: int,
    save_dir: Path,
    val_ppl: float,
    bandit_state: Optional[dict] = None,
    is_best: bool = False,
) -> None:
    """
    Save an HF-compatible checkpoint directory.

    Layout written:
      save_dir/ckpt-step{step:06d}/
        pytorch_model.bin     ← state_dict
        metadata.json         ← step, val_ppl, wall_time
    And optionally a symlink save_dir/ckpt-best → ckpt-step{best_step}.

    Uses state_dict (not full HF save_pretrained) to avoid saving tokenizer
    and config per checkpoint. Config + tokenizer live at save_dir root,
    written ONCE at start of training.
    """
    ckpt_dir = save_dir / f"ckpt-step{step:06d}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    torch.save(model.state_dict(), ckpt_dir / "pytorch_model.bin")

    meta = {
        "step": step,
        "val_ppl": float(val_ppl),
        "timestamp": time.time(),
    }
    if bandit_state is not None:
        torch.save(bandit_state, ckpt_dir / "bandit_state.pt")
        meta["has_bandit_state"] = True

    (ckpt_dir / "metadata.json").write_text(_json.dumps(meta, indent=2))

    if is_best:
        best_link = save_dir / "ckpt-best"
        if best_link.exists() or best_link.is_symlink():
            best_link.unlink()
        best_link.symlink_to(ckpt_dir.name)


def _init_save_dir(
    model: torch.nn.Module,
    save_dir: Path,
    model_name: str,
) -> None:
    """Write config + tokenizer to save_dir root, ONCE per run."""
    save_dir.mkdir(parents=True, exist_ok=True)

    if hasattr(model, "config"):
        model.config.save_pretrained(save_dir)

    try:
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(model_name)
        tok.save_pretrained(save_dir)
    except Exception as e:
        print(f"[ckpt] tokenizer save failed (non-fatal): {e}")


def _prune_checkpoints(
    save_dir: Path,
    max_keep: int,
    protected: set[str],
) -> None:
    """
    Keep only the `max_keep` most recent checkpoints, plus anything in
    `protected` (typically the best step directory name).
    """
    if max_keep <= 0:
        return
    ckpts = sorted(
        save_dir.glob("ckpt-step*"),
        key=lambda p: int(p.name.replace("ckpt-step", "")),
    )
    if len(ckpts) <= max_keep:
        return
    to_delete = ckpts[:-max_keep]
    for c in to_delete:
        if c.name in protected:
            continue
        for f in c.iterdir():
            f.unlink()
        c.rmdir()


def train(
    model:    torch.nn.Module,
    train_ds: MemmapDataset,
    val_ds:   MemmapDataset,
    cfg:      SieveConfig,
    dev_cfg:  DeviceCfg,
    sieve:    Optional[SieveState] = None,
) -> dict:
    """Fresh per-batch scoring and fixed full-token CE on every device.

    Weights remain fixed between evaluations; improvement credits that block.
    Host timers do not isolate asynchronous CUDA work.
    """
    try:
        from tqdm import tqdm
        _use_tqdm = True
    except ImportError:
        _use_tqdm = False

    device = dev_cfg.device
    optim  = torch.optim.AdamW(
        model.parameters(), lr=cfg.lr, weight_decay=0.1
    )

    history      = {"loss": [], "val_ppl": [], "weights": [], "variance": []}
    prev_loss    = 1.0
    prev_grad    = 0.0
    last_val_ppl = float("nan")
    best_ppl     = float("inf")
    best_step    = -1

    log_fn = _make_logger(cfg)

    save_dir: Optional[Path] = None
    if getattr(cfg, "save_dir", None):
        save_dir = Path(cfg.save_dir)
        _init_save_dir(model, save_dir, cfg.model_name)
        print(f"[ckpt] writing to {save_dir}")

    save_interval = getattr(cfg, "save_interval", 0)
    max_keep      = getattr(cfg, "max_checkpoints", 0)

    patience: Optional[PatienceMonitor] = None
    if getattr(cfg, "patience_enable", False):
        effective_min = max(cfg.patience_min_steps, 10 * cfg.eval_interval)
        patience = PatienceMonitor(
            window=cfg.patience_window,
            min_steps=effective_min,
            ppl_tol=cfg.patience_ppl_tol,
            weight_tol=cfg.patience_weight_tol,
            entropy_tol=cfg.patience_entropy_tol,
            mode=cfg.patience_mode,
        )
        print(
            f"[patience] enabled: mode={cfg.patience_mode} "
            f"window={cfg.patience_window} min_steps={effective_min}"
        )

    # ── Timing accumulators ────────────────────────────────────────────
    T = defaultdict(float)   # cumulative seconds per component
    N = defaultdict(int)     # call count per component

    # ── Progress bar ──────────────────────────────────────────────────
    step_range = range(cfg.max_steps)
    pbar       = tqdm(step_range, desc="SIEVE", dynamic_ncols=True) \
                 if _use_tqdm else step_range

    t_wall = time.time()
    pf: dict = {}

    # Establish the baseline before sampling the first interval's weights.
    initial_ppl = evaluate(model, val_ds, device, max_batches=50)
    history["val_ppl"].append(initial_ppl)
    last_val_ppl = initial_ppl
    if sieve is not None:
        sieve.on_eval(math.log(initial_ppl))
    last_eval_step = 0

    stopped_early = False
    n_loop_steps = 0

    for step in pbar:
        n_loop_steps = step + 1

        # ── Hook 1: sample weights at an evaluation boundary ──────────────
        t0 = perf_counter()
        if sieve is not None:
            sieve.step_begin(step, cfg.max_steps, prev_loss, prev_grad)
        T["bandit"] += perf_counter() - t0
        N["bandit"] += 1

        # ── Data: batch sampling + H2D transfer ───────────────────────
        t0 = perf_counter()
        batch      = train_ds.sample_batch(cfg.batch_size, device)
        input_ids  = batch["input_ids"]
        ref_losses = batch.get("ref_losses")
        T["data"] += perf_counter() - t0
        N["data"] += 1

        # ── Forward — labels=None so we control the loss ───────────────
        optim.zero_grad(set_to_none=True)
        t0 = perf_counter()
        with dev_cfg.autocast:
            out = model(input_ids=input_ids, labels=None, use_cache=False)
        logits = out.logits
        T["fwd"] += perf_counter() - t0
        N["fwd"] += 1
        t1 = perf_counter()
        mask = sieve.score_and_mask(logits.detach(), input_ids, ref_losses) if sieve is not None else None
        t_score_step = perf_counter() - t1
        if sieve is not None:
            T["score"] += t_score_step
            N["score"] += 1
            T["score_rescore"] += t_score_step
            N["score_rescore"] += 1
        loss = selective_loss(logits.float(), input_ids, mask)

        # ── Backward + optimizer ──────────────────────────────────────
        t0 = perf_counter()
        loss.backward()
        prev_grad = torch.nn.utils.clip_grad_norm_(
            model.parameters(), cfg.grad_clip
        ).item()
        optim.step()
        # Synchronising MPS every step hurts throughput significantly.
        # Keep it only when profiling is explicitly enabled.
        if device.type == "mps" and cfg.profile_dataloader:
            torch.mps.synchronize()
        T["bwd"] += perf_counter() - t0
        N["bwd"] += 1

        prev_loss = loss.item()

        # ── Logging ───────────────────────────────────────────────────
        if step % cfg.log_interval == 0:
            elapsed = time.time() - t_wall
            log = {
                "step":    step,
                "loss":    prev_loss,
                "grad":    prev_grad,
                "elapsed": elapsed,
            }
            if sieve is not None:
                log.update(sieve.log_dict())
            log_fn(log)
            history["loss"].append(prev_loss)
            if _has_bandit(sieve):
                history["weights"].append(
                    sieve.bandit.expected_weights(
                        sieve.bandit._current_bin
                    ).tolist()
                )
                variance = _bandit_variance_snapshot(sieve)
                if variance is not None:
                    history["variance"].append(variance)

        # ── tqdm postfix — live overhead monitoring ────────────────────
        if _use_tqdm and step % 50 == 0 and step > 0:
            total_t   = sum(T[k] for k in ["data","bandit","fwd","score","bwd"])
            score_pct = 100.0 * T["score"] / max(total_t, 1e-9)

            pf = {
                "loss":    f"{prev_loss:.3f}",
                "ppl":     f"{last_val_ppl:.1f}",
                "score%":  f"{score_pct:.2f}",
            }
            if _has_bandit(sieve):
                ew = sieve.bandit.expected_weights(sieve.bandit._current_bin)
                names = list(getattr(sieve, "scorer_names", ["S_E", "S_U", "S_L", "S_D"]))
                # dominant strategy — the paper's key interpretability signal
                pf["dom"] = names[int(ew.argmax())]
            pbar.set_postfix(pf)

        # ── Evaluation + Hook 3 ───────────────────────────────────────
        if (step + 1) % cfg.eval_interval == 0:
            completed = step + 1
            val_ppl = evaluate(model, val_ds, device, max_batches=50)
            last_val_ppl = val_ppl
            last_eval_step = step + 1
            history["val_ppl"].append(val_ppl)
            print(f"\n  step {completed:5d}  val_ppl={val_ppl:.2f}"
                  + (f"  dom={pf.get('dom', '?')}" if _has_bandit(sieve) else ""))

            w_snap: Optional[torch.Tensor] = None
            ent_snap: Optional[float] = None
            if _has_bandit(sieve):
                b = sieve.bandit._current_bin
                ew = sieve.bandit.expected_weights(b)
                w_snap = torch.tensor(ew, dtype=torch.float32)
                ent_snap = _dirichlet_entropy(sieve.bandit, b)
                if ent_snap is None:
                    ent_snap = float(sieve.bandit.posterior_entropy(b))

            if patience is not None:
                patience.update(completed, val_ppl, w_snap, ent_snap)

            if sieve is not None:
                sieve.on_eval(math.log(val_ppl))   # pass log-PPL as proxy loss

            ev_log = {"step": completed, "val/ppl": val_ppl}
            if patience is not None:
                ev_log.update(patience.log_dict())
            log_fn(ev_log)

            if (
                save_dir is not None
                and save_interval > 0
                and completed % save_interval == 0
            ):
                is_best = val_ppl < best_ppl
                if is_best:
                    best_ppl = val_ppl
                    best_step = completed
                _save_checkpoint(
                    model,
                    completed,
                    save_dir,
                    val_ppl,
                    bandit_state=None,
                    is_best=is_best,
                )
                if max_keep > 0:
                    prot: set[str] = set()
                    if best_step >= 0:
                        prot.add(f"ckpt-step{best_step:06d}")
                    _prune_checkpoints(save_dir, max_keep, prot)

            if patience is not None:
                stop, reason = patience.should_stop(completed)
                if stop:
                    print(f"\n[patience] early stop at step {completed}: {reason}")
                    history["early_stop"] = True
                    history["early_stop_step"] = completed
                    history["early_stop_reason"] = reason
                    stopped_early = True
                    break

    # Final evaluation (skipped if we already evaluated at the stop step)
    if stopped_early or last_eval_step == n_loop_steps:
        val_ppl = history["val_ppl"][-1]
        print(f"  [final] val_ppl={val_ppl:.2f}  (already evaluated)")
        log_fn({"step": n_loop_steps, "val/ppl": val_ppl})
        final_ckpt_step = n_loop_steps
    else:
        val_ppl = evaluate(model, val_ds, device, max_batches=50)
        history["val_ppl"].append(val_ppl)
        if sieve is not None:
            sieve.on_eval(math.log(val_ppl))
        print(f"  [final] val_ppl={val_ppl:.2f}")
        log_fn({"step": cfg.max_steps, "val/ppl": val_ppl})
        final_ckpt_step = cfg.max_steps

    if save_dir is not None:
        bandit_state = _collect_bandit_state(sieve, history)
        is_best = val_ppl < best_ppl
        if is_best:
            best_ppl = val_ppl
            best_step = final_ckpt_step
        _save_checkpoint(
            model,
            final_ckpt_step,
            save_dir,
            val_ppl,
            bandit_state=bandit_state,
            is_best=is_best,
        )
        print(
            f"[ckpt] final at step {final_ckpt_step}, "
            f"best at step {best_step} (val_ppl={best_ppl:.2f})"
        )

    # ── Timing summary at end ─────────────────────────────────────────
    variance = _bandit_variance_snapshot(sieve)
    if variance is not None:
        history["variance"].append(variance)
    _print_timing_summary(T, N, n_loop_steps)
    _print_variance_summary(history)
    history["timing"] = dict(T)
    if save_dir is not None:
        history["best_step"] = best_step
        history["best_ppl"] = best_ppl

    return history


def _print_variance_summary(history: dict) -> None:
    """Emit end-of-run variance analysis into stdout/train.log."""
    if not history.get("variance"):
        return
    v = history["variance"][-1]
    print("\n── Variance analysis ────────────────────────────────")
    print(
        f"  context_bin={v['context_bin']}  "
        f"posterior_var_sum={v['variance_sum']:.6f}  "
        f"weight_var={v['weight_variance']:.6f}  "
        f"weight_gap={v['weight_gap']:.6f}"
    )
    print(
        "  posterior_var: "
        f"S_E={v['variance_S_E']:.6f}  "
        f"S_U={v['variance_S_U']:.6f}  "
        f"S_L={v['variance_S_L']:.6f}  "
        f"S_D={v['variance_S_D']:.6f}"
    )
    print("─" * 52)


def _print_timing_summary(T: dict, N: dict, max_steps: int) -> dict:
    """
    Per-component timing table — produces numbers for paper Table efficiency.

    Expected result on A100 40GB (GPT-2 Large, B=8, T=512):
      data:    ~1-2 ms/step   (memmap is fast)
      bandit:  ~0.1 ms/step   (numpy dirichlet sample)
      fwd:     ~80-120 ms/step
      score:   ~0.5-1.5 ms/step amortised (5ms per rescore / Δ=50)
      bwd:     ~100-150 ms/step
      score%:  ~0.3-0.8% of total  ← paper claims ~1%, this is even better
    """
    components = ["data", "bandit", "fwd", "score", "bwd"]
    total      = sum(T.get(k, 0.0) for k in components)
    if total < 1e-9 or max_steps == 0:
        return {}

    print("\n── Timing breakdown ─────────────────────────────────")
    print(f"  {'component':<10} {'ms/step':>9}  {'% total':>8}  {'calls':>6}")
    print(f"  {'─'*10} {'─'*9}  {'─'*8}  {'─'*6}")
    result = {}
    for k in components:
        t   = T.get(k, 0.0)
        ms  = 1000.0 * t / max_steps
        pct = 100.0 * t / total
        n   = N.get(k, 0)
        print(f"  {k:<10} {ms:>9.2f}  {pct:>7.1f}%  {n:>6}")
        result[k] = {"ms_per_step": ms, "pct": pct}
    print(f"  {'─'*10} {'─'*9}  {'─'*8}  {'─'*6}")
    print(f"  {'TOTAL':<10} {1000.0*total/max_steps:>9.2f}  {'100.0%':>8}")

    # Rescore-step decomposition — the headline number for the paper
    if N.get("score_rescore", 0) > 0:
        ms_per_rescore   = 1000.0 * T["score_rescore"] / N["score_rescore"]
        amortised_ms     = 1000.0 * T["score_rescore"] / max_steps
        amortised_pct    = 100.0  * T["score_rescore"] / total
        rescore_fraction = 100.0  * N["score_rescore"] / max_steps
        print(f"\n  Rescore steps:          {N['score_rescore']}/{max_steps}"
              f" ({rescore_fraction:.1f}% of steps)")
        print(f"  Cost per rescore:       {ms_per_rescore:.2f} ms")
        print(f"  Amortised scorer:       {amortised_ms:.3f} ms/step "
              f"({amortised_pct:.2f}% of total)")
        status = "✓ within paper claim" if amortised_pct < 2.0 else "✗ too high"
        print(f"  Paper claims ~1%:       {status}")

    print("─" * 52)
    return result


def _make_logger(cfg: SieveConfig):
    """Returns a logging function — wandb if configured, else print."""
    try:
        if cfg.wandb_project:
            import wandb
            wandb.init(
                project = cfg.wandb_project,
                name    = cfg.run_name,
                config  = cfg.__dict__,
            )
            def log(d): wandb.log(d)
        else:
            raise ImportError
    except Exception:
        def log(d):
            step = d.get("step", "")
            loss = d.get("loss", "")
            ppl  = d.get("val/ppl", "")
            se   = d.get("sieve/w_S_E", "")
            su   = d.get("sieve/w_S_U", "")
            var  = d.get("sieve/variance_sum", "")
            if loss:
                print(f"step {step:5d}  loss={loss:.4f}  "
                      f"grad={d.get('grad', 0):.3f}"
                      + (f"  S_E={se:.3f} S_U={su:.3f}" if se != "" else "")
                      + (f"  var={var:.6f}" if var != "" else ""))
            if ppl:
                print(f"          val_ppl={ppl:.2f}")
    return log
