"""
sieve/loop.py

Framework-free SIEVE training loop.
Works on GPT-2 (any size), TinyLlama, any HF CausalLM.
Three hooks are explicit and labelled.

PATCH: Fine-grained per-component timing added.
  - tqdm progress bar with live loss/ppl/overhead postfix
  - Separate timers: t_data, t_bandit, t_fwd, t_score, t_bwd
  - End-of-run summary table for paper's ~1% overhead claim
  - Profiling adds ~0.05ms/step (negligible) via perf_counter
"""
from __future__ import annotations
import math, time
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
from .reference import reference_nll


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


def _lr_factor(step: int, cfg: SieveConfig) -> float:
    """Linear warmup over warmup_steps, then cosine decay to min_lr_ratio at max_steps."""
    if cfg.warmup_steps > 0 and step < cfg.warmup_steps:
        return (step + 1) / cfg.warmup_steps
    span = max(1, cfg.max_steps - cfg.warmup_steps)
    progress = min(1.0, (step - cfg.warmup_steps) / span)
    return cfg.min_lr_ratio + (1.0 - cfg.min_lr_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress))


def evaluate(
    model:       torch.nn.Module,
    dataset:     MemmapDataset,
    device:      torch.device,
    max_batches: Optional[int] = 50,
) -> float:
    """Validation PPL over at most max_batches batches of 8 sequences (None = whole set)."""
    model.eval()
    total_nll, total_tok, n = 0.0, 0, 0
    with torch.no_grad():
        for batch in dataset.val_batches(8, device):
            ids  = batch["input_ids"]
            B, T = ids.shape
            # Same bf16 autocast as training; cast logits to float32 for CE
            with torch.autocast(device.type, dtype=torch.bfloat16,
                                enabled=(device.type == 'cuda')):
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


def train(
    model:    torch.nn.Module,
    train_ds: MemmapDataset,
    val_ds:   MemmapDataset,
    cfg:      SieveConfig,
    dev_cfg:  DeviceCfg,
    sieve:    Optional[SieveState] = None,
    ref_model: Optional[torch.nn.Module] = None,
) -> dict:
    """
    Main SIEVE training loop with fine-grained profiling.

    Three hooks (labelled):
      Hook 1  step_begin()     — bandit context encode + Dirichlet sample
      Hook 2  score_and_mask() — token scoring + topk mask (every step)
      Hook 3  on_eval()        — r = max(0, L_prev − L_val) → α ← α + r·w

    Timing breakdown (all measured separately):
      t_data:   sample_batch + H2D transfer
      t_bandit: hook 1 (context encode + sampling) — should be <0.5ms
      t_fwd:    model forward (no labels — scorer uses logits directly)
      t_score:  hook 2 (scoring + topk) — every step, on the current batch
      t_bwd:    backward + gradient clip + optimizer step

    Expected overhead for SIEVE vs CLM:
      t_bandit: ~0.1ms/step (pure Python, O(K) numpy)
      t_score:  one scoring pass per step. The mask is never reused across
                steps: each step samples a new random batch.
    """
    try:
        from tqdm import tqdm
        _use_tqdm = True
    except ImportError:
        _use_tqdm = False

    device = dev_cfg.device
    optim  = torch.optim.AdamW(
        model.parameters(), lr=cfg.lr, betas=(cfg.beta1, cfg.beta2),
        weight_decay=cfg.weight_decay,
    )
    sched  = torch.optim.lr_scheduler.LambdaLR(optim, lambda s: _lr_factor(s, cfg))

    history      = {"loss": [], "log_steps": [], "lr": [],
                    "val_ppl": [], "eval_steps": [],
                    "weights": [], "variance": []}
    prev_loss    = 1.0
    prev_grad    = 0.0
    last_val_ppl = float("nan")
    log_fn = _make_logger(cfg)

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

    # Full CLM on CUDA uses the model's internal (Liger-fused) loss; any
    # selector needs the logits, so it always takes the standard path.
    use_fused_loss = (dev_cfg.backend == "cuda")

    def _build_masked_labels(input_ids: torch.Tensor,
                             mask: Optional[torch.Tensor],
                             ignore_index: int = -100) -> torch.Tensor:
        B, T = input_ids.shape
        labels = input_ids.clone()
        # First token never contributes (standard CLM shift)
        labels[:, 0] = ignore_index
        if mask is not None:
            m = mask[:, 1:]
            labels[:, 1:] = torch.where(m, labels[:, 1:], torch.full_like(labels[:, 1:], ignore_index))
        return labels

    stopped_early = False
    n_loop_steps = 0

    for step in pbar:
        n_loop_steps = step + 1

        # ── Hook 1: sample bandit weights if rescore due ──────────────
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

        # ── Reference losses for S_E, on the current batch ────────────
        if ref_model is not None:
            t0 = perf_counter()
            ref_losses = reference_nll(ref_model, input_ids, dev_cfg.autocast)
            T["ref"] += perf_counter() - t0
            N["ref"] += 1

        # ── Forward ───────────────────────────────────────────────────
        optim.zero_grad(set_to_none=True)
        t0 = perf_counter()

        if use_fused_loss and sieve is None:
            # Full CLM on CUDA: model's internal loss (patched by Liger),
            # logits are never materialised.
            labels = _build_masked_labels(input_ids, None)
            with dev_cfg.autocast:
                out = model(input_ids=input_ids, labels=labels,
                            output_hidden_states=False, use_cache=False, return_dict=True)
            T["fwd"] += perf_counter() - t0
            N["fwd"] += 1
            loss = out.loss
        else:
            # One forward with labels=None: the same logits are used to
            # score the current batch and to compute the loss.
            with dev_cfg.autocast:
                out = model(input_ids=input_ids, labels=None, use_cache=False)
            logits = out.logits
            T["fwd"] += perf_counter() - t0
            N["fwd"] += 1

            # ── Hook 2: score the current batch and build its mask ──
            # Done at every step: batches are sampled at random offsets,
            # so a mask computed on another batch selects unrelated tokens.
            mask = None
            if sieve is not None:
                t1 = perf_counter()
                with torch.no_grad():
                    mask = sieve.score_and_mask(logits.detach(), input_ids, ref_losses)
                T["score"] += perf_counter() - t1
                N["score"] += 1

            # ── Loss: mean NLL over the selected tokens (Eq. train_loss)
            loss = selective_loss(logits, input_ids, mask)

        # ── Backward + optimizer ──────────────────────────────────────
        t0 = perf_counter()
        loss.backward()
        prev_grad = torch.nn.utils.clip_grad_norm_(
            model.parameters(), cfg.grad_clip
        ).item()
        lr_now = optim.param_groups[0]["lr"]
        optim.step()
        sched.step()
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
                "lr":      lr_now,
                "elapsed": elapsed,
            }
            if sieve is not None:
                log.update(sieve.log_dict())
            log_fn(log)
            history["loss"].append(prev_loss)
            history["log_steps"].append(step)
            history["lr"].append(lr_now)
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
            total_t   = sum(T[k] for k in ["data","ref","bandit","fwd","score","bwd"])
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
        # Evaluate after every eval_interval completed steps, i.e. before
        # the next step re-samples the weights: the reward then credits the
        # mixture that was active over the last rescore window. (Evaluating
        # at step % eval_interval == 0 would credit a mixture sampled at the
        # start of that same step, which had driven a single update.)
        done = step + 1
        if done % cfg.eval_interval == 0:
            val_ppl = evaluate(model, val_ds, device, max_batches=cfg.eval_batches)
            last_val_ppl = val_ppl
            history["val_ppl"].append(val_ppl)
            history["eval_steps"].append(done)
            print(f"\n  step {done:5d}  val_ppl={val_ppl:.2f}"
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
                patience.update(done, val_ppl, w_snap, ent_snap)

            if sieve is not None:
                # Validation loss = mean token NLL = log(PPL)
                sieve.on_eval(math.log(val_ppl), step=done)

            ev_log = {"step": done, "val/ppl": val_ppl}
            if patience is not None:
                ev_log.update(patience.log_dict())
            log_fn(ev_log)

            if patience is not None:
                stop, reason = patience.should_stop(done)
                if stop:
                    print(f"\n[patience] early stop at step {done}: {reason}")
                    history["early_stop"] = True
                    history["early_stop_step"] = done
                    history["early_stop_reason"] = reason
                    stopped_early = True
                    break

    # ── Final evaluation on the whole validation set (reported number) ──
    final_ppl = evaluate(model, val_ds, device, max_batches=None)
    history["final_val_ppl"] = final_ppl
    history["final_val_loss"] = math.log(final_ppl)
    history["n_steps"] = n_loop_steps
    print(f"  [final] val_ppl={final_ppl:.3f}  (full validation set, {n_loop_steps} steps)"
          + ("  (patience early stop)" if stopped_early else ""))
    log_fn({"step": n_loop_steps, "val/ppl": final_ppl})

    # ── Run record + timing summary ───────────────────────────────────
    variance = _bandit_variance_snapshot(sieve)
    if variance is not None:
        history["variance"].append(variance)
    if sieve is not None:
        history["sample_log"] = list(getattr(sieve, "sample_log", []))
        history["reward_log"] = list(getattr(sieve, "reward_log", []))
    if _has_bandit(sieve):
        history["alpha_final"] = sieve.bandit.alpha[0].tolist()
        history["expected_weights_final"] = sieve.bandit.expected_weights().tolist()
        history["scorer_names"] = list(getattr(sieve, "scorer_names", []))
    _print_timing_summary(T, N, n_loop_steps)
    _print_variance_summary(history)
    history["timing"] = dict(T)
    history["timing_counts"] = dict(N)
    history["wall_time_s"] = time.time() - t_wall

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
      score:   scoring + top-k on the current batch, at every step
      bwd:     ~100-150 ms/step
    """
    components = ["data", "ref", "bandit", "fwd", "score", "bwd"]
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

    # Scoring check — selection baselines must score 100% of steps
    if N.get("score", 0) > 0:
        ms_per_score   = 1000.0 * T["score"] / N["score"]
        score_fraction = 100.0  * N["score"] / max_steps
        print(f"\n  Scored steps:           {N['score']}/{max_steps}"
              f" ({score_fraction:.1f}% of steps)")
        print(f"  Cost per scored step:   {ms_per_score:.2f} ms")

    print("─" * 52)
    return result


def _make_logger(cfg: SieveConfig):
    """Return a simple stdout logger."""
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
