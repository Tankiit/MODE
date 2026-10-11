"""
Training loop (paper Algorithm 1).

Each step: sample mixture weights if due → draw a batch → forward →
(score the current batch, top-k mask) → loss on the selected tokens →
AdamW step. After every `eval.interval` completed steps, the validation
loss is measured and, before the next step re-samples the weights, the
selector credits the mixture that was active over the last block.
"""
from __future__ import annotations

import logging
import time
from typing import Optional

import torch

from ..config import RunConfig
from ..data import MemmapDataset
from ..models import autocast, reference_nll
from ..scorers import ScoringInputs
from ..selection import TokenSelector
from .evaluate import evaluate, perplexity
from .loss import masked_mean_nll
from .optim import build_optimizer

log = logging.getLogger("sieve")


def train(model: torch.nn.Module, selector: Optional[TokenSelector],
          ref_model: Optional[torch.nn.Module], train_ds: MemmapDataset,
          val_ds: MemmapDataset, cfg: RunConfig, device: torch.device) -> dict:
    if selector is not None and selector.needs_reference and ref_model is None:
        raise ValueError("the selector uses excess_loss but no reference model was given")
    tc, ec = cfg.training, cfg.eval
    optim, sched = build_optimizer(model, tc)
    ac = lambda: autocast(device, tc.autocast)  # noqa: E731
    h: dict = {"log_steps": [], "loss": [], "lr": [], "grad_norm": [], "expected_weights": [],
               "eval_steps": [], "val_loss": [], "val_ppl": [],
               "slots": selector.slots if selector else [], "n_scored_steps": 0}
    t_start = time.time()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    model.train()

    for step in range(tc.steps):
        if selector is not None:
            selector.begin_step(step)
        input_ids = train_ds.sample_batch(cfg.data.batch_size, device)

        optim.zero_grad(set_to_none=True)
        with ac():
            logits = model(input_ids=input_ids, use_cache=False).logits

        mask = None
        if selector is not None:
            ref = reference_nll(ref_model, input_ids, ac()) if selector.needs_reference else None
            mask = selector.select(ScoringInputs(input_ids, logits.detach().float(), ref))
            h["n_scored_steps"] += 1

        loss = masked_mean_nll(logits, input_ids, mask)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), tc.grad_clip).item()
        lr = optim.param_groups[0]["lr"]
        optim.step()
        sched.step()

        if step % cfg.output.log_interval == 0:
            ew = selector.mixer.expected_weights().tolist() if selector else []
            for k, v in (("log_steps", step), ("loss", loss.item()), ("lr", lr),
                         ("grad_norm", grad_norm), ("expected_weights", ew)):
                h[k].append(v)
            log.info("step %6d  loss %.4f  lr %.2e  grad %.3f%s", step, loss.item(), lr, grad_norm,
                     ("  E[w] " + " ".join(f"{s}={w:.3f}" for s, w in zip(h["slots"], ew))) if ew else "")

        done = step + 1
        if done % ec.interval == 0:
            val = evaluate(model, val_ds, device, ec.batch_size, ec.batches, ac())
            h["eval_steps"].append(done)
            h["val_loss"].append(val)
            h["val_ppl"].append(perplexity(val))
            log.info("step %6d  val_ppl %.3f", done, perplexity(val))
            if selector is not None:
                selector.on_validation(val, done)

    final = evaluate(model, val_ds, device, ec.batch_size, None, ac())
    h.update({
        "final_val_loss": final,
        "final_val_ppl": perplexity(final),
        "n_steps": tc.steps,
        "wall_time_s": time.time() - t_start,
        "peak_memory_gb": (torch.cuda.max_memory_allocated(device) / 2**30
                           if device.type == "cuda" else None),
    })
    if selector is not None:
        h.update({"sample_log": selector.sample_log, "reward_log": selector.reward_log,
                  "expected_weights_final": selector.mixer.expected_weights().tolist(),
                  "mixer_state": selector.mixer.state()})
    log.info("final val_ppl %.3f (full validation set, %d steps, %.0f s)",
             h["final_val_ppl"], tc.steps, h["wall_time_s"])
    return h
