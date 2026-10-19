"""Local, full-logit scorer discrimination from a shared optimizer snapshot."""
import copy

import torch
import torch.nn.functional as F

from .loss import selective_loss
from .mask import MaskCache
from .scorers import TokenScorers


@torch.no_grad()
def individual_masks(logits, ids, frequency, alpha, generator, reference_logits=None):
    """All scores use the same checkpoint and causal contexts.

    Excess loss is included only with an explicit context-matched reference.
    Raw NLL is never silently relabelled as excess loss.
    """
    if frequency is None:
        raise ValueError("A corpus frequency table is required")
    scorers = TokenScorers(freq_table=frequency)
    nll = F.cross_entropy(logits[:, :-1].float().flatten(0, 1),
                          ids[:, 1:].flatten(), reduction="none").reshape_as(ids[:, 1:])
    scores = {
        "nll": F.pad(nll, (1, 0)),
        "entropy": scorers.score_entropy(logits.float()),
        "spread": scorers.score_logit_spread(logits.float()),
        "rarity": scorers.score_inv_freq(ids, ids.device),
        "random": torch.rand(ids.shape, generator=generator).to(ids.device),
    }
    if reference_logits is not None:
        ref = F.cross_entropy(reference_logits[:, :-1].float().flatten(0, 1),
                              ids[:, 1:].flatten(), reduction="none").reshape_as(nll)
        scores["excess"] = F.pad((nll-ref).clamp_min(0), (1, 0))
    cache = MaskCache(alpha, 1)
    masks = {name: cache.update(score, 0) for name, score in scores.items()}
    masks["clm"] = torch.ones_like(ids, dtype=torch.bool)
    masks["clm"][:, 0] = False
    return masks


def local_continuations(model, optimizer, optimizer_factory, batches, heldout,
                        frequency, alpha=0.7, seed=42, reference_model=None,
                        grad_clip=1.0):
    """Return paired losses and initial-checkpoint masks for every available arm.

    optimizer_factory receives cloned model parameters and must construct the
    same optimizer class. Its state and hyperparameters are then overwritten by
    the supplied optimizer snapshot. Inputs are fixed unpadded token batches;
    heldout must be independent of the controller's data. The caller owns this
    split (the Tinker CLI enforces a disjoint validation suffix).
    """
    if not batches or not heldout:
        raise ValueError("Provide training batches and independent evaluation batches")
    device = next(model.parameters()).device
    snapshot = copy.deepcopy(optimizer.state_dict())
    if reference_model is not None:
        reference = copy.deepcopy(reference_model).eval()
    else:
        reference = None

    @torch.no_grad()
    def evaluate(m):
        m.eval()
        total, count = 0.0, 0
        for ids in heldout:
            logits = m(input_ids=ids, use_cache=False).logits.float()
            total += float(F.cross_entropy(logits[:, :-1].flatten(0, 1),
                                          ids[:, 1:].flatten(), reduction="sum"))
            count += ids[:, 1:].numel()
        return total/count

    # Common-checkpoint diagnostics precede all continuations.
    probe = copy.deepcopy(model).eval()
    generator = torch.Generator().manual_seed(seed+1000)
    masks = {}
    with torch.no_grad():
        for ids in batches:
            logits = probe(input_ids=ids, use_cache=False).logits
            ref = reference(input_ids=ids, use_cache=False).logits if reference is not None else None
            for name, mask in individual_masks(logits, ids, frequency, alpha, generator, ref).items():
                masks.setdefault(name, []).append(mask[:, 1:].cpu())
    masks = {name: torch.cat(values) for name, values in masks.items()}
    del probe
    results = {}
    cuda_devices = list(range(torch.cuda.device_count())) if device.type == "cuda" else []
    for arm in masks:
        # Isolate dropout RNG and restore the exact model/optimizer for each arm.
        with torch.random.fork_rng(devices=cuda_devices):
            torch.manual_seed(seed)
            if cuda_devices:
                torch.cuda.manual_seed_all(seed)
            current = copy.deepcopy(model)
            opt = optimizer_factory(current.parameters())
            if type(opt) is not type(optimizer):
                raise ValueError("optimizer_factory must preserve optimizer class")
            opt.load_state_dict(copy.deepcopy(snapshot))
            initial = evaluate(current)
            current.train()
            generator = torch.Generator().manual_seed(seed+1000)
            for ids in batches:
                opt.zero_grad(set_to_none=True)
                logits = current(input_ids=ids, use_cache=False).logits
                with torch.no_grad():
                    ref = reference(input_ids=ids, use_cache=False).logits if reference is not None else None
                    mask = individual_masks(logits.detach(), ids, frequency, alpha, generator, ref)[arm]
                loss = selective_loss(logits.float(), ids, mask)
                if not torch.isfinite(loss):
                    raise ValueError("Non-finite continuation loss")
                loss.backward()
                if grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(current.parameters(), grad_clip)
                opt.step()
            final = evaluate(current)
            results[arm] = {"initial_nll": initial, "final_nll": final,
                            "improvement": initial-final}
            del current, opt
    return results, masks
