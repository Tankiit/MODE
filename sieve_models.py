from typing import Optional

import torch
from torch import Tensor


class StateEncoder:
    def __init__(self, dim: int = 16):
        self.dim = dim
        self.loss_history: list[float] = []
        self.grad_history: list[float] = []
        self.strategy_rewards: list[tuple[int, float]] = []

    def encode(self, step: int, total_steps: int,
               train_loss: float, grad_norm: float = 0.0) -> Tensor:
        ctx = torch.zeros(self.dim, dtype=torch.float64)
        p = step / max(total_steps, 1)

        ctx[0] = float(p < 0.10)
        ctx[1] = float(p < 0.30)
        ctx[2] = float(0.30 <= p < 0.70)
        ctx[3] = float(p >= 0.70)

        self.loss_history.append(train_loss)
        if len(self.loss_history) >= 5:
            r = self.loss_history[-5:]
            slope = (r[-1] - r[0]) / 4.0
            var = sum((x - sum(r)/5)**2 for x in r) / 5
            ctx[4] = float(slope < -0.01)
            ctx[5] = float(abs(slope) < 0.005)
            ctx[6] = float(slope > 0.01)
            ctx[7] = float(var > 0.05)

        if grad_norm > 0:
            self.grad_history.append(grad_norm)
            if len(self.grad_history) >= 3:
                recent = self.grad_history[-3:]
                mn = sum(recent) / len(recent)
                ctx[8]  = float(mn > 10.0)
                ctx[9]  = float(mn < 0.01)
                ctx[10] = float(grad_norm > 2 * mn)
                ctx[11] = float(len(self.grad_history) > 5 and
                               self.grad_history[-1] < self.grad_history[-5])

        if len(self.strategy_rewards) >= 3:
            bucket: dict[int, list[float]] = {}
            for idx, rw in self.strategy_rewards[-10:]:
                bucket.setdefault(idx, []).append(rw)
            for k in range(min(4, self.dim - 12)):
                if k in bucket and bucket[k]:
                    ctx[12 + k] = float(sum(bucket[k]) / len(bucket[k]) > 0)

        return ctx

    def record_reward(self, strategy_idx: int, reward: float):
        self.strategy_rewards.append((strategy_idx, reward))


class DirichletTS:
    def __init__(self, cfg):
        K = cfg.num_strategies
        self.K = K
        self.gamma = cfg.discount_gamma
        self.adaptive = cfg.adaptive_gamma
        self.num_bins = cfg.num_context_bins

        self.alphas = {
            i: torch.ones(K, dtype=torch.float64) * cfg.dirichlet_prior
            for i in range(self.num_bins)
        }

        self.prev_variance = None
        self.breakpoints: list[int] = []

        self.weight_log: list[list[float]] = []
        self.round = 0

    def _bin(self, ctx: Tensor) -> int:
        bits = (ctx[:4] > 0.5).int()
        idx = bits[0]*8 + bits[1]*4 + bits[2]*2 + bits[3]
        return min(idx.item(), self.num_bins - 1)

    def _detect_transition(self, alpha: Tensor) -> bool:
        total = alpha.sum()
        var = (alpha * (total - alpha) / (total**2 * (total + 1))).sum().item()
        is_transition = False
        if self.prev_variance is not None and var > 1.5 * self.prev_variance:
            is_transition = True
            self.breakpoints.append(self.round)
        self.prev_variance = var
        return is_transition

    def sample_weights(self, ctx: Tensor) -> Tensor:
        b = self._bin(ctx)
        alpha = self.alphas[b].clamp(min=0.01)

        try:
            w = torch.distributions.Dirichlet(alpha).sample()
        except Exception:
            w = torch.ones(self.K, dtype=torch.float64) / self.K

        self.weight_log.append(w.tolist())
        self.round += 1
        return w.float()

    def update(self, ctx: Tensor, weights: Tensor, reward: float):
        b = self._bin(ctx)

        g = self.gamma
        if self.adaptive and self._detect_transition(self.alphas[b]):
            g = self.gamma * 0.8

        self.alphas[b] *= g

        if reward > 0:
            self.alphas[b] += weights.double() * reward * 2.0
        else:
            self.alphas[b] += 0.1

        self.alphas[b].clamp_(min=0.01)


class SieveSelector:
    def __init__(self, cfg, device: str = "cuda"):
        self.cfg = cfg
        self.device = device

        self.encoder = StateEncoder(cfg.context_dim)
        self.bandit = DirichletTS(cfg)

        self.cached_mask: Optional[Tensor] = None
        self.cached_at_step: int = -9999
        self.current_weights: Optional[Tensor] = None
        self.current_context: Optional[Tensor] = None

        self.ref_logits: Optional[Tensor] = None

        self.last_val_loss: Optional[float] = None
        self.scoring_round: int = 0

        self.tokens_scored: int = 0
        self.tokens_trained: int = 0

    def needs_rescore(self, step: int) -> bool:
        if self.cfg.mode == "offline" and self.scoring_round > 0:
            return False
        if self.cfg.mode == "online":
            return True
        return (step - self.cached_at_step) >= self.cfg.rescore_every

    @torch.no_grad()
    def score_and_cache(self, step: int, total_steps: int,
                        model, inputs: Tensor, targets: Tensor,
                        sliding_window_num_blocks=None,
                        train_loss: float = 0.0,
                        grad_norm: float = 0.0) -> Tensor:
        seq_len = targets.size(0)
        num_select = max(1, int(seq_len * self.cfg.select_ratio))

        ctx = self.encoder.encode(step, total_steps, train_loss, grad_norm)
        self.current_context = ctx

        weights = self.bandit.sample_weights(ctx)
        self.current_weights = weights

        model.eval()

        logits = self._get_logits(model, inputs, targets, sliding_window_num_blocks)

        model.train()

        if self.ref_logits is None and self.cfg.snapshot_reference:
            self.ref_logits = logits.detach().clone()

        from sieve_scorers import compute_all_scores, normalize_scores, combine_scores
        scores = compute_all_scores(logits, targets, self.ref_logits)
        scores = normalize_scores(scores)

        combined = combine_scores(scores, weights, self.cfg.strategy_names)

        _, top_idx = torch.topk(combined, num_select)
        mask = torch.zeros(seq_len, dtype=torch.bool, device=self.device)
        mask[top_idx] = True

        self.cached_mask = mask
        self.cached_at_step = step
        self.scoring_round += 1
        self.tokens_scored += seq_len

        return mask

    def _get_logits(self, model, inputs, targets, sw_blocks=None) -> Tensor:
        if sw_blocks is None:
            # HF native path: standard forward, logits returned directly
            model.eval()
            with torch.no_grad():
                inp = inputs.unsqueeze(0) if inputs.dim() == 1 else inputs
                outputs = model(input_ids=inp)
                logits = outputs.logits
            model.train()
            if logits.dim() == 3:
                logits = logits.squeeze(0)
            return logits.detach()

        # ramenGPT path: hook on lm_head + softcap
        captured = {}

        def hook_fn(module, input, output):
            captured["logits"] = output.detach()

        handle = model.lm_head.register_forward_hook(hook_fn)

        try:
            _ = model(inputs, targets, sw_blocks)
        finally:
            handle.remove()

        logits = captured.get("logits")
        if logits is None:
            seq_len = targets.size(0)
            V = model.lm_head.weight.size(0)
            return torch.randn(seq_len, V, device=self.device)

        logits = model._logits_softcap_scale * torch.sigmoid(
            (logits + model._logits_softcap_shift) / model._logits_softcap_divisor
        )

        if logits.dim() == 3:
            logits = logits.squeeze(0)

        return logits

    def get_token_mask(self, step: int, total_steps: int,
                       model, inputs: Tensor, targets: Tensor,
                       sliding_window_num_blocks=None,
                       train_loss: float = 0.0,
                       grad_norm: float = 0.0) -> Tensor:
        if self.needs_rescore(step):
            mask = self.score_and_cache(
                step, total_steps, model, inputs, targets,
                sliding_window_num_blocks, train_loss, grad_norm
            )
        else:
            mask = self.cached_mask

        if mask is None or mask.size(0) != targets.size(0):
            seq_len = targets.size(0)
            num_select = max(1, int(seq_len * self.cfg.select_ratio))
            idx = torch.randperm(seq_len, device=self.device)[:num_select]
            mask = torch.zeros(seq_len, dtype=torch.bool, device=self.device)
            mask[idx] = True

        self.tokens_trained += mask.sum().item()
        return mask

    def observe_validation(self, step: int, val_loss: float):
        if self.last_val_loss is not None and self.current_context is not None:
            reward = self.last_val_loss - val_loss
            if self.current_weights is not None:
                self.bandit.update(self.current_context, self.current_weights, reward)
                self.encoder.record_reward(
                    self.current_weights.argmax().item(), reward
                )
        self.last_val_loss = val_loss

    def get_log_dict(self) -> dict:
        d: dict = {}
        if self.current_weights is not None:
            for i, name in enumerate(self.cfg.strategy_names):
                d[f"sieve/weight_{name}"] = self.current_weights[i].item()
        d["sieve/scoring_round"] = self.scoring_round
        d["sieve/tokens_scored"] = self.tokens_scored
        d["sieve/tokens_trained"] = self.tokens_trained
        if self.tokens_scored > 0:
            d["sieve/effective_ratio"] = self.tokens_trained / self.tokens_scored
        d["sieve/num_breakpoints"] = len(self.bandit.breakpoints)
        return d


class FullTrainingSelector:
    def get_token_mask(self, *args, **kwargs):
        return None
    def observe_validation(self, *args, **kwargs):
        pass
    def needs_rescore(self, *args, **kwargs):
        return False
    def get_log_dict(self):
        return {"sieve/method": "full"}


class RandomSelector:
    def __init__(self, ratio: float = 0.7, device: str = "cuda"):
        self.ratio = ratio
        self.device = device

    def get_token_mask(self, step, total_steps, model, inputs, targets, sw, **kw):
        n = targets.size(0)
        k = max(1, int(n * self.ratio))
        idx = torch.randperm(n, device=self.device)[:k]
        mask = torch.zeros(n, dtype=torch.bool, device=self.device)
        mask[idx] = True
        return mask

    def needs_rescore(self, step):
        return True

    def observe_validation(self, *args, **kwargs):
        pass

    def get_log_dict(self):
        return {"sieve/method": "random"}


class Rho1Selector:
    def __init__(self, ratio: float = 0.7, device: str = "cuda"):
        self.ratio = ratio
        self.device = device
        self.ref_logits: Optional[Tensor] = None
        self.cached_mask: Optional[Tensor] = None
        self.scored = False

    def needs_rescore(self, step):
        return not self.scored

    @torch.no_grad()
    def get_token_mask(self, step, total_steps, model, inputs, targets, sw, **kw):
        if self.cached_mask is not None and self.cached_mask.size(0) == targets.size(0):
            return self.cached_mask

        seq_len = targets.size(0)
        num_select = max(1, int(seq_len * self.ratio))

        captured = {}
        def hook_fn(module, input, output):
            captured["logits"] = output.detach()

        model.eval()
        handle = model.lm_head.register_forward_hook(hook_fn)
        try:
            _ = model(inputs, targets, sw)
        finally:
            handle.remove()
        model.train()

        logits = captured.get("logits")
        if logits is None:
            idx = torch.randperm(seq_len, device=self.device)[:num_select]
            mask = torch.zeros(seq_len, dtype=torch.bool, device=self.device)
            mask[idx] = True
            return mask

        logits = model._logits_softcap_scale * torch.sigmoid(
            (logits + model._logits_softcap_shift) / model._logits_softcap_divisor
        )
        if logits.dim() == 3:
            logits = logits.squeeze(0)

        if self.ref_logits is None:
            self.ref_logits = logits.clone()

        from sieve_scorers import score_excess_loss
        excess = score_excess_loss(logits, targets, self.ref_logits)
        _, top_idx = torch.topk(excess, num_select)
        mask = torch.zeros(seq_len, dtype=torch.bool, device=self.device)
        mask[top_idx] = True

        self.cached_mask = mask
        self.scored = True
        return mask

    def observe_validation(self, *args, **kwargs):
        pass

    def get_log_dict(self):
        return {"sieve/method": "rho1"}


class SingleStrategySelector:
    def __init__(self, strategy: str, ratio: float = 0.7,
                 rescore_every: int = 50, device: str = "cuda"):
        self.strategy = strategy
        self.ratio = ratio
        self.rescore_every = rescore_every
        self.device = device
        self.cached_mask: Optional[Tensor] = None
        self.cached_at: int = -9999

    def needs_rescore(self, step):
        return (step - self.cached_at) >= self.rescore_every

    @torch.no_grad()
    def get_token_mask(self, step, total_steps, model, inputs, targets, sw, **kw):
        if not self.needs_rescore(step) and self.cached_mask is not None \
                and self.cached_mask.size(0) == targets.size(0):
            return self.cached_mask

        seq_len = targets.size(0)
        num_select = max(1, int(seq_len * self.ratio))

        captured = {}
        def hook_fn(module, input, output):
            captured["logits"] = output.detach()

        model.eval()
        handle = model.lm_head.register_forward_hook(hook_fn)
        try:
            _ = model(inputs, targets, sw)
        finally:
            handle.remove()
        model.train()

        logits = captured.get("logits")
        if logits is None:
            idx = torch.randperm(seq_len, device=self.device)[:num_select]
            mask = torch.zeros(seq_len, dtype=torch.bool, device=self.device)
            mask[idx] = True
            return mask

        logits = model._logits_softcap_scale * torch.sigmoid(
            (logits + model._logits_softcap_shift) / model._logits_softcap_divisor
        )
        if logits.dim() == 3:
            logits = logits.squeeze(0)

        from sieve_scorers import score_excess_loss, score_uncertainty, score_attn_entropy, score_diversity
        score_fns = {
            "excess_loss": lambda: score_excess_loss(logits, targets),
            "uncertainty": lambda: score_uncertainty(logits),
            "attn_entropy": lambda: score_attn_entropy(logits),
            "diversity": lambda: score_diversity(targets, logits.size(-1)),
        }

        s = score_fns.get(self.strategy, score_fns["excess_loss"])()
        _, top_idx = torch.topk(s, num_select)
        mask = torch.zeros(seq_len, dtype=torch.bool, device=self.device)
        mask[top_idx] = True

        self.cached_mask = mask
        self.cached_at = step
        return mask

    def observe_validation(self, *args, **kwargs):
        pass

    def get_log_dict(self):
        return {"sieve/method": f"single_{self.strategy}"}
