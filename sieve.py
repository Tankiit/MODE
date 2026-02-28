from .sieve_config import SieveConfig
from .sieve_models import (
    SieveSelector,
    FullTrainingSelector,
    RandomSelector,
    Rho1Selector,
    SingleStrategySelector,
)
from .sieve_loss import sieve_masked_loss


def create_selector(method: str, ratio: float = 0.7,
                    rescore_every: int = 50, device: str | None = None,
                    **kwargs) -> object:
    if device is None:
        import torch
        device = (
            "cuda"
            if torch.cuda.is_available()
            else ("mps" if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available() else "cpu")
        )
    if method == "full":
        return FullTrainingSelector()
    elif method == "sieve":
        cfg = SieveConfig(select_ratio=ratio, rescore_every=rescore_every, **kwargs)
        return SieveSelector(cfg, device)
    elif method == "sieve_offline":
        cfg = SieveConfig(select_ratio=ratio, mode="offline", **kwargs)
        return SieveSelector(cfg, device)
    elif method == "sieve_online":
        cfg = SieveConfig(select_ratio=ratio, mode="online", rescore_every=1, **kwargs)
        return SieveSelector(cfg, device)
    elif method == "rho1":
        return Rho1Selector(ratio, device)
    elif method == "random":
        return RandomSelector(ratio, device)
    elif method.endswith("_only"):
        strategy = method.replace("_only", "")
        return SingleStrategySelector(strategy, ratio, rescore_every, device)
    else:
        raise ValueError(f"Unknown method: {method}. "
                         f"Choose from: full, sieve, sieve_offline, sieve_online, "
                         f"rho1, random, excess_loss_only, uncertainty_only, "
                         f"attn_entropy_only, diversity_only")


__all__ = [
    "SieveConfig",
    "SieveSelector",
    "FullTrainingSelector",
    "RandomSelector",
    "Rho1Selector",
    "SingleStrategySelector",
    "sieve_masked_loss",
    "create_selector",
]
