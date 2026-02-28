from dataclasses import dataclass


@dataclass
class SieveConfig:
    select_ratio: float = 0.70
    mode: str = "periodic"
    rescore_every: int = 50
    num_strategies: int = 4
    strategy_names: tuple = ("excess_loss", "uncertainty", "attn_entropy", "diversity")
    dirichlet_prior: float = 1.0
    discount_gamma: float = 0.95
    adaptive_gamma: bool = True
    context_dim: int = 16
    num_context_bins: int = 16
    snapshot_reference: bool = True
    reference_is_initial: bool = True
    scale_loss: bool = True
