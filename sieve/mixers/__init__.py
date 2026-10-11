"""Mixer registry: config names → mixer classes."""
from __future__ import annotations

from ..config import SelectionConfig
from .base import Mixer
from .dirichlet import DirichletMixer, VNDirichletMixer
from .fixed import FixedMixer

MIXERS = {"fixed": FixedMixer, "dirichlet": DirichletMixer, "vn_dirichlet": VNDirichletMixer}


def build_mixer(cfg: SelectionConfig, k: int, seed: int) -> Mixer:
    if cfg.mixer == "fixed":
        return FixedMixer(k, cfg.fixed_weights)
    if cfg.mixer == "dirichlet":
        return DirichletMixer(k, seed)
    if cfg.mixer == "vn_dirichlet":
        return VNDirichletMixer(k, seed, eps=cfg.vn_eps, sigma_warmup=cfg.vn_sigma_warmup)
    raise ValueError(f"unknown mixer {cfg.mixer!r}; known: {sorted(MIXERS)}")


__all__ = ["MIXERS", "Mixer", "build_mixer"]
