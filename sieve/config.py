"""
Run configuration: nested dataclasses loaded from YAML.

Several files are merged left to right (later values override earlier
ones); a file may list other files under `include:` (paths relative to that
file, merged before its own content); `--set section.key=value` overrides
are applied last. Unknown keys are errors, never silently ignored.

Defaults follow the paper (§3, App. B). A run's identity for "already done"
checks is `config_hash`, computed on everything except `name` and `output`.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Union, get_args, get_origin, get_type_hints

import yaml


@dataclass
class ModelConfig:
    name: str = "gpt2"            # Hugging Face id (gpt2, gpt2-large, ...) or "tiny" (tests)
    tiny_layers: int = 2          # only for name: tiny (randomly initialised small GPT-2)
    tiny_dim: int = 64
    tiny_heads: int = 2


@dataclass
class DataConfig:
    name: str = "wikitext103"     # label used by the analyses
    dir: str = "./data/wikitext103"
    seq_len: int = 1024
    batch_size: int = 8
    fraction: float = 1.0         # fraction of train.bin to sample from


@dataclass
class SelectionConfig:
    method: str = "clm"           # label used by the analyses (se_only, mix4, ...)
    alpha: Optional[float] = None # retained fraction per sequence; null = full CLM
    scorers: list = field(default_factory=list)   # names from sieve.scorers.SCORERS
    mixer: str = "fixed"          # fixed | dirichlet | vn_dirichlet
    fixed_weights: Optional[list] = None          # mixer fixed; null = uniform
    rescore_interval: int = 100   # T_rescore: steps between mixture-weight samples
    reference_model: str = "gpt2" # frozen reference for excess_loss (paper: GPT-2 Small)
    vn_eps: float = 1e-3          # VN-Mix ε (paper Eq. vnts)
    vn_sigma_warmup: int = 2      # VN-Mix: σ̂ = 1 until this many rewards
    noise_seed: Optional[int] = None  # noise scorer; null = run seed


@dataclass
class TrainingConfig:
    steps: int = 25_000
    lr: float = 3e-4
    beta1: float = 0.9
    beta2: float = 0.95
    weight_decay: float = 0.1
    warmup_steps: int = 500       # linear warmup, then cosine decay
    min_lr_ratio: float = 0.1     # cosine ends at 10% of the peak
    grad_clip: float = 1.0
    autocast: str = "bfloat16"    # CUDA autocast dtype; weights stay float32


@dataclass
class EvalConfig:
    interval: int = 1000          # T_val
    batches: int = 50             # batches per intermediate evaluation (final: all)
    batch_size: int = 8


@dataclass
class OutputConfig:
    root: str = "runs"
    log_interval: int = 100
    save_final: bool = False


@dataclass
class RunConfig:
    name: Optional[str] = None    # null = derived from data/model/method/alpha/steps/seed
    seed: int = 42
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    selection: SelectionConfig = field(default_factory=SelectionConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    output: OutputConfig = field(default_factory=OutputConfig)

    @property
    def run_name(self) -> str:
        if self.name:
            return self.name
        a = "full" if self.selection.alpha is None else f"a{round(self.selection.alpha * 100):03d}"
        model = self.model.name.split("/")[-1]
        return (f"{self.data.name}_{model}_{self.selection.method}_{a}"
                f"_st{self.training.steps}_s{self.seed}")


# ── Loading ───────────────────────────────────────────────────────────

def _read_yaml(path: Path, seen: tuple = ()) -> dict:
    path = path.resolve()
    if path in seen:
        raise ValueError(f"include cycle: {' -> '.join(map(str, seen + (path,)))}")
    data = yaml.safe_load(path.read_text()) or {}
    if not isinstance(data, dict):
        raise ValueError(f"{path}: top level must be a mapping")
    merged: dict = {}
    for inc in data.pop("include", []) or []:
        merged = deep_merge(merged, _read_yaml(path.parent / inc, seen + (path,)))
    return deep_merge(merged, data)


def deep_merge(base: dict, over: dict) -> dict:
    out = dict(base)
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def parse_override(text: str) -> dict:
    """'training.steps=300' → {'training': {'steps': 300}} (value parsed as YAML)."""
    if "=" not in text:
        raise ValueError(f"override must be key=value, got {text!r}")
    key, value = text.split("=", 1)
    node: Any = yaml.safe_load(value)
    for part in reversed(key.strip().split(".")):
        node = {part: node}
    return node


def _build(cls, data: dict, where: str):
    hints = get_type_hints(cls)
    names = {f.name for f in dataclasses.fields(cls)}
    unknown = set(data) - names
    if unknown:
        raise ValueError(f"unknown config key(s) in {where or 'root'}: {sorted(unknown)}")
    kwargs = {}
    for f in dataclasses.fields(cls):
        if f.name not in data:
            continue
        v, t = data[f.name], hints[f.name]
        path = f"{where}.{f.name}" if where else f.name
        if dataclasses.is_dataclass(t):
            if not isinstance(v, dict):
                raise ValueError(f"{path} must be a mapping")
            kwargs[f.name] = _build(t, v, path)
        else:
            kwargs[f.name] = _coerce(v, t, path)
    return cls(**kwargs)


def _coerce(v, t, path):
    if get_origin(t) is Union:                       # Optional[X]
        if v is None:
            return None
        t = next(a for a in get_args(t) if a is not type(None))
    if t is float and isinstance(v, int) and not isinstance(v, bool):
        return float(v)
    base = get_origin(t) or t
    if base is list:
        if not isinstance(v, list):
            raise ValueError(f"{path} must be a list, got {v!r}")
        return v
    if not isinstance(v, base) or (base is int and isinstance(v, bool)):
        raise ValueError(f"{path} must be {base.__name__}, got {v!r}")
    return v


def load_config(paths: list[str], overrides: list[str] = ()) -> RunConfig:
    merged: dict = {}
    for p in paths:
        merged = deep_merge(merged, _read_yaml(Path(p)))
    for o in overrides:
        merged = deep_merge(merged, parse_override(o))
    cfg = _build(RunConfig, merged, "")
    validate(cfg)
    return cfg


def validate(cfg: RunConfig) -> None:
    s = cfg.selection
    if s.alpha is None:
        if s.scorers:
            raise ValueError("selection.alpha is null (full CLM) but scorers are listed")
    else:
        if not 0.0 < s.alpha <= 1.0:
            raise ValueError(f"selection.alpha must be in (0, 1], got {s.alpha}")
        if not s.scorers:
            raise ValueError("selection.scorers is empty")
        if s.fixed_weights is not None and len(s.fixed_weights) != len(s.scorers):
            raise ValueError("selection.fixed_weights must have one weight per scorer")
    for path, v in [("selection.rescore_interval", s.rescore_interval),
                    ("eval.interval", cfg.eval.interval),
                    ("training.steps", cfg.training.steps),
                    ("data.batch_size", cfg.data.batch_size),
                    ("output.log_interval", cfg.output.log_interval)]:
        if v <= 0:
            raise ValueError(f"{path} must be > 0")


def to_dict(cfg: RunConfig) -> dict:
    return dataclasses.asdict(cfg)


def config_hash(cfg: RunConfig) -> str:
    """Identity of the experiment: everything except `name` and `output`."""
    d = to_dict(cfg)
    d.pop("name")
    d.pop("output")
    return hashlib.sha256(json.dumps(d, sort_keys=True).encode()).hexdigest()[:16]
