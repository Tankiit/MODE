"""Auto-detect best available backend. Import get_cfg() everywhere."""
from __future__ import annotations
import torch
from dataclasses import dataclass
from typing import Literal


@dataclass
class DeviceCfg:
    device:   torch.device
    dtype:    torch.dtype           # bf16 on CUDA, fp16 on MPS, fp32 on CPU
    backend:  Literal["cuda","mps","cpu"]
    use_amp:  bool                  # whether to wrap forward in autocast

    def to(self, x):
        """Move any tensor or dict of tensors to this device."""
        if isinstance(x, torch.Tensor):
            return x.to(self.device)
        if isinstance(x, dict):
            return {k: v.to(self.device) if isinstance(v, torch.Tensor)
                    else v for k, v in x.items()}
        return x

    @property
    def autocast(self):
        if self.backend == "cuda":
            return torch.autocast("cuda", dtype=self.dtype)
        return torch.autocast("cpu", enabled=False)   # MPS/CPU: no-op


_cfg: DeviceCfg | None = None

def get_cfg() -> DeviceCfg:
    global _cfg
    if _cfg is None:
        if torch.cuda.is_available():
            _cfg = DeviceCfg(
                device=torch.device("cuda"),
                dtype=torch.bfloat16,
                backend="cuda",
                use_amp=True,
            )
        elif torch.backends.mps.is_available():
            _cfg = DeviceCfg(
                device=torch.device("mps"),
                dtype=torch.float16,   # bf16 not supported on MPS
                backend="mps",
                use_amp=False,         # MPS autocast is unstable
            )
        else:
            _cfg = DeviceCfg(
                device=torch.device("cpu"),
                dtype=torch.float32,
                backend="cpu",
                use_amp=False,
            )
    return _cfg
