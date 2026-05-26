"""
SIEVE: Selective token-level training for language models.

A complete implementation of SIEVE token selection with contextual bandit
optimization over multiple scoring strategies.
"""

from .config import SieveConfig
from .state import SieveState
from .loop import train, evaluate
from .patience import PatienceMonitor
from .rho1 import Rho1Baseline
from .device import get_cfg

__version__ = "0.1.0"
__all__ = [
    "SieveConfig",
    "SieveState",
    "train",
    "evaluate",
    "PatienceMonitor",
    "Rho1Baseline",
    "get_cfg",
]
