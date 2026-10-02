from rosetta.adapters.base import BaseAdapter, l2_normalize
from rosetta.adapters.baseline import CCABaseline, PadTruncateBaseline
try:
    from rosetta.adapters.contrastive import ContrastiveAdapter
except ImportError:
    ContrastiveAdapter = None

from rosetta.adapters.procrustes import ProcrustesAdapter
from rosetta.adapters.ridge import RidgeAdapter

__all__ = [
    "BaseAdapter",
    "CCABaseline",
    "ContrastiveAdapter",
    "PadTruncateBaseline",
    "ProcrustesAdapter",
    "RidgeAdapter",
    "l2_normalize",
]


