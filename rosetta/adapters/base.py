from abc import ABC, abstractmethod

import numpy as np


def l2_normalize(x: np.ndarray) -> np.ndarray:
    """Normalize vectors along the last dimension to unit length."""
    if x.ndim == 1:
        norm = np.linalg.norm(x)
        return x / max(norm, 1e-12)
    norms = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / np.clip(norms, a_min=1e-12, a_max=None)


class BaseAdapter(ABC):
    """Abstract base class for cross-model vector space adapters."""

    def __init__(self):
        self.W: np.ndarray | None = None
        self.mu_x: np.ndarray | None = None
        self.mu_y: np.ndarray | None = None
        self.d_in: int | None = None
        self.d_out: int | None = None

    @abstractmethod
    def fit(self, X: np.ndarray, Y: np.ndarray, **kwargs) -> "BaseAdapter":
        """
        Fit mapping from source space X (n x d_in) to target space Y (n x d_out).
        """

    @abstractmethod
    def __call__(self, X: np.ndarray) -> np.ndarray:
        """
        Translate vectors from source space to target space. Output is L2-normalized.
        """

    def transform(self, X: np.ndarray) -> np.ndarray:
        return self(X)

    def to_dict(self) -> dict:
        return {
            "type": self.__class__.__name__,
            "d_in": self.d_in,
            "d_out": self.d_out,
            "has_mu_x": self.mu_x is not None,
            "has_mu_y": self.mu_y is not None,
        }
