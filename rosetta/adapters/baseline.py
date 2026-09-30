import numpy as np
from sklearn.cross_decomposition import CCA

from rosetta.adapters.base import BaseAdapter, l2_normalize


class PadTruncateBaseline(BaseAdapter):
    """
    Naive baseline: Pad with zeros or truncate coordinate axes to match dimension.
    """

    def fit(self, X: np.ndarray, Y: np.ndarray) -> "PadTruncateBaseline":
        self.d_in = X.shape[1]
        self.d_out = Y.shape[1]
        return self

    def __call__(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float32)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        n, d_in = X.shape
        if d_in == self.d_out:
            res = X.copy()
        elif d_in < self.d_out:
            res = np.zeros((n, self.d_out), dtype=np.float32)
            res[:, :d_in] = X
        else:
            res = X[:, : self.d_out].copy()
        return l2_normalize(res)


class CCABaseline(BaseAdapter):
    """
    Classical baseline: Canonical Correlation Analysis (CCA) projection.
    """

    def __init__(self, n_components: int | None = None):
        super().__init__()
        self.n_components = n_components
        self.cca: CCA | None = None

    def fit(self, X: np.ndarray, Y: np.ndarray) -> "CCABaseline":
        X = l2_normalize(np.asarray(X, dtype=np.float32))
        Y = l2_normalize(np.asarray(Y, dtype=np.float32))
        self.d_in = X.shape[1]
        self.d_out = Y.shape[1]

        n_comp = self.n_components or min(self.d_in, self.d_out, 64)
        self.cca = CCA(n_components=n_comp)
        self.cca.fit(X, Y)

        # Approximate linear mapping from CCA weights
        # X_rot = X @ x_rotations_, Y_approx = X_rot @ y_rotations_.T
        self.W = (self.cca.x_rotations_ @ self.cca.y_rotations_.T).astype(np.float32)
        return self

    def __call__(self, X: np.ndarray) -> np.ndarray:
        if self.W is None:
            raise RuntimeError("CCABaseline must be fitted before transform.")
        X = l2_normalize(np.asarray(X, dtype=np.float32))
        if X.ndim == 1:
            X = X.reshape(1, -1)
        projected = X @ self.W
        return l2_normalize(projected)
