import numpy as np

from rosetta.adapters.base import BaseAdapter, l2_normalize


class ProcrustesAdapter(BaseAdapter):
    """
    Semi-orthogonal Procrustes Adapter:
    Fits orthogonal / semi-orthogonal matrix W via SVD of cross-covariance matrix:
    X^T Y = U S V^T  ==>  W = U V^T
    Preserves inner products / geometric angles.
    """

    def fit(self, X: np.ndarray, Y: np.ndarray, center: bool = False) -> "ProcrustesAdapter":
        X = l2_normalize(np.asarray(X, dtype=np.float32))
        Y = l2_normalize(np.asarray(Y, dtype=np.float32))

        self.d_in = X.shape[1]
        self.d_out = Y.shape[1]

        if center:
            self.mu_x = np.mean(X, axis=0, keepdims=True)
            self.mu_y = np.mean(Y, axis=0, keepdims=True)
            Xc = X - self.mu_x
            Yc = Y - self.mu_y
        else:
            self.mu_x = np.zeros((1, self.d_in), dtype=np.float32)
            self.mu_y = np.zeros((1, self.d_out), dtype=np.float32)
            Xc = X
            Yc = Y

        M = Xc.T @ Yc
        U, _, Vt = np.linalg.svd(M, full_matrices=False)
        self.W = (U @ Vt).astype(np.float32)
        return self

    def __call__(self, X: np.ndarray) -> np.ndarray:
        if self.W is None:
            raise RuntimeError("ProcrustesAdapter must be fitted before transform.")
        X = l2_normalize(np.asarray(X, dtype=np.float32))
        if X.ndim == 1:
            X = X.reshape(1, -1)
        Xc = X - self.mu_x
        projected = Xc @ self.W + self.mu_y
        return l2_normalize(projected)
