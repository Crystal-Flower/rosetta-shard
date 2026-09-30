import numpy as np

from rosetta.adapters.base import BaseAdapter, l2_normalize


class RidgeAdapter(BaseAdapter):
    """
    Ridge Regression Adapter:
    Maps small-space vectors X to large-space vectors Y via regularized least squares:
    W = (X_c^T X_c + lambda * I)^(-1) X_c^T Y_c
    """

    def __init__(self, lam: float = 1e-3, center: bool = True):
        super().__init__()
        self.lam = lam
        self.center = center

    def fit(self, X: np.ndarray, Y: np.ndarray, lam: float | None = None) -> "RidgeAdapter":
        if lam is not None:
            self.lam = lam

        X = l2_normalize(np.asarray(X, dtype=np.float32))
        Y = l2_normalize(np.asarray(Y, dtype=np.float32))

        self.d_in = X.shape[1]
        self.d_out = Y.shape[1]

        if self.center:
            self.mu_x = np.mean(X, axis=0, keepdims=True)
            self.mu_y = np.mean(Y, axis=0, keepdims=True)
            Xc = X - self.mu_x
            Yc = Y - self.mu_y
        else:
            self.mu_x = np.zeros((1, self.d_in), dtype=np.float32)
            self.mu_y = np.zeros((1, self.d_out), dtype=np.float32)
            Xc = X
            Yc = Y

        reg = self.lam * np.eye(self.d_in, dtype=np.float32)
        # Solve (Xc^T @ Xc + lam * I) @ W = Xc^T @ Yc
        A = Xc.T @ Xc + reg
        B = Xc.T @ Yc
        self.W = np.linalg.solve(A, B).astype(np.float32)
        return self

    def __call__(self, X: np.ndarray) -> np.ndarray:
        if self.W is None:
            raise RuntimeError("RidgeAdapter must be fitted before transform.")
        X = l2_normalize(np.asarray(X, dtype=np.float32))
        if X.ndim == 1:
            X = X.reshape(1, -1)
        Xc = X - self.mu_x
        projected = Xc @ self.W + self.mu_y
        return l2_normalize(projected)

    def tune(
        self,
        X_train: np.ndarray,
        Y_train: np.ndarray,
        X_val: np.ndarray,
        Y_val: np.ndarray,
        candidates: list[float] | None = None,
    ) -> float:
        """Find lambda that maximizes cosine similarity on validation set."""
        if candidates is None:
            candidates = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0]

        best_lam = self.lam
        best_sim = -float("inf")

        Y_val_norm = l2_normalize(Y_val)

        for c in candidates:
            self.fit(X_train, Y_train, lam=c)
            Y_pred = self(X_val)
            # Average cosine similarity
            sim = float(np.mean(np.sum(Y_pred * Y_val_norm, axis=1)))
            if sim > best_sim:
                best_sim = sim
                best_lam = c

        self.fit(X_train, Y_train, lam=best_lam)
        return best_lam
