import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from rosetta.adapters.base import BaseAdapter, l2_normalize


class ContrastiveAdapter(BaseAdapter):
    """
    Retrieval-Aware Contrastive Adapter (Section 7.1c):
    Optimizes a linear projection W via in-batch InfoNCE and cosine alignment loss.
    Directly maximizes retrieval ranking fidelity rather than raw L2 reconstruction error.
    """

    def __init__(
        self,
        lr: float = 5e-3,
        epochs: int = 100,
        temperature: float = 0.07,
        lam_cos: float = 0.2,
        lam_reg: float = 1e-4,
        batch_size: int = 64,
        center: bool = False,
        init: str = "procrustes",
    ):
        super().__init__()
        self.lr = lr
        self.epochs = epochs
        self.temperature = temperature
        self.lam_cos = lam_cos
        self.lam_reg = lam_reg
        self.batch_size = batch_size
        self.center = center
        self.init = init

    def fit(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        center: bool | None = None,
        epochs: int | None = None,
        lr: float | None = None,
    ) -> "ContrastiveAdapter":
        if center is not None:
            self.center = center
        if epochs is not None:
            self.epochs = epochs
        if lr is not None:
            self.lr = lr

        X_np = l2_normalize(np.asarray(X, dtype=np.float32))
        Y_np = l2_normalize(np.asarray(Y, dtype=np.float32))

        n_samples, self.d_in = X_np.shape
        self.d_out = Y_np.shape[1]

        if self.center:
            self.mu_x = np.mean(X_np, axis=0, keepdims=True)
            self.mu_y = np.mean(Y_np, axis=0, keepdims=True)
            Xc = X_np - self.mu_x
            Yc = Y_np - self.mu_y
        else:
            self.mu_x = np.zeros((1, self.d_in), dtype=np.float32)
            self.mu_y = np.zeros((1, self.d_out), dtype=np.float32)
            Xc = X_np
            Yc = Y_np

        # Initialization
        if self.init == "procrustes":
            M = Xc.T @ Yc
            U, _, Vt = np.linalg.svd(M, full_matrices=False)
            W_init = (U @ Vt).astype(np.float32)
        elif self.init == "ridge":
            A = Xc.T @ Xc + 1e-2 * np.eye(self.d_in, dtype=np.float32)
            B = Xc.T @ Yc
            W_init = np.linalg.solve(A, B).astype(np.float32)
        else:
            rng = np.random.default_rng(42)
            W_init = (rng.standard_normal((self.d_in, self.d_out)) * 0.02).astype(np.float32)

        # PyTorch training module
        W_param = nn.Parameter(torch.from_numpy(W_init).float())
        optimizer = torch.optim.AdamW([W_param], lr=self.lr, weight_decay=self.lam_reg)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.epochs)

        X_tensor = torch.from_numpy(Xc).float()
        Y_tensor = torch.from_numpy(Yc).float()
        Y_target_norm = F.normalize(torch.from_numpy(Y_np).float(), p=2, dim=1)
        mu_y_tensor = torch.from_numpy(self.mu_y).float()

        effective_batch = min(self.batch_size, n_samples)
        n_batches = int(np.ceil(n_samples / effective_batch))

        for _ in range(self.epochs):
            perm = torch.randperm(n_samples)
            for b in range(n_batches):
                idx = perm[b * effective_batch : (b + 1) * effective_batch]
                if len(idx) < 2:
                    continue

                xb = X_tensor[idx]
                yb_true = Y_target_norm[idx]

                # Project and normalize
                y_hat = xb @ W_param
                if self.center:
                    y_hat = y_hat + mu_y_tensor
                y_hat_norm = F.normalize(y_hat, p=2, dim=1)

                # InfoNCE similarity matrix
                sims = torch.matmul(y_hat_norm, yb_true.T) / self.temperature
                labels = torch.arange(len(idx), device=sims.device)

                loss_nce = 0.5 * (
                    F.cross_entropy(sims, labels) + F.cross_entropy(sims.T, labels)
                )

                # Cosine alignment loss: 1 - cos(y_hat, y)
                loss_cos = (1.0 - (y_hat_norm * yb_true).sum(dim=1)).mean()

                loss = loss_nce + self.lam_cos * loss_cos

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            scheduler.step()

        self.W = W_param.detach().cpu().numpy().astype(np.float32)
        return self

    def __call__(self, X: np.ndarray) -> np.ndarray:
        if self.W is None:
            raise RuntimeError("ContrastiveAdapter must be fitted before transform.")
        X = l2_normalize(np.asarray(X, dtype=np.float32))
        if X.ndim == 1:
            X = X.reshape(1, -1)
        Xc = X - self.mu_x
        projected = Xc @ self.W + self.mu_y
        return l2_normalize(projected)
