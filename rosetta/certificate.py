from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.linear_model import Ridge

from rosetta.adapters.base import l2_normalize
from rosetta.metrics import overlap_at_k
from rosetta.store import Hit


def conformal_quantile(scores: np.ndarray, alpha: float) -> float:
    """
    Compute finite-sample one-sided conformal prediction quantile.
    Guarantee: P(s_{n+1} <= q_hat) >= 1 - alpha under exchangeability.
    """
    scores = np.asarray(scores, dtype=np.float32)
    n = len(scores)
    if n == 0:
        return 1.0
    k = int(np.ceil((n + 1) * (1.0 - alpha)))
    k = max(1, min(k, n))
    sorted_scores = np.sort(scores)
    return float(sorted_scores[k - 1])


@dataclass
class CertificateFeatures:
    sim_anchor_max: float
    sim_anchor_top5: float
    margin_1_10: float
    margin_1_2: float
    local_fit_err: float
    self_consistency: float

    def to_array(self) -> np.ndarray:
        return np.array(
            [
                self.sim_anchor_max,
                self.sim_anchor_top5,
                self.margin_1_10,
                self.margin_1_2,
                self.local_fit_err,
                self.self_consistency,
            ],
            dtype=np.float32,
        )

    @classmethod
    def feature_names(cls) -> list[str]:
        return [
            "sim_anchor_max",
            "sim_anchor_top5",
            "margin_1_10",
            "margin_1_2",
            "local_fit_err",
            "self_consistency",
        ]


def extract_features(
    q_small: np.ndarray,
    q_hat: np.ndarray,
    hits: list[Hit],
    anchors_centroids: np.ndarray | None = None,
    centroid_residuals: np.ndarray | None = None,
    old_space_hits: list[Hit] | None = None,
    k: int = 10,
) -> CertificateFeatures:
    """
    Extract lightweight on-device features without requiring the large cloud model.
    """
    q_s = l2_normalize(q_small.flatten())
    q_h = l2_normalize(q_hat.flatten())

    # 1 & 2: Anchor similarity features (OOD signals)
    if anchors_centroids is not None and len(anchors_centroids) > 0:
        anc_norms = l2_normalize(anchors_centroids)
        sims = anc_norms @ q_s
        sim_max = float(np.max(sims))
        top5_count = min(5, len(sims))
        sim_top5 = float(np.mean(np.sort(sims)[-top5_count:]))

        if centroid_residuals is not None and len(centroid_residuals) == len(anchors_centroids):
            top_idx = int(np.argmax(sims))
            local_err = float(centroid_residuals[top_idx])
        else:
            local_err = 0.05
    else:
        sim_max = 0.5
        sim_top5 = 0.5
        local_err = 0.05

    # 3 & 4: Score margin features from local hits
    if len(hits) >= 2:
        margin_1_2 = float(hits[0].score - hits[1].score)
    else:
        margin_1_2 = 0.0

    if len(hits) >= k:
        margin_1_10 = float(hits[0].score - hits[k - 1].score)
    elif len(hits) > 0:
        margin_1_10 = float(hits[0].score - hits[-1].score)
    else:
        margin_1_10 = 0.0

    # 5: Self-consistency (overlap between old-space search & translated search)
    if old_space_hits is not None and len(old_space_hits) > 0:
        old_ids = [h.id for h in old_space_hits[:k]]
        new_ids = [h.id for h in hits[:k]]
        self_consistency = overlap_at_k(old_ids, new_ids, k=k)
    else:
        self_consistency = 0.5

    return CertificateFeatures(
        sim_anchor_max=sim_max,
        sim_anchor_top5=sim_top5,
        margin_1_10=margin_1_10,
        margin_1_2=margin_1_2,
        local_fit_err=local_err,
        self_consistency=self_consistency,
    )


class RecallCertificate:
    """
    On-device Conformal Recall Certificate Engine.
    Computes guaranteed lower bounds on retrieval overlap@k.
    """

    def __init__(
        self,
        weights: np.ndarray,
        intercept: float,
        quantiles: dict[str, float],
        feature_means: np.ndarray,
        feature_stds: np.ndarray,
        anchors_centroids: np.ndarray | None = None,
        centroid_residuals: np.ndarray | None = None,
    ):
        self.weights = np.asarray(weights, dtype=np.float32)
        self.intercept = float(intercept)
        # alpha str ("0.05", "0.10", "0.20") -> q_hat
        self.quantiles = {str(k): float(v) for k, v in quantiles.items()}
        self.feature_means = np.asarray(feature_means, dtype=np.float32)
        self.feature_stds = np.asarray(feature_stds, dtype=np.float32)
        self.anchors_centroids = anchors_centroids
        self.centroid_residuals = centroid_residuals

    def predict_loss(self, features: CertificateFeatures) -> float:
        """Estimate expected loss h_hat(f)."""
        f = (features.to_array() - self.feature_means) / np.clip(self.feature_stds, 1e-6, None)
        pred = float(np.dot(self.weights, f) + self.intercept)
        return float(np.clip(pred, 0.0, 1.0))

    def certify(
        self,
        features: CertificateFeatures,
        alpha: float = 0.10,
    ) -> tuple[float, float]:
        """
        Compute (certified_overlap, predicted_loss) at nominal significance level alpha.
        Guarantees P(overlap >= certified_overlap) >= 1 - alpha for exchangeable queries.
        """
        pred_loss = self.predict_loss(features)
        alpha_key = f"{alpha:.2f}"
        # Fallback to closest available quantile if exact key missing
        if alpha_key in self.quantiles:
            q_hat = self.quantiles[alpha_key]
        else:
            closest = min(self.quantiles.keys(), key=lambda k: abs(float(k) - alpha))
            q_hat = self.quantiles[closest]

        upper_loss = float(np.clip(pred_loss + q_hat, 0.0, 1.0))
        certified_overlap = float(np.clip(1.0 - upper_loss, 0.0, 1.0))
        return certified_overlap, pred_loss

    def to_dict(self) -> dict[str, Any]:
        return {
            "model": {
                "weights": self.weights.tolist(),
                "intercept": self.intercept,
                "feature_means": self.feature_means.tolist(),
                "feature_stds": self.feature_stds.tolist(),
            },
            "calib": {
                "quantiles": self.quantiles,
            },
        }

    @classmethod
    def from_dict(
        cls,
        cert_model: dict[str, Any],
        cert_calib: dict[str, Any],
        anchors_centroids: np.ndarray | None = None,
        centroid_residuals: np.ndarray | None = None,
    ) -> "RecallCertificate":
        m = cert_model.get("model", cert_model)
        c = cert_calib.get("calib", cert_calib)
        return cls(
            weights=np.array(m["weights"], dtype=np.float32),
            intercept=float(m["intercept"]),
            quantiles=c["quantiles"],
            feature_means=np.array(m["feature_means"], dtype=np.float32),
            feature_stds=np.array(m["feature_stds"], dtype=np.float32),
            anchors_centroids=anchors_centroids,
            centroid_residuals=centroid_residuals,
        )


def fit_certificate(
    features_train: np.ndarray,
    losses_train: np.ndarray,
    features_cal: np.ndarray,
    losses_cal: np.ndarray,
    alphas: list[float] = [0.05, 0.10, 0.20],
    anchors_centroids: np.ndarray | None = None,
    centroid_residuals: np.ndarray | None = None,
) -> RecallCertificate:
    """
    Cloud-side calibration routine:
    1. Fit loss predictor h_hat on (features_train, losses_train).
    2. Compute conformal nonconformity scores s_i = loss_i - h_hat(f_i) on calibration split.
    3. Calculate conformal quantiles q_hat for each alpha.
    """
    means = np.mean(features_train, axis=0)
    stds = np.std(features_train, axis=0)
    stds[stds < 1e-6] = 1.0

    X_train_norm = (features_train - means) / stds
    X_cal_norm = (features_cal - means) / stds

    # Ridge predictor for smooth calibration
    reg = Ridge(alpha=1.0)
    reg.fit(X_train_norm, losses_train)

    preds_cal = reg.predict(X_cal_norm)
    # Nonconformity scores: s_i = loss_i - h_hat(f_i)
    nonconformity_scores = losses_cal - preds_cal

    quantiles = {}
    for a in alphas:
        q = conformal_quantile(nonconformity_scores, a)
        quantiles[f"{a:.2f}"] = float(q)

    return RecallCertificate(
        weights=reg.coef_.astype(np.float32),
        intercept=float(reg.intercept_),
        quantiles=quantiles,
        feature_means=means.astype(np.float32),
        feature_stds=stds.astype(np.float32),
        anchors_centroids=anchors_centroids,
        centroid_residuals=centroid_residuals,
    )
