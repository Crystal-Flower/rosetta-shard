import numpy as np
import pytest

from rosetta.certificate import (
    CertificateFeatures,
    RecallCertificate,
    conformal_quantile,
    fit_certificate,
)


def test_conformal_quantile_brute_force():
    scores = np.array([0.1, 0.3, 0.5, 0.7, 0.9], dtype=np.float32)
    # n = 5
    # For alpha = 0.2: ceil((5 + 1) * 0.8) = ceil(4.8) = 5 -> index 4 -> 0.9
    q = conformal_quantile(scores, alpha=0.2)
    assert q == pytest.approx(0.9, rel=1e-5)

    # For alpha = 0.5: ceil(6 * 0.5) = 3 -> index 2 -> 0.5
    q5 = conformal_quantile(scores, alpha=0.5)
    assert q5 == pytest.approx(0.5, rel=1e-5)


def test_conformal_quantile_monotonicity():
    rng = np.random.default_rng(42)
    scores = rng.standard_normal(200)

    alphas = [0.01, 0.05, 0.10, 0.20, 0.50]
    quantiles = [conformal_quantile(scores, a) for a in alphas]

    # As alpha increases, nominal coverage (1-alpha) decreases, so q_hat must be non-increasing
    for i in range(len(quantiles) - 1):
        assert quantiles[i] >= quantiles[i + 1]


def test_conformal_coverage_synthetic():
    """
    Finite-sample coverage property test:
    Under exchangeability, empirical coverage on test queries should be >= 1 - alpha
    within expected binomial sampling variance.
    """
    rng = np.random.default_rng(123)
    n_cal = 500
    n_test = 1000
    d_feat = 6

    # Features and linear loss + noise
    weights = rng.uniform(0.01, 0.1, size=d_feat)

    X_train = rng.uniform(0, 1, size=(200, d_feat))
    losses_train = np.clip(X_train @ weights + rng.normal(0, 0.05, size=200), 0.0, 1.0)

    X_cal = rng.uniform(0, 1, size=(n_cal, d_feat))
    losses_cal = np.clip(X_cal @ weights + rng.normal(0, 0.05, size=n_cal), 0.0, 1.0)

    X_test = rng.uniform(0, 1, size=(n_test, d_feat))
    losses_test = np.clip(X_test @ weights + rng.normal(0, 0.05, size=n_test), 0.0, 1.0)

    alpha = 0.10
    cert = fit_certificate(
        features_train=X_train,
        losses_train=losses_train,
        features_cal=X_cal,
        losses_cal=losses_cal,
        alphas=[alpha],
    )

    # Evaluate on test set
    covered = 0
    for i in range(n_test):
        feat = CertificateFeatures(
            sim_anchor_max=float(X_test[i, 0]),
            sim_anchor_top5=float(X_test[i, 1]),
            margin_1_10=float(X_test[i, 2]),
            margin_1_2=float(X_test[i, 3]),
            local_fit_err=float(X_test[i, 4]),
            self_consistency=float(X_test[i, 5]),
        )
        cert_overlap, _ = cert.certify(feat, alpha=alpha)
        true_overlap = 1.0 - losses_test[i]
        # Certificate claim: true_overlap >= cert_overlap
        if true_overlap >= cert_overlap - 1e-6:
            covered += 1

    emp_coverage = covered / n_test
    print(f"Empirical coverage: {emp_coverage:.4f} (nominal: {1.0 - alpha})")
    # For n_test=1000 and nominal 0.90, 99.9% confidence interval is ~[0.86, 0.94]
    assert emp_coverage >= 0.86


def test_certificate_serialization():
    weights = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6], dtype=np.float32)
    means = np.zeros(6, dtype=np.float32)
    stds = np.ones(6, dtype=np.float32)
    quantiles = {"0.05": 0.15, "0.10": 0.10, "0.20": 0.05}

    cert1 = RecallCertificate(
        weights=weights,
        intercept=0.05,
        quantiles=quantiles,
        feature_means=means,
        feature_stds=stds,
    )

    d = cert1.to_dict()
    cert2 = RecallCertificate.from_dict(cert_model=d["model"], cert_calib=d["calib"])

    feat = CertificateFeatures(0.8, 0.7, 0.1, 0.05, 0.02, 0.9)
    c1, p1 = cert1.certify(feat, alpha=0.10)
    c2, p2 = cert2.certify(feat, alpha=0.10)

    assert abs(c1 - c2) < 1e-6
    assert abs(p1 - p2) < 1e-6
