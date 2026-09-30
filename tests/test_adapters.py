import numpy as np
import pytest

from rosetta.adapters.baseline import CCABaseline, PadTruncateBaseline
from rosetta.adapters.procrustes import ProcrustesAdapter
from rosetta.adapters.ridge import RidgeAdapter


@pytest.fixture
def synthetic_data():
    rng = np.random.default_rng(42)
    n_samples = 200
    d_in = 384
    d_out = 1024

    X = rng.standard_normal((n_samples, d_in), dtype=np.float32)
    X = X / np.linalg.norm(X, axis=1, keepdims=True)

    # Correlated target space
    transform = rng.standard_normal((d_in, d_out), dtype=np.float32)
    Y = X @ transform + 0.1 * rng.standard_normal((n_samples, d_out), dtype=np.float32)
    Y = Y / np.linalg.norm(Y, axis=1, keepdims=True)

    return X, Y


def test_ridge_adapter(synthetic_data):
    X, Y = synthetic_data
    adapter = RidgeAdapter(lam=1e-2, center=True)
    adapter.fit(X[:150], Y[:150])

    assert adapter.W is not None
    assert adapter.W.shape == (384, 1024)

    preds = adapter(X[150:])
    assert preds.shape == (50, 1024)

    # Output must be unit normalized
    norms = np.linalg.norm(preds, axis=1)
    np.testing.assert_allclose(norms, 1.0, atol=1e-5)

    # Cosine similarity with ground truth should be positive and high
    cos_sims = np.sum(preds * Y[150:], axis=1)
    assert np.mean(cos_sims) > 0.5


def test_ridge_tuning(synthetic_data):
    X, Y = synthetic_data
    adapter = RidgeAdapter()
    best_lam = adapter.tune(
        X_train=X[:100],
        Y_train=Y[:100],
        X_val=X[100:150],
        Y_val=Y[100:150],
        candidates=[1e-4, 1e-2, 1.0],
    )
    assert best_lam in [1e-4, 1e-2, 1.0]
    preds = adapter(X[150:])
    assert preds.shape == (50, 1024)


def test_procrustes_adapter(synthetic_data):
    X, Y = synthetic_data
    adapter = ProcrustesAdapter()
    adapter.fit(X[:150], Y[:150], center=False)

    assert adapter.W is not None
    assert adapter.W.shape == (384, 1024)

    preds = adapter(X[150:])
    assert preds.shape == (50, 1024)

    norms = np.linalg.norm(preds, axis=1)
    np.testing.assert_allclose(norms, 1.0, atol=1e-5)

    cos_sims = np.sum(preds * Y[150:], axis=1)
    assert np.mean(cos_sims) > 0.35


def test_pad_truncate_baseline(synthetic_data):
    X, Y = synthetic_data
    # 384 -> 1024 (padding)
    pad = PadTruncateBaseline()
    pad.fit(X, Y)
    preds = pad(X)
    assert preds.shape == (len(X), 1024)
    np.testing.assert_allclose(np.linalg.norm(preds, axis=1), 1.0, atol=1e-5)
    np.testing.assert_allclose(preds[:, 384:], 0.0)

    # 1024 -> 384 (truncating)
    trunc = PadTruncateBaseline()
    trunc.fit(Y, X)
    preds_trunc = trunc(Y)
    assert preds_trunc.shape == (len(Y), 384)
    np.testing.assert_allclose(np.linalg.norm(preds_trunc, axis=1), 1.0, atol=1e-5)


def test_cca_baseline(synthetic_data):
    X, Y = synthetic_data
    cca = CCABaseline(n_components=32)
    cca.fit(X[:150], Y[:150])

    preds = cca(X[150:])
    assert preds.shape == (50, 1024)
    np.testing.assert_allclose(np.linalg.norm(preds, axis=1), 1.0, atol=1e-5)


def test_adapter_determinism(synthetic_data):
    X, Y = synthetic_data
    adapter1 = RidgeAdapter(lam=0.01)
    adapter1.fit(X[:100], Y[:100])

    adapter2 = RidgeAdapter(lam=0.01)
    adapter2.fit(X[:100], Y[:100])

    out1 = adapter1(X[100:120])
    out2 = adapter2(X[100:120])
    np.testing.assert_array_equal(out1, out2)


def test_contrastive_adapter(synthetic_data):
    from rosetta.adapters import ContrastiveAdapter

    X, Y = synthetic_data
    adapter = ContrastiveAdapter(epochs=20, lr=1e-2, center=False)
    adapter.fit(X[:150], Y[:150])

    assert adapter.W is not None
    assert adapter.W.shape == (384, 1024)

    preds = adapter(X[150:])
    assert preds.shape == (50, 1024)

    norms = np.linalg.norm(preds, axis=1)
    np.testing.assert_allclose(norms, 1.0, atol=1e-5)

    cos_sims = np.sum(preds * Y[150:], axis=1)
    assert np.mean(cos_sims) > 0.35

