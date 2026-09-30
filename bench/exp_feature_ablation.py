import json
import time
from typing import Any

import numpy as np
from sklearn.linear_model import Ridge

from bench.plots import plot_exp_feature_ablation
from rosetta.adapters import ContrastiveAdapter
from rosetta.certificate import extract_features
from rosetta.config import MODELS, RANDOM_SEED, RESULTS_DIR
from rosetta.data import load_dataset, split_queries
from rosetta.embed import embed_texts
from rosetta.metrics import overlap_at_k
from rosetta.store import Hit


def run_exp_feature_ablation(max_corpus: int = 500) -> dict[str, Any]:
    print("\n" + "=" * 60)
    print("Running E9: Certificate Loss Predictor Feature Ablation")
    print("=" * 60)

    ds = load_dataset("scifact", max_corpus=max_corpus)
    doc_ids = list(ds.corpus.keys())[:max_corpus]
    doc_texts = [ds.corpus[did] for did in doc_ids]

    splits = split_queries(ds, n_anchor=100, n_dev=50, n_cal_train=30, n_cal=30, n_test=50)

    m_small = MODELS["bge-small-en-v1.5"]
    m_large = MODELS["bge-large-en-v1.5"]

    all_anchor_texts = [q[1] for q in splits["anchor"]] + doc_texts[:150]
    X_pool = embed_texts(all_anchor_texts, model=m_small, is_query=False, cache_key="shared_train_s")
    Y_pool = embed_texts(all_anchor_texts, model=m_large, is_query=False, cache_key="shared_train_l")
    X_anc = X_pool[:100]
    Y_anc = Y_pool[:100]
    adapter = ContrastiveAdapter(
        epochs=40, lr=5e-3, temperature=0.05, center=False, init="procrustes"
    ).fit(X_anc, Y_anc)

    rng = np.random.default_rng(RANDOM_SEED)
    centroid_indices = rng.choice(len(X_anc), size=min(32, len(X_anc)), replace=False)
    anchors_centroids = X_anc[centroid_indices]
    centroid_residuals = np.clip(
        1.0 - np.sum(adapter(anchors_centroids) * Y_anc[centroid_indices], axis=1), 0.0, 1.0
    )

    Y_corpus = embed_texts(doc_texts, model=m_large, is_query=False, cache_key="shared_corpus_l")

    def eval_queries(q_list, cache_s=None, cache_l=None):
        X_q = embed_texts(q_list, model=m_small, is_query=True, cache_key=cache_s)
        Y_q = embed_texts(q_list, model=m_large, is_query=True, cache_key=cache_l)
        Q_hat = adapter(X_q)
        feats, losses = [], []
        for i in range(len(q_list)):
            true_sims = Y_corpus @ Y_q[i]
            true_top10 = [doc_ids[idx] for idx in np.argsort(-true_sims)[:10]]
            trans_sims = Y_corpus @ Q_hat[i]
            trans_idx = np.argsort(-trans_sims)[:10]
            trans_top10 = [doc_ids[idx] for idx in trans_idx]
            hits = [
                Hit(id=doc_ids[idx], score=float(trans_sims[idx]), payload={}) for idx in trans_idx
            ]
            feat = extract_features(
                q_small=X_q[i],
                q_hat=Q_hat[i],
                hits=hits,
                anchors_centroids=anchors_centroids,
                centroid_residuals=centroid_residuals,
                k=10,
            )
            overlap = overlap_at_k(trans_top10, true_top10, k=10)
            feats.append(feat.to_array())
            losses.append(1.0 - overlap)
        return np.array(feats, dtype=np.float32), np.array(losses, dtype=np.float32)

    F_train, L_train = eval_queries([q[1] for q in splits["cal_train"]], cache_s="e4_cal_train_s", cache_l="e4_cal_train_l")
    F_test, L_test = eval_queries([q[1] for q in splits["test"]], cache_s="e1_test_s", cache_l="e1_test_l")

    # Feature indices:
    # 0: sim_anchor_max
    # 1: sim_anchor_top5
    # 2: margin_1_10
    # 3: margin_1_2
    # 4: local_fit_err
    # 5: self_consistency

    feature_subsets = {
        "full_model": list(range(6)),
        "drop_anchor_sims": [2, 3, 4, 5],
        "drop_margins": [0, 1, 4, 5],
        "drop_local_fit_err": [0, 1, 2, 3, 5],
        "drop_self_consistency": [0, 1, 2, 3, 4],
        "only_anchor_sims": [0, 1],
        "constant_baseline": [],
    }

    ablations = {}
    print("\nFeature ablation results (Test Loss MAE):")
    for name, indices in feature_subsets.items():
        if not indices:
            # Predict mean
            pred = np.full_like(L_test, np.mean(L_train))
        else:
            X_tr = F_train[:, indices]
            X_te = F_test[:, indices]
            means = np.mean(X_tr, axis=0)
            stds = np.std(X_tr, axis=0)
            stds[stds < 1e-6] = 1.0

            reg = Ridge(alpha=1.0)
            reg.fit((X_tr - means) / stds, L_train)
            pred = reg.predict((X_te - means) / stds)
            pred = np.clip(pred, 0.0, 1.0)

        mae = float(np.mean(np.abs(pred - L_test)))
        ablations[name] = {"loss_mae": round(mae, 4), "n_features": len(indices)}
        print(f"{name:<25} | Features: {len(indices)} | Test MAE: {mae:.4f}")

    results = {
        "experiment": "E9_feature_ablation",
        "timestamp": time.time(),
        "random_seed": RANDOM_SEED,
        "n_train": len(F_train),
        "n_test": len(F_test),
        "ablations": ablations,
    }

    out_json = RESULTS_DIR / "exp_feature_ablation.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    out_png = RESULTS_DIR / "exp_feature_ablation.png"
    plot_exp_feature_ablation(results, out_png)
    print(f"\nSaved results to {out_json} and {out_png}")
    return results


if __name__ == "__main__":
    run_exp_feature_ablation()
