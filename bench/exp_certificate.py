import json
import time
from typing import Any

import numpy as np

from bench.plots import plot_exp_certificate
from rosetta.adapters import ContrastiveAdapter
from rosetta.certificate import (
    CertificateFeatures,
    extract_features,
    fit_certificate,
)
from rosetta.config import MODELS, RANDOM_SEED, RESULTS_DIR
from rosetta.data import load_dataset, split_queries
from rosetta.embed import embed_texts
from rosetta.metrics import overlap_at_k
from rosetta.store import Hit


def run_exp_certificate(max_corpus: int = 500) -> dict[str, Any]:
    print("\n" + "=" * 60)
    print("Running E4: Conformal Recall Certificate Calibration")
    print("=" * 60)

    ds = load_dataset("scifact", max_corpus=max_corpus)
    doc_ids = list(ds.corpus.keys())[:max_corpus]
    doc_texts = [ds.corpus[did] for did in doc_ids]

    splits = split_queries(ds, n_anchor=100, n_dev=50, n_cal_train=30, n_cal=30, n_test=50)

    m_small = MODELS["bge-small-en-v1.5"]
    m_large = MODELS["bge-large-en-v1.5"]

    # 1. Fit Selected Adapter (Contrastive on 100 query anchors)
    all_anchor_texts = [q[1] for q in splits["anchor"]] + doc_texts[:150]
    X_pool = embed_texts(all_anchor_texts, model=m_small, is_query=False, cache_key="shared_train_s")
    Y_pool = embed_texts(all_anchor_texts, model=m_large, is_query=False, cache_key="shared_train_l")
    X_anc = X_pool[:100]
    Y_anc = Y_pool[:100]

    adapter = ContrastiveAdapter(
        epochs=40, lr=5e-3, temperature=0.05, center=False, init="procrustes"
    ).fit(X_anc, Y_anc)

    # Precompute anchor centroids (k-means or random anchor subset)
    rng = np.random.default_rng(RANDOM_SEED)
    centroid_indices = rng.choice(len(X_anc), size=min(32, len(X_anc)), replace=False)
    anchors_centroids = X_anc[centroid_indices]

    # Precompute centroid residuals
    translated_anc = adapter(anchors_centroids)
    Y_centroids = Y_anc[centroid_indices]
    centroid_residuals = np.clip(1.0 - np.sum(translated_anc * Y_centroids, axis=1), 0.0, 1.0)

    # 2. Reference Corpus
    Y_corpus = embed_texts(doc_texts, model=m_large, is_query=False, cache_key="shared_corpus_l")

    # Helper function to compute features and losses for a query split
    def evaluate_split(split_name: str):
        queries = [q[1] for q in splits[split_name]]
        q_cache_s = "e1_test_s" if split_name == "test" else f"e4_{split_name}_s"
        q_cache_l = "e1_test_l" if split_name == "test" else f"e4_{split_name}_l"
        X_q = embed_texts(queries, model=m_small, is_query=True, cache_key=q_cache_s)
        Y_q = embed_texts(queries, model=m_large, is_query=True, cache_key=q_cache_l)
        Q_hat = adapter(X_q)

        features_list = []
        losses_list = []
        overlaps_list = []

        for i in range(len(queries)):
            # Large space true ranking
            true_sims = Y_corpus @ Y_q[i]
            true_top10 = [doc_ids[idx] for idx in np.argsort(-true_sims)[:10]]

            # Translated local ranking
            trans_sims = Y_corpus @ Q_hat[i]
            trans_idx = np.argsort(-trans_sims)[:10]
            trans_top10 = [doc_ids[idx] for idx in trans_idx]

            # Construct mock local hits for feature extraction
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
            loss = 1.0 - overlap

            features_list.append(feat.to_array())
            losses_list.append(loss)
            overlaps_list.append(overlap)

        return (
            np.array(features_list, dtype=np.float32),
            np.array(losses_list, dtype=np.float32),
            np.array(overlaps_list, dtype=np.float32),
        )

    print("Computing calibration features on Q_cal_train and Q_cal...")
    F_cal_train, L_cal_train, _ = evaluate_split("cal_train")
    F_cal, L_cal, _ = evaluate_split("cal")

    alphas = [0.05, 0.10, 0.20]
    print("Fitting Conformal Recall Certificate...")
    cert = fit_certificate(
        features_train=F_cal_train,
        losses_train=L_cal_train,
        features_cal=F_cal,
        losses_cal=L_cal,
        alphas=alphas,
        anchors_centroids=anchors_centroids,
        centroid_residuals=centroid_residuals,
    )

    print("\nEvaluating empirical coverage on held-out Q_test queries...")
    F_test, L_test, O_test = evaluate_split("test")

    empirical_coverages = []
    mean_certified = []

    for a in alphas:
        covered = 0
        certs = []
        for i in range(len(F_test)):
            feat = CertificateFeatures(
                sim_anchor_max=float(F_test[i, 0]),
                sim_anchor_top5=float(F_test[i, 1]),
                margin_1_10=float(F_test[i, 2]),
                margin_1_2=float(F_test[i, 3]),
                local_fit_err=float(F_test[i, 4]),
                self_consistency=float(F_test[i, 5]),
            )
            c_overlap, _ = cert.certify(feat, alpha=a)
            certs.append(c_overlap)
            if O_test[i] >= c_overlap - 1e-6:
                covered += 1

        emp_cov = covered / len(F_test)
        empirical_coverages.append(round(emp_cov, 4))
        mean_certified.append(round(float(np.mean(certs)), 4))
        print(
            f"Alpha: {a:4.2f} | Nominal: {(1 - a) * 100:4.1f}% | Empirical Coverage: {emp_cov * 100:5.1f}% | Mean Certified Overlap: {np.mean(certs) * 100:4.1f}%"
        )

    results = {
        "experiment": "E4_certificate",
        "timestamp": time.time(),
        "random_seed": RANDOM_SEED,
        "n_cal_train": len(F_cal_train),
        "n_cal": len(F_cal),
        "n_test": len(F_test),
        "alphas": alphas,
        "empirical_coverage": empirical_coverages,
        "mean_certified_overlap": mean_certified,
        "mean_true_overlap": round(float(np.mean(O_test)), 4),
    }

    out_json = RESULTS_DIR / "exp_certificate.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    out_png = RESULTS_DIR / "exp_certificate.png"
    plot_exp_certificate(results, out_png)
    print(f"\nSaved results to {out_json} and {out_png}")
    return results


if __name__ == "__main__":
    run_exp_certificate()
