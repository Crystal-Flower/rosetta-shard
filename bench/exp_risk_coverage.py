import json
import time
from typing import Any

import numpy as np

from bench.plots import plot_exp_risk_coverage
from rosetta.adapters import ContrastiveAdapter
from rosetta.certificate import CertificateFeatures, extract_features, fit_certificate
from rosetta.config import MODELS, RANDOM_SEED, RESULTS_DIR
from rosetta.data import load_dataset, split_queries
from rosetta.embed import embed_texts
from rosetta.metrics import overlap_at_k
from rosetta.store import Hit


def run_exp_risk_coverage(max_corpus: int = 500) -> dict[str, Any]:
    print("\n" + "=" * 60)
    print("Running E5: Risk–Coverage Tradeoff with Gated Edge Routing")
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
    translated_anc = adapter(anchors_centroids)
    centroid_residuals = np.clip(
        1.0 - np.sum(translated_anc * Y_anc[centroid_indices], axis=1), 0.0, 1.0
    )

    Y_corpus = embed_texts(doc_texts, model=m_large, is_query=False, cache_key="shared_corpus_l")

    def eval_queries(q_list, split_name="test"):
        q_cache_s = "e1_test_s" if split_name == "test" else f"e4_{split_name}_s"
        q_cache_l = "e1_test_l" if split_name == "test" else f"e4_{split_name}_l"
        X_q = embed_texts(q_list, model=m_small, is_query=True, cache_key=q_cache_s)
        Y_q = embed_texts(q_list, model=m_large, is_query=True, cache_key=q_cache_l)
        Q_hat = adapter(X_q)
        feats, losses, overlaps = [], [], []
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
            overlaps.append(overlap)
        return (
            np.array(feats, dtype=np.float32),
            np.array(losses, dtype=np.float32),
            np.array(overlaps, dtype=np.float32),
        )

    F_train, L_train, _ = eval_queries([q[1] for q in splits["cal_train"]], "cal_train")
    F_cal, L_cal, _ = eval_queries([q[1] for q in splits["cal"]], "cal")
    F_test, L_test, O_test = eval_queries([q[1] for q in splits["test"]], "test")

    cert = fit_certificate(
        features_train=F_train,
        losses_train=L_train,
        features_cal=F_cal,
        losses_cal=L_cal,
        alphas=[0.10],
        anchors_centroids=anchors_centroids,
        centroid_residuals=centroid_residuals,
    )

    certified_overlaps = []
    for i in range(len(F_test)):
        feat = CertificateFeatures(
            sim_anchor_max=float(F_test[i, 0]),
            sim_anchor_top5=float(F_test[i, 1]),
            margin_1_10=float(F_test[i, 2]),
            margin_1_2=float(F_test[i, 3]),
            local_fit_err=float(F_test[i, 4]),
            self_consistency=float(F_test[i, 5]),
        )
        c_overlap, _ = cert.certify(feat, alpha=0.10)
        certified_overlaps.append(c_overlap)

    certified_overlaps = np.array(certified_overlaps)

    taus = [0.00, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
    served_fractions = []
    mean_overlaps = []
    p10_overlaps = []

    print("\nSweeping confidence threshold tau_serve:")
    for tau in taus:
        served_mask = certified_overlaps >= tau
        n_served = np.sum(served_mask)
        fraction = float(n_served / len(certified_overlaps))
        served_fractions.append(round(fraction, 4))

        if n_served > 0:
            served_true_overlaps = O_test[served_mask]
            m_overlap = float(np.mean(served_true_overlaps))
            p10 = float(np.percentile(served_true_overlaps, 10))
        else:
            m_overlap = 1.0
            p10 = 1.0

        mean_overlaps.append(round(m_overlap, 4))
        p10_overlaps.append(round(p10, 4))
        print(
            f"tau: {tau:.2f} | Served Locally: {fraction * 100:5.1f}% | Mean Overlap: {m_overlap * 100:5.1f}% | Worst-Decile (p10): {p10 * 100:5.1f}%"
        )

    results = {
        "experiment": "E5_risk_coverage",
        "timestamp": time.time(),
        "random_seed": RANDOM_SEED,
        "tau_thresholds": taus,
        "local_served_fraction": served_fractions,
        "mean_overlap_served": mean_overlaps,
        "p10_overlap_served": p10_overlaps,
    }

    out_json = RESULTS_DIR / "exp_risk_coverage.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    out_png = RESULTS_DIR / "exp_risk_coverage.png"
    plot_exp_risk_coverage(results, out_png)
    print(f"\nSaved results to {out_json} and {out_png}")
    return results


if __name__ == "__main__":
    run_exp_risk_coverage()
