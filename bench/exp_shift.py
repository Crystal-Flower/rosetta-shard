import json
import time
from typing import Any

import numpy as np

from bench.plots import plot_exp_shift
from rosetta.adapters import ContrastiveAdapter
from rosetta.certificate import CertificateFeatures, extract_features, fit_certificate
from rosetta.config import MODELS, RANDOM_SEED, RESULTS_DIR
from rosetta.data import load_dataset, split_queries
from rosetta.embed import embed_texts
from rosetta.metrics import overlap_at_k
from rosetta.store import Hit


def run_exp_shift(max_corpus: int = 500) -> dict[str, Any]:
    print("\n" + "=" * 60)
    print("Running E6: Distribution Shift & OOD Detection Behavior")
    print("=" * 60)

    # In-domain: SciFact (biomedical claims)
    ds_scifact = load_dataset("scifact", max_corpus=max_corpus)
    doc_ids = list(ds_scifact.corpus.keys())[:max_corpus]
    doc_texts = [ds_scifact.corpus[did] for did in doc_ids]

    splits = split_queries(ds_scifact, n_anchor=100, n_dev=50, n_cal_train=30, n_cal=30, n_test=50)

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

    def eval_queries(q_list, split_name=None):
        q_cache_s = "e1_test_s" if split_name == "test" else (f"e4_{split_name}_s" if split_name else None)
        q_cache_l = "e1_test_l" if split_name == "test" else (f"e4_{split_name}_l" if split_name else None)
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

    # 1. Fit on in-domain SciFact
    cert = fit_certificate(
        features_train=F_train,
        losses_train=L_train,
        features_cal=F_cal,
        losses_cal=L_cal,
        alphas=[0.10],
        anchors_centroids=anchors_centroids,
        centroid_residuals=centroid_residuals,
    )

    # 2. Evaluate In-Domain Test
    F_in_test, _, O_in_test = eval_queries([q[1] for q in splits["test"]], "test")
    in_covered = 0
    for i in range(len(F_in_test)):
        feat = CertificateFeatures(
            sim_anchor_max=float(F_in_test[i, 0]),
            sim_anchor_top5=float(F_in_test[i, 1]),
            margin_1_10=float(F_in_test[i, 2]),
            margin_1_2=float(F_in_test[i, 3]),
            local_fit_err=float(F_in_test[i, 4]),
            self_consistency=float(F_in_test[i, 5]),
        )
        c_overlap, _ = cert.certify(feat, alpha=0.10)
        if O_in_test[i] >= c_overlap - 1e-6:
            in_covered += 1

    in_domain_coverage = in_covered / len(F_in_test)

    # 3. Evaluate Out-of-Distribution Queries (Financial / General queries)
    ood_query_samples = [
        "What is the average return on investment for high-yield bond portfolios?",
        "How do capital gains taxes affect cryptocurrency exchange transactions?",
        "Federal reserve interest rate hike impact on mortgage rates and real estate",
        "Quarterly earnings EBITDA margins comparison between retail and tech sectors",
        "Options trading implied volatility strike price arbitrage strategies",
        "Convertible debt covenants debt-to-equity ratio restructuring",
        "Venture capital convertible notes discount rate valuation cap terms",
        "Index fund expense ratios dividend reinvestment passive investing",
        "Treasury bills liquidity risk yield curve inversion macroeconomic indicators",
        "Automated algorithmic trading latency order routing limit books",
    ] * 5

    F_ood, _, O_ood = eval_queries(ood_query_samples)
    ood_covered = 0
    ood_flagged_count = 0

    for i in range(len(F_ood)):
        sim_max = float(F_ood[i, 0])
        feat = CertificateFeatures(
            sim_anchor_max=sim_max,
            sim_anchor_top5=float(F_ood[i, 1]),
            margin_1_10=float(F_ood[i, 2]),
            margin_1_2=float(F_ood[i, 3]),
            local_fit_err=float(F_ood[i, 4]),
            self_consistency=float(F_ood[i, 5]),
        )
        c_overlap, _ = cert.certify(feat, alpha=0.10)
        if O_ood[i] >= c_overlap - 1e-6:
            ood_covered += 1

        # OOD gate check: sim_anchor_max low or certificate very low
        if sim_max < 0.40 or c_overlap < 0.30:
            ood_flagged_count += 1

    shifted_coverage = ood_covered / len(F_ood)
    ood_escalation_rate = ood_flagged_count / len(F_ood)

    print(f"\nIn-Domain Coverage: {in_domain_coverage * 100:.1f}% (Nominal Guarantee: 90.0%)")
    print(f"Shifted OOD Coverage: {shifted_coverage * 100:.1f}% (Honest degradation under shift)")
    print(f"OOD Flagged & Escalated: {ood_escalation_rate * 100:.1f}% (Router safety catch)")

    results = {
        "experiment": "E6_shift",
        "timestamp": time.time(),
        "random_seed": RANDOM_SEED,
        "nominal_guarantee": 0.90,
        "in_domain_coverage": round(in_domain_coverage, 4),
        "shifted_coverage": round(shifted_coverage, 4),
        "ood_escalation_rate": round(ood_escalation_rate, 4),
    }

    out_json = RESULTS_DIR / "exp_shift.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    out_png = RESULTS_DIR / "exp_shift.png"
    plot_exp_shift(results, out_png)
    print(f"\nSaved results to {out_json} and {out_png}")
    return results


if __name__ == "__main__":
    run_exp_shift()
