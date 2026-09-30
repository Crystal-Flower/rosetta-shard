import json
import time
from typing import Any

import numpy as np

from bench.plots import plot_exp_anchor_curve
from rosetta.adapters import ContrastiveAdapter, ProcrustesAdapter, RidgeAdapter
from rosetta.config import MODELS, RANDOM_SEED, RESULTS_DIR
from rosetta.data import load_dataset, split_queries
from rosetta.embed import embed_texts
from rosetta.metrics import overlap_at_k


def run_exp_anchor_curve(max_corpus: int = 500, n_queries: int = 50) -> dict[str, Any]:
    print("\n" + "=" * 60)
    print("Running E3: Anchor Sample Efficiency Curve (Data Economy)")
    print("=" * 60)

    ds = load_dataset("scifact", max_corpus=max_corpus)
    doc_ids = list(ds.corpus.keys())[:max_corpus]
    doc_texts = [ds.corpus[did] for did in doc_ids]

    splits = split_queries(ds, n_anchor=100, n_dev=50, n_cal_train=30, n_cal=30, n_test=n_queries)
    test_queries = splits["test"][:n_queries]
    test_texts = [q[1] for q in test_queries]

    all_anchor_texts = [q[1] for q in splits["anchor"]] + doc_texts[:150]

    m_small = MODELS["bge-small-en-v1.5"]
    m_large = MODELS["bge-large-en-v1.5"]

    print("Embedding full anchor pool...")
    X_pool = embed_texts(all_anchor_texts, model=m_small, is_query=False, cache_key="shared_train_s")
    Y_pool = embed_texts(all_anchor_texts, model=m_large, is_query=False, cache_key="shared_train_l")

    X_test = embed_texts(test_texts, model=m_small, is_query=True, cache_key="e1_test_s")
    Y_test = embed_texts(test_texts, model=m_large, is_query=True, cache_key="e1_test_l")
    Y_corpus = embed_texts(doc_texts, model=m_large, is_query=False, cache_key="shared_corpus_l")

    gt_top10 = []
    for i in range(len(test_texts)):
        sims = Y_corpus @ Y_test[i]
        gt_top10.append([doc_ids[idx] for idx in np.argsort(-sims)[:10]])

    anchor_counts = [25, 50, 100, 150, 200, 250]
    ridge_overlaps = []
    proc_overlaps = []
    contrastive_overlaps = []

    print("\nEvaluating efficiency across anchor budget:")
    for n in anchor_counts:
        X_sub = X_pool[:n]
        Y_sub = Y_pool[:n]

        # 1. Ridge
        ridge = RidgeAdapter(lam=1e-2, center=False).fit(X_sub, Y_sub)
        pred_ridge = ridge(X_test)
        r_scores = [
            overlap_at_k([doc_ids[idx] for idx in np.argsort(-(Y_corpus @ pred_ridge[i]))[:10]], gt_top10[i], k=10)
            for i in range(len(test_texts))
        ]
        mean_r = float(np.mean(r_scores))
        ridge_overlaps.append(round(mean_r, 4))

        # 2. Procrustes
        proc = ProcrustesAdapter().fit(X_sub, Y_sub, center=False)
        pred_proc = proc(X_test)
        p_scores = [
            overlap_at_k([doc_ids[idx] for idx in np.argsort(-(Y_corpus @ pred_proc[i]))[:10]], gt_top10[i], k=10)
            for i in range(len(test_texts))
        ]
        mean_p = float(np.mean(p_scores))
        proc_overlaps.append(round(mean_p, 4))

        # 3. Contrastive
        contra = ContrastiveAdapter(
            epochs=30, lr=5e-3, temperature=0.05, center=False, init="procrustes"
        ).fit(X_sub, Y_sub)
        pred_contra = contra(X_test)
        c_scores = [
            overlap_at_k([doc_ids[idx] for idx in np.argsort(-(Y_corpus @ pred_contra[i]))[:10]], gt_top10[i], k=10)
            for i in range(len(test_texts))
        ]
        mean_c = float(np.mean(c_scores))
        contrastive_overlaps.append(round(mean_c, 4))

        print(
            f"Anchors: {n:3d} | Ridge: {mean_r * 100:4.1f}% | Procrustes: {mean_p * 100:4.1f}% | Contrastive: {mean_c * 100:4.1f}%"
        )

    results = {
        "experiment": "E3_anchor_curve",
        "timestamp": time.time(),
        "random_seed": RANDOM_SEED,
        "anchor_counts": anchor_counts,
        "ridge_overlaps": ridge_overlaps,
        "procrustes_overlaps": proc_overlaps,
        "contrastive_overlaps": contrastive_overlaps,
    }

    out_json = RESULTS_DIR / "exp_anchor_curve.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    out_png = RESULTS_DIR / "exp_anchor_curve.png"
    plot_exp_anchor_curve(results, out_png)
    print(f"\nSaved results to {out_json} and {out_png}")
    return results


if __name__ == "__main__":
    run_exp_anchor_curve()
