import json
import time
from typing import Any

import numpy as np

from bench.plots import plot_exp_adapter_ablation
from rosetta.adapters import (
    CCABaseline,
    ContrastiveAdapter,
    PadTruncateBaseline,
    ProcrustesAdapter,
    RidgeAdapter,
)
from rosetta.config import MODELS, RANDOM_SEED, RESULTS_DIR
from rosetta.data import load_dataset, split_queries
from rosetta.embed import embed_texts
from rosetta.metrics import ndcg_at_k, overlap_at_k, recall_at_k


def run_exp_adapter_ablation(max_corpus: int = 500, n_queries: int = 50) -> dict[str, Any]:
    print("\n" + "=" * 60)
    print("Running E2: Adapter Architecture Ablation (Overlap, Recall, nDCG)")
    print("=" * 60)

    ds = load_dataset("scifact", max_corpus=max_corpus)
    doc_ids = list(ds.corpus.keys())[:max_corpus]
    doc_texts = [ds.corpus[did] for did in doc_ids]

    splits = split_queries(ds, n_anchor=100, n_dev=50, n_cal_train=30, n_cal=30, n_test=n_queries)
    test_queries = splits["test"][:n_queries]
    test_qids = [q[0] for q in test_queries]
    test_texts = [q[1] for q in test_queries]

    train_queries = [q[1] for q in splits["anchor"]]
    all_train_texts = train_queries + doc_texts[:150]

    m_small = MODELS["bge-small-en-v1.5"]
    m_large = MODELS["bge-large-en-v1.5"]

    print("Loading training pairs (selected 100 query anchors)...")
    X_pool = embed_texts(all_train_texts, model=m_small, is_query=False, cache_key="shared_train_s")
    Y_pool = embed_texts(all_train_texts, model=m_large, is_query=False, cache_key="shared_train_l")
    X_train = X_pool[:100]
    Y_train = Y_pool[:100]

    print("Embedding test queries & reference corpus...")
    X_test = embed_texts(test_texts, model=m_small, is_query=True, cache_key="e1_test_s")
    Y_test = embed_texts(test_texts, model=m_large, is_query=True, cache_key="e1_test_l")
    Y_corpus = embed_texts(doc_texts, model=m_large, is_query=False, cache_key="shared_corpus_l")

    # Fit candidate adapters
    print("Fitting candidate adapters...")
    pad_base = PadTruncateBaseline().fit(X_train, Y_train)
    cca_base = CCABaseline(n_components=64).fit(X_train, Y_train)
    proc_adapter = ProcrustesAdapter().fit(X_train, Y_train, center=False)
    ridge_adapter = RidgeAdapter(lam=1e-2, center=False).fit(X_train, Y_train)
    contrastive_adapter = ContrastiveAdapter(
        epochs=40, lr=5e-3, temperature=0.05, center=False, init="procrustes"
    ).fit(X_train, Y_train)

    methods = {
        "random_baseline": np.random.default_rng(RANDOM_SEED).standard_normal(
            Y_test.shape, dtype=np.float32
        ),
        "pad_truncate": pad_base(X_test),
        "cca_baseline": cca_base(X_test),
        "procrustes": proc_adapter(X_test),
        "ridge_adapter": ridge_adapter(X_test),
        "contrastive": contrastive_adapter(X_test),
        "native_large_ceiling": Y_test,
    }
    # Normalize random
    methods["random_baseline"] = methods["random_baseline"] / np.linalg.norm(
        methods["random_baseline"], axis=1, keepdims=True
    )

    # Compute ground truth top-10 in large space
    gt_rankings = []
    for i in range(len(test_texts)):
        sims = Y_corpus @ Y_test[i]
        gt_rankings.append([doc_ids[idx] for idx in np.argsort(-sims)[:10]])

    results = {
        "experiment": "E2_adapter_ablation",
        "timestamp": time.time(),
        "random_seed": RANDOM_SEED,
        "n_anchors": len(X_train),
        "n_docs": len(doc_ids),
        "n_queries": len(test_texts),
        "methods": {},
    }

    print("\n--- Ablation Results ---")
    for name, pred_vectors in methods.items():
        overlaps = []
        recalls = []
        ndcgs = []

        for i, qid in enumerate(test_qids):
            sims = Y_corpus @ pred_vectors[i]
            retrieved_ids = [doc_ids[idx] for idx in np.argsort(-sims)[:10]]

            # Overlap against ceiling
            overlaps.append(overlap_at_k(retrieved_ids, gt_rankings[i], k=10))

            # Task recall & ndcg against qrels if available
            rel_dict = ds.qrels.get(qid, {})
            recalls.append(recall_at_k(retrieved_ids, rel_dict, k=10))
            ndcgs.append(ndcg_at_k(retrieved_ids, rel_dict, k=10))

        m_res = {
            "overlap_at_10": round(float(np.mean(overlaps)), 4),
            "recall_at_10": round(float(np.mean(recalls)), 4),
            "ndcg_at_10": round(float(np.mean(ndcgs)), 4),
        }
        results["methods"][name] = m_res
        print(
            f"{name:<22} | Overlap@10: {m_res['overlap_at_10'] * 100:.1f}% | Recall@10: {m_res['recall_at_10']:.4f} | nDCG@10: {m_res['ndcg_at_10']:.4f}"
        )

    out_json = RESULTS_DIR / "exp_adapter_ablation.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    out_png = RESULTS_DIR / "exp_adapter_ablation.png"
    plot_exp_adapter_ablation(results, out_png)
    print(f"\nSaved results to {out_json} and {out_png}")
    return results


if __name__ == "__main__":
    run_exp_adapter_ablation()
