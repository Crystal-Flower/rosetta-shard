import json
import time
from typing import Any

import numpy as np

from bench.plots import plot_exp_repair
from rosetta.adapters import ContrastiveAdapter
from rosetta.config import MODELS, RANDOM_SEED, RESULTS_DIR
from rosetta.data import load_dataset, split_queries
from rosetta.embed import embed_texts
from rosetta.metrics import overlap_at_k


def run_exp_repair(max_corpus: int = 500, n_queries: int = 50) -> dict[str, Any]:
    print("\n" + "=" * 60)
    print("Running E7: Byte-Budgeted Hot-Set Repair Efficiency")
    print("=" * 60)

    ds = load_dataset("scifact", max_corpus=max_corpus)
    doc_ids = list(ds.corpus.keys())[:max_corpus]
    doc_texts = [ds.corpus[did] for did in doc_ids]

    splits = split_queries(ds, n_anchor=100, n_dev=50, n_cal_train=30, n_cal=30, n_test=n_queries)
    test_queries = splits["test"][:n_queries]
    test_texts = [q[1] for q in test_queries]

    m_small = MODELS["bge-small-en-v1.5"]
    m_large = MODELS["bge-large-en-v1.5"]

    train_texts = [q[1] for q in splits["anchor"]] + doc_texts[:150]
    X_pool = embed_texts(train_texts, model=m_small, is_query=False, cache_key="shared_train_s")
    Y_pool = embed_texts(train_texts, model=m_large, is_query=False, cache_key="shared_train_l")
    X_train = X_pool[:100]
    Y_train = Y_pool[:100]
    adapter = ContrastiveAdapter(
        epochs=40, lr=5e-3, temperature=0.05, center=False, init="procrustes"
    ).fit(X_train, Y_train)

    # Embed corpus in both spaces
    X_corpus = embed_texts(doc_texts, model=m_small, is_query=False, cache_key="shared_corpus_s")
    Y_corpus = embed_texts(doc_texts, model=m_large, is_query=False, cache_key="shared_corpus_l")

    # Translated corpus on edge
    Y_corpus_translated = adapter(X_corpus)

    # Cloud precomputes doc error map e_i = 1 - cos(ŷ_i, y_i)
    doc_error_map = {}
    for i, did in enumerate(doc_ids):
        err = float(np.clip(1.0 - np.dot(Y_corpus_translated[i], Y_corpus[i]), 0.0, 1.0))
        doc_error_map[did] = err

    # Queries
    Y_test = embed_texts(test_texts, model=m_large, is_query=True, cache_key="e1_test_l")
    X_test = embed_texts(test_texts, model=m_small, is_query=True, cache_key="e1_test_s")
    Q_hat = adapter(X_test)

    # Ground truth ranking
    gt_top10 = []
    for i in range(len(test_texts)):
        sims = Y_corpus @ Y_test[i]
        gt_top10.append([doc_ids[idx] for idx in np.argsort(-sims)[:10]])

    # Count observed query hits on un-repaired translated corpus
    hit_counts = {did: 0 for did in doc_ids}
    for i in range(len(test_texts)):
        sims = Y_corpus_translated @ Q_hat[i]
        for idx in np.argsort(-sims)[:10]:
            hit_counts[doc_ids[idx]] += 1

    # Bytes per 1024-d float32 vector: 1024 * 4 + 64 bytes = ~4.16 KB
    bytes_per_vec = 1024 * 4 + 64
    budgets_kb = [0, 25, 50, 100, 200, 400]
    policies = ["priority", "frequency", "error", "random"]

    policy_results = {p: {"overlaps": []} for p in policies}

    for pol in policies:
        print(f"\nEvaluating repair policy: {pol}")
        for b_kb in budgets_kb:
            budget_bytes = b_kb * 1024
            max_vectors = budget_bytes // bytes_per_vec

            # Candidate selection
            if pol == "random":
                rng = np.random.default_rng(RANDOM_SEED)
                shuffled = rng.permutation(doc_ids).tolist()
                selected_ids = shuffled[:max_vectors]
            elif pol == "frequency":
                sorted_docs = sorted(doc_ids, key=lambda d: hit_counts.get(d, 0), reverse=True)
                selected_ids = sorted_docs[:max_vectors]
            elif pol == "error":
                sorted_docs = sorted(doc_ids, key=lambda d: doc_error_map.get(d, 0.0), reverse=True)
                selected_ids = sorted_docs[:max_vectors]
            else:  # priority: hits * error
                scored = [(d, hit_counts.get(d, 1) * doc_error_map.get(d, 0.05)) for d in doc_ids]
                scored.sort(key=lambda x: x[1], reverse=True)
                selected_ids = [x[0] for x in scored[:max_vectors]]

            selected_set = set(selected_ids)

            # Construct repaired corpus
            repaired_corpus = Y_corpus_translated.copy()
            for i, did in enumerate(doc_ids):
                if did in selected_set:
                    repaired_corpus[i] = Y_corpus[i]  # Exact vector from cloud

            # Evaluate top-10 overlap
            overlaps = []
            for i in range(len(test_texts)):
                sims = repaired_corpus @ Q_hat[i]
                top10 = [doc_ids[idx] for idx in np.argsort(-sims)[:10]]
                overlaps.append(overlap_at_k(top10, gt_top10[i], k=10))

            mean_ov = float(np.mean(overlaps))
            policy_results[pol]["overlaps"].append(round(mean_ov, 4))
            print(
                f"Budget: {b_kb:3d} KB ({len(selected_ids):2d} docs) | Overlap@10: {mean_ov * 100:.1f}%"
            )

    results = {
        "experiment": "E7_repair",
        "timestamp": time.time(),
        "random_seed": RANDOM_SEED,
        "budgets_kb": budgets_kb,
        "bytes_per_vector": bytes_per_vec,
        "policies": policy_results,
    }

    out_json = RESULTS_DIR / "exp_repair.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    out_png = RESULTS_DIR / "exp_repair.png"
    plot_exp_repair(results, out_png)
    print(f"\nSaved results to {out_json} and {out_png}")
    return results


if __name__ == "__main__":
    run_exp_repair()
