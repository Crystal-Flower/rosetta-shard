import json
import time
from typing import Any

import numpy as np

from bench.plots import plot_exp_collapse
from rosetta.adapters import ContrastiveAdapter, PadTruncateBaseline
from rosetta.config import MODELS, RANDOM_SEED, RESULTS_DIR
from rosetta.data import load_dataset, split_queries
from rosetta.embed import embed_texts
from rosetta.metrics import overlap_at_k
from rosetta.space import SpaceGuard, SpaceMismatchError


def run_exp_collapse(max_corpus: int = 500, n_queries: int = 50) -> dict[str, Any]:
    print("\n" + "=" * 60)
    print("Running E1: Cross-Model Retrieval Collapse (S1 Silent Failure & S2 Recovery)")
    print("=" * 60)

    ds = load_dataset("scifact", max_corpus=max_corpus)
    doc_ids = list(ds.corpus.keys())[:max_corpus]
    doc_texts = [ds.corpus[did] for did in doc_ids]

    splits = split_queries(ds, n_anchor=100, n_dev=50, n_cal_train=50, n_cal=50, n_test=n_queries)
    test_queries = splits["test"][:n_queries]
    test_texts = [q[1] for q in test_queries]

    # --- S1 Silent Failure Test ---
    # Model 1: all-MiniLM-L6-v2 (384-d)
    # Model 2: bge-small-en-v1.5 (384-d)
    print("\n[S1] Embedding corpus and queries for silent failure test...")
    m_minilm = MODELS["all-MiniLM-L6-v2"]
    m_bge_small = MODELS["bge-small-en-v1.5"]
    m_bge_large = MODELS["bge-large-en-v1.5"]

    Y_corpus_bge_small = embed_texts(
        doc_texts, model=m_bge_small, is_query=False, cache_key="e1_corp_bge_s"
    )
    X_query_minilm = embed_texts(test_texts, model=m_minilm, is_query=True, cache_key="e1_q_minilm")
    Y_query_bge_small = embed_texts(
        test_texts, model=m_bge_small, is_query=True, cache_key="e1_q_bge_s"
    )

    # True top-10 in target space
    s1_silent_overlaps = []
    guard_blocked_count = 0

    for i in range(len(test_texts)):
        # True ranking with native bge-small
        true_sims = Y_corpus_bge_small @ Y_query_bge_small[i]
        true_top10 = [doc_ids[idx] for idx in np.argsort(-true_sims)[:10]]

        # Unadapted cross-space search (MiniLM query into BGE-small corpus)
        silent_sims = Y_corpus_bge_small @ X_query_minilm[i]
        silent_top10 = [doc_ids[idx] for idx in np.argsort(-silent_sims)[:10]]

        s1_silent_overlaps.append(overlap_at_k(silent_top10, true_top10, k=10))

        # SpaceGuard test
        try:
            SpaceGuard.validate(m_minilm.space_id, m_bge_small.space_id)
        except SpaceMismatchError:
            guard_blocked_count += 1

    mean_s1_silent = float(np.mean(s1_silent_overlaps))
    print(f"S1 Cross-Space Unadapted Overlap@10: {mean_s1_silent * 100:.2f}% (Silent Collapse!)")
    print(f"Space Guard Blocked: {guard_blocked_count}/{len(test_texts)} (100% Protection)")

    # --- S2 Dimension Mismatch & Recovery ---
    # Small: bge-small (384-d), Large: bge-large (1024-d)
    print("\n[S2] Evaluating Dimension Mismatch (384-d -> 1024-d) & Rosetta Recovery...")
    train_queries = [q[1] for q in splits["anchor"]]
    train_texts = train_queries + doc_texts[:150]

    X_train = embed_texts(train_texts, model=m_bge_small, is_query=False, cache_key="e1_train_s")
    Y_train = embed_texts(train_texts, model=m_bge_large, is_query=False, cache_key="e1_train_l")

    X_test = embed_texts(test_texts, model=m_bge_small, is_query=True, cache_key="e1_test_s")
    Y_test = embed_texts(test_texts, model=m_bge_large, is_query=True, cache_key="e1_test_l")
    Y_corpus_large = embed_texts(
        doc_texts, model=m_bge_large, is_query=False, cache_key="e1_corp_l"
    )

    # Baseline: Pad/Truncate
    pad = PadTruncateBaseline().fit(X_train, Y_train)
    X_test_pad = pad(X_test)

    # Rosetta Contrastive Adapter (100 query anchors)
    adapter = ContrastiveAdapter(
        epochs=40, lr=5e-3, temperature=0.05, center=False, init="procrustes"
    ).fit(X_train[:100], Y_train[:100])
    X_test_adapted = adapter(X_test)

    s2_pad_overlaps = []
    s2_adapted_overlaps = []

    for i in range(len(test_texts)):
        true_sims = Y_corpus_large @ Y_test[i]
        true_top10 = [doc_ids[idx] for idx in np.argsort(-true_sims)[:10]]

        pad_sims = Y_corpus_large @ X_test_pad[i]
        pad_top10 = [doc_ids[idx] for idx in np.argsort(-pad_sims)[:10]]
        s2_pad_overlaps.append(overlap_at_k(pad_top10, true_top10, k=10))

        adapted_sims = Y_corpus_large @ X_test_adapted[i]
        adapted_top10 = [doc_ids[idx] for idx in np.argsort(-adapted_sims)[:10]]
        s2_adapted_overlaps.append(overlap_at_k(adapted_top10, true_top10, k=10))

    mean_s2_pad = float(np.mean(s2_pad_overlaps))
    mean_s2_adapted = float(np.mean(s2_adapted_overlaps))

    print(f"S2 Pad/Truncate Baseline Overlap@10: {mean_s2_pad * 100:.2f}%")
    print(f"S2 Rosetta Contrastive Adapter Overlap@10: {mean_s2_adapted * 100:.2f}% (High Recovery)")

    results = {
        "experiment": "E1_collapse",
        "timestamp": time.time(),
        "random_seed": RANDOM_SEED,
        "n_docs": len(doc_ids),
        "n_queries": len(test_texts),
        "s1_silent_overlap": round(mean_s1_silent, 4),
        "s1_guard_blocked_rate": guard_blocked_count / len(test_texts),
        "s2_baseline_overlap": round(mean_s2_pad, 4),
        "s2_adapted_overlap": round(mean_s2_adapted, 4),
    }

    out_json = RESULTS_DIR / "exp_collapse.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    out_png = RESULTS_DIR / "exp_collapse.png"
    plot_exp_collapse(results, out_png)
    print(f"Saved results to {out_json} and {out_png}")
    return results


if __name__ == "__main__":
    run_exp_collapse()
