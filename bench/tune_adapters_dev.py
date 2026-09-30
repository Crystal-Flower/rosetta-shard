import json
from typing import Any

import numpy as np

from rosetta.adapters import ContrastiveAdapter, ProcrustesAdapter, RidgeAdapter
from rosetta.config import MODELS, RESULTS_DIR
from rosetta.data import load_dataset, split_queries
from rosetta.embed import embed_texts
from rosetta.metrics import overlap_at_k


def run_tune_adapters_dev(max_corpus: int = 500) -> dict[str, Any]:
    print("\n" + "=" * 70)
    print("TUNING ADAPTERS ON DEV SPLIT (Ridge vs Procrustes vs Contrastive)")
    print("=" * 70)

    ds = load_dataset("scifact", max_corpus=max_corpus)
    doc_ids = list(ds.corpus.keys())[:max_corpus]
    doc_texts = [ds.corpus[did] for did in doc_ids]

    splits = split_queries(ds, n_anchor=100, n_dev=50, n_cal_train=30, n_cal=30, n_test=50)
    dev_queries = splits["dev"]
    dev_texts = [q[1] for q in dev_queries]

    all_anchor_texts = [q[1] for q in splits["anchor"]] + doc_texts[:150]

    m_small = MODELS["bge-small-en-v1.5"]
    m_large = MODELS["bge-large-en-v1.5"]

    print("Embedding full anchor pool and dev queries...")
    X_pool = embed_texts(all_anchor_texts, model=m_small, is_query=False, cache_key="shared_train_s")
    Y_pool = embed_texts(all_anchor_texts, model=m_large, is_query=False, cache_key="shared_train_l")

    X_dev = embed_texts(dev_texts, model=m_small, is_query=True, cache_key="shared_dev_s")
    Y_dev = embed_texts(dev_texts, model=m_large, is_query=True, cache_key="shared_dev_l")
    Y_corpus = embed_texts(doc_texts, model=m_large, is_query=False, cache_key="shared_corpus_l")

    # Ground truth top-10 in large space on dev split
    gt_dev_top10 = []
    for i in range(len(dev_texts)):
        sims = Y_corpus @ Y_dev[i]
        gt_dev_top10.append([doc_ids[idx] for idx in np.argsort(-sims)[:10]])

    def evaluate(adapter):
        preds = adapter(X_dev)
        scores = []
        for i in range(len(dev_texts)):
            sims = Y_corpus @ preds[i]
            top10 = [doc_ids[idx] for idx in np.argsort(-sims)[:10]]
            scores.append(overlap_at_k(top10, gt_dev_top10[i], k=10))
        return float(np.mean(scores))

    anchor_options = [50, 100, 250]
    results = []

    for n_anc in anchor_options:
        X_sub = X_pool[:n_anc]
        Y_sub = Y_pool[:n_anc]
        desc = "queries only" if n_anc <= 100 else f"100 queries + {n_anc - 100} docs"
        print(f"\n--- Anchor Count: {n_anc} ({desc}) ---")

        # 1. Ridge center=False
        for lam in [1e-4, 1e-2, 1e-1]:
            r_nocenter = RidgeAdapter(lam=lam, center=False).fit(X_sub, Y_sub)
            score = evaluate(r_nocenter)
            results.append({
                "adapter": "Ridge",
                "center": False,
                "n_anchors": n_anc,
                "hparams": f"lam={lam}",
                "dev_overlap": round(score, 4),
            })
            print(f"Ridge (center=False, lam={lam:<5}): Dev Overlap@10 = {score * 100:.2f}%")

        # 2. Ridge center=True
        for lam in [1e-4, 1e-2, 1e-1]:
            r_center = RidgeAdapter(lam=lam, center=True).fit(X_sub, Y_sub)
            score = evaluate(r_center)
            results.append({
                "adapter": "Ridge",
                "center": True,
                "n_anchors": n_anc,
                "hparams": f"lam={lam}",
                "dev_overlap": round(score, 4),
            })
            print(f"Ridge (center=True,  lam={lam:<5}): Dev Overlap@10 = {score * 100:.2f}%")

        # 3. Procrustes center=False
        p_nocenter = ProcrustesAdapter().fit(X_sub, Y_sub, center=False)
        score = evaluate(p_nocenter)
        results.append({
            "adapter": "Procrustes",
            "center": False,
            "n_anchors": n_anc,
            "hparams": "orthonormal",
            "dev_overlap": round(score, 4),
        })
        print(f"Procrustes (center=False):         Dev Overlap@10 = {score * 100:.2f}%")

        # 4. Procrustes center=True
        p_center = ProcrustesAdapter().fit(X_sub, Y_sub, center=True)
        score = evaluate(p_center)
        results.append({
            "adapter": "Procrustes",
            "center": True,
            "n_anchors": n_anc,
            "hparams": "orthonormal",
            "dev_overlap": round(score, 4),
        })
        print(f"Procrustes (center=True):          Dev Overlap@10 = {score * 100:.2f}%")

        # 5. ContrastiveAdapter center=False
        for temp in [0.05, 0.10]:
            for lr in [1e-3, 5e-3]:
                c_nocenter = ContrastiveAdapter(
                    epochs=40, lr=lr, temperature=temp, lam_cos=0.2, center=False, init="procrustes"
                ).fit(X_sub, Y_sub)
                score = evaluate(c_nocenter)
                results.append({
                    "adapter": "Contrastive",
                    "center": False,
                    "n_anchors": n_anc,
                    "hparams": f"init=procrustes,temp={temp},lr={lr}",
                    "dev_overlap": round(score, 4),
                })
                print(f"Contrastive (center=False, temp={temp}, lr={lr}): Dev Overlap@10 = {score * 100:.2f}%")

        # 6. ContrastiveAdapter center=True
        for temp in [0.05, 0.10]:
            c_center = ContrastiveAdapter(
                epochs=40, lr=5e-3, temperature=temp, lam_cos=0.2, center=True, init="procrustes"
            ).fit(X_sub, Y_sub)
            score = evaluate(c_center)
            results.append({
                "adapter": "Contrastive",
                "center": True,
                "n_anchors": n_anc,
                "hparams": f"init=procrustes,temp={temp},lr=5e-3",
                "dev_overlap": round(score, 4),
            })
            print(f"Contrastive (center=True,  temp={temp}):          Dev Overlap@10 = {score * 100:.2f}%")

    # Sort results by dev overlap descending
    results.sort(key=lambda r: r["dev_overlap"], reverse=True)

    print("\n" + "=" * 70)
    print("TOP 10 CONFIGURATIONS ON DEV SPLIT:")
    print("=" * 70)
    for i, r in enumerate(results[:10]):
        print(f"{i+1:2d}. {r['adapter']:<12} | Anchors: {r['n_anchors']:3d} | Center: {r['center']!s:<5} | {r['hparams']:<30} | Dev Overlap@10: {r['dev_overlap']*100:.2f}%")

    out_file = RESULTS_DIR / "adapter_dev_tuning.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump({"top_configs": results[:10], "all_configs": results}, f, indent=2)

    return {"best_config": results[0], "all_results": results}


if __name__ == "__main__":
    run_tune_adapters_dev()
