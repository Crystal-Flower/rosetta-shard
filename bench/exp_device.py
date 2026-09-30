import json
import os
import platform
import shutil
import tempfile
import time
from typing import Any

import numpy as np

from bench.plots import plot_exp_device
from rosetta.adapters import ContrastiveAdapter
from rosetta.bundle import AdapterBundle
from rosetta.config import MODELS, RANDOM_SEED, RESULTS_DIR
from rosetta.embed import embed_texts
from rosetta.migrate import migrate_shard
from rosetta.store.edge_store import EdgeStore


def run_exp_device() -> dict[str, Any]:
    print("\n" + "=" * 60)
    print("Running E8: Real Hardware Device Benchmark (Latency & Migration)")
    print("=" * 60)

    system_info = {
        "os": platform.system(),
        "release": platform.release(),
        "architecture": platform.machine(),
        "processor": platform.processor(),
        "python_version": platform.python_version(),
    }
    print(f"Platform: {system_info['os']} {system_info['release']} ({system_info['architecture']})")
    print(f"Processor: {system_info['processor']}")
    print(f"Python: {system_info['python_version']}")

    # 1. Query path latency breakdown
    m_small = MODELS["bge-small-en-v1.5"]
    m_large = MODELS["bge-large-en-v1.5"]

    dummy_adapter = ContrastiveAdapter(epochs=10, lr=1e-2, center=False).fit(
        np.random.randn(50, 384).astype(np.float32),
        np.random.randn(50, 1024).astype(np.float32),
    )

    tmp_dir = tempfile.mkdtemp()
    try:
        shard_path = os.path.join(tmp_dir, "bench_edge_shard")
        edge_store = EdgeStore(path=shard_path, space_id=m_large.space_id, dim=1024)

        # Upsert 1,000 dummy points to test real search latency
        n_warmup_pts = 1000
        rng = np.random.default_rng(RANDOM_SEED)
        pts = rng.standard_normal((n_warmup_pts, 1024), dtype=np.float32)
        pts = pts / np.linalg.norm(pts, axis=1, keepdims=True)
        edge_store.upsert(
            ids=[f"doc_{i}" for i in range(n_warmup_pts)],
            vectors=pts,
            payloads=[{"doc_id": f"doc_{i}"} for i in range(n_warmup_pts)],
        )

        sample_queries = [
            "What causes genetic mutations in neural tissue?",
            "Treatment options for chronic inflammatory conditions",
            "Deep learning transformer attention mechanisms",
            "High performance vector database indexing on edge devices",
            "Statistical guarantees in conformal prediction intervals",
        ] * 10

        embed_times = []
        adapter_times = []
        search_times = []

        print("\nMeasuring query stage latencies (50 iterations)...")
        for q_text in sample_queries:
            # Stage 1: Embed
            t0 = time.perf_counter()
            q_emb = embed_texts([q_text], model=m_small, is_query=True)
            t1 = time.perf_counter()
            embed_times.append((t1 - t0) * 1000.0)

            # Stage 2: Adapter transform
            t0 = time.perf_counter()
            q_hat = dummy_adapter(q_emb)
            t1 = time.perf_counter()
            adapter_times.append((t1 - t0) * 1000.0)

            # Stage 3: Edge nearest neighbor search
            t0 = time.perf_counter()
            hits = edge_store.query(q_hat, k=10)
            t1 = time.perf_counter()
            search_times.append((t1 - t0) * 1000.0)

        edge_store.close()
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    query_metrics = {
        "embed_p50_ms": round(float(np.percentile(embed_times, 50)), 2),
        "embed_p95_ms": round(float(np.percentile(embed_times, 95)), 2),
        "adapter_p50_ms": round(float(np.percentile(adapter_times, 50)), 2),
        "adapter_p95_ms": round(float(np.percentile(adapter_times, 95)), 2),
        "search_p50_ms": round(float(np.percentile(search_times, 50)), 2),
        "search_p95_ms": round(float(np.percentile(search_times, 95)), 2),
    }
    total_p50 = (
        query_metrics["embed_p50_ms"]
        + query_metrics["adapter_p50_ms"]
        + query_metrics["search_p50_ms"]
    )
    query_metrics["total_p50_ms"] = round(total_p50, 2)

    print(
        f"Embed Latency (Small Model):   p50={query_metrics['embed_p50_ms']:.2f} ms | p95={query_metrics['embed_p95_ms']:.2f} ms"
    )
    print(
        f"Adapter Projection:            p50={query_metrics['adapter_p50_ms']:.2f} ms | p95={query_metrics['adapter_p95_ms']:.2f} ms"
    )
    print(
        f"EdgeStore Search (k=10):       p50={query_metrics['search_p50_ms']:.2f} ms | p95={query_metrics['search_p95_ms']:.2f} ms"
    )
    print(f"Total Query Path Latency:      p50={query_metrics['total_p50_ms']:.2f} ms")

    # 2. Migration throughput vs Re-embedding
    print("\nMeasuring shard migration throughput...")
    point_counts = [500, 1000, 2000]
    migration_times = []
    reembed_times_extrapolated = []

    # Measure re-embed throughput of large model on 10 texts (actual neural forward pass)
    from fastembed import TextEmbedding
    model_fe = TextEmbedding(m_large.name)
    test_reembed_texts = [f"Unique text for neural forward pass measurement {i} {time.time()}" for i in range(10)]
    t0 = time.perf_counter()
    list(model_fe.embed(test_reembed_texts, batch_size=10))
    time_per_large_embed = (time.perf_counter() - t0) / 10.0

    bundle = AdapterBundle(
        from_space=m_small.space_id,
        to_space=m_large.space_id,
        W=dummy_adapter.W,
    )

    for n in point_counts:
        tmp_mig_dir = tempfile.mkdtemp()
        try:
            m_path = os.path.join(tmp_mig_dir, "mig_shard")
            s = EdgeStore(path=m_path, space_id=m_small.space_id, dim=384)
            raw = rng.standard_normal((n, 384), dtype=np.float32)
            raw = raw / np.linalg.norm(raw, axis=1, keepdims=True)
            s.upsert([f"d_{i}" for i in range(n)], raw, [{"doc_id": f"d_{i}"} for i in range(n)])
            s.close()

            # Time migration
            t_start = time.perf_counter()
            migrated = migrate_shard(
                live_shard_path=m_path,
                bundle=bundle,
                target_dim=1024,
                batch_size=512,
            )
            mig_time = time.perf_counter() - t_start
            migrated.close()

            migration_times.append(round(mig_time, 3))
            reembed_est = round(n * time_per_large_embed, 2)
            reembed_times_extrapolated.append(reembed_est)

            print(
                f"N={n:4d} points | Migration: {mig_time:.2f}s ({n / mig_time:.0f} pts/sec) | Large Re-embed (Extrapolated): {reembed_est:.2f}s ({reembed_est / mig_time:.1f}x speedup)"
            )

        finally:
            shutil.rmtree(tmp_mig_dir, ignore_errors=True)

    results = {
        "experiment": "E8_device",
        "timestamp": time.time(),
        "random_seed": RANDOM_SEED,
        "hardware": system_info,
        "query": query_metrics,
        "migration": {
            "points": point_counts,
            "time_sec": migration_times,
            "reembed_time_sec": reembed_times_extrapolated,
            "reembed_is_extrapolated": True,
        },
    }

    out_json = RESULTS_DIR / "exp_device.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    out_png = RESULTS_DIR / "exp_device.png"
    plot_exp_device(results, out_png)
    print(f"\nSaved results to {out_json} and {out_png}")
    return results


if __name__ == "__main__":
    run_exp_device()
