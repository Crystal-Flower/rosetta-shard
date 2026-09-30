import os
import shutil
import tempfile
import time

import numpy as np

from rosetta.metrics import overlap_at_k
from rosetta.store.edge_store import EdgeStore
from rosetta.store.numpy_store import NumPyStore


def test_edge_vs_numpy_store_5k():
    """
    Phase 1 Acceptance Test:
    - Build a shard of 5,000 vectors in under 60 seconds.
    - Top-10 from the EdgeStore matches brute-force NumPyStore.
    """
    n_docs = 5000
    dim = 64
    space_id = "test-model@main/64/l2"

    rng = np.random.default_rng(42)
    # Generate random normalized vectors
    raw_vectors = rng.standard_normal((n_docs, dim), dtype=np.float32)
    norms = np.linalg.norm(raw_vectors, axis=1, keepdims=True)
    vectors = raw_vectors / norms

    doc_ids = [f"doc_{i}" for i in range(n_docs)]
    payloads = [{"doc_id": doc_ids[i], "text": f"text content {i}"} for i in range(n_docs)]

    tmp_dir = tempfile.mkdtemp()
    shard_path = os.path.join(tmp_dir, "test_shard_5k")

    try:
        edge_store = EdgeStore(path=shard_path, space_id=space_id, dim=dim)
        numpy_store = NumPyStore(space_id=space_id, dim=dim)

        start_time = time.perf_counter()

        # Batch upsert into edge store
        batch_size = 1000
        for i in range(0, n_docs, batch_size):
            edge_store.upsert(
                ids=doc_ids[i : i + batch_size],
                vectors=vectors[i : i + batch_size],
                payloads=payloads[i : i + batch_size],
            )
            numpy_store.upsert(
                ids=doc_ids[i : i + batch_size],
                vectors=vectors[i : i + batch_size],
                payloads=payloads[i : i + batch_size],
            )

        build_time = time.perf_counter() - start_time
        print(f"\nBuilt 5,000 doc shard in {build_time:.2f} seconds")
        assert build_time < 60.0, f"Shard build took {build_time:.2f}s, expected < 60s"
        assert edge_store.count() == n_docs
        assert numpy_store.count() == n_docs

        # Query comparison on 10 random query vectors
        query_vectors = rng.standard_normal((10, dim), dtype=np.float32)
        query_vectors = query_vectors / np.linalg.norm(query_vectors, axis=1, keepdims=True)

        overlaps = []
        for q_vec in query_vectors:
            edge_hits = edge_store.query(q_vec, k=10)
            numpy_hits = numpy_store.query(q_vec, k=10)

            edge_ids = [h.id for h in edge_hits]
            numpy_ids = [h.id for h in numpy_hits]

            overlap = overlap_at_k(edge_ids, numpy_ids, k=10)
            overlaps.append(overlap)

        mean_overlap = np.mean(overlaps)
        print(f"Mean top-10 overlap between EdgeStore and NumPyStore: {mean_overlap:.4f}")
        # With exact cosine distance and same vectors, overlap should be 1.0 (exact match)
        assert mean_overlap >= 0.99, f"Overlap was {mean_overlap}, expected >= 0.99"

        # Check iteration
        iter_count = 0
        for b_ids, b_vecs, b_payloads in edge_store.iterate(batch=1024):
            iter_count += len(b_ids)
        assert iter_count == n_docs

        # Check size_bytes
        assert edge_store.size_bytes() > 0

        edge_store.close()
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
