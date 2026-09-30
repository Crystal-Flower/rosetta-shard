import os
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pytest

from rosetta.bundle import AdapterBundle
from rosetta.migrate import migrate_shard, rollback_shard
from rosetta.space import SpaceMismatchError
from rosetta.store.edge_store import EdgeStore


@pytest.fixture
def temp_workspace():
    tmp_dir = tempfile.mkdtemp()
    yield Path(tmp_dir) if "Path" in globals() else tmp_dir
    shutil.rmtree(tmp_dir, ignore_errors=True)


def test_atomic_migration_and_rollback():
    tmp_dir = tempfile.mkdtemp()
    try:
        shard_path = os.path.join(tmp_dir, "test_shard")
        os.makedirs(shard_path, exist_ok=True)

        n_points = 100
        d_in = 384
        d_out = 1024
        from_space = "small-model@main/384/l2"
        to_space = "large-model@main/1024/l2"

        rng = np.random.default_rng(42)
        raw_vecs = rng.standard_normal((n_points, d_in), dtype=np.float32)
        vecs = raw_vecs / np.linalg.norm(raw_vecs, axis=1, keepdims=True)
        doc_ids = [f"doc_{i}" for i in range(n_points)]
        payloads = [{"doc_id": doc_ids[i], "origin": "shared"} for i in range(n_points)]

        # Create source shard
        store = EdgeStore(path=shard_path, space_id=from_space, dim=d_in)
        store.upsert(ids=doc_ids, vectors=vecs, payloads=payloads)
        assert store.count() == n_points
        store.close()

        # Create dummy adapter bundle
        W = rng.standard_normal((d_in, d_out), dtype=np.float32)
        bundle = AdapterBundle(
            from_space=from_space,
            to_space=to_space,
            W=W,
            git_sha="test_sha_123",
        )

        # 1. Execute atomic migration
        migrated_store = migrate_shard(
            live_shard_path=shard_path,
            bundle=bundle,
            target_dim=d_out,
            batch_size=32,
            keep_rollback_vector=True,
        )

        assert migrated_store.count() == n_points
        assert migrated_store.space_id == to_space
        assert migrated_store.dim == d_out

        # Verify iterated payloads
        all_ids = set()
        for b_ids, b_vecs, b_payloads in migrated_store.iterate():
            for pid, vec, pld in zip(b_ids, b_vecs, b_payloads):
                all_ids.add(pid)
                assert pld["translated"] is True
                assert pld["adapter_id"] == "test_sha_123"
                assert pld["space_id"] == to_space
                prim_vec = (
                    vec.get("primary", list(vec.values())[0]) if isinstance(vec, dict) else vec
                )
                np.testing.assert_allclose(np.linalg.norm(prim_vec), 1.0, atol=1e-5)

        assert all_ids == set(doc_ids)

        # Space guard check: old space vector query raises
        with pytest.raises(SpaceMismatchError):
            migrated_store.query(vecs[0], space_id=from_space)

        # New space query succeeds
        test_q = np.ones(d_out, dtype=np.float32) / np.sqrt(d_out)
        hits = migrated_store.query(test_q, space_id=to_space, k=5)
        assert len(hits) == 5

        migrated_store.close()

        # 2. Test rollback to old space
        rolled_back_store = rollback_shard(
            live_shard_path=shard_path,
            original_space=from_space,
        )
        assert rolled_back_store.count() == n_points
        assert rolled_back_store.space_id == from_space
        rolled_back_store.close()

    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def test_migration_crash_safety():
    """Verify that a crash halfway through leaves the original shard intact."""
    tmp_dir = tempfile.mkdtemp()
    try:
        shard_path = os.path.join(tmp_dir, "crash_shard")
        os.makedirs(shard_path, exist_ok=True)

        n_points = 60
        d_in = 384
        d_out = 1024
        from_space = "small-model@main/384/l2"
        to_space = "large-model@main/1024/l2"

        rng = np.random.default_rng(42)
        raw_vecs = rng.standard_normal((n_points, d_in), dtype=np.float32)
        vecs = raw_vecs / np.linalg.norm(raw_vecs, axis=1, keepdims=True)
        doc_ids = [f"doc_{i}" for i in range(n_points)]
        payloads = [{"doc_id": doc_ids[i]} for i in range(n_points)]

        store = EdgeStore(path=shard_path, space_id=from_space, dim=d_in)
        store.upsert(ids=doc_ids, vectors=vecs, payloads=payloads)
        store.close()

        bundle = AdapterBundle(
            from_space=from_space,
            to_space=to_space,
            W=rng.standard_normal((d_in, d_out), dtype=np.float32),
        )

        def crash_hook(batch_idx):
            if batch_idx == 1:
                raise RuntimeError("Simulated crash mid-migration!")

        with pytest.raises(RuntimeError) as exc_info:
            migrate_shard(
                live_shard_path=shard_path,
                bundle=bundle,
                target_dim=d_out,
                batch_size=20,
                crash_hook=crash_hook,
            )

        assert "Simulated crash mid-migration!" in str(exc_info.value)

        # Original shard must still exist, have original space_id, and all 60 points!
        reopened = EdgeStore(
            path=shard_path, space_id=from_space, dim=d_in, create_if_missing=False
        )
        assert reopened.count() == n_points
        assert reopened.space_id == from_space
        reopened.close()

    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
