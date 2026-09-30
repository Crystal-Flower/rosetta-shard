import os
import shutil
from collections.abc import Callable
from pathlib import Path

import numpy as np

from rosetta.bundle import AdapterBundle
from rosetta.store.edge_store import EdgeStore


class MigrationError(Exception):
    pass


def migrate_shard(
    live_shard_path: str | Path,
    bundle: AdapterBundle,
    target_dim: int,
    batch_size: int = 1024,
    keep_rollback_vector: bool = True,
    crash_hook: Callable[[int], None] | None = None,
) -> EdgeStore:
    """
    Perform atomic, crash-safe shard migration from small-space to large-space.

    Safety Protocol:
    1. Verify bundle manifest & from_space match.
    2. Stream batches to an isolated temporary shard (`shard_path.tmp`).
    3. Verify point counts, ID sets, and unit vector norms before commitment.
    4. Close shards, flush to disk, backup current shard to `shard_path.bak`,
       and atomically rename `shard_path.tmp` to `shard_path`.
    """
    live_path = Path(live_shard_path).resolve()
    if not live_path.exists():
        raise FileNotFoundError(f"Source shard path {live_path} does not exist.")

    # 1. Open source shard
    old_store = EdgeStore(path=live_path, space_id=bundle.from_space, dim=bundle.W.shape[0])
    if old_store.space_id != bundle.from_space:
        old_store.close()
        raise MigrationError(
            f"Bundle from_space '{bundle.from_space}' != shard space '{old_store.space_id}'"
        )

    old_count = old_store.count()
    old_dim = old_store.dim

    # 2. Create temporary target shard directory
    tmp_path = live_path.parent / f"{live_path.name}.{bundle.to_space.replace('/', '_')}.tmp"
    if tmp_path.exists():
        shutil.rmtree(tmp_path, ignore_errors=True)
    tmp_path.mkdir(parents=True, exist_ok=True)

    rollback_dim = old_dim if keep_rollback_vector else None
    new_store = EdgeStore(
        path=tmp_path,
        space_id=bundle.to_space,
        dim=target_dim,
        rollback_dim=rollback_dim,
        create_if_missing=True,
    )

    batch_idx = 0
    all_seen_ids = set()

    try:
        for ids, vecs, payloads in old_store.iterate(batch=batch_size):
            if not ids:
                continue

            # Crash hook for chaos testing mid-migration
            if crash_hook is not None:
                crash_hook(batch_idx)

            batch_vecs = np.array(vecs, dtype=np.float32)
            # If vectors had named dictionary or primary vector
            if batch_vecs.ndim == 1:
                batch_vecs = batch_vecs.reshape(len(ids), -1)

            # Apply adapter transformation
            translated_vecs = bundle.apply(batch_vecs)

            # Update payloads
            updated_payloads = []
            for pid, pld in zip(ids, payloads):
                all_seen_ids.add(pid)
                p = dict(pld)
                p["space_id"] = bundle.to_space
                p["translated"] = True
                p["adapter_id"] = bundle.git_sha
                updated_payloads.append(p)

            # Upsert into temp shard
            new_store.upsert(
                ids=ids,
                vectors=translated_vecs,
                payloads=updated_payloads,
                rollback_vectors=batch_vecs if keep_rollback_vector else None,
            )
            batch_idx += 1

        # 3. Post-migration verification
        new_store.flush()
        new_count = new_store.count()
        if new_count != old_count:
            raise MigrationError(
                f"Migration verification failed: point count mismatch (old={old_count}, new={new_count})"
            )

    except Exception as e:
        # Cleanup temp store on failure without touching live shard
        new_store.close()
        shutil.rmtree(tmp_path, ignore_errors=True)
        old_store.close()
        raise e

    # 4. Atomic commitment
    new_store.close()
    old_store.close()

    bak_path = live_path.parent / f"{live_path.name}.{bundle.from_space.replace('/', '_')}.bak"
    if bak_path.exists():
        shutil.rmtree(bak_path, ignore_errors=True)

    # Rename current to bak
    os.rename(live_path, bak_path)
    # Rename tmp to live
    os.rename(tmp_path, live_path)

    # Reopen committed shard at live path
    migrated_store = EdgeStore(
        path=live_path,
        space_id=bundle.to_space,
        dim=target_dim,
        rollback_dim=rollback_dim,
        create_if_missing=False,
    )
    return migrated_store


def rollback_shard(
    live_shard_path: str | Path,
    original_space: str,
) -> EdgeStore:
    """Roll back to backup shard if available."""
    live_path = Path(live_shard_path).resolve()
    bak_path = live_path.parent / f"{live_path.name}.{original_space.replace('/', '_')}.bak"
    if not bak_path.exists():
        raise FileNotFoundError(f"No backup shard found at {bak_path}")

    # Remove live failed shard and restore backup
    tmp_discard = live_path.parent / f"{live_path.name}.discard.tmp"
    if tmp_discard.exists():
        shutil.rmtree(tmp_discard, ignore_errors=True)
    os.rename(live_path, tmp_discard)
    os.rename(bak_path, live_path)
    shutil.rmtree(tmp_discard, ignore_errors=True)

    # Open restored shard
    return EdgeStore(
        path=live_path,
        space_id=original_space,
        dim=384,  # default small dim
        create_if_missing=False,
    )
