from collections.abc import Generator
from pathlib import Path
from typing import Any

import numpy as np
from qdrant_edge import (
    CountRequest,
    Distance,
    EdgeConfig,
    EdgeShard,
    EdgeVectorParams,
    Point,
    Query,
    QueryRequest,
    ScrollRequest,
    UpdateOperation,
)

from rosetta.store import Hit, StoreProtocol


class EdgeStore(StoreProtocol):
    """
    Production Qdrant Edge wrapper implementing StoreProtocol on native rust bindings.
    """

    def __init__(
        self,
        path: str | Path,
        space_id: str,
        dim: int,
        rollback_dim: int | None = None,
        create_if_missing: bool = True,
    ):
        self.path = Path(path).resolve()
        self.space_id = space_id
        self.dim = dim
        self.rollback_dim = rollback_dim

        if not self.path.exists() or not any(self.path.iterdir() if self.path.is_dir() else []):
            if create_if_missing:
                self.path.mkdir(parents=True, exist_ok=True)
                if rollback_dim:
                    config = EdgeConfig(
                        vectors={
                            "primary": EdgeVectorParams(size=dim, distance=Distance.Cosine),
                            "rollback": EdgeVectorParams(
                                size=rollback_dim, distance=Distance.Cosine
                            ),
                        }
                    )
                else:
                    config = EdgeConfig(
                        vectors=EdgeVectorParams(size=dim, distance=Distance.Cosine)
                    )
                self.shard = EdgeShard.create(str(self.path), config)
            else:
                raise FileNotFoundError(f"Edge shard directory {self.path} does not exist.")
        else:
            self.shard = EdgeShard.load(str(self.path))

    def upsert(
        self,
        ids: list[int | str],
        vectors: np.ndarray | list[list[float]],
        payloads: list[dict[str, Any]],
        rollback_vectors: np.ndarray | list[list[float]] | None = None,
    ) -> None:
        """Upsert a batch of points into the EdgeShard."""
        if isinstance(vectors, np.ndarray):
            vectors = vectors.tolist()
        if rollback_vectors is not None and isinstance(rollback_vectors, np.ndarray):
            rollback_vectors = rollback_vectors.tolist()

        points = []
        for i, (pid, vec, pld) in enumerate(zip(ids, vectors, payloads)):
            # Qdrant Edge requires unsigned 64-bit integer or UUID string
            int_id = (
                int(pid)
                if isinstance(pid, int) or (isinstance(pid, str) and pid.isdigit())
                else (hash(str(pid)) & 0x7FFFFFFF)
            )

            # Embed space_id and doc_id into payload
            merged_payload = dict(pld)
            merged_payload.setdefault("doc_id", str(pid))
            merged_payload.setdefault("space_id", self.space_id)

            if rollback_vectors is not None and self.rollback_dim:
                vec_data = {
                    "primary": vec,
                    "rollback": rollback_vectors[i],
                }
            else:
                vec_data = vec

            points.append(Point(id=int_id, vector=vec_data, payload=merged_payload))

        if points:
            self.shard.update(UpdateOperation.upsert_points(points))

    def query(
        self,
        vector: np.ndarray | list[float] | Any,
        k: int = 10,
        flt: Any | None = None,
        space_id: str | None = None,
    ) -> list[Hit]:
        """Query nearest neighbors using cosine similarity with SpaceGuard validation."""
        from rosetta.space import QueryVector, SpaceGuard

        if isinstance(vector, QueryVector):
            SpaceGuard.validate(vector.space_id, self.space_id)
            vec_list = vector.vector.flatten().tolist()
        else:
            if space_id is not None:
                SpaceGuard.validate(space_id, self.space_id)
            if isinstance(vector, np.ndarray):
                vec_list = vector.flatten().tolist()
            else:
                vec_list = vector

        if len(vec_list) != self.dim:
            raise ValueError(
                f"Query vector dimension {len(vec_list)} does not match store dimension {self.dim}."
            )

        query_obj = Query.Nearest(vec_list, using="primary" if self.rollback_dim else None)
        qr = QueryRequest(
            limit=k,
            query=query_obj,
            filter=flt,
            with_payload=True,
            with_vector=True,
        )

        res = self.shard.query(qr)
        hits = []
        for scored in res:
            hits.append(
                Hit(
                    id=scored.payload.get("doc_id", scored.id) if scored.payload else scored.id,
                    score=float(scored.score),
                    payload=dict(scored.payload) if scored.payload else {},
                    vector=scored.vector if scored.vector is not None else None,
                )
            )
        return hits

    def iterate(
        self, batch: int = 1024
    ) -> Generator[tuple[list[Any], list[Any], list[dict[str, Any]]], None, None]:
        """Iterate all points in batches (ids, vectors, payloads)."""
        offset = None
        while True:
            sr = ScrollRequest(
                offset=offset,
                limit=batch,
                with_payload=True,
                with_vector=True,
            )
            records, next_offset = self.shard.scroll(sr)
            if not records:
                break

            ids = [r.payload.get("doc_id", r.id) if r.payload else r.id for r in records]
            vectors = [r.vector for r in records]
            payloads = [dict(r.payload) if r.payload else {} for r in records]

            yield ids, vectors, payloads

            if next_offset is None:
                break
            offset = next_offset

    def count(self) -> int:
        """Return total point count in shard."""
        return int(self.shard.count(CountRequest(exact=True)))

    def size_bytes(self) -> int:
        """Calculate total directory footprint in bytes."""
        total = 0
        if self.path.exists():
            for f in self.path.rglob("*"):
                if f.is_file():
                    total += f.stat().st_size
        return total

    def flush(self) -> None:
        self.shard.flush()

    def close(self) -> None:
        if hasattr(self, "shard") and self.shard is not None:
            self.shard.close()
