from collections.abc import Generator
from typing import Any

import numpy as np

from rosetta.store import Hit, StoreProtocol


class NumPyStore(StoreProtocol):
    """
    In-memory brute-force NumPy vector store for testing and exact ground-truth verification.
    """

    def __init__(self, space_id: str, dim: int):
        self.space_id = space_id
        self.dim = dim
        self.ids: list[Any] = []
        self.vectors: np.ndarray = np.empty((0, dim), dtype=np.float32)
        self.payloads: list[dict[str, Any]] = []

    def upsert(
        self,
        ids: list[int | str],
        vectors: np.ndarray | list[list[float]],
        payloads: list[dict[str, Any]],
    ) -> None:
        new_vecs = np.array(vectors, dtype=np.float32)
        if new_vecs.ndim == 1:
            new_vecs = new_vecs.reshape(1, -1)
        if new_vecs.shape[1] != self.dim:
            raise ValueError(f"Vector dim {new_vecs.shape[1]} != store dim {self.dim}")

        # L2-normalize stored vectors
        norms = np.linalg.norm(new_vecs, axis=1, keepdims=True)
        new_vecs = new_vecs / np.clip(norms, a_min=1e-12, a_max=None)

        for pid, vec, pld in zip(ids, new_vecs, payloads):
            merged_pld = dict(pld)
            merged_pld.setdefault("doc_id", str(pid))
            merged_pld.setdefault("space_id", self.space_id)

            if pid in self.ids:
                idx = self.ids.index(pid)
                self.vectors[idx] = vec
                self.payloads[idx] = merged_pld
            else:
                self.ids.append(pid)
                self.vectors = (
                    np.vstack([self.vectors, vec.reshape(1, -1)])
                    if len(self.vectors) > 0
                    else vec.reshape(1, -1)
                )
                self.payloads.append(merged_pld)

    def query(
        self,
        vector: np.ndarray | list[float] | Any,
        k: int = 10,
        flt: Any | None = None,
        space_id: str | None = None,
    ) -> list[Hit]:
        from rosetta.space import QueryVector, SpaceGuard

        if isinstance(vector, QueryVector):
            SpaceGuard.validate(vector.space_id, self.space_id)
            q = vector.vector.flatten()
        else:
            if space_id is not None:
                SpaceGuard.validate(space_id, self.space_id)
            q = np.array(vector, dtype=np.float32).flatten()

        if len(self.ids) == 0:
            return []
        q_norm = np.linalg.norm(q)
        if q_norm > 1e-12:
            q = q / q_norm

        scores = self.vectors @ q
        top_k_indices = np.argsort(-scores)[:k]

        hits = []
        for idx in top_k_indices:
            hits.append(
                Hit(
                    id=self.payloads[idx].get("doc_id", self.ids[idx]),
                    score=float(scores[idx]),
                    payload=self.payloads[idx],
                    vector=self.vectors[idx].tolist(),
                )
            )
        return hits

    def iterate(
        self, batch: int = 1024
    ) -> Generator[tuple[list[Any], list[Any], list[dict[str, Any]]], None, None]:
        total = len(self.ids)
        for i in range(0, total, batch):
            yield (
                self.ids[i : i + batch],
                self.vectors[i : i + batch].tolist(),
                self.payloads[i : i + batch],
            )

    def count(self) -> int:
        return len(self.ids)

    def size_bytes(self) -> int:
        return int(self.vectors.nbytes)

    def close(self) -> None:
        pass
