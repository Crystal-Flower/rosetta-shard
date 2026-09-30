from collections.abc import Generator
from typing import Any

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.http import models as rest_models

from rosetta.store import Hit, StoreProtocol


class CloudStore(StoreProtocol):
    """
    Cloud Qdrant collection wrapper using qdrant-client.
    Connects to remote Qdrant server or in-memory instance.
    """

    def __init__(
        self,
        collection_name: str,
        space_id: str,
        dim: int,
        url: str | None = None,
        host: str | None = None,
        port: int = 6333,
        in_memory: bool = False,
    ):
        self.collection_name = collection_name
        self.space_id = space_id
        self.dim = dim

        if in_memory:
            self.client = QdrantClient(":memory:")
        elif url:
            self.client = QdrantClient(url=url)
        elif host:
            self.client = QdrantClient(host=host, port=port)
        else:
            # Default to localhost:6333 or fallback to in_memory if server unavailable
            try:
                self.client = QdrantClient(host="localhost", port=port, timeout=2.0)
                # Check connection
                self.client.get_collections()
            except Exception:
                # In-memory fallback for local execution without Docker
                self.client = QdrantClient(":memory:")

        # Ensure collection exists
        collections = [c.name for c in self.client.get_collections().collections]
        if self.collection_name not in collections:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=rest_models.VectorParams(
                    size=dim,
                    distance=rest_models.Distance.COSINE,
                ),
            )

    def upsert(
        self,
        ids: list[int | str],
        vectors: np.ndarray | list[list[float]],
        payloads: list[dict[str, Any]],
    ) -> None:
        if isinstance(vectors, np.ndarray):
            vectors = vectors.tolist()

        points = []
        for pid, vec, pld in zip(ids, vectors, payloads):
            int_id = (
                int(pid)
                if isinstance(pid, int) or (isinstance(pid, str) and pid.isdigit())
                else (hash(str(pid)) & 0x7FFFFFFF)
            )
            merged_pld = dict(pld)
            merged_pld.setdefault("doc_id", str(pid))
            merged_pld.setdefault("space_id", self.space_id)

            points.append(
                rest_models.PointStruct(
                    id=int_id,
                    vector=vec,
                    payload=merged_pld,
                )
            )

        if points:
            self.client.upsert(collection_name=self.collection_name, points=points)

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
            vec_list = vector.vector.flatten().tolist()
        else:
            if space_id is not None:
                SpaceGuard.validate(space_id, self.space_id)
            if isinstance(vector, np.ndarray):
                vec_list = vector.flatten().tolist()
            else:
                vec_list = vector

        res = self.client.search(
            collection_name=self.collection_name,
            query_vector=vec_list,
            limit=k,
            query_filter=flt,
            with_payload=True,
            with_vectors=True,
        )

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
        offset = None
        while True:
            records, next_offset = self.client.scroll(
                collection_name=self.collection_name,
                offset=offset,
                limit=batch,
                with_payload=True,
                with_vectors=True,
            )
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
        info = self.client.get_collection(self.collection_name)
        return info.points_count or 0

    def size_bytes(self) -> int:
        return 0

    def close(self) -> None:
        self.client.close()
