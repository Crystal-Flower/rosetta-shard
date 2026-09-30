import os
import shutil
import tempfile

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


def test_edge_shard_smoke():
    tmp_dir = tempfile.mkdtemp()
    try:
        shard_path = os.path.join(tmp_dir, "smoke_shard")
        os.makedirs(shard_path, exist_ok=True)
        config = EdgeConfig(vectors=EdgeVectorParams(size=4, distance=Distance.Cosine))
        shard = EdgeShard.create(shard_path, config)

        points = [
            Point(id=i, vector=[float(i), 1.0, 0.5, 0.2], payload={"doc_id": f"doc_{i}"})
            for i in range(1, 6)
        ]
        shard.update(UpdateOperation.upsert_points(points))
        assert shard.count(CountRequest()) == 5

        # Query
        res = shard.query(QueryRequest(limit=2, query=Query.Nearest([1.0, 1.0, 0.5, 0.2])))
        assert len(res) == 2
        assert res[0].id == 1

        # Iterate all points
        scrolled_pts, _ = shard.scroll(ScrollRequest(limit=10, with_payload=True, with_vector=True))
        assert len(scrolled_pts) == 5
        assert {p.id for p in scrolled_pts} == {1, 2, 3, 4, 5}

        shard.close()
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
