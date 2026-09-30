import os
from typing import Dict, List, Optional
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
import numpy as np

from rosetta.config import BUNDLE_CACHE_DIR, MODELS
from rosetta.store.cloud_store import CloudStore


app = FastAPI(title="Rosetta Shard Cloud API", version="0.1.0")

# Active cloud store instance (in-memory or connecting to local Qdrant server)
cloud_store: Optional[CloudStore] = None
# In-memory storage of exact cloud vectors for repair
exact_vectors_db: Dict[str, List[float]] = {}


class QueryRequestModel(BaseModel):
    vector: List[float]
    space_id: str
    k: int = 10


class HitModel(BaseModel):
    id: str
    score: float
    payload: dict


@app.on_event("startup")
def startup_event():
    global cloud_store
    cloud_model = MODELS["bge-large-en-v1.5"]
    cloud_store = CloudStore(
        collection_name="cloud_reference_corpus",
        space_id=cloud_model.space_id,
        dim=cloud_model.dim,
        in_memory=True,  # Default to in-memory for testing, or connects to docker
    )


@app.get("/status")
def get_status():
    global cloud_store
    count = cloud_store.count() if cloud_store else 0
    space_id = cloud_store.space_id if cloud_store else ""
    return {
        "status": "online",
        "active_space_id": space_id,
        "indexed_points": count,
    }


@app.get("/bundle/{scenario}")
def get_bundle_manifest(scenario: str = "S2"):
    """Return bundle manifest for requested scenario."""
    bundle_dir = BUNDLE_CACHE_DIR / scenario
    manifest_file = bundle_dir / "manifest.json"
    if not manifest_file.exists():
        raise HTTPException(status_code=404, detail=f"Bundle for scenario {scenario} not found.")

    import json
    with open(manifest_file, "r", encoding="utf-8") as f:
        return json.load(f)


@app.post("/query", response_model=List[HitModel])
def query_cloud_index(req: QueryRequestModel):
    """Execute search on cloud reference index."""
    global cloud_store
    if not cloud_store:
        raise HTTPException(status_code=503, detail="Cloud index not initialized.")

    if req.space_id != cloud_store.space_id:
        raise HTTPException(
            status_code=400,
            detail=f"Space mismatch: Query space {req.space_id} != Cloud space {cloud_store.space_id}",
        )

    hits = cloud_store.query(req.vector, k=req.k)
    return [
        HitModel(id=str(h.id), score=h.score, payload=h.payload)
        for h in hits
    ]


@app.get("/vectors")
def get_exact_vectors(ids: List[str] = Query(...)):
    """Fetch exact large-model vectors for byte-budgeted hot-set repair."""
    results = {}
    for doc_id in ids:
        if doc_id in exact_vectors_db:
            results[doc_id] = exact_vectors_db[doc_id]
    return {"vectors": results}
