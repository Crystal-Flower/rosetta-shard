# Day-0 Verification Record
**Rosetta Shard — Code Cubicle 6.0 · Theme 3: AI-Powered Edge Memory & Intelligence Platform (Qdrant Edge)**
**Date:** 2026-09-30

---

### Item 1: Theme 3 Problem Statement & Contest Context
- **Theme:** Theme 3: AI-Powered Edge Memory & Intelligence Platform (Qdrant Edge).
- **Core Challenge:** Edge memory compatibility across model upgrades and heterogeneous edge-to-cloud architectures.
- **Judging Focus:** Practicality, low-resource efficiency, mathematical rigor of guarantees, real device execution without raw text leakage.
- **Deliverables:** Working repository, benchmark outputs (`results/*.json`), reproducible demo with Qdrant Edge, clear architectural documentation (`DESIGN.md`, `PREREG.md`, `LIMITATIONS.md`).

---

### Item 2: `qdrant-edge-py` Platform Support & Installation
- **Status:** Verified (2026-09-30).
- **Version:** `qdrant-edge-py==0.8.0`.
- **Target OS:** Windows 10/11 x86_64, Linux ARM64 / x86_64 compatible wheels.
- **Reference Docs:** https://qdrant.tech/documentation/edge/
- **Verification Script:** Successfully created, loaded, upserted, and queried shard with native Rust bindings.

---

### Item 3: Edge API Surface
- **Create Shard:** `EdgeShard.create(path, config: EdgeConfig)`. Note: On Windows, parent directories must exist before WAL creation.
- **Load Shard:** `EdgeShard.load(path)`.
- **Count:** `shard.count(CountRequest(exact=True))` -> returns integer count.
- **Upsert Points:** `shard.update(UpdateOperation.upsert_points([Point(id=..., vector=..., payload=...)]))`.
- **Delete Points:** `shard.update(UpdateOperation.delete_points([id1, id2]))`.
- **Query / Search:**
  - `shard.query(QueryRequest(limit=k, query=Query.Nearest(vector), with_payload=True, with_vector=True))` -> returns list of `ScoredPoint` (has `.id`, `.score`, `.payload`, `.vector`).
  - `shard.search(SearchRequest(...))` also supported.
- **Scroll / Iterate All:**
  - `shard.scroll(ScrollRequest(offset=next_offset, limit=batch_size, with_payload=True, with_vector=True))` -> returns `(points: list[Record], next_offset: Optional[PointId])`.
- **On-disk Size:** Computed via directory traversal of segment and WAL files: `sum(f.stat().st_size for f in Path(path).rglob('*') if f.is_file())`.
- **Sync API:** Snapshot manifest and snapshot unpack methods supported via `shard.snapshot_manifest()` and `shard.update_from_snapshot()`.

---

### Item 4: Multiple Named Vectors with Heterogeneous Dimensions
- **Status:** Verified (2026-09-30).
- **Finding:** YES. `EdgeConfig(vectors={'primary': EdgeVectorParams(size=384, distance=Distance.Cosine), 'rollback': EdgeVectorParams(size=1024, distance=Distance.Cosine)})` is natively supported in a single shard.
- **Implication:** The optional rollback vector for atomic rollbacks and self-consistency checking can reside directly in the primary shard without requiring two separate shard processes.

---

### Item 5: FastEmbed Supported Models
- **Status:** Verified via `TextEmbedding.list_supported_models()` (2026-09-30).
- **Available Models:**
  - `sentence-transformers/all-MiniLM-L6-v2` (dim=384)
  - `BAAI/bge-small-en-v1.5` (dim=384, prefix: query `"Represent this sentence for searching relevant passages: "`, passage `""`)
  - `BAAI/bge-large-en-v1.5` (dim=1024, prefix: query `"Represent this sentence for searching relevant passages: "`, passage `""`)
  - `mixedbread-ai/mxbai-embed-large-v1` (dim=1024, prefix: query `"Represent this sentence for searching relevant passages: "`, passage `""`)
- **Scenarios Supported:**
  - **S1 (Silent Failure):** `all-MiniLM-L6-v2` (384-d) -> `bge-small-en-v1.5` (384-d). Same dimension; without Space Guard, results collapse into noise without throwing an error.
  - **S2 (Primary Migration):** `bge-small-en-v1.5` (384-d) -> `bge-large-en-v1.5` (1024-d). Dimension expansion, edge-to-cloud adaptation.
  - **S3 (Stretch Cross-Family):** `all-MiniLM-L6-v2` (384-d) -> `mxbai-embed-large-v1` (1024-d). Cross-architecture translation.

---

### Item 6: Dataset and Model Licenses
- **Datasets:**
  - BEIR / SciFact / FiQA: Open research licenses (Apache 2.0 / CC-BY-SA).
  - No datasets committed directly to repository; downloaded programmatically via `rosetta/data.py` with deterministic caching.
- **Model Weights:**
  - FastEmbed ONNX runtimes distributed under Apache 2.0 / MIT.
  - Cached to `.cache/models/` without redistributing binary weights in git.

---

### Item 7: Physical Device Profile
- **Environment:** Edge Host runtime (Python 3.11/3.13, x86_64, Windows/Linux agnostic).
- **Edge Embedder:** `fastembed` ONNX Runtime with quantized CPU inference for ultra-fast, low-memory edge embedding.
- **Edge Vector Database:** Native `qdrant-edge-py` embedded shard (zero external daemon required on device).
