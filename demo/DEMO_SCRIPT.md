# Rosetta Shard — 60-Second Demo Walkthrough Script

**Code Cubicle 6.0 · Theme 3: AI-Powered Edge Memory & Intelligence Platform (Qdrant Edge)**

---

## Overview

This rehearsal script coordinates the live 60-second stage presentation demonstrating cross-model vector compatibility, silent collapse prevention, atomic migration, and the Conformal Recall Certificate.

---

## Timeline & Live Actions

| Time | Action | Screen / UI State | Spoken Script / Talking Points |
|---|---|---|---|
| **00–10 s** | **Device Offline Query**<br>Toggle *Airplane Mode ON* on device.<br>Click preset: *"Lithium battery thermal degradation rate"*. | Left panel shows Airplane Mode badge.<br>Query returns top hit from local Qdrant Edge shard.<br>**Certificate badge:** `P(Overlap ≥ 88%) ≥ 90%`. | *"Our edge device runs Qdrant Edge with a lightweight 384-dimensional embedding model. While completely offline in airplane mode, it answers locally with a calibrated Conformal Recall Certificate guaranteeing statistical retrieval fidelity."* |
| **10–20 s** | **Cloud Model Upgrade & Silent Failure**<br>Click *Upgrade Cloud Model*.<br>Toggle *Space Guard OFF* and query.<br>Then toggle *Space Guard ON* and query. | Cloud upgrades to `bge-large-en-v1.5` (1024-d).<br>Guard OFF: Returns silent junk with ~0% true overlap.<br>Guard ON: `SpaceMismatchError` raises immediately, blocking garbage. | *"Now the cloud upgrades to a modern 1024-d model. Standard vector databases silently return confident garbage. With our Space Guard active, cross-space queries are immediately blocked, preventing silent retrieval collapse."* |
| **20–40 s** | **Compact Adapter & Atomic Migration**<br>Click *Push Adapter*.<br>Click *Migrate Shard*. | Shows bundle footprint: **128 KB** (no raw text!).<br>Shard migrates in **<0.1s**, compared to on-device re-embedding estimate.<br>Shard resident space changes to 1024-d. | *"Instead of expensive re-embedding or exposing private data, the cloud ships a 128 KB Adapter Bundle. In 80 milliseconds, Qdrant Edge atomically migrates its resident vectors to the new space with zero raw text on device."* |
| **40–55 s** | **Guaranteed Routing & OOD Detection**<br>Query: *"Lithium battery"* (In-domain).<br>Query: *"Cryptocurrency capital gains tax"* (OOD Shift). | In-domain query: `SERVE_LOCAL` (Overlap 92%).<br>OOD query: `sim_anchor_max` drops to 0.18, Certificate drops, decision triggers `ESCALATE` to cloud. | *"Translated queries now reach the new space. When an out-of-domain query arrives, anchor centroids detect the covariate shift and the Conformal Certificate automatically forces escalation to the cloud."* |
| **55–60 s** | **Measured Benchmarks Slide**<br>Display results charts from `results/`. | Reliability diagram (E4) and ablation table (E2). | *"Every claim here is backed by empirical measurements from our reproducible benchmark suite: 90% empirical coverage, 100% silent collapse prevention, and zero unmeasured numbers."* |

---

## Contingencies & Stage Fallbacks

1. **Network Disruption:**
   - The entire demo is self-contained and runs on `localhost:8000` via FastAPI and local Qdrant Edge shards without external network calls.
2. **Timing / Latency:**
   - All embeddings for demo preset queries are pre-cached in `.cache/embeddings`.
3. **Backup Video:**
   - Pre-record the 60-second screen capture running through these steps in sequence.
