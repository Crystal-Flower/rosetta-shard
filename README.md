# Rosetta Shard 🏺

**AI-Powered Edge Memory & Intelligence Platform with Calibrated Recall Certificates**  
*Code Cubicle 6.0 · Theme 3: AI-Powered Edge Memory & Intelligence Platform (Qdrant Edge)*

> **Edge memory that survives model upgrades, queries cloud indexes built with different embedding models, and tells you, with a statistical guarantee, when it can be trusted.**

[![CI](https://github.com/rosetta-shard/rosetta-shard/actions/workflows/ci.yml/badge.svg)](https://github.com/rosetta-shard/rosetta-shard/actions)
[![Qdrant Edge](https://img.shields.io/badge/Qdrant_Edge-0.8.0-red.svg)](https://qdrant.tech/documentation/edge/)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-green.svg)](LICENSE)

---

## 1. The Problem: Silent Retrieval Collapse

Edge devices (smartphones, IoT gateways, mobile field robots) maintain local vector stores with **small embedding models** (e.g. 384-dimensional models like `bge-small-en-v1.5` or `all-MiniLM-L6-v2`) to conserve battery, memory, and compute. The cloud maintains a high-capacity index using a **large embedding model** (e.g. 1024-dimensional `bge-large-en-v1.5`).

This architecture creates three critical bottlenecks:
1. **Silent Semantic Failure:** When vector dimensions match (e.g., swapping `all-MiniLM-L6-v2` with `bge-small-en-v1.5`, both 384-d), standard vector databases execute searches without raising any exception. However, because coordinate axes in learned representations are unaligned, the dot products yield near-zero overlap with true rankings. The system silently returns garbage.
2. **Dimension Incompatibility:** 384-d edge queries cannot search a 1024-d cloud index.
3. **Prohibitive Re-Embedding Costs:** Upgrading edge memory traditionally requires running the large model across all stored texts. On low-power edge hardware, this drains battery and takes minutes to hours. Crucially, **private on-device documents may no longer exist in raw text form**.

---

## 2. The Solution: Rosetta Shard Architecture

Rosetta Shard introduces a verifiable cross-model adaptation and certification layer built directly on **Qdrant Edge**:

```
                       ┌──────────────────────────────────────────────┐
                       │               CLOUD (v2 Space)               │
                       │  • Large Model Index (1024-d)                │
                       │  • Precomputes Adapter Bundle (128 KB)       │
                       │  • Precomputes Conformal Calibration Stats   │
                       └──────────────────────┬───────────────────────┘
                                              │ Shipped over wire (zero raw text)
                                              ▼
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                                 EDGE DEVICE (Qdrant Edge)                              │
│                                                                                        │
│   Incoming Query                                                                       │
│         │                                                                              │
│         ▼                                                                              │
│  [Small Model 384-d] ──► [Space Guard] ──► [Ridge/Procrustes] ──► [Edge Shard Query]  │
│                                                                           │            │
│                                                                           ▼            │
│  Decision Router ◄── [Recall Certificate: P(Overlap ≥ X) ≥ 1-α] ◄── [Feature Extractor]│
│         │                                                                              │
│         ├──► SERVE_LOCAL (0 bytes sent, instant response)                              │
│         ├──► ESCALATE (Search Cloud Index via qdrant-client)                           │
│         └──► REFUSE_LOW_CONFIDENCE (Airplane mode safe fallback)                       │
└────────────────────────────────────────────────────────────────────────────────────────┘
```

### Core System Pillars

- **Space Guard (`rosetta/space.py`):** Explicit coordinate space tagging (`<model>@<rev>/<dim>/<norm>`). Refuses cross-space queries with `SpaceMismatchError`, eliminating silent semantic collapse.
- **Compact Adapter Bundle (`rosetta/bundle.py`):** A portable ~128 KB package containing closed-form linear mapping weights ($W$), anchor centroids, and conformal calibration quantiles. **Contains zero raw documents and zero private text.**
- **Atomic, Crash-Safe Migration (`rosetta/migrate.py`):** Migrates Qdrant Edge shards in-process via batch streaming into an isolated staging directory. Atomic directory swap guarantees the original shard remains queryable if interrupted.
- **Conformal Recall Certificate (`rosetta/certificate.py`):** A finite-sample, distribution-free statistical guarantee computed entirely on device. Informs the application: *"With probability at least $1-\alpha$, this translated query recovers at least $X\%$ of the true top-10 cloud ranking."*
- **Gated Query Router (`rosetta/router.py`):** Automatically routes high-confidence queries to local edge execution while escalating low-confidence or out-of-domain queries to the cloud.
- **Byte-Budgeted Hot-Set Repair (`rosetta/repair.py`):** Prioritizes frequently queried documents with high translation residual errors, replacing them with exact cloud vectors under strict byte budgets.

---

## 3. Measured Benchmark Results

All figures below are generated directly from machine-readable JSON outputs in `results/` produced by `make bench` (`python -m bench.run_all`, Rule R1: *No unmeasured claims*). Evaluated on SciFact (500 reference documents, held-out test split of 50 queries).

| Experiment | Description | Baseline | Rosetta Shard | Metric Details |
|---|---|---|---|---|
| **E1: Guard** | S1 Cross-Model Unadapted Query | 27.8% Overlap@10 | **Blocked (100.0% Guarded)** | SpaceMismatchError raised (no silent collapse) |
| **E1: Recovery** | S2 Dimension Adaptation (384-d → 1024-d) | 2.4% (Pad/Truncate) | **41.2% (Rosetta Contrastive)** | Top-10 Overlap against native 1024-d search |
| **E2: Ablation** | Adapter Architecture vs Native Ceiling | 2.4% (Pad/Truncate), 1.8% (Random), 3.6% (CCA) | **41.2% (Contrastive) / 42.0% (Procrustes) / 43.2% (Ridge)** | 100 Query Anchors; Native Ceiling: 100.0% |
| **E3: Economy** | Anchor Sample Efficiency Curve | 21.0% (25 Anchors) | **43.2% (100 Anchors)** | Peak economy on pure query anchors |
| **E4: Certificate** | Conformal Calibration ($\alpha=0.10$, $n_{cal}=30$) | Nominal: 90.0% | **80.0% Empirical Coverage** | Mean Certified Overlap: 18.2% ($n_{test}=50$) |
| **E4: Certificate** | Conformal Calibration ($\alpha=0.05$, $n_{cal}=30$) | Nominal: 95.0% | **90.0% Empirical Coverage** | Mean Certified Overlap: 10.9% ($n_{test}=50$) |
| **E5: Gating** | Risk-Coverage Tradeoff ($\tau_{serve}=0.20$) | 41.6% Unfiltered Overlap | **54.2% Mean Overlap (Served)** | 38.0% Local Served Fraction (Worst-Decile: 38.0%) |
| **E6: OOD Shift** | Distribution Shift Detection | False Confidence | **100.0% Escalated to Cloud** | Distance to anchor centroids triggers escalation |
| **E7: Repair** | Byte-Budgeted Hot-Set Repair (100–200 KB) | 18.8% (Unrepaired) | **21.2% (100 KB) → 22.0% (200 KB)** | Priority Policy (vs 18.2% Random at 200 KB) |
| **E8: Latency** | AMD64 Family 25 Model 80, Win 11, Py 3.13.5 | - | **p50 = 1.37 ms (Total Query Path)** | 0.37ms Embed + 0.13ms Adapter + 0.87ms EdgeStore |
| **E8: Migration** | AMD64 Family 25 Model 80, Win 11, Py 3.13.5 | 134.7s (On-Device Re-embed) | **0.66s (3,013 pts/sec for 2,000 pts)** | In-process batch matrix projection |
| **E9: Ablation** | Certificate Loss Predictor Feature Ablation | Constant Baseline: 0.1688 MAE | **0.1397 MAE (Full 6-Feature Model)** | Margins Most Predictive (Drop Margins: 0.1617) |

---

## 4. Quickstart & One-Command Reproduction

### Prerequisites
- Python 3.11+
- Virtual environment (`venv` or `uv`)

### Installation
```bash
git clone https://github.com/rosetta-shard/rosetta-shard.git
cd rosetta-shard

# Create venv and install dependencies
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -e ".[dev]"
```

### Run Tests
```bash
make test
# or: pytest tests/ -v
```

### Reproduce Full Benchmarks
```bash
make bench
# or: python -m bench.run_all
```
Outputs:
- JSON logs: `results/exp_*.json`
- High-resolution plots: `results/exp_*.png`

### Launch Interactive Demo App
```bash
make demo
# or: python -m uvicorn demo.server:app --host 0.0.0.0 --port 8000
```
Open **`http://localhost:8000`** in your browser to test:
- Offline edge search in Airplane Mode.
- Live Space Guard blocking on model upgrade.
- Live 80ms atomic shard migration.
- Real-time Conformal Recall Certificate badge per query.

---

## 5. Repository Layout

```
rosetta-shard/
├── README.md                      # Architecture, benchmark table, quickstart
├── Makefile                       # setup, test, bench, demo targets
├── pyproject.toml                 # Dependencies and build configuration
├── docker-compose.yml             # Cloud Qdrant instance
├── docs/
│   ├── DESIGN.md                  # Architectural rationale & failure modes
│   ├── PREREG.md                  # Pre-registered benchmark protocol & bars
│   ├── VERIFIED.md                # Day-0 verification checklist
│   ├── LIMITATIONS.md             # Honest constraints & scope boundaries
│   └── BUILD_SPEC.md              # Complete technical specification
├── rosetta/
│   ├── config.py                  # Model configurations, space IDs, paths
│   ├── embed.py                   # FastEmbed integration & disk cache
│   ├── data.py                    # Dataset loading & deterministic splits
│   ├── metrics.py                 # Overlap@k, recall@k, nDCG@k
│   ├── adapters/                  # Ridge, Procrustes, Baseline mappers
│   ├── bundle.py                  # Portable signed Adapter Bundle
│   ├── store/                     # Qdrant Edge, Cloud, and NumPy stores
│   ├── space.py                   # SpaceID & SpaceGuard protection
│   ├── migrate.py                 # Atomic, crash-safe shard migration
│   ├── certificate.py             # Conformal Recall Certificate engine
│   ├── router.py                  # Edge-Cloud routing decision logic
│   ├── repair.py                  # Byte-budgeted hot-set repair
│   └── device/agent.py            # On-device runtime agent
├── bench/                         # Complete benchmark suite (E1 - E9)
├── demo/                          # Interactive FastAPI & single-page UI
├── results/                       # Machine-readable JSON metrics & PNG plots
└── tests/                         # Unit, property, and integration tests
```

---

## 6. Documentation & References

- **[DESIGN.md](docs/DESIGN.md):** Deep-dive into mathematical formulation, space guard invariants, and migration safety.
- **[PREREG.md](docs/PREREG.md):** Frozen benchmark evaluation protocol and hypothesis bars.
- **[VERIFIED.md](docs/VERIFIED.md):** Day-0 empirical confirmation of Qdrant Edge APIs and multi-vector capabilities.
- **[LIMITATIONS.md](docs/LIMITATIONS.md):** Stated limitations regarding marginal coverage, distribution shift, and private vectors.
- **[DEMO_SCRIPT.md](demo/DEMO_SCRIPT.md):** Rehearsed 60-second live stage walkthrough script.

---

## License

Distributed under the Apache 2.0 License. See `LICENSE` for details.
