<USER_REQUEST>
# Rosetta Shard — Build Spec
**Code Cubicle 6.0 · Theme 3: AI-Powered Edge Memory & Intelligence Platform (Qdrant Edge)**

> Edge memory that survives model upgrades, talks to a cloud index built with a *different* embedding model, and **tells you, with a statistical guarantee, when it can be trusted.**

---

## 0. How to use this file

- Paste this whole file into a coding agent (e.g. Claude Code) as the project brief, or follow it by hand.
- Work **phase by phase** (Section 6). Do not start phase N+1 until phase N's acceptance criteria pass.
- Ground rules, never break these:
  - **R1. No unmeasured claims.** Every number in the README, video and LinkedIn post must come from a file in `results/` produced by `make bench`.
  - **R2. Never demo with the NumPy fallback store.** The fallback exists for unit tests only. The demo must run on real Qdrant Edge.
  - **R3. Anything marked `VERIFY` is unconfirmed.** Check it against the docs on Day 0 and record the answer in `docs/VERIFIED.md`.
  - **R4. `main` is always runnable.** Work on branches, merge only what passes CI.
  - **R5. Publish failures.** If a bar is missed, report the real number and explain why. Do not quietly change the bar.

---

## 1. The project in one paragraph

Edge devices (phones, kiosks, robots) keep a local Qdrant Edge shard of vectors from a **small** embedding model (cheap on battery and RAM). The cloud keeps a **large** model's index (better quality). The two vector spaces are incompatible, so the device cannot query the cloud without running the large model, and a model upgrade silently breaks every stored vector. Re-embedding is expensive, and private on-device documents may no longer exist in raw form.

Rosetta Shard learns a small **adapter** (a few hundred KB) that translates the small-model space into the large-model space. It is used to:

1. **Migrate** an existing edge shard to the new model space with no raw text and no large model on the device.
2. **Translate queries** so the edge can search the cloud index directly.
3. **Certify** each translated query: a calibrated, conformal-prediction **Recall Certificate** says how much of the true top-k the translated search will recover, so the device can answer locally, escalate, or refuse.

### Why a judge should care (the wedge)

- Silent failure is the real danger: two 384-d models give no error, only wrong answers.
- The existing Theme 3 projects I could find publicly are about consolidation/decay, priority sync and conflict merge. Embedding-space compatibility is a different problem.
- The novelty is **not** "we fit a linear map". It is the system around it plus the guarantee: Space Guard, atomic crash-safe migration, byte-budgeted repair, and calibrated per-query certificates. Every piece is benchmarked.

---

## 2. Day-0 verification checklist (do before writing code)

Record each answer in `docs/VERIFIED.md` with a date and a link.

| # | Verify | Why it matters |
|---|---|---|
| 1 | Official Theme 3 problem statement text, judging rubric, submission format, deadline, hosted-URL requirement, team rules | The rubric is the source of truth. If it explicitly wants LLM answers or a cloud dashboard, add thin versions (see Section 15). |
| 2 | `qdrant-edge-py` installs and works on your OS and on the target device | Everything depends on it. Docs: `qdrant.tech/documentation/edge`. Examples: the `lib/edge/publish/examples` folder in the Qdrant GitHub repo. |
| 3 | Edge API surface: create/load shard, upsert with payload, query, filter, **scroll/iterate all points**, delete, on-disk size, sync API | Migration needs iterate-all plus bulk upsert. |
| 4 | Can one shard hold **multiple named vectors with different dimensions**? | Needed for the optional rollback vector. If not, use two shards. |
| 5 | `fastembed` supported model list: `TextEmbedding.list_supported_models()` | Confirms which model pairs you can actually run. |
| 6 | Licenses of datasets and models you will redistribute or show | Do not commit datasets. Download via script. |
| 7 | Which physical device you will demo on, and whether it can run the small embedder | Drives the device benchmark in Section 9. |

---

## 3. Repository layout

```
rosetta-shard/
├── README.md                      # results table + GIF + one-command reproduce
├── Makefile
├── pyproject.toml
├── docker-compose.yml             # cloud Qdrant + cloud API
├── .github/workflows/ci.yml
├── docs/
│   ├── DESIGN.md                  # why each decision, what fails
│   ├── PREREG.md                  # frozen benchmark protocol + bars (Section 8)
│   ├── VERIFIED.md                # Day-0 answers
│   └── LIMITATIONS.md
├── rosetta/
│   ├── config.py                  # model pairs, paths, seeds
│   ├── embed.py                   # fastembed wrappers, prefixes, caching
│   ├── data.py                    # dataset download + split logic
│   ├── metrics.py                 # overlap@k, recall@k, nDCG@k
│   ├── adapters/
│   │   ├── base.py
│   │   ├── ridge.py
│   │   ├── procrustes.py
│   │   └── contrastive.py         # optional retrieval-aware adapter (torch)
│   ├── bundle.py                  # Adapter Bundle build/load/sign
│   ├── store/
│   │   ├── edge_store.py          # Qdrant Edge wrapper (THE real store)
│   │   ├── cloud_store.py         # qdrant-client wrapper
│   │   └── numpy_store.py         # tests only
│   ├── space.py                   # SpaceID, registry, Space Guard
│   ├── migrate.py                 # atomic, crash-safe shard migration
│   ├── certificate.py             # features, predictor, conformal calibration  ← novel
│   ├── router.py                  # local / escalate / refuse decision
│   ├── repair.py                  # byte-budgeted hot-set repair
│   └── device/
│       └── agent.py               # on-device runtime (query, migrate, log)
├── cloud/
│   └── api.py                     # FastAPI: bundle download, query, exact vectors
├── bench/
│   ├── run_all.py
│   ├── exp_collapse.py
│   ├── exp_anchor_curve.py
│   ├── exp_adapter_ablation.py
│   ├── exp_certificate.py
│   ├── exp_repair.py
│   ├── exp_device.py
│   └── plots.py
├── demo/
│   ├── server.py                  # demo backend
│   ├── static/index.html          # single-page, vanilla JS
│   └── DEMO_SCRIPT.md
├── results/                       # committed JSON + PNG outputs of the bench
└── tests/
```

---

## 4. Environment

- Python 3.11+, virtualenv via `uv` or `venv`.
- Dependencies (pin exact versions after first successful run):
  - `qdrant-edge-py` (edge), `qdrant-client` (cloud)
  - `fastembed`, `numpy`, `scipy`, `scikit-learn`, `pandas`, `matplotlib`
  - `torch` (only for the optional contrastive adapter)
  - `fastapi`, `uvicorn`, `httpx`, `pydantic`, `typer`
  - `pytest`, `ruff`, `pytrec_eval` (or implement nDCG yourself)
- Docker for the cloud Qdrant server.
- Seeds fixed in `config.py`. Every experiment logs the seed, package versions and git SHA into its result JSON.

### Model pairs (check availability with `list_supported_models()`, `VERIFY`)

| Scenario | Edge model (v1) | Cloud/new model (v2) | Purpose |
|---|---|---|---|
| **S1: silent failure** | `all-MiniLM-L6-v2` (384-d) | `bge-small-en-v1.5` (384-d) | Same dimension, so no exception is raised, only garbage. Shows why Space Guard exists. |
| **S2: primary** | `bge-small-en-v1.5` (384-d) | `bge-large-en-v1.5` (1024-d) | Dimension mismatch. The main edge to cloud case. |
| **S3: stretch** | `all-MiniLM-L6-v2` (384-d) | `mxbai-embed-large-v1` (1024-d) | Cross-family. Shows generality and honest degradation. |

Respect each model's query and passage prefix conventions consistently. A wrong prefix will look like an adapter failure.

---

## 5. Data contracts

### 5.1 Space ID
`space_id = "<model-name>@<revision>/<dim>/<normalization>"`, for example `bge-small-en-v1.5@main/384/l2`.
Every vector payload and every query object carries its `space_id`.

### 5.2 Point payload (edge and cloud)
```json
{
  "doc_id": "string",
  "space_id": "bge-large-en-v1.5@main/1024/l2",
  "origin": "shared | private",
  "translated": true,
  "adapter_id": "sha256:...",
  "text_ref": "optional pointer, may be absent on device"
}
```
- `origin=shared` means the same doc exists in the cloud, so its exact vector can be fetched for repair.
- `origin=private` means device-only memory. It can only be translated, never repaired from the cloud.

### 5.3 Adapter Bundle (`bundle.py`), the only artifact shipped cloud to device
```
bundle/
├── manifest.json        # from_space, to_space, created, git_sha, sha256 of each file
├── W.npy                # (d_small × d_large) float16 or float32
├── mu_x.npy, mu_y.npy   # optional centering means
├── anchors_centroids.npy# k-means centroids of anchor embeddings in SMALL space (m × d_small)
├── doc_error_map.npy    # per-shared-doc translation error e_i, float16 (cloud-computed)
├── cert_model.json      # predictor coefficients + feature config
└── cert_calib.json      # conformal quantiles per alpha, n_cal, feature stats
```
Report the total bundle size in KB in the README. **The bundle contains no documents and no raw text.** It does contain anchor centroids (embedding statistics), which `LIMITATIONS.md` must note.

---

## 6. Phases

### Phase 0: Scaffold and verification (2–3 h)
- Create repo, CI (ruff + pytest), Makefile targets: `setup`, `test`, `bench`, `demo`.
- Complete the Day-0 checklist.
- **Accept:** `make test` passes on an empty test; `docs/VERIFIED.md` answers items 1–7; a 10-line script creates an Edge shard, upserts 5 points, queries, and iterates all points.

### Phase 1: Stores, embeddings, data (4 h)
- `embed.py`: cached embedding of any list of texts for a given model, with correct prefixes. Cache to `.cache/` as `.npy` keyed by model and text hash.
- `data.py`: download script for the datasets in Section 8 and deterministic splits.
- `edge_store.py` exposes a minimal interface:
  ```python
  class EdgeStore(Protocol):
      space_id: str
      def upsert(self, ids, vectors, payloads): ...
      def query(self, vector, k, flt=None) -> list[Hit]: ...
      def iterate(self, batch=1024): ...   # yields (ids, vectors, payloads)
      def count(self) -> int: ...
      def size_bytes(self) -> int: ...
  ```
  Implement against `qdrant-edge-py` exactly as the docs show (`VERIFY` every call). `numpy_store.py` implements the same interface for unit tests.
- **Accept:** build a shard of 5k docs in under a minute; exact top-10 from the shard matches brute-force NumPy top-10 (ties aside).

### Phase 2: Adapters and benchmark harness (6 h) ← core of the project
- Implement ridge and Procrustes (Section 7.1). Implement the harness that, for any adapter, computes overlap@10, recall@10 and nDCG@10 on the test queries.
- Run the **collapse experiment** and the **anchor-efficiency curve** early. If the adapter does not work at all, find out now.
- **Accept:** S2 adapter beats "no adapter / pad-truncate" baselines by a large margin on the dev split, and results land in `results/*.json`. If recovery is poor, go straight to the contrastive adapter (7.1c) before building anything else.

### Phase 3: Migration and Space Guard (4 h)
- `space.py`: Space Guard (7.4). `migrate.py`: atomic migration (7.3).
- **Accept:** migration preserves point count, IDs and payloads; a mid-migration kill leaves the old shard queryable; a mismatched-space query raises `SpaceMismatchError`.

### Phase 4: Routing and Recall Certificate (8 h) ← the novel feature
- Implement `certificate.py` and `router.py` (7.5, 7.6). Run `exp_certificate.py`.
- **Accept:** empirical coverage on the held-out test queries tracks the nominal level within sampling error for in-distribution queries; you also produce the under-shift result (Section 8) and report it honestly.

### Phase 5: Hot-set repair (4 h)
- `repair.py` and `cloud/api.py` exact-vector endpoint (7.7).
- **Accept:** a recall-vs-KB-downloaded curve exists, and repair beats random selection at the same byte budget.

### Phase 6: Demo app and real device (8 h)
- Demo server and page (Section 10), device benchmark (Section 9), offline behavior verified on hardware.
- **Accept:** the 60-second script runs end to end three times in a row without intervention, and a pre-recorded fallback video exists.

### Phase 7: Freeze, document, submit (6 h)
- Freeze `PREREG.md`, run `make bench` clean, regenerate README tables, record the video, tag the release (Section 12).

### Stretch (only if everything above is green): adapter chains
- Registry-based path finding v1→v2→v3, with chain-level recalibration (7.4). Do not claim it in the pitch unless it is benchmarked.

### Cut lines if time runs short
1. First cut: contrastive adapter, S3, adapter chains.
2. Second cut: hot-set repair polish (keep a simple version).
3. **Never cut:** S1 collapse demo, S2 migration with measured recovery, Space Guard, Recall Certificate, the real-device run, the README results table.

---

## 7. Module specifications

### 7.1 Adapters

All adapters map L2-normalized small-space vectors to L2-normalized large-space vectors.

**Training data ("anchors").** Texts embedded in **both** models, computed cloud-side. Use a mix of document texts and real query texts. Split all query texts into four disjoint sets: `Q_anchor`, `Q_dev`, `Q_cal`, `Q_test`. Only `Q_anchor` and sampled document texts train the adapter.

**(a) Ridge (baseline to beat)**
```python
import numpy as np

def l2n(x):
    return x / np.linalg.norm(x, axis=1, keepdims=True).clip(1e-12)

class RidgeAdapter:
    def fit(self, X, Y, lam, center=True):
        X, Y = l2n(X), l2n(Y)
        self.mu_x = X.mean(0) if center else 0.0
        self.mu_y = Y.mean(0) if center else 0.0
        Xc, Yc = X - self.mu_x, Y - self.mu_y
        d = Xc.shape[1]
        self.W = np.linalg.solve(Xc.T @ Xc + lam * np.eye(d), Xc.T @ Yc)
        return self

    def __call__(self, X):
        return l2n((l2n(X) - self.mu_x) @ self.W + self.mu_y)
```
Choose `lam` on `Q_dev` by maximizing mean overlap@10, not by reconstruction error.

**(b) Semi-orthogonal Procrustes (works for d_small < d_large)**
```python
class ProcrustesAdapter:
    def fit(self, X, Y):
        X, Y = l2n(X), l2n(Y)
        U, S, Vt = np.linalg.svd(X.T @ Y, full_matrices=False)
        self.W = U @ Vt            # (d_small x d_large), rows orthonormal
        return self
    def __call__(self, X):
        return l2n(l2n(X) @ self.W)
```
It has fewer degrees of freedom, so it may win at very low anchor counts. Test that rather than assume it.

**(c) Retrieval-aware adapter (optional, torch)**
- Small linear or 2-layer residual MLP.
- Loss: in-batch InfoNCE. Translated query `ŷ_q` should score the large-model vector of its own source text (or a labeled relevant doc) above other in-batch docs. Optionally add `1 - cos(ŷ, y)` as a regularizer.
- This optimizes retrieval rather than reconstruction, which is the ablation judges will ask about.

**(d) Classical baseline:** CCA or PCA-then-least-squares. Include it so nobody says you skipped the obvious.

### 7.2 Metrics (`metrics.py`)
```python
def overlap_at_k(a_ids, b_ids, k=10):
    return len(set(a_ids[:k]) & set(b_ids[:k])) / k

def loss_at_k(a_ids, b_ids, k=10):
    return 1.0 - overlap_at_k(a_ids, b_ids, k)
```
- **overlap@k** measures fidelity: translated-query top-k vs the native large-model top-k over the same corpus. It is label-free.
- **recall@k / nDCG@k** use dataset relevance labels (qrels).
- Always report both, plus the native large-model score as the ceiling.

### 7.3 Atomic, crash-safe migration (`migrate.py`)
1. Load bundle. Verify `manifest.json` hashes and that `from_space == shard.space_id`.
2. Create a **new shard directory** `shard.<to_space>.tmp/`.
3. `for ids, vecs, payloads in old.iterate(batch=1024)`: apply the adapter, L2-normalize, set `space_id`, `translated=true`, `adapter_id`, and upsert into the new shard.
4. Verify: same count, same ID set, payloads equal apart from the three migration fields, all norms ≈ 1.
5. Flush and fsync, then atomically rename the tmp directory to the live path. Keep the old shard as `shard.<from_space>.bak/` until `rollback_window` passes or the user confirms.
6. Optional rollback vector: keep the old-space vector as a second named vector (`VERIFY` item 4) so `rollback()` is exact and the **self-consistency feature** (7.5) is available.

**Tests:** kill the process at random points during step 3 (subprocess plus SIGKILL, 20 trials). The old shard must always be queryable and the new one must never be half-live.

### 7.4 Space Guard and registry (`space.py`)
- Every query is an object `{vector, space_id}`. `EdgeStore.query` and `CloudStore.query` **refuse** a query whose `space_id` differs from the store's, raising `SpaceMismatchError`. No silent cross-space search, ever.
- `AdapterRegistry` holds adapters as edges of a graph between `space_id`s. `translate(query, to_space)` finds the path. Single-hop is required. Multi-hop is the stretch goal.
- For chains, **recalibrate** the certificate per chain. Do not assume errors compose additively.
- **S1 test (the dramatic one):** with the guard disabled, S1 returns results with no error and near-zero overlap. With it enabled, the same call raises. Show both in the demo.

### 7.5 Recall Certificate (`certificate.py`), the novel feature

**Goal.** For each translated query, give a number the device can trust: *"with probability at least 1−α, the translated search recovers at least X of the true top-k."* Claims of this kind are a calibrated guarantee, not a vibe.

**Definitions**
- `ℓ(q) = 1 − overlap@k(top-k of translated query, top-k of the native large-model query)` over the shared reference corpus. It lies in `[0, 1]`. It needs the large model, so it is computed **cloud-side** for calibration queries only.
- The device must estimate `ℓ(q)` without the large model, using only signals it already has.

**Device-available features `f(q)`**
1. `sim_anchor_max`: max cosine similarity between the small-space query and the anchor centroids (an out-of-distribution signal).
2. `sim_anchor_top5`: mean of the top-5 centroid similarities.
3. `margin_1_10`: score gap between local top-1 and top-10 after translation.
4. `margin_1_2`: score gap between local top-1 and top-2.
5. `local_fit_err`: interpolated adapter residual of the nearest anchors (store per-centroid mean residual in the bundle).
6. `self_consistency`: if the rollback vector exists, overlap@k between top-k from **old-space** search (native small model) and top-k from **new-space** translated search. Cheap, and often highly informative. Optional because it doubles vector storage.

**Predictor `ĥ(f)`.** Start with ridge regression on `f`. Try a shallow gradient-boosted tree if it helps, and keep inference in pure NumPy on the device. Train on `Q_cal_train`, a split of calibration data separate from the conformal calibration split.

**Split conformal calibration (one-sided, finite-sample).**
```python
import numpy as np

def conformal_quantile(scores, alpha):
    """scores: nonconformity on the calibration set. Returns q_hat."""
    n = len(scores)
    k = int(np.ceil((n + 1) * (1 - alpha)))
    k = min(k, n)
    return np.sort(scores)[k - 1]

# calibration (cloud side), on Q_cal (disjoint from adapter training,
# predictor training, dev, and test):
#   s_i   = loss_i - h_hat(f_i)
#   q_hat = conformal_quantile(s, alpha)
# device side, for a new query:
#   upper_loss = clip(h_hat(f) + q_hat, 0, 1)
#   certified_overlap = 1 - upper_loss
```
- **Guarantee:** if the new query is exchangeable with the calibration queries, `P(ℓ(q) ≤ ĥ(f) + q̂) ≥ 1−α`. This is **marginal** (averaged over queries), not per-query, and only for in-distribution queries. Say so in the docs.
- **Improvement (if time):** normalized conformal scores `s_i = (loss_i − ĥ_i)/σ̂_i`, with a second small model predicting `σ̂` so the bound tightens on easy queries.
- **Ship:** `ĥ` coefficients, `q̂` for a few α values (0.05, 0.1, 0.2), and feature normalization stats in the bundle.

**What is and is not certified.** The certificate is about **translation fidelity over the shared reference corpus**. It does not certify that the local shard contains the answer. Local-shard coverage is handled by the score margins and the escalate path. State this scoping explicitly so it survives technical questioning.

**Shift behavior (must be measured, not hand-waved).** Calibrate on dataset A queries, test on dataset B queries. Report (a) coverage degradation, and (b) how often `sim_anchor_max` flags the shift and pushes the router to escalate. A certificate that admits its own breakdown is more credible than one that claims universality.

### 7.6 Router (`router.py`)
```
q_small  = embed_small(query)
q_hat    = adapter(q_small)                     # query in the new space
hits     = edge.query(q_hat, k)                 # local translated search
cert     = certificate(q_small, q_hat, hits)    # certified_overlap at chosen alpha
if cert >= tau_serve:                 -> SERVE_LOCAL
elif network_available:               -> ESCALATE (cloud.query(q_hat, k), log bytes)
else:                                 -> REFUSE_LOW_CONFIDENCE (return hits flagged)
```
- `tau_serve` and `alpha` are config values, reported in every result.
- Log per query: decision, certificate, bytes sent/received, latency. This log feeds the risk–coverage plot and the repair policy.

### 7.7 Byte-budgeted hot-set repair (`repair.py`)
**Idea.** Translated vectors are approximate. For *shared* docs that matter most, pull the **exact** large-model vector from the cloud by `doc_id`. No raw text and no large model on the device.

- The cloud precomputes `e_i = 1 − cos(adapter(x_i), y_i)` for every shared doc and ships it as `doc_error_map.npy` (2 bytes per doc in float16).
- The device counts how often each doc appears in local top-k (`hit_i`) from the router log.
- Priority: `p_i = hit_i · e_i / bytes_i`, filled greedily under a byte budget `B`.
- Cloud endpoint `GET /vectors?ids=...` returns exact vectors (float32, or float16 if you measure its effect).
- Repaired points get `translated=false`. Private docs are never repaired from the cloud. Report this as a stated limitation.
- **Baselines:** random selection, frequency-only, error-only. Plot recall vs KB downloaded.

---

## 8. Benchmark protocol (write `docs/PREREG.md` **before** the final run)

### Datasets
- **Primary in-domain:** a BEIR-style set with many real queries (FiQA is a good candidate), so the four-way query split has enough data. `VERIFY` the license before any redistribution, and only download via script.
- **Shift set:** a different domain (for example SciFact) for the out-of-distribution test.
- **Demo domain corpus:** a small corpus you may legally redistribute (public-domain or openly licensed technical manuals or field guides). Use it for the live demo only, and keep the benchmark numbers on the public sets.
- Pseudo-queries (random sentences or titles) may add volume for adapter training, but **calibration and test use real queries only**.

### Splits (fixed seed)
`Q_anchor` / `Q_dev` / `Q_cal_train` / `Q_cal` / `Q_test`, all disjoint. Document anchors are sampled from the corpus. Test queries are touched once, at the final run.

### Experiments and what each shows

| ID | Experiment | Shows |
|---|---|---|
| E1 | **Collapse:** S1 old-space queries vs new-space index, no adapter | Silent failure is real |
| E2 | **Adapter ablation:** none, pad/truncate, CCA, ridge, Procrustes, contrastive vs native ceiling | The method works and which variant wins |
| E3 | **Anchor curve:** n ∈ {50, 100, 200, 500, 1000, 2000, 5000} | How little data is needed |
| E4 | **Certificate calibration:** nominal 1−α vs empirical coverage, α ∈ {0.05, 0.1, 0.2} | The guarantee holds in-distribution |
| E5 | **Risk–coverage:** fraction served locally vs mean and worst-decile overlap, sweeping `tau_serve` | The gate is useful, not just correct |
| E6 | **Shift test:** calibrate on A, test on B | Honest failure behavior and detection |
| E7 | **Repair curve:** recall vs KB downloaded, policy vs baselines | Repair is worth its bytes |
| E8 | **Device:** migration and query cost on real hardware (Section 9) | It runs where claimed |
| E9 | **Feature ablation:** drop each certificate feature | Which signals carry the certificate |

### Metrics
overlap@10, recall@10, nDCG@10, certificate coverage, local-serve fraction, bytes transferred, migration time, p50/p95 query latency, peak RAM, shard size on disk, bundle size.

### Bars
Set initial bars after a **pilot on the dev split**, then write them into `PREREG.md` and freeze them. Suggested bar types, with numbers to be chosen from the pilot:
- Adapter reaches a stated fraction of native large-model nDCG@10 using at most N anchors.
- Certificate empirical coverage is within a stated tolerance of nominal.
- Repair at budget B recovers a stated fraction of the remaining gap.

These are hypotheses, not promises. Publish the result whether it passes or fails (R5).

### Output
Each experiment writes `results/<id>.json` (metrics, seeds, versions, git SHA) and `results/<id>.png`. `bench/plots.py` and a README-table script render everything from these files.

---

## 9. Device measurement protocol (E8)

Use the actual demo device (Raspberry Pi, phone-class board, or old laptop).

1. Record hardware model, RAM, OS, Python version, Edge version.
2. **Migration:** time migrating N ∈ {5k, 20k, 50k} vectors. Record wall time, peak RSS, and shard size before and after.
3. **Query path:** p50 and p95 latency for embed, translate, and local search, separately.
4. **Comparison:** re-embedding the same N texts with the large model on the same device. If the device cannot run it, measure throughput on a small subset and **label the result an extrapolation**. Do not present an estimate as a measurement.
5. Energy: only report it if you have an actual USB power meter. Otherwise do not mention energy.
6. Run each measurement at least 5 times and report median and spread.

---

## 10. Demo app

- Backend: FastAPI (`demo/server.py`). Frontend: one static `index.html`, vanilla JS, no framework.
- Two panels:
  - **Device (Qdrant Edge)** on the left: shard space ID, point count, network state, query box, result list with a **certificate badge** per query.
  - **Cloud** on the right: active model, index size, bytes sent and received.
- Controls: **Airplane mode** toggle (the device agent refuses to call the cloud and counts blocked attempts), **Upgrade cloud model** button, **Push adapter** button, **Migrate** button with live progress.
- Live widgets: overlap bar (before, after migration), certificate gauge, bytes counter, migration timer next to the re-embed estimate (clearly labeled as estimate or measurement).

### 60-second script (`demo/DEMO_SCRIPT.md`)

| Time | Action | Must be visible |
|---|---|---|
| 0–10 s | Device offline. Ask a question. Local shard answers. | Airplane badge, result with certificate |
| 10–20 s | Cloud upgrades its model. Guard off: stale queries return confident junk (S1). Guard on: `SpaceMismatchError`. | Overlap collapse number |
| 20–40 s | Push the adapter (size in KB). Migrate live. Timer runs against re-embed. Overlap jumps. | Bundle KB, migration seconds, before/after overlap |
| 40–55 s | Translated query reaches the cloud without the large model on the device. An out-of-domain query gets a low certificate and escalates. | Certificate values, escalation decision, bytes |
| 55–60 s | Results slide: calibration plot plus recovery table. | Numbers read from `results/` |

**Fallbacks:** a pre-recorded video of a full successful run, cached results in `results/`, and `docker compose up` working on a laptop hotspot. Rehearse the failure path too (what you say if the Wi-Fi dies).

---

## 11. Testing and CI

- **Unit:** adapters (shapes, normalization, determinism), metrics, conformal quantile (check against a brute-force computation), Space Guard.
- **Property:** translating then normalizing always gives unit-norm vectors. Conformal quantile is monotone in α.
- **Integration:** migrate a 500-point shard, compare IDs, payloads, counts, and top-k agreement.
- **Crash test:** SIGKILL during migration, 20 trials (7.3).
- **Coverage test:** synthetic data where the true coverage is known. The certificate must hit nominal within tolerance.
- **CI** (GitHub Actions): ruff, pytest, plus a tiny-corpus smoke run of the whole pipeline. Add the badge to the README.

---

## 12. Submission checklist

**Repo**
- [ ] README top: title line, results table (auto-generated), one GIF, one-command reproduce
- [ ] Every README number traces to `results/*.json`
- [ ] `docs/DESIGN.md`, `docs/PREREG.md`, `docs/LIMITATIONS.md`, `docs/VERIFIED.md`
- [ ] Pinned dependency versions, Dockerfile or compose, license, dataset and model license notes
- [ ] No keys, no `.env`, no model weights or datasets committed
- [ ] CI green on `main`; release tag (for example `v1.0-submission`) linked in the form and the post
- [ ] Commit history readable and spread across the work period

**Video (90 s max)**
- [ ] Follows the 60-second script, plus 30 s on the guarantee and the limits
- [ ] Shows a real device and real terminal output

**LinkedIn post template**
> Swap an embedding model and every vector you've stored silently breaks, with no error, only wrong answers. We built **Rosetta Shard** for Qdrant Edge: a KB-sized adapter that migrates on-device memory to a new model *without the raw data*, lets edge queries reach a cloud index built with a different model, and attaches a calibrated **Recall Certificate** to each query. Results: [chart from `results/`]. Code, benchmark, and our honest failure cases: [link]. #CodeCubicle

Only include numbers that exist in `results/`.

---

## 13. Judge Q&A cheat sheet

| Question | Honest answer |
|---|---|
| Why not just re-embed? | On-device it costs time and battery. Private documents may no longer exist in raw form. We measure both costs (E8). |
| Why not run the same model on both sides? | The large model does not fit the device's RAM and latency budget, and the cloud wants the better model. The mismatch is the premise. |
| Isn't this just linear regression? | The adapter can be, and we benchmark nonlinear and retrieval-aware variants. The contribution is the system (Space Guard, atomic migration, byte-budgeted repair) and the calibrated certificate. |
| Does the guarantee really hold? | Marginally, for queries exchangeable with calibration, at a nominal level we test empirically (E4). Under shift it degrades, and we measure that too (E6). |
| What exactly is certified? | Translation fidelity against the large model over the shared corpus. Not whether the local shard contains the answer. |
| What if the adapter is wrong for a query? | The certificate drops, the router escalates or refuses, and Space Guard blocks cross-space searches. |
| What about private data? | The bundle has no documents or raw text. It has anchor centroids, which are embedding statistics, and we list that in `LIMITATIONS.md`. |
| Why Qdrant Edge specifically? | In-process shard, payload filtering, multiple named vectors (verified per `VERIFIED.md`), and a server sync path. |
| Where is the LLM? | Not needed for the core claim. See Section 15 if the rubric requires generation. |

---

## 14. Risks and mitigations

| Risk | Mitigation |
|---|---|
| Adapter recovery is mediocre on real data | Pilot early (Phase 2). Switch to the contrastive adapter. Lead the pitch with migration cost and privacy, which hold even at partial recovery. Use repair and the certificate to cover weak regions. |
| Conformal guarantee looks weak under shift | Show it openly. Use `sim_anchor_max` as an OOD flag that forces escalation. |
| `qdrant-edge-py` lacks a needed API (iterate-all, multi named vectors) | Found on Day 0. Use two shards, or iterate via ID ranges. Note the workaround in `VERIFIED.md`. |
| Demo failure on stage | Pre-recorded video, cached results, compose file, rehearsed failure script. |
| Looks like "just infra" | Real device, live collapse-then-recovery, a human scenario in the first 10 seconds, and a clean one-command benchmark. |
| Scope creep | Follow the cut lines in Section 6. |

---

## 15. Non-goals, and rubric-driven add-ons

**Non-goals:** a chatbot, a general RAG wrapper, a dashboard for its own sake, a custom vector database, cloud-scale sync.

**If the official rubric (Day-0 item 1) demands more, add only the thin version:**
- *Needs generated answers:* add an extractive answer panel that quotes the top local hit with its certificate. No LLM is required for the core claim.
- *Needs visible sync:* use the Qdrant Edge sync patterns from the docs (`VERIFY`) to show the translated shard reconciling with the cloud collection.
- *Needs a hosted URL:* host the demo server and the cloud API only. Keep the benchmark as a script.

---

## 16. First hour, concretely

1. Create the repo, CI, and `docs/VERIFIED.md`.
2. Install `qdrant-edge-py` and `fastembed`. Create a shard, upsert 5 points, query, iterate all.
3. Embed 5k docs with two models and cache to `.cache/`.
4. Fit a ridge adapter on 500 anchors and print overlap@10 on 100 dev queries.

If step 4 shows meaningful recovery, the project is viable. If it does not, switch to the contrastive adapter before building anything else.
</USER_REQUEST>
<ADDITIONAL_METADATA>
The current local time is: 2026-09-30T13:41:26+05:30.
</ADDITIONAL_METADATA>
<USER_SETTINGS_CHANGE>
The user changed setting `Model Selection` from None to Gemini 3.8 Flash (High). No need to comment on this change if the user doesn't ask about it. If reporting what model you are, please use a human readable name instead of the exact string.
</USER_SETTINGS_CHANGE>