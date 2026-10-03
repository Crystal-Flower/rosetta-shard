# Architectural Design & Rationale
**Rosetta Shard: AI-Powered Edge Memory & Intelligence Platform**

---

## 1. Problem Statement & The Threat of Silent Failure
Edge AI deployments (smartphones, IoT sensors, robotics, point-of-sale systems) maintain local vector stores using small, fast embedding models (e.g. 384-dimensional models such as `all-MiniLM-L6-v2` or `bge-small-en-v1.5`). These models consume minimal RAM and battery. In contrast, cloud indexes use large, high-capacity models (e.g. 1024-dimensional `bge-large-en-v1.5` or `mxbai-embed-large-v1`) that produce significantly higher semantic fidelity.

When vector dimensions match (such as replacing `all-MiniLM-L6-v2` with `bge-small-en-v1.5`, both 384 dimensions), standard vector databases will execute searches without throwing any error. However, because coordinate axes in learned neural representations are unaligned, the dot products yield near-zero overlap with the true nearest neighbors. This **silent semantic collapse** renders edge memories useless without alerting the system.

When vector dimensions do not match (e.g. 384-d edge vs 1024-d cloud), edge devices cannot search the cloud, and upgrading models requires re-embedding every stored vector. Re-embedding requires:
1. Significant edge battery, compute, and latency.
2. The presence of the raw text on-device (which may have been deleted for privacy or storage reasons).

---

## 2. Core Architectural Pillars

### Pillar 1: Space Guard (`rosetta/space.py`)
To prevent silent collapse, all vectors and query requests carry an explicit `SpaceID`:
`SpaceID = "<model_name>@<revision>/<dim>/<normalization>"`
Every vector store wrapper strictly validates that incoming queries match the store's resident `SpaceID`. Cross-space queries immediately raise `SpaceMismatchError` rather than silently returning corrupted results.

### Pillar 2: Lightweight Anchor-Trained Adapter (`rosetta/adapters/`)
Instead of re-embedding the corpus, Rosetta Shard trains an adapter mapping:
$$W \in \mathbb{R}^{d_{\mathrm{small}} \times d_{\mathrm{large}}}$$
trained on paired anchor vectors $(x_i, y_i)$ embedded cloud-side.
Two core closed-form adapters are supported:
1. **Ridge Regression:** Minimizes regularized reconstruction error $\|XW - Y\|_F^2 + \lambda \|W\|_F^2$.
2. **Semi-Orthogonal Procrustes:** Enforces orthonormality via SVD ($X^T Y = U \Sigma V^T \implies W = U V^T$), preserving geometric angles even at extremely small anchor counts.

### Pillar 3: Atomic, Crash-Safe Migration (`rosetta/migrate.py`)
To migrate edge memory without downtime or corruption:
1. Load adapter bundle and verify SHA-256 signatures.
2. Stream existing vectors via `shard.scroll()` in batches.
3. Project vectors to the new space and normalize: $\hat{y} = \frac{x W}{\|x W\|_2}$.
4. Insert into a temporary shard (`shard.<to_space>.tmp`).
5. Atomically replace the active shard upon completion, preserving rollback capabilities.

### Pillar 4: Calibrated Conformal Recall Certificate (`rosetta/certificate.py`)
Linear projection cannot preserve 100% of high-dimensional geometry. Therefore, every translated query must be certified:
- Let loss $\ell(q) = 1 - \mathrm{overlap}_k(\hat{q}, q^*)$ over a reference corpus.
- The edge device extracts lightweight features $f(q)$ without requiring the large model:
  1. Maximum similarity to anchor centroids (`sim_anchor_max`).
  2. Top-5 centroid similarity mean (`sim_anchor_top5`).
  3. Score margins between top hits (`margin_1_10`, `margin_1_2`).
  4. Local fit error of nearest anchor centroids.
- A calibrated predictor $\hat{h}(f)$ estimates expected loss.
- Using **Split Conformal Prediction**, we compute a nonconformity quantile $\hat{q}_\alpha$ on held-out calibration queries $Q_{\mathrm{cal}}$ such that:
$$P\left(\ell(q) \le \hat{h}(f(q)) + \hat{q}_\alpha\right) \ge 1 - \alpha$$
The edge device outputs a guaranteed Certified Recall:
$$\text{Certified Overlap}_k = 1 - \min\left(1, \max\left(0, \hat{h}(f(q)) + \hat{q}_\alpha\right)\right)$$

### Pillar 5: Intelligent Router (`rosetta/router.py`)
Based on the certified recall:
- If $\text{Certified Recall} \ge \tau_{\mathrm{serve}}$: **Serve locally** (0 network bytes, instant response).
- If below threshold and network available: **Escalate** to cloud index search.
- If offline and below threshold: **Refuse low confidence** or serve with an explicit warning banner.

### Pillar 6: Byte-Budgeted Hot-Set Repair (`rosetta/repair.py`)
The cloud precomputes per-document translation error $e_i = 1 - \cos(\hat{y}_i, y_i)$.
The edge tracks frequent query hits $h_i$. Under a byte budget $B$, the edge requests exact vectors for the highest priority documents:
$$p_i = \frac{h_i \cdot e_i}{\text{bytes}}$$
This repairs the hot set to 100% native accuracy while transferring minimal bytes.
