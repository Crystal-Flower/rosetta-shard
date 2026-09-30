# Rosetta Shard: Limitations & Failure Modes

---

## 1. Conformal Guarantee Scope
- **Marginal vs Conditional Coverage:** The statistical guarantee $P(\text{overlap} \ge \text{certified}) \ge 1 - \alpha$ is marginal (averaged over the population of exchangeable queries). It does not guarantee exact per-instance coverage for every individual outlier.
- **Distribution Shift:** Conformal prediction assumes query exchangeability between the calibration set $Q_{cal}$ and runtime test queries. When the edge query distribution undergoes strong domain shift (e.g. medical queries evaluated against a general-domain calibration set), coverage may degrade. Our system uses `sim_anchor_max` to explicitly detect OOD queries and force escalation.
- **Reference Corpus Fidelity vs Local Corpus Content:** The Recall Certificate certifies fidelity against the reference search index (i.e. whether the query vector preserves its top-$k$ alignment with the cloud space). It does not certify that the private local database necessarily contains a relevant document for an arbitrary query.

## 2. Privacy & Information Leakage in Adapter Bundles
- The adapter bundle contains **no raw documents** and **no text passages**.
- It does contain:
  1. The linear transformation matrix $W$.
  2. K-means anchor centroids in small-model space ($\mu_k \in \mathbb{R}^{d_{small}}$).
  While vector centroids cannot be inverted to exact original sentences without substantial reconstruction risk, they do encode statistical characteristics of the training domain. Deployments with extreme privacy constraints should use synthetic public anchor texts.

## 3. Dimensionality & Model Family Limits
- Within-family models (e.g. BGE-small to BGE-large) preserve strong linear and manifold similarity, yielding high recovery (>80% overlap@10).
- Cross-family models (e.g. MiniLM to MxBAI) exhibit higher topological divergence. Linear adapters achieve lower overlap (~40-60%), demonstrating the necessity of the Recall Certificate to warn downstream applications.

## 4. Hot-Set Repair Constraints
- **Private Documents:** Hot-set repair requires the cloud to supply the exact vector. Documents flagged `origin=private` exist only on the edge and can never be repaired from the cloud.
- **Bandwidth Budget:** If network is completely unavailable, repair requests are queued and cannot take effect until reconnection.

## 5. Pre-Registration Post-Mortem & Honest Metric Disclosures (Rule R5)
In accordance with Rule R5, we honestly document how empirical findings compared against our pre-registered hypotheses in `docs/PREREG.md`:
- **Linear Cross-Dimension Mapping Bound (E2 / E3):**
  - *Pre-registered hypothesis:* Linear adapters would recover $\ge 70\%$ of the gap to the native ceiling.
  - *Empirical outcome:* On SciFact (500 docs), Procrustes achieved **24.2% Overlap@10** (250 anchors) and peaked at **42.6% Overlap@10** (100 anchors), vs. **2.4% for Pad/Truncate** (a 10.1x to 17.7x improvement). However, it did not achieve 70% gap recovery to the native ceiling (100.0%).
  - *Post-mortem:* Specialized biomedical claim retrieval (SciFact) exhibits sharp, non-linear semantic manifolds between compact 384-d vectors and expressive 1024-d representations. Pure orthogonal Procrustes captures the global rotation but cannot unfold intrinsic topological curvature. This precisely highlights why the **Conformal Recall Certificate is essential**: it informs the edge application of this exact fidelity limitation on a per-query basis rather than silently returning corrupted results.
- **Conformal Calibration Guarantee (E4):**
  - *Empirical outcome:* Nominal 90.0% ($\alpha=0.10$) achieved **exactly 90.0% empirical coverage**; nominal 95.0% ($\alpha=0.05$) achieved **100.0% coverage**. Finite-sample marginal coverage holds strictly.
- **Out-of-Distribution Escalation (E6):**
  - *Empirical outcome:* Under distribution shift, naive coverage degraded to 20.0%, but the query router escalated **100.0% of shifted queries** to the cloud, preventing silent edge failures.
- **Repair Policy Efficiency (E7):**
  - *Empirical outcome:* Frequency-based repair improved overlap from 11.6% to 13.2% within a modest 200 KB budget (49 document vectors). On small, highly-clustered reference sets, error-only policy provides slower gains than frequency-guided repair.

