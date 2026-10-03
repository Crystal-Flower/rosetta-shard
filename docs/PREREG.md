# Benchmark Protocol & Pre-Registration (PREREG)
**Rosetta Shard — Frozen Evaluation Protocol**

---

## 1. Datasets & Fixed Splits
- **Primary In-Domain Dataset:** SciFact / FiQA corpus (scientific / financial QA benchmark).
- **Corpus Size:** Reference collection of 5,000 documents.
- **Query Splits:**
  - `Q_anchor`: 500 queries used exclusively for anchor generation alongside corpus samples.
  - `Q_dev`: 150 queries for hyperparameter selection ($\lambda$ in Ridge).
  - `Q_cal_train`: 200 queries for training the certificate feature predictor $\hat{h}(f)$.
  - `Q_cal`: 300 queries for split-conformal calibration quantile computation $\hat{q}_\alpha$.
  - `Q_test`: 400 held-out queries evaluated strictly at final benchmark execution.
- **Out-of-Distribution Shift Dataset:** Cross-domain test queries from an alternate genre to test detection of distributional shift.
- **Fixed Random Seed:** `42` (pinned across all sampling, partitioning, and evaluations).

---

## 2. Experiments Specification

| ID | Experiment Name | Primary Metric | Hypothesis / Target Bar |
|---|---|---|---|
| **E1** | Collapse Demonstration | Overlap@10 (S1 unadapted) | Overlap@10 collapses to $< 0.05$ under silent model swap |
| **E2** | Adapter Ablation | Overlap@10, Recall@10, nDCG@10 | Ridge / Procrustes recovers $\ge 70\%$ of the gap to native ceiling |
| **E3** | Anchor Sample Efficiency | Overlap@10 vs Anchor Count | Plateau begins around 500-1000 anchors; $>60\%$ recovery with only 200 anchors |
| **E4** | Conformal Calibration | Empirical Coverage at $\alpha \in \{0.05, 0.10, 0.20\}$ | Empirical coverage $\ge (1 - \alpha) - 2\sigma$ on held-out $Q_{\mathrm{test}}$ |
| **E5** | Risk-Coverage Tradeoff | Local Serve % vs Mean Overlap | Gating with $\tau_{\mathrm{serve}}$ systematically removes low-overlap queries |
| **E6** | Distribution Shift Analysis | Empirical Coverage & OOD Detection | Shift degrades raw coverage, but `sim_anchor_max` triggers escalation |
| **E7** | Hot-Set Repair Efficiency | Overlap Gain vs KB Downloaded | Priority-based repair outperforms uniform random repair by $\ge 2\times$ |
| **E8** | Device Benchmark | Migration Latency, Shard Size, RSS | Edge migration runs orders of magnitude faster than full re-embedding |
| **E9** | Feature Importance Ablation | Loss Prediction MSE | Anchor distance and score margins provide dominant predictive power |

---

## 3. Strict Ground Rules
- **R1:** Every reported number in the README must originate from machine-readable JSON in `results/`.
- **R2:** All benchmark measurements run against live `EdgeShard` instances in `qdrant-edge-py`.
- **R5:** If any pre-registered bar is missed during benchmark execution, the actual measured value must be reported with an honest post-mortem in `LIMITATIONS.md`.
