from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")  # Non-interactive headless backend
import matplotlib.pyplot as plt
import numpy as np


def plot_exp_collapse(data: dict[str, Any], out_path: Path):
    plt.style.use(
        "seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default"
    )
    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=150)

    categories = [
        "S1 Cross-Space\n(No Guard/Adapter)",
        "S1 with\nSpace Guard",
        "S2 Pad/Truncate\nBaseline",
        "S2 Ridge Adapter\n(Rosetta)",
        "S2 Native Ceiling\n(Same Model)",
    ]
    overlaps = [
        data.get("s1_silent_overlap", 0.0),
        0.0,  # Blocked / 0% false positives
        data.get("s2_baseline_overlap", 0.0),
        data.get("s2_adapted_overlap", 0.0),
        1.0,
    ]
    colors = ["#e74c3c", "#95a5a6", "#e67e22", "#2ecc71", "#3498db"]

    bars = ax.bar(categories, overlaps, color=colors, width=0.55, edgecolor="black", linewidth=0.8)
    ax.set_ylabel("Top-10 Overlap with True Ranking", fontsize=11, fontweight="bold")
    ax.set_title(
        "E1: Cross-Model Retrieval Collapse vs. Rosetta Shard Recovery",
        fontsize=12,
        fontweight="bold",
        pad=12,
    )
    ax.set_ylim(0, 1.1)

    for bar, val in zip(bars, overlaps):
        height = bar.get_height()
        label = f"{val * 100:.1f}%" if bar != bars[1] else "BLOCKED\n(Exception)"
        ax.annotate(
            label,
            xy=(bar.get_x() + bar.get_width() / 2, max(height, 0.05)),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
        )

    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_exp_adapter_ablation(data: dict[str, Any], out_path: Path):
    plt.style.use(
        "seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default"
    )
    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=150)

    methods = list(data["methods"].keys())
    overlaps = [data["methods"][m]["overlap_at_10"] for m in methods]
    labels = [m.replace("_", " ").title() for m in methods]

    colors = ["#e74c3c", "#e67e22", "#f1c40f", "#3498db", "#2ecc71", "#27ae60"]
    if len(colors) < len(methods):
        colors = colors * 2

    bars = ax.bar(
        labels, overlaps, color=colors[: len(methods)], width=0.55, edgecolor="black", linewidth=0.8
    )
    ax.set_ylabel("Fidelity (Overlap@10)", fontsize=11, fontweight="bold")
    ax.set_title(
        "E2: Cross-Model Adapter Architecture Ablation", fontsize=12, fontweight="bold", pad=12
    )
    ax.set_ylim(0, 1.1)

    for bar, val in zip(bars, overlaps):
        ax.annotate(
            f"{val * 100:.1f}%",
            xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
        )

    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_exp_anchor_curve(data: dict[str, Any], out_path: Path):
    plt.style.use(
        "seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default"
    )
    fig, ax = plt.subplots(figsize=(7.5, 4.5), dpi=150)

    anchor_counts = data["anchor_counts"]
    ridge_overlaps = data["ridge_overlaps"]
    proc_overlaps = data.get("procrustes_overlaps", [])

    ax.plot(
        anchor_counts,
        ridge_overlaps,
        marker="o",
        linewidth=2.2,
        color="#2ecc71",
        label="Ridge Adapter",
    )
    if proc_overlaps:
        ax.plot(
            anchor_counts,
            proc_overlaps,
            marker="s",
            linewidth=2.0,
            color="#3498db",
            linestyle="--",
            label="Procrustes Adapter",
        )
    contrastive_overlaps = data.get("contrastive_overlaps", [])
    if contrastive_overlaps:
        ax.plot(
            anchor_counts,
            contrastive_overlaps,
            marker="^",
            linewidth=2.0,
            color="#9b59b6",
            linestyle="-.",
            label="Contrastive Adapter",
        )

    ax.set_xscale("log")
    ax.set_xlabel("Number of Anchors (Mix of Queries + Docs)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Top-10 Overlap", fontsize=11, fontweight="bold")
    ax.set_title(
        "E3: Anchor Sample Efficiency Curve (Data Economy)", fontsize=12, fontweight="bold", pad=12
    )
    ax.set_ylim(0, 1.05)
    ax.legend(loc="lower right", frameon=True)

    for x, y in zip(anchor_counts, ridge_overlaps):
        ax.annotate(
            f"{y * 100:.1f}%",
            (x, y),
            textcoords="offset points",
            xytext=(0, 6),
            ha="center",
            fontsize=8,
        )

    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_exp_certificate(data: dict[str, Any], out_path: Path):
    plt.style.use(
        "seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default"
    )
    fig, ax = plt.subplots(figsize=(6.5, 4.5), dpi=150)

    alphas = data["alphas"]
    nominal = [1.0 - a for a in alphas]
    empirical = data["empirical_coverage"]

    x = np.arange(len(alphas))
    width = 0.35

    rects1 = ax.bar(
        x - width / 2,
        nominal,
        width,
        label="Nominal Guarantee (1 - α)",
        color="#3498db",
        edgecolor="black",
        linewidth=0.8,
    )
    rects2 = ax.bar(
        x + width / 2,
        empirical,
        width,
        label="Empirical Coverage (Held-out Test)",
        color="#2ecc71",
        edgecolor="black",
        linewidth=0.8,
    )

    ax.set_ylabel("Coverage Probability", fontsize=11, fontweight="bold")
    ax.set_title(
        "E4: Conformal Recall Certificate Calibration", fontsize=12, fontweight="bold", pad=12
    )
    ax.set_xticks(x)
    ax.set_xticklabels([f"α = {a}\n(Nominal {int((1 - a) * 100)}%)" for a in alphas])
    ax.set_ylim(0, 1.15)
    ax.legend(loc="lower left", frameon=True)

    for rect in rects1:
        h = rect.get_height()
        ax.annotate(
            f"{h * 100:.1f}%",
            (rect.get_x() + rect.get_width() / 2, h),
            textcoords="offset points",
            xytext=(0, 3),
            ha="center",
            fontsize=8,
        )
    for rect in rects2:
        h = rect.get_height()
        ax.annotate(
            f"{h * 100:.1f}%",
            (rect.get_x() + rect.get_width() / 2, h),
            textcoords="offset points",
            xytext=(0, 3),
            ha="center",
            fontsize=8,
            fontweight="bold",
        )

    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_exp_risk_coverage(data: dict[str, Any], out_path: Path):
    plt.style.use(
        "seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default"
    )
    fig, ax1 = plt.subplots(figsize=(7.5, 4.5), dpi=150)

    taus = data["tau_thresholds"]
    served_pct = [s * 100 for s in data["local_served_fraction"]]
    mean_overlaps = [m * 100 for m in data["mean_overlap_served"]]
    p10_overlaps = [p * 100 for p in data["p10_overlap_served"]]

    ax1.plot(taus, served_pct, color="#2980b9", linewidth=2.2, marker="o", label="% Served Locally")
    ax1.set_xlabel("Confidence Gating Threshold (τ_serve)", fontsize=11, fontweight="bold")
    ax1.set_ylabel(
        "% Queries Served Locally on Edge", color="#2980b9", fontsize=11, fontweight="bold"
    )
    ax1.tick_params(axis="y", labelcolor="#2980b9")
    ax1.set_ylim(0, 105)

    ax2 = ax1.twinx()
    ax2.plot(
        taus, mean_overlaps, color="#27ae60", linewidth=2.2, marker="s", label="Mean Overlap@10"
    )
    ax2.plot(
        taus,
        p10_overlaps,
        color="#e67e22",
        linewidth=2.0,
        linestyle="--",
        marker="^",
        label="10th Percentile Overlap (Worst-Decile)",
    )
    ax2.set_ylabel(
        "Retrieval Fidelity of Served Queries (%)", color="#27ae60", fontsize=11, fontweight="bold"
    )
    ax2.tick_params(axis="y", labelcolor="#27ae60")
    ax2.set_ylim(0, 105)

    ax1.set_title(
        "E5: Risk–Coverage Tradeoff with Gated Edge Routing", fontsize=12, fontweight="bold", pad=12
    )
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="lower left", frameon=True)

    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_exp_shift(data: dict[str, Any], out_path: Path):
    plt.style.use(
        "seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default"
    )
    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=150)

    cats = [
        "In-Domain\n(Exchangeable)",
        "Out-of-Distribution\n(Domain Shift)",
        "OOD Flagged &\nEscalated to Cloud",
    ]
    vals = [
        data["in_domain_coverage"] * 100,
        data["shifted_coverage"] * 100,
        data["ood_escalation_rate"] * 100,
    ]
    colors = ["#2ecc71", "#e74c3c", "#3498db"]

    bars = ax.bar(cats, vals, color=colors, width=0.5, edgecolor="black", linewidth=0.8)
    ax.set_ylabel("Percentage (%)", fontsize=11, fontweight="bold")
    ax.set_title(
        "E6: Distribution Shift & OOD Detection Behavior", fontsize=12, fontweight="bold", pad=12
    )
    ax.set_ylim(0, 115)

    for bar, val in zip(bars, vals):
        ax.annotate(
            f"{val:.1f}%",
            xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
        )

    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_exp_repair(data: dict[str, Any], out_path: Path):
    plt.style.use(
        "seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default"
    )
    fig, ax = plt.subplots(figsize=(7.5, 4.5), dpi=150)

    budgets_kb = data["budgets_kb"]
    policies = data["policies"]

    markers = {"priority": "o", "frequency": "s", "error": "^", "random": "x"}
    colors = {
        "priority": "#2ecc71",
        "frequency": "#3498db",
        "error": "#9b59b6",
        "random": "#e74c3c",
    }

    for pol, res in policies.items():
        ax.plot(
            budgets_kb,
            res["overlaps"],
            label=f"{pol.capitalize()} Policy",
            marker=markers.get(pol, "o"),
            color=colors.get(pol, "gray"),
            linewidth=2.0 if pol == "priority" else 1.5,
        )

    ax.set_xlabel("Hot-Set Download Budget (Kilobytes)", fontsize=11, fontweight="bold")
    ax.set_ylabel("System Overlap@10", fontsize=11, fontweight="bold")
    ax.set_title(
        "E7: Byte-Budgeted Hot-Set Repair Efficiency", fontsize=12, fontweight="bold", pad=12
    )
    ax.legend(loc="lower right", frameon=True)

    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_exp_device(data: dict[str, Any], out_path: Path):
    plt.style.use(
        "seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default"
    )
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.2), dpi=150)

    # 1. Migration time vs Re-embed time
    n_points = [str(n) for n in data["migration"]["points"]]
    migrate_times = data["migration"]["time_sec"]
    reembed_times = data["migration"]["reembed_time_sec"]

    x = np.arange(len(n_points))
    width = 0.35

    ax1.bar(
        x - width / 2,
        migrate_times,
        width,
        label="Rosetta Migration",
        color="#2ecc71",
        edgecolor="black",
        linewidth=0.8,
    )
    ax1.bar(
        x + width / 2,
        reembed_times,
        width,
        label="Full Large Re-embed",
        color="#e74c3c",
        edgecolor="black",
        linewidth=0.8,
    )
    ax1.set_xlabel("Shard Point Count", fontsize=10, fontweight="bold")
    ax1.set_ylabel("Time (seconds)", fontsize=10, fontweight="bold")
    ax1.set_title("Migration vs Re-embedding Time", fontsize=11, fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels(n_points)
    ax1.set_yscale("log")
    ax1.legend(loc="upper left")

    # 2. Query Latency Breakdown
    stages = ["Small Embed", "Adapter Proj", "Edge Shard Query"]
    p50s = [
        data["query"]["embed_p50_ms"],
        data["query"]["adapter_p50_ms"],
        data["query"]["search_p50_ms"],
    ]
    colors = ["#3498db", "#f1c40f", "#2ecc71"]

    ax2.bar(stages, p50s, color=colors, width=0.5, edgecolor="black", linewidth=0.8)
    ax2.set_ylabel("Latency (ms, p50)", fontsize=10, fontweight="bold")
    ax2.set_title("On-Device Query Latency Breakdown", fontsize=11, fontweight="bold")
    for bar, val in zip(ax2.patches, p50s):
        ax2.annotate(
            f"{val:.2f} ms",
            (bar.get_x() + bar.get_width() / 2, bar.get_height()),
            textcoords="offset points",
            xytext=(0, 3),
            ha="center",
            fontsize=9,
            fontweight="bold",
        )

    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_exp_feature_ablation(data: dict[str, Any], out_path: Path):
    plt.style.use(
        "seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default"
    )
    fig, ax = plt.subplots(figsize=(7.5, 4.5), dpi=150)

    configs = list(data["ablations"].keys())
    maes = [data["ablations"][c]["loss_mae"] for c in configs]
    labels = [c.replace("_", " ").title() for c in configs]

    bars = ax.bar(labels, maes, color="#3498db", width=0.5, edgecolor="black", linewidth=0.8)
    ax.set_ylabel("Certificate Prediction MAE (Lower is Better)", fontsize=11, fontweight="bold")
    ax.set_title(
        "E9: Certificate Loss Predictor Feature Ablation", fontsize=12, fontweight="bold", pad=12
    )
    plt.xticks(rotation=20, ha="right")

    for bar, val in zip(bars, maes):
        ax.annotate(
            f"{val:.4f}",
            (bar.get_x() + bar.get_width() / 2, bar.get_height()),
            textcoords="offset points",
            xytext=(0, 3),
            ha="center",
            fontsize=8,
            fontweight="bold",
        )

    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
