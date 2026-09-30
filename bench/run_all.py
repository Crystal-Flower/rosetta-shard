import time

from bench.exp_adapter_ablation import run_exp_adapter_ablation
from bench.exp_anchor_curve import run_exp_anchor_curve
from bench.exp_certificate import run_exp_certificate
from bench.exp_collapse import run_exp_collapse
from bench.exp_device import run_exp_device
from bench.exp_feature_ablation import run_exp_feature_ablation
from bench.exp_repair import run_exp_repair
from bench.exp_risk_coverage import run_exp_risk_coverage
from bench.exp_shift import run_exp_shift
from rosetta.config import RESULTS_DIR


def main():
    print("=" * 80)
    print("ROSETTA SHARD — COMPREHENSIVE BENCHMARK SUITE")
    print("Code Cubicle 6.0 | Theme 3: AI-Powered Edge Memory & Intelligence Platform")
    print("=" * 80)

    t0 = time.time()

    e1 = run_exp_collapse()
    e2 = run_exp_adapter_ablation()
    e3 = run_exp_anchor_curve()
    e4 = run_exp_certificate()
    e5 = run_exp_risk_coverage()
    e6 = run_exp_shift()
    e7 = run_exp_repair()
    e8 = run_exp_device()
    e9 = run_exp_feature_ablation()

    total_time = time.time() - t0

    print("\n" + "=" * 80)
    print("BENCHMARK EXECUTION SUMMARY (ALL 9 EXPERIMENTS COMPLETED)")
    print(f"Total benchmark elapsed time: {total_time:.1f} seconds")
    print("=" * 80)
    print(f"| {'Exp':<4} | {'Name':<28} | {'Key Metric':<24} | {'Value':<12} |")
    print(f"|{'-' * 6}|{'-' * 30}|{'-' * 26}|{'-' * 14}|")
    print(
        f"| E1   | Retrieval Collapse           | S1 Silent Overlap        | {e1['s1_silent_overlap'] * 100:.1f}%        |"
    )
    print(
        f"| E1   | Space Guard Protection       | S1 Mismatch Blocked      | {e1['s1_guard_blocked_rate'] * 100:.1f}%       |"
    )
    print(
        f"| E2   | Adapter Ablation             | Contrastive Overlap@10   | {e2['methods']['contrastive']['overlap_at_10'] * 100:.1f}%        |"
    )
    print(
        f"| E2   | Adapter Ablation             | Pad/Truncate Baseline    | {e2['methods']['pad_truncate']['overlap_at_10'] * 100:.1f}%        |"
    )
    print(
        f"| E3   | Sample Efficiency            | Max Ridge Overlap@10     | {max(e3['ridge_overlaps']) * 100:.1f}%        |"
    )
    print(
        f"| E4   | Certificate Calibration (90%)| Empirical Test Coverage  | {e4['empirical_coverage'][1] * 100:.1f}%        |"
    )
    print(
        f"| E5   | Gated Risk-Coverage (tau=.2) | Local Served Fraction    | {e5['local_served_fraction'][4] * 100:.1f}%        |"
    )
    print(
        f"| E6   | Distribution Shift           | OOD Escalation Rate      | {e6['ood_escalation_rate'] * 100:.1f}%        |"
    )
    print(
        f"| E7   | Byte-Budgeted Repair (100KB) | Priority Policy Overlap  | {e7['policies']['priority']['overlaps'][3] * 100:.1f}%        |"
    )
    print(
        f"| E8   | Real Hardware Latency        | Total Query p50          | {e8['query']['total_p50_ms']:.2f} ms     |"
    )
    print(
        f"| E9   | Feature Ablation             | Full Model Test MAE     | {e9['ablations']['full_model']['loss_mae']:.4f}       |"
    )
    print("=" * 80)
    print(f"All metric JSONs and plots committed to: {RESULTS_DIR.resolve()}\n")


if __name__ == "__main__":
    main()
