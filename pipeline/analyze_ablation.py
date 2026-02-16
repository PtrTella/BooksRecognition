#!/usr/bin/env python3
"""
Analyzes ablation_results.csv and identifies the best value for each parameter.

Criteria:
1. Highest F1 score.
2. If tied, highest Precision (fewer false positives).
3. If tied, highest Recall (fewer false negatives).
4. If tied, the default value (if it exists in the ties).

Usage:
    python pipeline/analyze_ablation.py
"""

import csv
import os
from pathlib import Path
from collections import defaultdict

RESULTS_CSV = Path(__file__).resolve().parent.parent / "output_ablation" / "ablation_results.csv"

def main():
    if not RESULTS_CSV.exists():
        print(f"Error: {RESULTS_CSV} not found. Run ablation.py first.")
        return

    # Load results
    experiments = []
    with open(RESULTS_CSV, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Convert types
            row["f1"] = float(row["f1"])
            row["precision"] = float(row["precision"])
            row["recall"] = float(row["recall"])
            row["is_default"] = row["is_default"].lower() == "true"
            experiments.append(row)

    if not experiments:
        print("Error: No data in CSV.")
        return

    # Group by parameter
    by_param = defaultdict(list)
    baseline = None
    for exp in experiments:
        if exp["parameter"] == "BASELINE":
            baseline = exp
            continue
        by_param[exp["parameter"]].append(exp)

    print(f"{'='*80}")
    print(f"{'PARAM':<25} | {'DEFAULT':<10} | {'BEST':<10} | {'F1':<6} | {'P':<6} | {'R':<6} | {'DELTA'}")
    print(f"{'─'*80}")

    best_config = {}

    # Identify best for each param
    for param, trials in sorted(by_param.items()):
        # Sorting criteria: F1 desc, Precision desc, Recall desc, is_default desc
        # (True > False, so is_default desc puts the default value first among ties)
        sorted_trials = sorted(
            trials,
            key=lambda x: (x["f1"], x["precision"], x["recall"], x["is_default"]),
            reverse=True
        )
        
        best = sorted_trials[0]
        default_row = next((t for t in trials if t["is_default"]), None)
        default_val = default_row["value"] if default_row else "???"
        
        delta = best["f1"] - (baseline["f1"] if baseline else 0)
        delta_str = f"{delta:+.4f}" if delta != 0 else "0.0000"
        if delta > 0.0001:
            delta_str = f"\033[92m{delta_str}\033[0m" # Green
        elif delta < -0.0001:
            delta_str = f"\033[91m{delta_str}\033[0m" # Red

        print(f"{param:<25} | {str(default_val):<10} | {str(best['value']):<10} | {best['f1']:.3f} | {best['precision']:.3f} | {best['recall']:.3f} | {delta_str}")
        
        best_config[param] = best["value"]

    print(f"{'─'*80}")
    if baseline:
        print(f"BASELINE F1: {baseline['f1']:.3f} (P={baseline['precision']:.3f}, R={baseline['recall']:.3f})")
    print(f"{'='*80}\n")

    # ── Top 5 Overall ───────────────────────────────────────────────────
    print(f"TOP 5 OVERALL CONFIGURATIONS (from individual trials):")
    print(f"{'RANK':<5} | {'PARAMETER':<20} | {'VALUE':<10} | {'F1':<6} | {'P':<6} | {'R':<6} | {'RESULTS'}")
    print(f"{'─'*80}")
    
    # All experiments including baseline
    all_experiments = experiments
    top_5 = sorted(
        all_experiments,
        key=lambda x: (x["f1"], x["precision"], x["recall"]),
        reverse=True
    )[:5]

    for i, exp in enumerate(top_5, 1):
        name = exp["parameter"] if exp["parameter"] != "BASELINE" else "BASELINE"
        res_str = f"TP={exp['tp']}, FP={exp['fp']}, FN={exp['fn']}"
        print(f"#{i:<4} | {name:<20} | {str(exp['value']):<10} | {exp['f1']:.3f} | {exp['precision']:.3f} | {exp['recall']:.3f} | {res_str}")
    
    print(f"{'─'*80}\n")

    print("Suggested Best Configuration (assembled from locally-best values):")
    for p, v in sorted(best_config.items()):
        # Find the experiment for this specific param and value
        exp = next(t for t in by_param[p] if str(t["value"]) == str(v))
        score_str = f"F1={exp['f1']:.3f} (P={exp['precision']:.3f}, R={exp['recall']:.3f})"
        print(f"    {p:<20} = {v:<10}  | {score_str}")



if __name__ == "__main__":
    main()
