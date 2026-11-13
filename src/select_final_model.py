from __future__ import annotations

import argparse
import json
from pathlib import Path
import pandas as pd


def includes_zero(low: float, high: float) -> bool:
    return low <= 0 <= high


def main():
    ap = argparse.ArgumentParser(description="Reproduce the documented final model-selection decision from locked result tables.")
    ap.add_argument("--performance", default="results/model_performance_8_configs.csv")
    ap.add_argument("--incremental", default="results/incremental_value_m1_vs_m0.csv")
    ap.add_argument("--pairwise", default="results/pairwise_m1_model_comparisons.csv")
    ap.add_argument("--output", default="results/model_selection_decision.json")
    args = ap.parse_args()

    perf = pd.read_csv(args.performance)
    inc = pd.read_csv(args.incremental)
    pair = pd.read_csv(args.pairwise)

    # 1) M1 must show consistent incremental value over M0 across algorithms.
    if not ((inc["AUROC_CI_low"] > 0).all() and (inc["AUPRC_CI_low"] > 0).all() and (inc["Brier_CI_high"] < 0).all()):
        raise RuntimeError("Locked criterion failed: M1 incremental benefit is not consistently supported.")

    m1 = perf[perf["configuration"] == "M1"].set_index("model")
    if "RF" not in m1.index or "XGBoost" not in m1.index:
        raise RuntimeError("Expected M1 RF and XGBoost rows not found.")

    # 2) XGBoost has numerically higher ranking metrics, but paired CIs versus RF cross zero.
    xgb_rf = pair[pair["comparison"] == "XGBoost_minus_RF"].set_index("metric")
    discrim_not_clearly_different = (
        includes_zero(xgb_rf.loc["AUROC", "CI_low"], xgb_rf.loc["AUROC", "CI_high"])
        and includes_zero(xgb_rf.loc["AUPRC", "CI_low"], xgb_rf.loc["AUPRC", "CI_high"])
    )
    if not discrim_not_clearly_different:
        raise RuntimeError("Locked criterion failed: XGBoost vs RF discrimination difference no longer crosses zero.")

    # 3) RF has the lowest Brier score among M1 candidates.
    brier_winner = m1["Brier"].idxmin()
    if brier_winner != "RF":
        raise RuntimeError(f"Locked criterion failed: lowest M1 Brier is {brier_winner}, not RF.")

    decision = {
        "selected_configuration": "M1-RF",
        "selected_algorithm": "Random Forest",
        "selection_basis": [
            "M1 showed consistent incremental predictive value over M0 across all four algorithms.",
            "XGBoost had numerically higher AUROC/AUPRC, but paired 95% CIs for its discrimination differences versus RF included zero.",
            "RF had the lowest Brier score among M1 models.",
            "RF provided a strong balance of discrimination, calibration, stability, interpretability, and parsimony."
        ],
        "important_constraint": "This is a documented multi-criterion selection decision, not a claim that RF is statistically superior to every alternative on every metric.",
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(decision, indent=2), encoding="utf-8")
    print(json.dumps(decision, indent=2))


if __name__ == "__main__":
    main()
