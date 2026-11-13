from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import shap

from data_utils import load_and_prepare_dataset


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def clean_feature_name(name: str) -> str:
    name = name.replace("age__", "").replace("prev_dx__", "").replace("functional__", "")
    name = name.replace("cat__sex_", "Sex=").replace("cat__education_", "Education=")
    name = name.replace("cat__employment_", "Employment=")
    name = name.replace("previous_anxiety_diagnosis", "Previous anxiety diagnosis")
    name = name.replace("functional_impairment", "Functional impairment")
    name = name.replace("age", "Age")
    return name


def positive_class_shap(explainer, X_transformed):
    values = explainer.shap_values(X_transformed)
    # SHAP API compatibility across versions.
    if isinstance(values, list):
        return np.asarray(values[1] if len(values) > 1 else values[0])
    arr = np.asarray(values)
    if arr.ndim == 3:
        # (samples, features, classes)
        return arr[:, :, 1]
    return arr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--outdir", default="artifacts")
    ap.add_argument("--figdir", default="figures")
    args = ap.parse_args()

    data_path = Path(args.data)
    model_path = Path(args.model)
    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    figdir = Path(args.figdir); figdir.mkdir(parents=True, exist_ok=True)

    X, y, _, audit = load_and_prepare_dataset(data_path)
    X_m1 = X[["age", "sex", "education", "employment", "previous_anxiety_diagnosis", "functional_impairment"]]

    pipeline = joblib.load(model_path)
    prep = pipeline.named_steps["preprocess"]
    rf = pipeline.named_steps["model"]
    Xt = prep.transform(X_m1)
    raw_names = list(prep.get_feature_names_out())
    feature_names = [clean_feature_name(x) for x in raw_names]

    explainer = shap.TreeExplainer(rf)
    sv = positive_class_shap(explainer, Xt)
    mean_abs = np.mean(np.abs(sv), axis=0)
    order = np.argsort(mean_abs)[::-1]

    ranked = [
        {"rank": int(i + 1), "feature": feature_names[j], "mean_abs_shap": float(mean_abs[j])}
        for i, j in enumerate(order)
    ]

    shap_json = {
        "model": "M1-Random Forest",
        "n": int(len(y)),
        "method": "SHAP TreeExplainer",
        "target_class": "GAD-7 >=10",
        "model_sha256": sha256(model_path),
        "dataset_sha256": sha256(data_path),
        "qc": audit,
        "global_feature_importance": ranked,
        "interpretation_constraint": "SHAP values quantify predictive attribution within this fitted model and must not be interpreted as causal effects.",
    }
    json_path = outdir / "final_rf_m1_shap.json"
    json_path.write_text(json.dumps(shap_json, indent=2, ensure_ascii=False), encoding="utf-8")

    top_n = min(12, len(order))
    idx = order[:top_n][::-1]
    plt.figure(figsize=(8, 6))
    plt.barh([feature_names[i] for i in idx], mean_abs[idx])
    plt.xlabel("Mean |SHAP value|")
    plt.ylabel("Predictor / encoded feature")
    plt.title("Global SHAP Importance — Final M1 Random Forest")
    plt.tight_layout()
    fig_path = figdir / "shap_global_rf_m1.png"
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(json.dumps({
        "json": str(json_path),
        "figure": str(fig_path),
        "top_features": ranked[:10],
        "model_sha256": sha256(model_path),
        "shap_json_sha256": sha256(json_path),
        "figure_sha256": sha256(fig_path),
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
