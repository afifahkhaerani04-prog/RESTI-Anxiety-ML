from __future__ import annotations

import argparse
import hashlib
import json
import platform
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from data_utils import load_and_prepare_dataset

SEED = 42
FINAL_PARAMS = {
    "n_estimators": 200,
    "max_depth": 10,
    "min_samples_split": 2,
    "min_samples_leaf": 2,
    "max_features": "sqrt",
    "class_weight": "balanced",
    "random_state": SEED,
    "n_jobs": -1,
}

# Locked RF search space, used only to verify the full-data refit tuning choice.
RF_GRID = {
    "model__n_estimators": [200, 500],
    "model__max_depth": [None, 5, 10],
    "model__min_samples_split": [2, 5],
    "model__min_samples_leaf": [1, 2, 4],
    "model__max_features": ["sqrt", "log2"],
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_pipeline() -> Pipeline:
    prep = ColumnTransformer(
        transformers=[
            ("age", "passthrough", ["age"]),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), ["sex", "education", "employment"]),
            ("prev_dx", "passthrough", ["previous_anxiety_diagnosis"]),
            ("functional", "passthrough", ["functional_impairment"]),
        ],
        remainder="drop",
        sparse_threshold=0,
    )
    rf = RandomForestClassifier(**FINAL_PARAMS)
    return Pipeline([("preprocess", prep), ("model", rf)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="Private raw .xlsx dataset; do not commit it publicly.")
    ap.add_argument("--outdir", default="artifacts")
    ap.add_argument("--verify-grid", action="store_true", help="Optionally rerun the locked RF grid on full n=510 before final refit.")
    args = ap.parse_args()

    data_path = Path(args.data)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    X, y, _, audit = load_and_prepare_dataset(data_path)
    X_m1 = X[["age", "sex", "education", "employment", "previous_anxiety_diagnosis", "functional_impairment"]]

    verification = None
    if args.verify_grid:
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
        search = GridSearchCV(build_pipeline(), RF_GRID, scoring="roc_auc", cv=cv, n_jobs=-1, refit=True)
        search.fit(X_m1, y)
        verification = {
            "full_data_inner_cv_mean_AUROC": float(search.best_score_),
            "best_params": search.best_params_,
        }
        # The verification result is diagnostic only. Generalization performance remains outer-CV based.

    pipeline = build_pipeline()
    pipeline.fit(X_m1, y)

    model_path = outdir / "final_rf_m1_pipeline.joblib"
    joblib.dump(pipeline, model_path)

    metadata = {
        "model": "M1-Random Forest",
        "training_n": int(len(y)),
        "class_counts": {"negative": int((y == 0).sum()), "positive": int((y == 1).sum())},
        "predictors": list(X_m1.columns),
        "parameters": FINAL_PARAMS,
        "random_seed": SEED,
        "dataset_sha256": sha256(data_path),
        "model_sha256": sha256(model_path),
        "qc": audit,
        "verification": verification,
        "software": {
            "python": platform.python_version(),
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "scikit-learn": sklearn.__version__,
            "joblib": joblib.__version__,
        },
        "important_note": "Full-data refit is for final artifact generation. Predictive performance must be reported from the locked outer nested cross-validation, not from training/refit performance.",
    }
    meta_path = outdir / "final_rf_m1_metadata.json"
    meta_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")

    print(json.dumps(metadata, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
