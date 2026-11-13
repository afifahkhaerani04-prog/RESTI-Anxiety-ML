from __future__ import annotations

import argparse
import json
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier

from data_utils import load_and_prepare_dataset

warnings.filterwarnings("ignore")
SEED = 42
BOOTSTRAP_B = 2000


class BalancedXGBClassifier(BaseEstimator, ClassifierMixin):
    """XGBoost wrapper that recomputes scale_pos_weight inside every current fit."""
    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=3, min_child_weight=1,
                 subsample=1.0, colsample_bytree=1.0, gamma=0.0, reg_alpha=0.0,
                 reg_lambda=1.0, tree_method="hist", random_state=42):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_child_weight = min_child_weight
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.gamma = gamma
        self.reg_alpha = reg_alpha
        self.reg_lambda = reg_lambda
        self.tree_method = tree_method
        self.random_state = random_state

    def fit(self, X, y):
        y = np.asarray(y)
        pos = int((y == 1).sum())
        neg = int((y == 0).sum())
        ratio = float(neg / pos) if pos else 1.0
        self.model_ = XGBClassifier(
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            max_depth=self.max_depth,
            min_child_weight=self.min_child_weight,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            gamma=self.gamma,
            reg_alpha=self.reg_alpha,
            reg_lambda=self.reg_lambda,
            scale_pos_weight=ratio,
            tree_method=self.tree_method,
            random_state=self.random_state,
            objective="binary:logistic",
            eval_metric="logloss",
            n_jobs=1,
            verbosity=0,
        )
        self.model_.fit(X, y)
        self.classes_ = self.model_.classes_
        return self

    def predict_proba(self, X):
        return self.model_.predict_proba(X)

    def predict(self, X):
        return self.model_.predict(X)


def make_preprocessor(config: str, scaled_age: bool):
    transformers = [
        ("age", StandardScaler() if scaled_age else "passthrough", ["age"]),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), ["sex", "education", "employment"]),
        ("diag", "passthrough", ["previous_anxiety_diagnosis"]),
    ]
    if config == "M1":
        transformers.append(("func", "passthrough", ["functional_impairment"]))
    return ColumnTransformer(transformers, remainder="drop", sparse_threshold=0)


def model_spec(name: str, config: str):
    prep = make_preprocessor(config, scaled_age=name in {"LR", "SVM"})
    if name == "LR":
        est = LogisticRegression(solver="liblinear", class_weight="balanced", max_iter=5000, random_state=SEED)
        grid = {"model__penalty": ["l1", "l2"], "model__C": [0.01, 0.1, 1, 10, 100]}
        return Pipeline([("prep", prep), ("model", est)]), GridSearchCV, grid, {}
    if name == "SVM":
        est = SVC(class_weight="balanced", probability=True, random_state=SEED)
        grid = [
            {"model__kernel": ["linear"], "model__C": [0.1, 1, 10, 100]},
            {"model__kernel": ["rbf"], "model__C": [0.1, 1, 10, 100], "model__gamma": ["scale", 0.001, 0.01, 0.1, 1]},
        ]
        return Pipeline([("prep", prep), ("model", est)]), GridSearchCV, grid, {}
    if name == "RF":
        est = RandomForestClassifier(class_weight="balanced", random_state=SEED, n_jobs=1)
        grid = {
            "model__n_estimators": [200, 500],
            "model__max_depth": [None, 5, 10],
            "model__min_samples_split": [2, 5],
            "model__min_samples_leaf": [1, 2, 4],
            "model__max_features": ["sqrt", "log2"],
        }
        return Pipeline([("prep", prep), ("model", est)]), GridSearchCV, grid, {}
    if name == "XGBoost":
        est = BalancedXGBClassifier(random_state=SEED)
        grid = {
            "model__n_estimators": [100, 300, 500],
            "model__learning_rate": [0.01, 0.05, 0.10],
            "model__max_depth": [2, 3, 5],
            "model__min_child_weight": [1, 3, 5],
            "model__subsample": [0.70, 0.90, 1.00],
            "model__colsample_bytree": [0.70, 0.90, 1.00],
            "model__gamma": [0, 0.10, 0.50],
            "model__reg_alpha": [0, 0.10, 1.00],
            "model__reg_lambda": [1, 5, 10],
        }
        return Pipeline([("prep", prep), ("model", est)]), RandomizedSearchCV, grid, {"n_iter": 40, "random_state": SEED}
    raise ValueError(name)


def calibration_intercept_slope(y, p):
    eps = 1e-6
    p = np.clip(np.asarray(p, float), eps, 1 - eps)
    z = np.log(p / (1 - p)).reshape(-1, 1)
    model = LogisticRegression(C=1e6, solver="lbfgs", max_iter=10000)
    model.fit(z, y)
    return float(model.intercept_[0]), float(model.coef_[0, 0])


def metric_set(y, p):
    pred = (np.asarray(p) >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    cint, cslope = calibration_intercept_slope(y, p)
    return {
        "AUROC": float(roc_auc_score(y, p)),
        "AUPRC": float(average_precision_score(y, p)),
        "Sensitivity": float(tp / (tp + fn)),
        "Specificity": float(tn / (tn + fp)),
        "Precision": float(precision_score(y, pred, zero_division=0)),
        "F1": float(f1_score(y, pred, zero_division=0)),
        "BalancedAccuracy": float(balanced_accuracy_score(y, pred)),
        "Brier": float(brier_score_loss(y, p)),
        "CalibrationIntercept": cint,
        "CalibrationSlope": cslope,
        "TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp),
    }


def bootstrap_ci(y, p, rng, B=BOOTSTRAP_B):
    vals = {"AUROC": [], "AUPRC": [], "Brier": []}
    n = len(y)
    for _ in range(B):
        idx = rng.integers(0, n, n)
        yy = y[idx]
        if len(np.unique(yy)) < 2:
            continue
        pp = p[idx]
        vals["AUROC"].append(roc_auc_score(yy, pp))
        vals["AUPRC"].append(average_precision_score(yy, pp))
        vals["Brier"].append(brier_score_loss(yy, pp))
    return {k: [float(np.quantile(v, 0.025)), float(np.quantile(v, 0.975))] for k, v in vals.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="Private raw dataset path. Never commit raw participant data publicly.")
    ap.add_argument("--outdir", default="results/recomputed")
    args = ap.parse_args()
    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)

    X, y, _, qc = load_and_prepare_dataset(args.data)
    predictors = {
        "M0": ["age", "sex", "education", "employment", "previous_anxiety_diagnosis"],
        "M1": ["age", "sex", "education", "employment", "previous_anxiety_diagnosis", "functional_impairment"],
    }

    outer = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    outer_splits = list(outer.split(X, y))
    models = ["LR", "SVM", "RF", "XGBoost"]
    results = {}
    oof_table = pd.DataFrame({"row_id": np.arange(len(y)), "outcome": y})
    start = time.time()

    for config in ["M0", "M1"]:
        Xc = X[predictors[config]]
        for model_name in models:
            key = f"{config}_{model_name}"
            oof = np.full(len(y), np.nan)
            fold_info = []
            print(f"START {key}", flush=True)
            for fold, (tr, va) in enumerate(outer_splits, start=1):
                inner = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
                pipe, Search, params, extra = model_spec(model_name, config)
                search = Search(
                    pipe, params, scoring="roc_auc", cv=inner, n_jobs=-1,
                    refit=True, error_score="raise", **extra
                )
                search.fit(Xc.iloc[tr], y[tr])
                p = search.predict_proba(Xc.iloc[va])[:, 1]
                oof[va] = p
                fold_info.append({
                    "fold": fold,
                    "outer_AUROC": float(roc_auc_score(y[va], p)),
                    "best_inner_AUROC": float(search.best_score_),
                    "best_params": search.best_params_,
                })
            if np.isnan(oof).any():
                raise RuntimeError(f"Missing OOF prediction in {key}")
            results[key] = {"oof": oof.tolist(), "folds": fold_info}
            oof_table[key] = oof
            print(f"DONE {key}: AUROC={roc_auc_score(y, oof):.4f}", flush=True)

    summary = {k: metric_set(y, np.asarray(v["oof"])) for k, v in results.items()}
    rng = np.random.default_rng(SEED)
    ci = {k: bootstrap_ci(y, np.asarray(v["oof"]), rng) for k, v in results.items()}

    deltas = {}
    for model_name in models:
        p0 = np.asarray(results[f"M0_{model_name}"]["oof"])
        p1 = np.asarray(results[f"M1_{model_name}"]["oof"])
        rr = np.random.default_rng(SEED)
        boots = {"DeltaAUROC": [], "DeltaAUPRC": [], "DeltaBrier": []}
        for _ in range(BOOTSTRAP_B):
            idx = rr.integers(0, len(y), len(y))
            yy = y[idx]
            if len(np.unique(yy)) < 2:
                continue
            boots["DeltaAUROC"].append(roc_auc_score(yy, p1[idx]) - roc_auc_score(yy, p0[idx]))
            boots["DeltaAUPRC"].append(average_precision_score(yy, p1[idx]) - average_precision_score(yy, p0[idx]))
            boots["DeltaBrier"].append(brier_score_loss(yy, p1[idx]) - brier_score_loss(yy, p0[idx]))
        deltas[model_name] = {
            "point": {
                "DeltaAUROC": summary[f"M1_{model_name}"]["AUROC"] - summary[f"M0_{model_name}"]["AUROC"],
                "DeltaAUPRC": summary[f"M1_{model_name}"]["AUPRC"] - summary[f"M0_{model_name}"]["AUPRC"],
                "DeltaBrier": summary[f"M1_{model_name}"]["Brier"] - summary[f"M0_{model_name}"]["Brier"],
            },
            "bootstrap95": {k: [float(np.quantile(v, .025)), float(np.quantile(v, .975))] for k, v in boots.items()},
        }

    rows = []
    for config in ["M0", "M1"]:
        for model_name in models:
            key = f"{config}_{model_name}"
            rows.append({"configuration": config, "model": model_name, **summary[key]})
    pd.DataFrame(rows).to_csv(outdir / "model_performance_8_configs.csv", index=False)
    oof_table.to_csv(outdir / "oof_predictions.csv", index=False)

    output = {
        "random_seed": SEED,
        "outer_cv": "StratifiedKFold(5, shuffle=True, random_state=42)",
        "inner_cv": "StratifiedKFold(5, shuffle=True, random_state=42)",
        "bootstrap_resamples": BOOTSTRAP_B,
        "qc": qc,
        "summary": summary,
        "bootstrap95": ci,
        "paired_deltas_M1_minus_M0": deltas,
        "details": results,
        "runtime_seconds": time.time() - start,
    }
    (outdir / "nested_cv_results.json").write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"Saved to {outdir}")


if __name__ == "__main__":
    main()
