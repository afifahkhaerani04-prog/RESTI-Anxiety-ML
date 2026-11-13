# Parsimonious Explainable Machine Learning for Anxiety Risk Stratification

This repository contains the reproducibility materials for the final RESTI manuscript analysis. The locked analytical design compares eight configurations: M0/M1 × Logistic Regression, SVM, Random Forest, and XGBoost using nested stratified 5-fold outer × 5-fold inner cross-validation with random seed 42.

## Final analytical design

- Initial responses: 514
- Repeated submissions excluded: 4
- Final analytical cohort: 510
- Outcome: GAD-7 <10 vs GAD-7 >=10
- Negative class: 393 (77.06%)
- Positive class: 117 (22.94%)
- M0: age, sex, education, employment, previous anxiety diagnosis
- M1: M0 + functional impairment
- Class imbalance: `class_weight="balanced"` for LR/SVM/RF; XGBoost recomputes `scale_pos_weight` within the current training fit
- SMOTE: not used
- Primary tuning metric: AUROC
- Classification threshold for reported classification metrics: 0.50

## Final model selection

The final configuration is **M1–Random Forest**. Selection is based on discrimination, calibration, stability, interpretability, and parsimony, not AUROC alone. Implementation feasibility and the web prototype are intentionally excluded from the revised model-selection rationale to match the final manuscript scope.

M1-RF outer-CV performance:

- AUROC: 0.9105
- AUPRC: 0.7737
- Sensitivity: 0.7521
- Specificity: 0.8982
- Precision: 0.6875
- F1: 0.7184
- Balanced accuracy: 0.8252
- Brier score: 0.0992
- Calibration intercept: -0.6335
- Calibration slope: 0.8728

The calibration metrics indicate that raw probabilities should not be described as externally validated absolute clinical risk. External validation and possible recalibration are required before clinical use.

## Repository structure

```text
src/
  data_utils.py
  nested_cv_pipeline.py
  select_final_model.py
  final_refit_m1_rf.py
  explain_shap_m1_rf.py
results/
  model_performance_8_configs.csv
  incremental_value_m1_vs_m0.csv
  pairwise_m1_model_comparisons.csv
  Anxiety_ML_Final_Model_Selection_Report_REVISED.md
  Anxiety_ML_Final_Model_Selection_Lock_REVISED.json
artifacts/original/
  final_rf_m1_pipeline.joblib   # original locked binary
artifacts/reproduced/
  final_rf_m1_pipeline.joblib   # independent reproduction
  final_rf_m1_metadata.json
  final_rf_m1_shap.json
  shap_global_rf_m1.png
original_locked_files/
  Anxiety_ML_Final_Locked_Pipeline.py
  Anxiety_ML_Final_Model_Selection_Report_original.md
  Anxiety_ML_Final_Model_Selection_Lock_original.json
data/
  data_dictionary.csv
  README_DATA_ACCESS.md
requirements.txt
DATA_AVAILABILITY_STATEMENT.txt
SHA256SUMS.txt
```

## Artifact provenance and integrity

**The original locked M1-Random Forest binary is included** at `artifacts/original/final_rf_m1_pipeline.joblib`. Its SHA-256 matches the original model-selection lock exactly:

`2a4cd502efcc6ff7d9778ba86c915e4f80f4a37ba6de381785aa8ff866668a7b`

The independent refit binary at `artifacts/reproduced/final_rf_m1_pipeline.joblib` is retained separately; its SHA-256 is `b8cb99798f7682536fb41bafef08a825ad54dec0320d136fe32857f3019bfd1e`. Different serialized hashes do not by themselves demonstrate different predictive behavior. **Do not substitute the reproduced model for the original locked model when citing the locked artifact.**

The original SHAP JSON and PNG are **not available** in the supplied files. The SHAP outputs in `artifacts/reproduced/` are reproduced materials, not byte-identical original SHAP artifacts. Original SHAP expected hashes are documented in `original_locked_files/Anxiety_ML_Final_Model_Selection_Lock_original.json`. The reproduced values match the recorded global feature-importance summary, but this does not establish full computational equivalence.

The original locked source and lock documentation under `original_locked_files/` are preserved unchanged. The revised lock under `results/` is an editorially revised copy; its metrics and model hyperparameters have not been changed. All tracked repository files are listed in `SHA256SUMS.txt`.

The original model is a Python joblib/pickle artifact. Only load it in a trusted, isolated environment with compatible dependencies; never load untrusted serialized models.

## Environment

Use Python 3.13.5 and install:

```bash
pip install -r requirements.txt
```

The locked environment uses pandas 2.2.3, NumPy 2.3.5, scikit-learn 1.8.0, XGBoost 3.1.3, SHAP 0.50.0, Matplotlib 3.10.8, and random seed 42.

## Reproduce the full nested analysis

Keep the participant-level workbook outside Git. Then run:

```bash
python src/nested_cv_pipeline.py --data /secure/path/dataset_anxiety.xlsx --outdir results/recomputed
```

The full grid is computationally expensive. Do not modify the fold structure, seed, predictor definitions, search spaces, or imbalance handling if the objective is exact manuscript reproduction.

## Reproduce the documented model-selection decision

```bash
python src/select_final_model.py
```

This verifies that M1 has consistent incremental value, that the XGBoost–RF paired discrimination confidence intervals cross zero, and that RF has the lowest M1 Brier score.

## Refit M1-RF

```bash
python src/final_refit_m1_rf.py --data /secure/path/dataset_anxiety.xlsx --outdir artifacts/reproduced
```

This full-data refit is for the final artifact only. Do not report its training performance as generalization performance.

## Recompute SHAP

```bash
python src/explain_shap_m1_rf.py \
  --data /secure/path/dataset_anxiety.xlsx \
  --model artifacts/reproduced/final_rf_m1_pipeline.joblib \
  --outdir artifacts/reproduced \
  --figdir artifacts/reproduced
```

The locked global ranking starts with functional impairment (mean |SHAP| 0.1645) and age (0.1096). SHAP values are predictive attributions, not causal effects.

## Verification and limitations

The repository includes locked evaluation tables, the original final model, independent reproduced artifacts, and scripts for rerunning analyses. Python syntax and the documented model-selection checks can be verified without private data. The entire nested CV analysis and SHAP computations require the original private dataset and an appropriately compatible environment; packaging and hash verification alone are not proof of end-to-end numerical reproducibility.

Note: `results/shap_run.json` provides repo-relative paths to the *reproduced* SHAP output files. It does not attest to original SHAP artifact hashes. `results/final_refit_run.json` likewise documents the reproduced binary.

## Data privacy

Do **not** upload the raw workbook to GitHub. See `data/README_DATA_ACCESS.md` for the recommended controlled-access strategy.
