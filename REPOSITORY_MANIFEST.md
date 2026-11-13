# Repository artifact status

| Component | File | Status |
|---|---|---|
| Locked nested-CV source | `original_locked_files/Anxiety_ML_Final_Locked_Pipeline.py` | Original library artifact |
| Portable nested-CV source | `src/nested_cv_pipeline.py` | Added for public reproducibility |
| Final model-selection code | `src/select_final_model.py` | Added |
| Final RF refit code | `src/final_refit_m1_rf.py` | Added |
| SHAP computation code | `src/explain_shap_m1_rf.py` | Added |
| Revised selection report | `results/Anxiety_ML_Final_Model_Selection_Report_REVISED.md` | Revised; web/implementation-feasibility rationale removed |
| Revised model lock | `results/Anxiety_ML_Final_Model_Selection_Lock_REVISED.json` | Revised narrative only; metrics/parameters unchanged |
| Model performance table | `results/model_performance_8_configs.csv` | Locked results |
| M1 vs M0 incremental table | `results/incremental_value_m1_vs_m0.csv` | Locked results |
| Pairwise M1 comparisons | `results/pairwise_m1_model_comparisons.csv` | Locked results |
| **Original locked final `.joblib`** | `artifacts/original/final_rf_m1_pipeline.joblib` | Present; SHA-256 exactly matches original locked file |
| Reproduced final `.joblib` | `artifacts/reproduced/final_rf_m1_pipeline.joblib` | Recomputed copy, not a replacement for original binary |
| Original `final_rf_m1_shap.json` | Not available in current workspace | Expected SHA in lock |
| Reproduced SHAP JSON | `artifacts/reproduced/final_rf_m1_shap.json` | Recomputed; feature values match locked results |
| Original `shap_global_rf_m1.png` | Not available in current workspace | Expected SHA in lock |
| Reproduced SHAP figure | `artifacts/reproduced/shap_global_rf_m1.png` | Recomputed |
| README | `README.md` | Added |
| Dependencies | `requirements.txt` | Added |
| Privacy/data access | `data/README_DATA_ACCESS.md` | Added |

## Integrity and execution scope

- The original locked model is included unchanged (SHA-256 `2a4cd502efcc6ff7d9778ba86c915e4f80f4a37ba6de381785aa8ff866668a7b`).
- `artifacts/reproduced/` is an explicitly separate set of independent refit and SHAP results.
- Original locked SHAP JSON and figure were not supplied and must not be mislabeled as reproduced-file checksums.
- `results/final_refit_run.json` and `results/shap_run.json` record reproduced outputs only.
- Raw participant-level Excel files are excluded from the public ZIP.
- Source, syntax, package file integrity, and decision logic are checked in packaging; full numerical reproduction requires a separate environment/data execution.
