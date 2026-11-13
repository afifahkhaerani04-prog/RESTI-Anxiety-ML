# Final Model Selection Report — Revised for RESTI Repository

**Selected final configuration:** M1–Random Forest (RF)

The M1 predictor configuration was retained because adding functional impairment consistently improved discrimination and overall probabilistic accuracy across all four algorithms. For RF, M1 increased AUROC by 0.0660 (95% CI 0.0386–0.0958) and AUPRC by 0.1592 (95% CI 0.0984–0.2110), while reducing the Brier score by 0.0470 (95% CI −0.0599 to −0.0343).

Among M1 models, XGBoost achieved the highest AUROC (0.9166) and AUPRC (0.7959), but its paired advantage over RF was small: ΔAUROC=0.0061 (95% CI −0.0064 to 0.0184) and ΔAUPRC=0.0222 (95% CI −0.0110 to 0.0564). RF achieved the lowest Brier score (0.0992), with AUROC=0.9105, AUPRC=0.7737, sensitivity=0.7521, specificity=0.8982, precision=0.6875, F1=0.7184, and balanced accuracy=0.8252. Relative to LR, RF showed no clear discrimination difference but a lower Brier score (ΔBrier=−0.0235, 95% CI −0.0368 to −0.0102). RF was therefore selected as the best overall trade-off under the prespecified criteria of discrimination, calibration, stability, interpretability, and parsimony. This selection does not imply that RF was statistically superior to every alternative on every metric.

The final M1-RF model was refitted on all 510 participants using `n_estimators=200`, `max_depth=10`, `min_samples_split=2`, `min_samples_leaf=2`, `max_features="sqrt"`, `class_weight="balanced"`, and `random_state=42`. The full-data inner-CV mean AUROC of 0.9188 is a tuning diagnostic only and must not replace outer-CV performance estimates.

Global SHAP analysis identified functional impairment as the largest contributor (mean |SHAP|=0.1645), followed by age (0.1096), education=SMA/Sederajat (0.0485), previous anxiety diagnosis (0.0297), sex=Perempuan (0.0255), and employment=Pelajar/Mahasiswa (0.0249). These values represent predictive attribution rather than causal effects.

**Calibration caution:** RF calibration remained imperfect (intercept −0.6335; slope 0.8728). The model output should therefore be described as internally validated risk stratification, not externally validated absolute clinical risk. External validation and potential recalibration are required before clinical use.

**Revision note:** references to implementation feasibility and a web prototype were removed from the selection rationale to match the final manuscript scope. No model parameter, performance estimate, or final selection was changed.
