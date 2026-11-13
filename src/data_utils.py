from __future__ import annotations

import re
from pathlib import Path
import numpy as np
import pandas as pd

GAD7_ITEM_INDICES = list(range(9, 16))
FUNCTIONAL_MAP = {
    "Tidak sulit sama sekali": 0,
    "Agak sulit": 1,
    "Sangat sulit": 2,
    "Sangat mengganggu aktivitas sehari-hari": 3,
}


def normalize_age(value) -> int:
    """Deterministically convert mixed age formats (e.g., 29, '29 Tahun') to integer."""
    if pd.isna(value):
        raise ValueError("Missing age encountered; locked dataset is expected to have no missing age.")
    if isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(value, bool):
        return int(float(value))
    match = re.search(r"\d+", str(value))
    if not match:
        raise ValueError(f"Age cannot be parsed: {value!r}")
    return int(match.group())


def _norm_text(value) -> str:
    return re.sub(r"\s+", " ", str(value).strip())


def _norm_email(value) -> str:
    return _norm_text(value).lower()


def load_and_prepare_dataset(path: str | Path):
    """
    Apply the locked QC/outcome rules.

    Duplicate rule: repeated submission only when normalized email AND core demographics
    (numeric age, sex, education, employment) are identical. The earliest row is retained.
    Email is used only for QC and is never returned as a predictor.
    """
    path = Path(path)
    raw = pd.read_excel(path)
    if raw.shape[1] != 17:
        raise ValueError(f"Expected 17 columns, found {raw.shape[1]}.")

    cols = list(raw.columns)
    work = raw.copy()
    work["_email_norm"] = work.iloc[:, 1].map(_norm_email)
    work["_age_numeric"] = work.iloc[:, 3].map(normalize_age)

    # Preserve source order; for identical repeated profiles retain first response.
    duplicate_mask = work.duplicated(
        subset=["_email_norm", "_age_numeric", cols[5], cols[6], cols[7]],
        keep="first",
    )
    excluded_indices = work.index[duplicate_mask].tolist()
    clean = work.loc[~duplicate_mask].copy().reset_index(drop=True)

    if len(raw) != 514 or len(excluded_indices) != 4 or len(clean) != 510:
        raise ValueError(
            f"Locked QC expectation failed: initial={len(raw)}, excluded={len(excluded_indices)}, final={len(clean)}"
        )

    gad7_items = clean.iloc[:, 9:16].apply(pd.to_numeric, errors="raise")
    clean["GAD7_total"] = gad7_items.sum(axis=1).astype(int)
    clean["outcome"] = (clean["GAD7_total"] >= 10).astype(int)

    previous_dx = clean.iloc[:, 8].map(lambda x: 1 if _norm_text(x).lower() == "ya" else 0).astype(int)
    functional = clean.iloc[:, 16].map(lambda x: FUNCTIONAL_MAP[_norm_text(x)]).astype(int)

    X = pd.DataFrame({
        "age": clean["_age_numeric"].astype(float),
        "sex": clean.iloc[:, 5].map(_norm_text),
        "education": clean.iloc[:, 6].map(_norm_text),
        "employment": clean.iloc[:, 7].map(_norm_text),
        "previous_anxiety_diagnosis": previous_dx,
        "functional_impairment": functional,
    })
    y = clean["outcome"].astype(int).to_numpy()

    audit = {
        "initial_responses": int(len(raw)),
        "excluded_repeated_submissions": int(len(excluded_indices)),
        "final_n": int(len(clean)),
        "negative_n": int((y == 0).sum()),
        "positive_n": int((y == 1).sum()),
        "age_min": int(X["age"].min()),
        "age_max": int(X["age"].max()),
        "missing_values": int(X.isna().sum().sum()),
        "excluded_source_row_indices_zero_based": [int(i) for i in excluded_indices],
    }
    return X, y, clean, audit
