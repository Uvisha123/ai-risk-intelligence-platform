"""
Tabular preprocessing: missing values and one-hot encoding alignment.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from app.ml.feature_engineering import engineer_features


CATEGORICAL_COLS = ["education", "self_employed"]
NUMERIC_COLS = [
    "no_of_dependents",
    "income_annum",
    "loan_amount",
    "loan_term",
    "cibil_score",
    "residential_assets_value",
    "commercial_assets_value",
    "luxury_assets_value",
    "bank_asset_value",
]


def fill_missing_numeric(df: pd.DataFrame, medians: Optional[Dict[str, float]] = None) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """
    Impute numeric columns with training medians or column medians.

    Args:
        df: Input features.
        medians: Optional precomputed medians from training.

    Returns:
        Tuple of filled DataFrame and medians used.
    """
    out = df.copy()
    used: Dict[str, float] = dict(medians or {})
    for col in NUMERIC_COLS:
        if col not in out.columns:
            continue
        m = used.get(col, float(out[col].median()))
        if np.isnan(m):
            m = 0.0
        used[col] = m
        out[col] = out[col].fillna(m)
    return out, used


def preprocess_for_model(
    record: Dict[str, Any],
    feature_columns: List[str],
    medians: Optional[Dict[str, float]] = None,
) -> pd.DataFrame:
    """
    Build a single-row design matrix aligned to training columns.

    Args:
        record: Raw loan application fields.
        feature_columns: Column order produced during training after get_dummies.
        medians: Numeric imputation lookup from training.

    Returns:
        One-row DataFrame reindexed to feature_columns.
    """
    row = {k: record.get(k) for k in NUMERIC_COLS + CATEGORICAL_COLS if k in record or k in NUMERIC_COLS}
    df = pd.DataFrame([row])
    df, _ = fill_missing_numeric(df, medians)
    df = engineer_features(df)
    df = pd.get_dummies(df, columns=CATEGORICAL_COLS, drop_first=False)
    df = df.reindex(columns=feature_columns, fill_value=0)
    return df


def dataframe_to_design_matrix(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """
    Training-time pipeline: impute, engineer, one-hot encode without reindex yet.

    Args:
        df: Raw feature columns including categoricals.

    Returns:
        Tuple of dummy-expanded frame and median dictionary.
    """
    out, medians = fill_missing_numeric(df)
    out = engineer_features(out)
    out = pd.get_dummies(out, columns=CATEGORICAL_COLS, drop_first=False)
    return out, medians
