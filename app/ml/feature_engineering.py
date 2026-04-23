"""
Derived features for credit and behavioral modeling.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add total assets, debt-to-income, and asset coverage ratio.

    Args:
        df: Raw or partially processed feature rows (single or batch).

    Returns:
        DataFrame with additional engineered columns.
    """
    out = df.copy()
    asset_cols = [
        "residential_assets_value",
        "commercial_assets_value",
        "luxury_assets_value",
        "bank_asset_value",
    ]
    out["total_assets"] = out[asset_cols].sum(axis=1)
    denom_income = out["income_annum"].replace({0: np.nan})
    out["debt_to_income_ratio"] = out["loan_amount"] / denom_income
    median_dti = float(out["debt_to_income_ratio"].median()) if len(out) else 0.0
    out["debt_to_income_ratio"] = out["debt_to_income_ratio"].fillna(median_dti)
    out["asset_ratio"] = out["total_assets"] / (out["loan_amount"].replace({0: np.nan}) + 1.0)
    out["asset_ratio"] = out["asset_ratio"].replace([np.inf, -np.inf], np.nan).fillna(0.0)

    income_safe = out["income_annum"].replace({0: np.nan})
    out["income_stability_index"] = np.log1p(income_safe).fillna(0.0)
    asset_std = out[asset_cols].std(axis=1).fillna(0.0)
    asset_mean = out[asset_cols].mean(axis=1).replace({0: np.nan})
    out["asset_consistency_score"] = 1.0 - (asset_std / (asset_mean + 1.0))
    out["asset_consistency_score"] = out["asset_consistency_score"].clip(0.0, 1.0).fillna(0.5)

    return out


def behavioral_scores(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute normalized behavioral scores from engineered features.

    Args:
        df: DataFrame after engineer_features.

    Returns:
        DataFrame with behavioral_score in [0, 1].
    """
    out = df.copy()
    inc = out["income_stability_index"]
    if len(inc) > 1:
        inc_norm = (inc - inc.min()) / (inc.max() - inc.min() + 1e-9)
    else:
        inc_norm = np.clip(out["income_annum"].apply(lambda x: np.log1p(max(x, 0)) / 25.0), 0.0, 1.0)
    ast = out["asset_consistency_score"]
    out["behavioral_score"] = (0.5 * ast + 0.5 * inc_norm.clip(0, 1)).clip(0, 1)
    return out


def single_row_behavioral_score(record: dict) -> float:
    """
    Compute behavioral score [0, 1] for one application using fixed scaling.

    Args:
        record: Loan application fields.

    Returns:
        Behavioral score where higher values indicate stronger profile stability.
    """
    df = engineer_features(pd.DataFrame([record]))
    df = behavioral_scores(df)
    return float(df["behavioral_score"].iloc[0])
