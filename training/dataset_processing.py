"""
Load and normalize the raw loan dataset for offline training jobs.
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import pandas as pd


def project_root() -> Path:
    """Return repository root (parent of ``training``)."""
    return Path(__file__).resolve().parents[1]


def load_and_clean_dataset(csv_name: str = "loan_data.csv") -> pd.DataFrame:
    """
    Read CSV, normalize headers, drop rows with missing targets or key numerics.

    Args:
        csv_name: Filename under ``data/``.

    Returns:
        Cleaned dataframe including ``loan_status`` for credit training.
    """
    root = project_root()
    path = root / "data" / csv_name
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    if "education" in df.columns:
        df["education"] = df["education"].astype(str).str.strip()
    if "self_employed" in df.columns:
        df["self_employed"] = df["self_employed"].astype(str).str.strip()
    if "loan_status" in df.columns:
        df["loan_status"] = df["loan_status"].astype(str).str.strip()
    df = df.dropna(subset=["loan_status"])
    required = [
        "no_of_dependents",
        "education",
        "self_employed",
        "income_annum",
        "loan_amount",
        "loan_term",
        "cibil_score",
        "residential_assets_value",
        "commercial_assets_value",
        "luxury_assets_value",
        "bank_asset_value",
    ]
    df = df.dropna(subset=required)
    return df


def attach_fraud_proxy_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build a heuristic fraud proxy label for supervised fraud-model training.

    Args:
        df: Clean loan frame.

    Returns:
        Same frame with ``fraud_proxy`` binary column.
    """
    out = df.copy()
    ratio = out["loan_amount"] / out["income_annum"].replace({0: pd.NA})
    ratio = ratio.fillna(ratio.median())
    suspicious = (
        (ratio > 18)
        | (out["cibil_score"] < 380)
        | ((out["luxury_assets_value"] + out["bank_asset_value"]) < out["loan_amount"] * 0.02)
    )
    out["fraud_proxy"] = suspicious.astype(int)
    return out
