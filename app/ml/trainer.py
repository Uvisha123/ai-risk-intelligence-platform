"""
Training utilities and artifact construction for sklearn models.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from app.ml.evaluator import evaluate_binary
from app.ml.preprocessing import dataframe_to_design_matrix


def build_xy(
    df: pd.DataFrame,
    target_column: str,
    positive_label: str,
) -> Tuple[pd.DataFrame, np.ndarray, List[str], Dict[str, float]]:
    """
    Prepare design matrix X and binary y (1 = positive class).

    Args:
        df: Dataset including target column.
        target_column: Name of the label column.
        positive_label: Label value that maps to y=1.

    Returns:
        X (DataFrame), y (ndarray), feature column list, median imputation map.
    """
    y = (df[target_column].astype(str) == positive_label).astype(int).values
    X_raw = df.drop(columns=[target_column])
    if "loan_id" in X_raw.columns:
        X_raw = X_raw.drop(columns=["loan_id"])
    X, medians = dataframe_to_design_matrix(X_raw)
    feature_columns = list(X.columns)
    return X, y, feature_columns, medians


def train_random_forest_binary(
    X: pd.DataFrame,
    y: np.ndarray,
    random_state: int = 42,
) -> Tuple[RandomForestClassifier, Dict[str, float]]:
    """
    Fit a RandomForestClassifier and return metrics on the hold-out set.

    Args:
        X: Feature matrix.
        y: Binary labels.
        random_state: Seed.

    Returns:
        Fitted model and validation metrics.
    """
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=random_state, stratify=y if len(np.unique(y)) > 1 else None
    )
    clf = RandomForestClassifier(
        n_estimators=150,
        max_depth=12,
        min_samples_leaf=4,
        class_weight="balanced_subsample",
        random_state=random_state,
        n_jobs=-1,
    )
    clf.fit(X_train, y_train)
    val_proba = clf.predict_proba(X_val)[:, 1]
    val_pred = clf.predict(X_val)
    metrics = evaluate_binary(y_val, val_pred, val_proba)
    return clf, metrics


def save_loan_artifact(
    path: Path,
    model: RandomForestClassifier,
    feature_columns: List[str],
    medians: Dict[str, float],
    metrics: Dict[str, float],
    classes: List[str],
    positive_class_index: int,
) -> None:
    """
    Persist loan risk model bundle to disk.

    Args:
        path: Destination .pkl path.
        model: Fitted estimator.
        feature_columns: Column order for inference.
        medians: Numeric imputation values.
        metrics: Validation metrics dict.
        classes: Class labels from estimator.
        positive_class_index: Index of the class used as risk (rejection) probability.
    """
    artifact: Dict[str, Any] = {
        "model": model,
        "feature_columns": feature_columns,
        "medians": medians,
        "metrics": metrics,
        "classes": classes,
        "positive_class_index": positive_class_index,
        "kind": "loan_credit_risk",
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, path)


def save_fraud_artifact(path: Path, artifact: Dict[str, Any]) -> None:
    """Persist fraud detection bundle."""
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, path)
