"""
Train the credit-risk RandomForest model and persist ``loan_model.pkl``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.ml.trainer import (  # noqa: E402
    build_xy,
    save_loan_artifact,
    train_random_forest_binary,
)
from training.dataset_processing import load_and_clean_dataset  # noqa: E402


def main() -> None:
    """Entry point executed from project root."""
    df = load_and_clean_dataset()
    X, y, feature_columns, medians = build_xy(df, "loan_status", "Rejected")
    model, metrics = train_random_forest_binary(X, y)

    classes_list = list(model.classes_)
    pos_idx = int(np.where(model.classes_ == 1)[0][0]) if 1 in classes_list else len(classes_list) - 1
    out_path = ROOT / "app" / "models" / "loan_model.pkl"
    save_loan_artifact(
        path=out_path,
        model=model,
        feature_columns=feature_columns,
        medians=medians,
        metrics=metrics,
        classes=[str(c) for c in model.classes_],
        positive_class_index=pos_idx,
    )
    print("Loan model saved to", out_path)
    print("Validation metrics:", metrics)


if __name__ == "__main__":
    main()
