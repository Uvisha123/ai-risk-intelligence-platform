"""
Train a lightweight fraud suspicion model and persist ``fraud_model.pkl``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.ml.preprocessing import dataframe_to_design_matrix  # noqa: E402
from app.ml.trainer import save_fraud_artifact, train_random_forest_binary  # noqa: E402
from training.dataset_processing import attach_fraud_proxy_labels, load_and_clean_dataset  # noqa: E402


def main() -> None:
    """Train fraud classifier on heuristic labels and save artifact."""
    df = attach_fraud_proxy_labels(load_and_clean_dataset())
    y = df["fraud_proxy"].astype(int).values
    X_raw = df.drop(columns=["fraud_proxy", "loan_status"])
    if "loan_id" in X_raw.columns:
        X_raw = X_raw.drop(columns=["loan_id"])
    X, medians = dataframe_to_design_matrix(X_raw)
    feature_columns = list(X.columns)

    model, metrics = train_random_forest_binary(X, y)

    if len(model.classes_) > 1:
        pos_idx = int(np.where(model.classes_ == 1)[0][0])
    else:
        pos_idx = 0

    artifact = {
        "model": model,
        "feature_columns": feature_columns,
        "medians": medians,
        "metrics": metrics,
        "classes": [str(c) for c in model.classes_],
        "positive_class_index": pos_idx,
        "kind": "fraud_suspicion",
    }
    out_path = ROOT / "app" / "models" / "fraud_model.pkl"
    save_fraud_artifact(out_path, artifact)
    print("Fraud model saved to", out_path)
    print("Hold-out metrics:", metrics)


if __name__ == "__main__":
    main()
