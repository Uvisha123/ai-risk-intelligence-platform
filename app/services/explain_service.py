"""
Explainable AI helpers: feature importance and human-readable narratives.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import pandas as pd

from app.ml.preprocessing import preprocess_for_model
from app.models.loader import load_loan_artifact


def _humanize_feature(name: str) -> str:
    """Map column name to a readable label."""
    mapping = {
        "debt_to_income_ratio": "Debt-to-income ratio",
        "asset_ratio": "Assets relative to loan size",
        "total_assets": "Total declared assets",
        "cibil_score": "Credit bureau (CIBIL) score",
        "income_annum": "Annual income",
        "loan_amount": "Requested loan amount",
        "loan_term": "Loan term",
        "no_of_dependents": "Number of dependents",
        "income_stability_index": "Income stability (log scale)",
        "asset_consistency_score": "Consistency of asset values",
        "education_Graduate": "Education: Graduate",
        "education_Not Graduate": "Education: Not Graduate",
        "self_employed_Yes": "Self employed: Yes",
        "self_employed_No": "Self employed: No",
    }
    return mapping.get(name, name.replace("_", " ").title())


def explain_prediction(record: Dict[str, Any], top_k: int = 5) -> Dict[str, Any]:
    """
    Build global feature-importance explanation for the trained credit model.

    Args:
        record: Loan application dict.
        top_k: Number of contributing features to return.

    Returns:
        Dict with ranked features and human explanations.
    """
    art = load_loan_artifact()
    model = art["model"]
    columns = art["feature_columns"]
    medians = art.get("medians") or {}
    X = preprocess_for_model(record, columns, medians)

    importances = getattr(model, "feature_importances_", None)
    if importances is None:
        return {"top_features": [], "notes": "Model does not expose feature_importances_"}

    pairs: List[Tuple[str, float, float]] = []
    row = X.iloc[0].values.astype(float)
    names = list(X.columns)
    for fname, imp, val in zip(names, importances, row):
        contribution = float(imp * abs(val))
        pairs.append((fname, float(imp), contribution))

    pairs.sort(key=lambda x: x[2], reverse=True)
    ranked = []
    for fname, imp, contrib in pairs[:top_k]:
        ranked.append(
            {
                "feature": fname,
                "display_name": _humanize_feature(fname),
                "importance": round(imp, 6),
                "weighted_signal": round(contrib, 6),
                "value": float(X[fname].iloc[0]),
            }
        )

    narrative_parts = [
        f"{item['display_name']} has importance {item['importance']:.4f} with observed value {item['value']:.4f}"
        for item in ranked[:3]
    ]
    narrative = (
        "The model leans most heavily on: " + "; ".join(narrative_parts) + "."
        if narrative_parts
        else "Insufficient detail to narrate."
    )

    return {
        "top_features": ranked,
        "summary": narrative,
        "method": "tree_feature_importance_weighted_by_input_magnitude",
    }


def local_reason_codes(record: Dict[str, Any]) -> List[str]:
    """
    Provide simple rule-based reason codes aligned with analyst expectations.

    Args:
        record: Loan application dict.

    Returns:
        Bullet reasons independent of model internals.
    """
    reasons: List[str] = []
    income = float(record.get("income_annum") or 0)
    loan = float(record.get("loan_amount") or 0)
    cibil = int(record.get("cibil_score") or 0)
    term = int(record.get("loan_term") or 0)

    if cibil < 650:
        reasons.append("Credit bureau score is below typical approval comfort levels")
    if income > 0 and loan / income > 0.6:
        reasons.append("Loan size is high relative to annual income")
    if term > 240:
        reasons.append("Very long tenor increases uncertainty and exposure window")
    if int(record.get("no_of_dependents") or 0) > 4:
        reasons.append("Household obligations appear elevated")

    df = pd.DataFrame([record])
    from app.ml.feature_engineering import engineer_features

    df = engineer_features(df)
    dti = float(df["debt_to_income_ratio"].iloc[0])
    if dti > 8:
        reasons.append("Debt-to-income ratio signals severe repayment stress")

    return reasons


def full_explanation(record: Dict[str, Any]) -> Dict[str, Any]:
    """
    Combine model-based explanation and heuristic reason codes.

    Args:
        record: Application data.

    Returns:
        Explainability payload suitable for JSON responses.
    """
    expl = explain_prediction(record)
    expl["heuristic_reasons"] = local_reason_codes(record)
    return expl
