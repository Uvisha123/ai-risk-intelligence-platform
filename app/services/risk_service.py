"""
Credit risk estimation, fraud checks, behavioral scoring, and early warnings.
"""

from __future__ import annotations

from typing import Any, Dict, List

import pandas as pd

from app.core.config import get_settings
from app.core.exceptions import PredictionError
from app.ml.feature_engineering import engineer_features, single_row_behavioral_score
from app.ml.preprocessing import preprocess_for_model
from app.models.loader import load_fraud_artifact, load_loan_artifact
from app.services.decision_service import classify_risk_level, decision_from_score
from app.services.explain_service import local_reason_codes
from app.utils.helpers import clip01


def _predict_proba_positive(record: Dict[str, Any], bundle_key: str) -> float:
    """
    Run binary classifier and return probability of the positive (adverse) class.

    Args:
        record: Application feature dict.
        bundle_key: Which artifact loader to use — 'loan' or 'fraud'.

    Returns:
        Probability in [0, 1].
    """
    try:
        if bundle_key == "loan":
            art = load_loan_artifact()
        else:
            art = load_fraud_artifact()
        model = art["model"]
        columns = art["feature_columns"]
        medians = art.get("medians") or {}
        X = preprocess_for_model(record, columns, medians)
        proba = model.predict_proba(X)[0]
        idx = int(art.get("positive_class_index", 1))
        idx = min(max(idx, 0), len(proba) - 1)
        return float(proba[idx])
    except KeyError as exc:
        raise PredictionError("Malformed model artifact", details={"error": str(exc)}) from exc
    except Exception as exc:  # pragma: no cover — numpy/sklearn defensive
        raise PredictionError("Inference failed", details={"error": str(exc)}) from exc


def predict_credit_risk_probability(record: Dict[str, Any]) -> float:
    """
    Predict probability of loan rejection / default risk using the loan model.

    Args:
        record: Loan application fields.

    Returns:
        risk_score between 0 and 1 (higher = riskier).
    """
    return clip01(_predict_proba_positive(record, "loan"))


def predict_fraud_probability(record: Dict[str, Any]) -> float:
    """
    Estimate fraud suspicion probability using the trained fraud classifier.

    Args:
        record: Loan application fields.

    Returns:
        Fraud score in [0, 1].
    """
    return clip01(_predict_proba_positive(record, "fraud"))


def apply_fraud_rules(record: Dict[str, Any]) -> List[str]:
    """
    Apply lightweight rule-based fraud indicators on top of the model.

    Args:
        record: Application payload.

    Returns:
        Human-readable fraud concern strings.
    """
    flags: List[str] = []
    income = float(record.get("income_annum") or 0)
    loan = float(record.get("loan_amount") or 0)
    cibil = int(record.get("cibil_score") or 0)
    if income > 0 and loan / income > 18:
        flags.append("Loan amount extremely high versus reported annual income")
    if income < 150_000 and loan > 5_000_000:
        flags.append("Large loan relative to very low stated income")
    if cibil < 400 and loan > 10_000_000:
        flags.append("Very large loan with critically low CIBIL score")
    assets = sum(
        float(record.get(k, 0) or 0)
        for k in (
            "residential_assets_value",
            "commercial_assets_value",
            "luxury_assets_value",
            "bank_asset_value",
        )
    )
    if assets < loan * 0.05:
        flags.append("Total disclosed assets are very small relative to loan size")
    return flags


def build_early_warning(record: Dict[str, Any], risk_score: float) -> Dict[str, Any]:
    """
    Produce mock early-warning signals and simple trend heuristics.

    Args:
        record: Loan application dict.
        risk_score: Latest credit risk estimate.

    Returns:
        Structured early warning payload including alerts.
    """
    settings = get_settings()
    df = engineer_features(pd.DataFrame([record]))
    dti = float(df["debt_to_income_ratio"].iloc[0])
    alerts: List[Dict[str, str]] = []

    if risk_score >= settings.threshold_medium_max:
        alerts.append({"severity": "high", "code": "RISK_ELEVATED", "message": "Overall credit risk is in the upper band"})
    elif risk_score >= settings.threshold_low_max:
        alerts.append({"severity": "medium", "code": "RISK_MONITOR", "message": "Credit risk elevated; monitor closely"})

    if dti > 12:
        alerts.append({"severity": "high", "code": "DTI_STRESS", "message": "Debt-to-income suggests severe leverage"})

    if int(record.get("no_of_dependents") or 0) >= 4 and dti > 6:
        alerts.append({"severity": "medium", "code": "HOUSEHOLD_STRESS", "message": "High dependents with aggressive leverage"})

    # Mock trend: hash-based stable pseudo-trend for demo reproducibility
    trend_score = (hash(str(record.get("loan_id", ""))) % 100) / 100.0
    trend = "WORSENING" if risk_score > 0.55 and trend_score > 0.7 else "STABLE"

    return {"alerts": alerts, "trend_indicator": trend, "mock_trend_score": round(trend_score, 3)}


def behavioral_analysis(record: Dict[str, Any]) -> Dict[str, Any]:
    """
    Derive behavioral risk features and a composite behavioral score.

    Args:
        record: Application information.

    Returns:
        Dict with behavioral_score and explanatory sub-fields.
    """
    df = engineer_features(pd.DataFrame([record]))
    dti = float(df["debt_to_income_ratio"].iloc[0])
    ast_cons = float(df["asset_consistency_score"].iloc[0])
    score = single_row_behavioral_score(record)
    return {
        "behavioral_score": score,
        "income_stability_index": float(df["income_stability_index"].iloc[0]),
        "asset_consistency": ast_cons,
        "debt_to_income_ratio": dti,
    }


def assess_credit_risk_package(record: Dict[str, Any]) -> Dict[str, Any]:
    """
    Full credit risk package used by API layer: scores, fraud, warnings, level, decision.

    Args:
        record: Validated loan application dictionary.

    Returns:
        Aggregated analysis dict.
    """
    risk_score = predict_credit_risk_probability(record)
    level = classify_risk_level(risk_score)
    decision = decision_from_score(risk_score)
    fraud_score = predict_fraud_probability(record)
    rules = apply_fraud_rules(record)
    behavior = behavioral_analysis(record)

    fraud_block = {
        "fraud_score": fraud_score,
        "model_flagged": fraud_score >= 0.45,
        "rule_flags": rules,
    }

    ew = build_early_warning(record, risk_score)
    drivers = local_reason_codes(record)

    return {
        "risk_score": risk_score,
        "risk_level": level,
        "decision": decision,
        "risk_drivers": drivers,
        "behavioral_score": behavior["behavioral_score"],
        "behavioral_detail": behavior,
        "fraud_assessment": fraud_block,
        "early_warning": ew,
    }


def compare_scenarios_ranking(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Rank multiple labeled scenarios by lowest risk_score (best first).

    Args:
        rows: Each item must include label, risk_score, risk_level, decision.

    Returns:
        Comparison summary with ranks and best label.
    """
    sorted_rows = sorted(rows, key=lambda r: r["risk_score"])
    best = sorted_rows[0]["label"] if sorted_rows else ""
    out = []
    for i, row in enumerate(sorted_rows, start=1):
        out.append({**row, "rank": i})
    return {"results": out, "best_option_label": best}
