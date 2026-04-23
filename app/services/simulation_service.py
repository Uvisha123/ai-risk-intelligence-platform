"""
What-if simulations and scenario comparison utilities.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, List

from app.services.decision_service import classify_risk_level, decision_from_score
from app.services.risk_service import compare_scenarios_ranking, predict_credit_risk_probability


def run_default_simulations(record: Dict[str, Any]) -> Dict[str, Any]:
    """
    Simulate income boost, loan reduction, and credit score improvements.

    Args:
        record: Baseline validated application.

    Returns:
        Baseline metrics plus scenario list with new decisions.
    """
    baseline_score = predict_credit_risk_probability(record)
    baseline_decision = decision_from_score(baseline_score)

    scenarios: List[Dict[str, Any]] = []

    inc = deepcopy(record)
    inc["income_annum"] = float(record["income_annum"]) * 1.2
    s1 = predict_credit_risk_probability(inc)
    scenarios.append(
        {
            "scenario_name": "income_increase_20pct",
            "description": "Annual income increases by 20%",
            "adjusted_fields": {"income_annum": inc["income_annum"]},
            "risk_score": s1,
            "risk_level": classify_risk_level(s1),
            "decision": decision_from_score(s1),
        }
    )

    loan_down = deepcopy(record)
    loan_down["loan_amount"] = float(record["loan_amount"]) * 0.85
    s2 = predict_credit_risk_probability(loan_down)
    scenarios.append(
        {
            "scenario_name": "loan_reduction_15pct",
            "description": "Requested loan reduced by 15%",
            "adjusted_fields": {"loan_amount": loan_down["loan_amount"]},
            "risk_score": s2,
            "risk_level": classify_risk_level(s2),
            "decision": decision_from_score(s2),
        }
    )

    cibil_up = deepcopy(record)
    cibil_up["cibil_score"] = min(900, int(record["cibil_score"]) + 40)
    s3 = predict_credit_risk_probability(cibil_up)
    scenarios.append(
        {
            "scenario_name": "cibil_plus_40",
            "description": "CIBIL score improves by 40 points (capped at 900)",
            "adjusted_fields": {"cibil_score": cibil_up["cibil_score"]},
            "risk_score": s3,
            "risk_level": classify_risk_level(s3),
            "decision": decision_from_score(s3),
        }
    )

    return {
        "baseline_risk_score": baseline_score,
        "baseline_decision": baseline_decision,
        "scenarios": scenarios,
    }


def compare_loan_scenarios(base: Dict[str, Any], scenarios: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Evaluate multiple hypothetical application vectors and rank by risk.

    Args:
        base: Baseline application fields.
        scenarios: Iterable of scenario dicts with keys: label, income_annum, loan_amount, cibil_score.

    Returns:
        Ranking metadata from compare_scenarios_ranking.
    """
    rows: List[Dict[str, Any]] = []
    for sc in scenarios:
        payload = deepcopy(base)
        payload["income_annum"] = float(sc.get("income_annum", base["income_annum"]))
        payload["loan_amount"] = float(sc.get("loan_amount", base["loan_amount"]))
        payload["cibil_score"] = int(sc.get("cibil_score", base["cibil_score"]))
        score = predict_credit_risk_probability(payload)
        rows.append(
            {
                "label": str(sc.get("label", "scenario")),
                "risk_score": score,
                "risk_level": classify_risk_level(score),
                "decision": decision_from_score(score),
            }
        )
    return compare_scenarios_ranking(rows)
