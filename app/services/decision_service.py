"""
Map calibrated risk scores to discrete levels and lending decisions.
"""

from typing import Literal

from app.core.config import get_settings
from app.utils.scoring import decision_from_risk_level, risk_level_from_score

RiskLevel = Literal["LOW", "MEDIUM", "HIGH"]
Decision = Literal["APPROVE", "REVIEW", "REJECT"]


def classify_risk_level(risk_score: float) -> RiskLevel:
    """
    Classify a continuous score into LOW, MEDIUM, or HIGH using configured cutoffs.

    Args:
        risk_score: Probability-style risk in [0, 1].

    Returns:
        Risk band label.
    """
    settings = get_settings()
    return risk_level_from_score(risk_score, settings.threshold_low_max, settings.threshold_medium_max)


def decision_from_score(risk_score: float) -> Decision:
    """
    Convert score to APPROVE / REVIEW / REJECT via risk level thresholds.

    Args:
        risk_score: Credit risk probability.

    Returns:
        Lending decision suggestion.
    """
    return decision_from_risk_level(classify_risk_level(risk_score))


def decide_from_payload(risk_score: float) -> dict:
    """
    Serialize decision bundle for APIs.

    Args:
        risk_score: Model risk score.

    Returns:
        Dictionary with risk_level and decision keys.
    """
    level = classify_risk_level(risk_score)
    decision = decision_from_risk_level(level)
    return {"risk_score": risk_score, "risk_level": level, "decision": decision}
