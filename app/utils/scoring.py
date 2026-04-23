"""
Risk banding and score normalization helpers.
"""

from typing import Literal, Tuple

RiskLevel = Literal["LOW", "MEDIUM", "HIGH"]
Decision = Literal["APPROVE", "REVIEW", "REJECT"]
TrustCategory = Literal["STRONG", "MODERATE", "WEAK"]


def risk_level_from_score(score: float, low_max: float, medium_max: float) -> RiskLevel:
    """
    Map a probability-style risk score into LOW / MEDIUM / HIGH bands.

    Args:
        score: Risk probability in [0, 1].
        low_max: Upper bound (exclusive) for LOW band.
        medium_max: Upper bound (inclusive) for MEDIUM band.

    Returns:
        Risk level label.
    """
    if score < low_max:
        return "LOW"
    if score <= medium_max:
        return "MEDIUM"
    return "HIGH"


def decision_from_risk_level(level: RiskLevel) -> Decision:
    """
    Default decision mapping used when thresholds align with bands.

    Args:
        level: LOW, MEDIUM, or HIGH.

    Returns:
        APPROVE, REVIEW, or REJECT.
    """
    if level == "LOW":
        return "APPROVE"
    if level == "MEDIUM":
        return "REVIEW"
    return "REJECT"


def scale_probability_to_trust_component(p: float) -> float:
    """Map probability [0,1] to contribution scale; invert so lower risk increases trust."""
    return float(max(0.0, min(1.0, 1.0 - p)))


def trust_category(score_0_100: float) -> TrustCategory:
    """
    Bucket a 0–100 trust score into labels.

    Args:
        score_0_100: Combined trust score.

    Returns:
        STRONG, MODERATE, or WEAK.
    """
    if score_0_100 >= 75:
        return "STRONG"
    if score_0_100 >= 50:
        return "MODERATE"
    return "WEAK"


def normalize_weights(w_credit: float, w_fraud: float, w_behavioral: float) -> Tuple[float, float, float]:
    """Normalize three weights to sum to 1."""
    s = w_credit + w_fraud + w_behavioral
    if s <= 0:
        return (1 / 3, 1 / 3, 1 / 3)
    return (w_credit / s, w_fraud / s, w_behavioral / s)
