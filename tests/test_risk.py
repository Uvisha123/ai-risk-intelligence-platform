"""Unit tests for pure scoring helpers (no ML IO)."""

from app.utils.scoring import decision_from_risk_level, risk_level_from_score, trust_category


def test_risk_level_mapping():
    """Band thresholds classify LOW/MEDIUM/HIGH."""
    assert risk_level_from_score(0.2, 0.33, 0.66) == "LOW"
    assert risk_level_from_score(0.5, 0.33, 0.66) == "MEDIUM"
    assert risk_level_from_score(0.9, 0.33, 0.66) == "HIGH"


def test_decision_mapping():
    """Risk levels translate to lending decisions."""
    assert decision_from_risk_level("LOW") == "APPROVE"
    assert decision_from_risk_level("MEDIUM") == "REVIEW"
    assert decision_from_risk_level("HIGH") == "REJECT"


def test_trust_categories():
    """Trust buckets align with numeric ranges."""
    assert trust_category(80) == "STRONG"
    assert trust_category(60) == "MODERATE"
    assert trust_category(30) == "WEAK"
