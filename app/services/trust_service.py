"""
Trust score combining credit, fraud suspicion, and behavioral signals.
"""

from __future__ import annotations

from typing import Any, Dict

from app.core.config import get_settings
from app.services.risk_service import behavioral_analysis, predict_credit_risk_probability, predict_fraud_probability
from app.utils.helpers import clip01
from app.utils.scoring import normalize_weights, trust_category


def compute_trust_score(record: Dict[str, Any]) -> Dict[str, Any]:
    """
    Combine credit risk, fraud risk, and behavioral score into a 0–100 trust index.

    Args:
        record: Loan application dictionary.

    Returns:
        Trust score details including category and component breakdown.
    """
    settings = get_settings()
    credit_risk = predict_credit_risk_probability(record)
    fraud_risk = predict_fraud_probability(record)
    behavior = behavioral_analysis(record)
    behavioral_score = float(behavior["behavioral_score"])

    w_c, w_f, w_b = normalize_weights(
        settings.trust_weight_credit,
        settings.trust_weight_fraud,
        settings.trust_weight_behavioral,
    )

    # Higher risk lowers trust; higher behavioral_score raises trust
    credit_component = clip01(1.0 - credit_risk)
    fraud_component = clip01(1.0 - fraud_risk)
    behavioral_component = clip01(behavioral_score)

    combined = w_c * credit_component + w_f * fraud_component + w_b * behavioral_component
    trust_0_100 = round(float(combined * 100), 2)
    category = trust_category(trust_0_100)

    return {
        "trust_score": trust_0_100,
        "trust_category": category,
        "components": {
            "credit_trust": round(credit_component * 100, 2),
            "fraud_trust": round(fraud_component * 100, 2),
            "behavioral_trust": round(behavioral_component * 100, 2),
        },
        "weights": {"credit": w_c, "fraud": w_f, "behavioral": w_b},
        "raw_signals": {"credit_risk": credit_risk, "fraud_risk": fraud_risk, "behavioral_score": behavioral_score},
    }
