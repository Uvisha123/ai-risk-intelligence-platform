"""
Lending decision mapping endpoint.
"""

from fastapi import APIRouter

from app.schemas.decision_schema import DecisionRequest, DecisionResponse
from app.services.decision_service import classify_risk_level, decision_from_score
from app.services.risk_service import predict_credit_risk_probability

router = APIRouter(tags=["decision"])


@router.post("/decision", response_model=DecisionResponse)
def lending_decision(body: DecisionRequest):
    """
    Map credit risk into APPROVE / REVIEW / REJECT using configured thresholds.

    Args:
        body: Application payload and optional risk_score override.

    Returns:
        DecisionResponse with computed or overridden score.
    """
    app_dict = body.application.model_dump()
    score = float(body.risk_score_override) if body.risk_score_override is not None else predict_credit_risk_probability(app_dict)
    level = classify_risk_level(score)
    decision = decision_from_score(score)
    return DecisionResponse(risk_score=score, risk_level=level, decision=decision)
