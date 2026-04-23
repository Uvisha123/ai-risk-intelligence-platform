"""
Endpoints for credit risk prediction and composite trust scoring.
"""

from fastapi import APIRouter

from app.schemas.risk_schema import LoanApplicationBase
from app.services.risk_service import assess_credit_risk_package
from app.services.trust_service import compute_trust_score

router = APIRouter(tags=["risk"])


@router.post("/predict-risk")
def predict_risk(payload: LoanApplicationBase):
    """
    Predict credit risk, fraud suspicion, behavioral signals, and early warnings.

    Args:
        payload: Validated loan application body.

    Returns:
        Standard envelope with full analysis payload.
    """
    data = assess_credit_risk_package(payload.model_dump())
    return {"status": "success", "data": data}


@router.post("/trust-score")
def trust_score(payload: LoanApplicationBase):
    """
    Compute blended trust score across credit, fraud, and behavioral dimensions.

    Args:
        payload: Loan application body.

    Returns:
        Trust score breakdown JSON envelope.
    """
    data = compute_trust_score(payload.model_dump())
    return {"status": "success", "data": data}
