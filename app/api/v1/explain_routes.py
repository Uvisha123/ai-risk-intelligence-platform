"""
Explainability (XAI) endpoints.
"""

from fastapi import APIRouter

from app.schemas.risk_schema import LoanApplicationBase
from app.services.explain_service import full_explanation

router = APIRouter(tags=["explain"])


@router.post("/explain")
def explain(payload: LoanApplicationBase):
    """
    Return global feature-importance narrative and heuristic reason codes.

    Args:
        payload: Loan application payload.

    Returns:
        Explanation JSON document.
    """
    data = full_explanation(payload.model_dump())
    return {"status": "success", "data": data}
