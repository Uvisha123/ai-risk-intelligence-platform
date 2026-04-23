"""
Recommendation suggestions based on applicant profile heuristics.
"""

from fastapi import APIRouter

from app.schemas.risk_schema import LoanApplicationBase
from app.services.suggestion_service import generate_suggestions

router = APIRouter(tags=["suggestions"])


@router.post("/recommend")
def recommend(payload: LoanApplicationBase):
    """
    Produce actionable suggestions (loan size, income, bureau, tenor).

    Args:
        payload: Loan application payload.

    Returns:
        List of structured suggestions.
    """
    items = generate_suggestions(payload.model_dump())
    return {"status": "success", "data": {"suggestions": items}}
