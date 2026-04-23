"""
Schemas for decision engine responses.
"""

from typing import Literal, Optional

from pydantic import BaseModel, Field

from app.schemas.risk_schema import LoanApplicationBase


class DecisionRequest(BaseModel):
    """Input for decision-only endpoint."""

    application: LoanApplicationBase
    risk_score_override: Optional[float] = Field(default=None, ge=0, le=1)


class DecisionResponse(BaseModel):
    """Structured decision outcome."""

    risk_score: float = Field(ge=0, le=1)
    risk_level: Literal["LOW", "MEDIUM", "HIGH"]
    decision: Literal["APPROVE", "REVIEW", "REJECT"]
    rationale: str = Field(default="Thresholds from application configuration")
