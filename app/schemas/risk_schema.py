"""
Pydantic schemas for loan applications and risk responses.
"""

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field, field_validator


class LoanApplicationBase(BaseModel):
    """Core loan application attributes used across endpoints."""

    loan_id: Optional[int] = Field(default=None, description="Optional client reference for traceability")
    no_of_dependents: int = Field(ge=0, le=20)
    education: Literal["Graduate", "Not Graduate"]
    self_employed: Literal["Yes", "No"]
    income_annum: float = Field(gt=0)
    loan_amount: float = Field(gt=0)
    loan_term: int = Field(gt=0, le=600)
    cibil_score: int = Field(ge=300, le=900)
    residential_assets_value: float = Field(ge=0)
    commercial_assets_value: float = Field(ge=0)
    luxury_assets_value: float = Field(ge=0)
    bank_asset_value: float = Field(ge=0)

    @field_validator("education", "self_employed", mode="before")
    @classmethod
    def strip_strings(cls, v: Any) -> Any:
        """Strip whitespace from categorical fields."""
        if isinstance(v, str):
            return v.strip()
        return v


class RiskAnalysisResponse(BaseModel):
    """Full risk intelligence payload for predict-risk."""

    risk_score: float = Field(ge=0, le=1, description="Probability of adverse credit outcome")
    risk_level: Literal["LOW", "MEDIUM", "HIGH"]
    decision: Literal["APPROVE", "REVIEW", "REJECT"]
    risk_drivers: List[str] = Field(default_factory=list)
    behavioral_score: float = Field(ge=0, le=1)
    fraud_assessment: Dict[str, Any]
    early_warning: Dict[str, Any]
    scenario_hint: Optional[str] = None


class CompareScenario(BaseModel):
    """One scenario for side-by-side comparison."""

    label: str
    income_annum: float
    loan_amount: float
    cibil_score: int


class ScenarioCompareRequest(BaseModel):
    """Request body to compare multiple loan scenarios."""

    base_application: LoanApplicationBase
    scenarios: List[CompareScenario] = Field(min_length=1, max_length=10)
