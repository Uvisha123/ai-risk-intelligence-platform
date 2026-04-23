"""
Schemas for what-if simulation and comparison results.
"""

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


class SimulationScenarioResult(BaseModel):
    """Outcome of one what-if tweak."""

    scenario_name: str
    description: str
    adjusted_fields: Dict[str, Any]
    risk_score: float
    risk_level: Literal["LOW", "MEDIUM", "HIGH"]
    decision: Literal["APPROVE", "REVIEW", "REJECT"]


class SimulationResponse(BaseModel):
    """Grouped simulation response."""

    baseline_risk_score: float
    baseline_decision: Literal["APPROVE", "REVIEW", "REJECT"]
    scenarios: List[SimulationScenarioResult]


class ComparisonRow(BaseModel):
    """Single row in scenario comparison."""

    label: str
    risk_score: float
    risk_level: str
    decision: str
    rank: int


class ComparisonResponse(BaseModel):
    """Best-option analysis across scenarios."""

    results: List[ComparisonRow]
    best_option_label: str
    notes: Optional[str] = None
