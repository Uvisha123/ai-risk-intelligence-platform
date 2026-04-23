"""
What-if simulation and scenario comparison endpoints.
"""

from fastapi import APIRouter

from app.schemas.risk_schema import LoanApplicationBase, ScenarioCompareRequest
from app.schemas.simulation_schema import ComparisonResponse, SimulationResponse
from app.services.simulation_service import compare_loan_scenarios, run_default_simulations

router = APIRouter(tags=["simulation"])


@router.post("/simulate", response_model=SimulationResponse)
def simulate(payload: LoanApplicationBase):
    """
    Simulate income, loan, and bureau improvements against the baseline profile.

    Args:
        payload: Loan application snapshot.

    Returns:
        Baseline metrics plus scenario outcomes.
    """
    result = run_default_simulations(payload.model_dump())
    return SimulationResponse(**result)


@router.post("/compare-scenarios", response_model=ComparisonResponse)
def compare_scenarios(body: ScenarioCompareRequest):
    """
    Compare alternative loan structures and identify the lowest-risk option.

    Args:
        body: Base profile plus an array of candidate scenarios.

    Returns:
        Ranked comparison table and recommended label.
    """
    summary = compare_loan_scenarios(
        body.base_application.model_dump(),
        [s.model_dump() for s in body.scenarios],
    )
    rows = summary["results"]
    comparison_rows = [
        {
            "label": r["label"],
            "risk_score": r["risk_score"],
            "risk_level": r["risk_level"],
            "decision": r["decision"],
            "rank": r["rank"],
        }
        for r in rows
    ]
    return ComparisonResponse(
        results=comparison_rows,
        best_option_label=summary["best_option_label"],
        notes="Ranked by lowest modeled credit risk score",
    )
