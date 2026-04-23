"""
Actionable recommendation engine for improving application strength.
"""

from __future__ import annotations

from typing import Any, Dict, List


def generate_suggestions(record: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Create prioritized, human-readable suggestions using simple heuristics.

    Args:
        record: Loan application dictionary.

    Returns:
        List of suggestion objects with expected impact hints.
    """
    income = float(record.get("income_annum") or 0)
    loan = float(record.get("loan_amount") or 0)
    cibil = int(record.get("cibil_score") or 0)

    suggestions: List[Dict[str, Any]] = []

    if income > 0 and loan / income > 0.5:
        target = loan * 0.85
        suggestions.append(
            {
                "category": "loan_structure",
                "title": "Reduce requested loan amount",
                "detail": "Lowering principal improves debt-to-income and collateral coverage.",
                "example_change": {"loan_amount": round(target, 2)},
                "expected_impact": "medium",
            }
        )

    if income < 600_000:
        suggestions.append(
            {
                "category": "income",
                "title": "Increase verifiable income",
                "detail": "Additional documented income strengthens repayment capacity.",
                "example_change": {"income_annum": round(income * 1.15, 2)},
                "expected_impact": "medium",
            }
        )

    if cibil < 720:
        suggestions.append(
            {
                "category": "credit_history",
                "title": "Improve CIBIL score",
                "detail": "Reduce revolving utilization and resolve delinquencies before re-applying.",
                "example_change": {"cibil_score": max(cibil + 40, 720)},
                "expected_impact": "high",
            }
        )

    if int(record.get("loan_term") or 0) > 180:
        suggestions.append(
            {
                "category": "loan_structure",
                "title": "Shorten loan tenor if cash-flow allows",
                "detail": "Shorter term reduces lender exposure horizon.",
                "example_change": {"loan_term": min(int(record.get("loan_term") or 0), 120)},
                "expected_impact": "low",
            }
        )

    if not suggestions:
        suggestions.append(
            {
                "category": "general",
                "title": "Maintain current profile",
                "detail": "No major structural weaknesses detected via rules; ensure documentation is complete.",
                "example_change": {},
                "expected_impact": "low",
            }
        )

    return suggestions

