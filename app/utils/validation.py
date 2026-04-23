"""
Additional validation helpers beyond Pydantic schemas.
"""

from typing import Any, Dict, List


def collect_numeric_bounds_violations(record: Dict[str, Any]) -> List[str]:
    """
    Check obvious numeric domain violations for loan applications.

    Args:
        record: Application payload.

    Returns:
        List of human-readable violation messages (empty if none).
    """
    issues: List[str] = []
    if record.get("cibil_score", 0) < 300 or record.get("cibil_score", 900) > 900:
        issues.append("cibil_score should typically be between 300 and 900")
    if record.get("loan_term", 0) <= 0:
        issues.append("loan_term must be positive")
    if record.get("income_annum", 0) < 0 or record.get("loan_amount", 0) < 0:
        issues.append("income and loan_amount must be non-negative")
    return issues


def assert_known_categories(record: Dict[str, Any]) -> List[str]:
    """Warn on unexpected categorical labels (non-blocking hints)."""
    hints: List[str] = []
    edu = str(record.get("education", "")).strip()
    if edu and edu not in ("Graduate", "Not Graduate"):
        hints.append(f"education value '{edu}' will be encoded as unseen category via dummies")
    emp = str(record.get("self_employed", "")).strip()
    if emp and emp not in ("Yes", "No"):
        hints.append(f"self_employed value '{emp}' may not match training categories")
    return hints
