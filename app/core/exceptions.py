"""
Domain and API exceptions with HTTP mapping.
"""

from typing import Any, Dict, Optional


class RiskIntelligenceError(Exception):
    """Base exception for the risk intelligence domain."""

    def __init__(self, message: str, code: str = "DOMAIN_ERROR", details: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.message = message
        self.code = code
        self.details = details or {}


class ModelLoadError(RiskIntelligenceError):
    """Raised when an ML artifact cannot be loaded."""

    def __init__(self, message: str = "Failed to load ML model", details: Optional[Dict[str, Any]] = None):
        super().__init__(message, code="MODEL_LOAD_ERROR", details=details)


class PredictionError(RiskIntelligenceError):
    """Raised when inference fails."""

    def __init__(self, message: str = "Prediction failed", details: Optional[Dict[str, Any]] = None):
        super().__init__(message, code="PREDICTION_ERROR", details=details)


class ValidationError(RiskIntelligenceError):
    """Raised when business validation fails beyond Pydantic schema."""

    def __init__(self, message: str = "Validation failed", details: Optional[Dict[str, Any]] = None):
        super().__init__(message, code="VALIDATION_ERROR", details=details)


def to_http_payload(exc: RiskIntelligenceError) -> Dict[str, Any]:
    """Serialize exception for JSON error responses."""
    return {"code": exc.code, "message": exc.message, "details": exc.details}
