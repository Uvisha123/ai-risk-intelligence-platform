"""
Load persisted sklearn model artifacts from disk using joblib.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any, Dict

import joblib

from app.core.config import ML_MODELS_DIR
from app.core.exceptions import ModelLoadError


def _path(name: str) -> Path:
    """Resolve a model filename under app/models."""
    return ML_MODELS_DIR / name


def _load_bundle(path: Path, label: str) -> Dict[str, Any]:
    """
    Load a joblib-serialized artifact dictionary.

    Args:
        path: Absolute path to .pkl file.
        label: Logical model name for error messages.

    Returns:
        Artifact dictionary.

    Raises:
        ModelLoadError: If file missing or corrupt.
    """
    if not path.exists():
        raise ModelLoadError(f"Missing model file: {path}", details={"model": label})
    try:
        bundle = joblib.load(path)
        if not isinstance(bundle, dict):
            raise ModelLoadError(f"Invalid artifact format for {label}")
        return bundle
    except ModelLoadError:
        raise
    except Exception as exc:  # pragma: no cover — defensive
        raise ModelLoadError(f"Could not load {label}", details={"error": str(exc)}) from exc


@lru_cache
def load_loan_artifact() -> Dict[str, Any]:
    """Load and cache the loan credit risk artifact."""
    return _load_bundle(_path("loan_model.pkl"), "loan")


@lru_cache
def load_fraud_artifact() -> Dict[str, Any]:
    """Load and cache the fraud scoring artifact."""
    return _load_bundle(_path("fraud_model.pkl"), "fraud")


def clear_loader_cache() -> None:
    """Invalidate caches (mainly for tests)."""
    load_loan_artifact.cache_clear()
    load_fraud_artifact.cache_clear()
