"""
Miscellaneous helpers for routing and payloads.
"""

import json
from typing import Any, Dict


def safe_json_dict(obj: Any) -> Dict[str, Any]:
    """Best-effort conversion of an object to a JSON-serializable dict."""
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    if isinstance(obj, dict):
        return obj
    try:
        return json.loads(json.dumps(obj, default=str))
    except Exception:
        return {"value": str(obj)}


def clip01(x: float) -> float:
    """Clamp a float to [0, 1]."""
    return max(0.0, min(1.0, float(x)))
