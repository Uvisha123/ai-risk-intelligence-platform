"""
Application settings and decision thresholds.
"""

from functools import lru_cache
from pathlib import Path
from typing import List

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parent.parent.parent
LOGS_DIR = BASE_DIR / "logs"
DATA_DIR = BASE_DIR / "data"
ML_MODELS_DIR = BASE_DIR / "app" / "models"


class Settings(BaseSettings):
    """Environment-driven configuration."""

    model_config = SettingsConfigDict(
        env_file=str(BASE_DIR / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_name: str = Field(default="AI Loan & Financial Risk Intelligence System")
    debug: bool = Field(default=False)
    api_v1_prefix: str = Field(default="/api/v1")

    database_url: str = Field(default=f"sqlite:///{BASE_DIR / 'data' / 'risk_intel.db'}")

    # Decision thresholds on risk_score in [0, 1]
    threshold_low_max: float = Field(default=0.33, description="Scores below this are LOW risk")
    threshold_medium_max: float = Field(default=0.66, description="Scores up to this are MEDIUM")

    # Trust score weights (must sum to ~1.0 for interpretability)
    trust_weight_credit: float = Field(default=0.5)
    trust_weight_fraud: float = Field(default=0.3)
    trust_weight_behavioral: float = Field(default=0.2)

    cors_origins: List[str] = Field(default_factory=lambda: ["*"])

    log_file: Path = Field(default=LOGS_DIR / "app.log")


@lru_cache
def get_settings() -> Settings:
    """Return cached settings singleton."""
    return Settings()
