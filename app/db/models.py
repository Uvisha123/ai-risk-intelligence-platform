"""
ORM models for audit and optional operational storage.
"""

from datetime import datetime
from typing import Optional

from sqlalchemy import DateTime, Float, Integer, JSON, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from app.db.database import Base


class PredictionAudit(Base):
    """Stores compact prediction outcomes for analytics (optional writes)."""

    __tablename__ = "prediction_audit"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    risk_score: Mapped[float] = mapped_column(Float)
    decision: Mapped[str] = mapped_column(String(32))
    payload_summary: Mapped[str] = mapped_column(Text, default="")
    extra: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
