from datetime import date, datetime

from sqlalchemy import BigInteger, Date, DateTime, Float, ForeignKey, Integer, String, UniqueConstraint
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.core.database import Base


class StockPrediction(Base):
    __tablename__ = "stock_predictions"
    __table_args__ = (
        UniqueConstraint(
            "stock_code",
            "horizon_days",
            name="uq_stock_prediction_horizon",
        ),
    )

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    stock_code: Mapped[str] = mapped_column(
        ForeignKey("stocks.code", ondelete="CASCADE"),
        index=True,
    )
    horizon_days: Mapped[int] = mapped_column(Integer, default=5, index=True)

    model_version: Mapped[str] = mapped_column(String(40), default="phase4-v1")
    model_type: Mapped[str] = mapped_column(String(80), default="gradient-boosting")
    model_path: Mapped[str | None] = mapped_column(String(500), nullable=True)
    model_run_id: Mapped[int | None] = mapped_column(BigInteger, ForeignKey("model_runs.id", ondelete="SET NULL"), nullable=True)
    prediction_date: Mapped[date | None] = mapped_column(Date, nullable=True)
    target_date: Mapped[date | None] = mapped_column(Date, nullable=True)

    predicted_return: Mapped[float] = mapped_column(Float, default=0.0)
    predicted_price: Mapped[float | None] = mapped_column(Float, nullable=True)
    prediction_lower: Mapped[float | None] = mapped_column(Float, nullable=True)
    prediction_upper: Mapped[float | None] = mapped_column(Float, nullable=True)
    upside_probability: Mapped[float] = mapped_column(Float, default=0.0)
    confidence_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    ml_score: Mapped[float] = mapped_column(Float, default=0.0, index=True)

    training_rows: Mapped[int] = mapped_column(Integer, default=0)
    validation_mae: Mapped[float | None] = mapped_column(Float, nullable=True)
    validation_accuracy: Mapped[float | None] = mapped_column(Float, nullable=True)

    feature_date: Mapped[date] = mapped_column(Date, index=True)
    feature_json: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    trained_at: Mapped[datetime] = mapped_column(DateTime(timezone=False), default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=False),
        default=datetime.utcnow,
        onupdate=datetime.utcnow,
    )

    stock = relationship("Stock", back_populates="predictions")
