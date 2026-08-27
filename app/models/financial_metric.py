from datetime import datetime
from decimal import Decimal

from sqlalchemy import DateTime, Float, ForeignKey, Numeric, String, Text, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.core.database import Base


class FinancialMetric(Base):
    __tablename__ = "financial_metrics"
    __table_args__ = (
        UniqueConstraint(
            "stock_code",
            "business_year",
            "report_code",
            "fs_div",
            name="uq_financial_metric_snapshot",
        ),
    )

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)

    stock_code: Mapped[str] = mapped_column(
        ForeignKey("stocks.code", ondelete="CASCADE"),
        index=True,
    )
    business_year: Mapped[str] = mapped_column(String(4), index=True)
    report_code: Mapped[str] = mapped_column(String(5), index=True)
    fs_div: Mapped[str] = mapped_column(String(10), default="CFS")

    revenue: Mapped[Decimal | None] = mapped_column(Numeric(30, 2), nullable=True)
    operating_income: Mapped[Decimal | None] = mapped_column(Numeric(30, 2), nullable=True)
    net_income: Mapped[Decimal | None] = mapped_column(Numeric(30, 2), nullable=True)
    total_assets: Mapped[Decimal | None] = mapped_column(Numeric(30, 2), nullable=True)
    total_liabilities: Mapped[Decimal | None] = mapped_column(Numeric(30, 2), nullable=True)
    total_equity: Mapped[Decimal | None] = mapped_column(Numeric(30, 2), nullable=True)
    operating_cash_flow: Mapped[Decimal | None] = mapped_column(Numeric(30, 2), nullable=True)

    previous_revenue: Mapped[Decimal | None] = mapped_column(Numeric(30, 2), nullable=True)
    previous_operating_income: Mapped[Decimal | None] = mapped_column(Numeric(30, 2), nullable=True)
    previous_net_income: Mapped[Decimal | None] = mapped_column(Numeric(30, 2), nullable=True)

    revenue_growth: Mapped[float | None] = mapped_column(Float, nullable=True)
    operating_income_growth: Mapped[float | None] = mapped_column(Float, nullable=True)
    net_income_growth: Mapped[float | None] = mapped_column(Float, nullable=True)
    operating_margin: Mapped[float | None] = mapped_column(Float, nullable=True)
    net_margin: Mapped[float | None] = mapped_column(Float, nullable=True)
    roe: Mapped[float | None] = mapped_column(Float, nullable=True)
    roa: Mapped[float | None] = mapped_column(Float, nullable=True)
    debt_ratio: Mapped[float | None] = mapped_column(Float, nullable=True)
    equity_ratio: Mapped[float | None] = mapped_column(Float, nullable=True)
    operating_cash_flow_margin: Mapped[float | None] = mapped_column(Float, nullable=True)
    cash_flow_quality: Mapped[float | None] = mapped_column(Float, nullable=True)

    growth_score: Mapped[float] = mapped_column(Float, default=0.0)
    profitability_score: Mapped[float] = mapped_column(Float, default=0.0)
    stability_score: Mapped[float] = mapped_column(Float, default=0.0)
    cash_flow_score: Mapped[float] = mapped_column(Float, default=0.0)
    valuation_score: Mapped[float] = mapped_column(Float, default=0.0)
    quality_score: Mapped[float] = mapped_column(Float, default=0.0)
    sector_relative_score: Mapped[float] = mapped_column(Float, default=0.0)
    financial_score: Mapped[float] = mapped_column(Float, default=0.0, index=True)
    data_completeness: Mapped[float] = mapped_column(Float, default=0.0)

    analysis_note: Mapped[str | None] = mapped_column(Text, nullable=True)

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=False), default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=False),
        default=datetime.utcnow,
        onupdate=datetime.utcnow,
    )

    stock = relationship("Stock", back_populates="financial_metrics")
