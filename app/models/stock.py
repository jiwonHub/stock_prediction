from datetime import date, datetime

from sqlalchemy import BigInteger, Boolean, Date, DateTime, Float, String
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.core.database import Base


class Stock(Base):
    __tablename__ = "stocks"

    code: Mapped[str] = mapped_column(
        String(6),
        primary_key=True,
    )

    corp_code: Mapped[str | None] = mapped_column(
        String(8),
        unique=True,
        index=True,
        nullable=True,
    )

    name: Mapped[str] = mapped_column(
        String(120),
        index=True,
    )

    english_name: Mapped[str | None] = mapped_column(
        String(200),
        nullable=True,
    )

    market: Mapped[str] = mapped_column(
        String(40),
        default="KRX",
    )

    current_price: Mapped[float | None] = mapped_column(
        Float,
        nullable=True,
    )

    change: Mapped[float | None] = mapped_column(
        Float,
        nullable=True,
    )

    change_rate: Mapped[float | None] = mapped_column(
        Float,
        nullable=True,
    )

    market_cap: Mapped[float | None] = mapped_column(
        Float,
        nullable=True,
    )

    per: Mapped[float | None] = mapped_column(
        Float,
        nullable=True,
    )

    pbr: Mapped[float | None] = mapped_column(
        Float,
        nullable=True,
    )

    eps: Mapped[float | None] = mapped_column(
        Float,
        nullable=True,
    )

    bps: Mapped[float | None] = mapped_column(
        Float,
        nullable=True,
    )

    sector_name: Mapped[str | None] = mapped_column(
        String(120),
        nullable=True,
    )

    sector_code: Mapped[str | None] = mapped_column(String(40), nullable=True)
    industry_name: Mapped[str | None] = mapped_column(String(160), nullable=True)
    listing_date: Mapped[date | None] = mapped_column(Date, nullable=True)
    shares_outstanding: Mapped[int | None] = mapped_column(BigInteger, nullable=True)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)

    modified_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=False),
        nullable=True,
    )

    last_price_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=False),
        nullable=True,
    )

    last_financial_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=False), nullable=True)
    last_news_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=False), nullable=True)
    last_prediction_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=False), nullable=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=False),
        default=datetime.utcnow,
    )

    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=False),
        default=datetime.utcnow,
        onupdate=datetime.utcnow,
    )

    prices = relationship(
        "StockPrice",
        back_populates="stock",
        cascade="all, delete-orphan",
    )

    financial_statements = relationship(
        "FinancialStatement",
        back_populates="stock",
        cascade="all, delete-orphan",
    )

    financial_metrics = relationship(
        "FinancialMetric",
        back_populates="stock",
        cascade="all, delete-orphan",
    )

    predictions = relationship(
        "StockPrediction",
        back_populates="stock",
        cascade="all, delete-orphan",
    )
