from datetime import date, datetime

from sqlalchemy import BigInteger, Date, DateTime, Float, ForeignKey, Numeric, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.core.database import Base


class StockPrice(Base):
    __tablename__ = "stock_prices"
    __table_args__ = (
        UniqueConstraint(
            "stock_code",
            "trade_date",
            name="uq_stock_price_code_date",
        ),
    )

    id: Mapped[int] = mapped_column(
        primary_key=True,
        autoincrement=True,
    )

    stock_code: Mapped[str] = mapped_column(
        ForeignKey(
            "stocks.code",
            ondelete="CASCADE",
        ),
        index=True,
    )

    trade_date: Mapped[date] = mapped_column(
        Date,
        index=True,
    )

    open: Mapped[float] = mapped_column(Float)
    high: Mapped[float] = mapped_column(Float)
    low: Mapped[float] = mapped_column(Float)
    close: Mapped[float] = mapped_column(Float)
    adjusted_close: Mapped[float | None] = mapped_column(Float, nullable=True)
    trading_value: Mapped[float | None] = mapped_column(Numeric(30, 2), nullable=True)

    volume: Mapped[int] = mapped_column(
        BigInteger,
        default=0,
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=False),
        default=datetime.utcnow,
    )

    stock = relationship(
        "Stock",
        back_populates="prices",
    )
