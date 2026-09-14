from datetime import date, datetime
from decimal import Decimal

from sqlalchemy import (
    BigInteger,
    Date,
    DateTime,
    Float,
    ForeignKey,
    Numeric,
    String,
    UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from app.core.database import Base


class MarketIndexPrice(Base):
    __tablename__ = "market_index_prices"
    __table_args__ = (
        UniqueConstraint(
            "index_code",
            "trade_date",
            name="uq_market_index_price",
        ),
    )

    id: Mapped[int] = mapped_column(
        BigInteger,
        primary_key=True,
        autoincrement=True,
    )

    index_code: Mapped[str] = mapped_column(
        String(40),
        index=True,
    )

    index_name: Mapped[str] = mapped_column(
        String(120),
    )

    market: Mapped[str] = mapped_column(
        String(30),
        index=True,
    )

    trade_date: Mapped[date] = mapped_column(
        Date,
        index=True,
    )

    open: Mapped[float | None] = mapped_column(
        Float,
        nullable=True,
    )

    high: Mapped[float | None] = mapped_column(
        Float,
        nullable=True,
    )

    low: Mapped[float | None] = mapped_column(
        Float,
        nullable=True,
    )

    close: Mapped[float] = mapped_column(
        Float,
    )

    change: Mapped[float | None] = mapped_column(
        Float,
        nullable=True,
    )

    change_rate: Mapped[float | None] = mapped_column(
        Float,
        nullable=True,
    )

    volume: Mapped[int | None] = mapped_column(
        BigInteger,
        nullable=True,
    )

    trading_value: Mapped[Decimal | None] = mapped_column(
        Numeric(30, 2),
        nullable=True,
    )

    source: Mapped[str] = mapped_column(
        String(40),
        default="KIS",
    )

    raw_json: Mapped[dict | None] = mapped_column(
        JSONB,
        nullable=True,
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=datetime.utcnow,
    )

    updated_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=datetime.utcnow,
        onupdate=datetime.utcnow,
    )


class MacroIndicator(Base):
    __tablename__ = "macro_indicators"
    __table_args__ = (
        UniqueConstraint(
            "indicator_code",
            "observed_date",
            name="uq_macro_indicator",
        ),
    )

    id: Mapped[int] = mapped_column(
        BigInteger,
        primary_key=True,
        autoincrement=True,
    )

    indicator_code: Mapped[str] = mapped_column(
        String(60),
        index=True,
    )

    indicator_name: Mapped[str] = mapped_column(
        String(160),
    )

    indicator_type: Mapped[str] = mapped_column(
        String(40),
        index=True,
    )

    frequency: Mapped[str] = mapped_column(
        String(20),
        default="D",
    )

    observed_date: Mapped[date] = mapped_column(
        Date,
        index=True,
    )

    value: Mapped[float] = mapped_column(
        Float,
    )

    unit: Mapped[str | None] = mapped_column(
        String(40),
        nullable=True,
    )

    source: Mapped[str] = mapped_column(
        String(40),
    )

    metadata_json: Mapped[dict | None] = mapped_column(
        JSONB,
        nullable=True,
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=datetime.utcnow,
    )

    updated_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=datetime.utcnow,
        onupdate=datetime.utcnow,
    )


class MarketInvestorFlow(Base):
    __tablename__ = "market_investor_flows"
    __table_args__ = (
        UniqueConstraint(
            "market",
            "trade_date",
            name="uq_market_investor_flow",
        ),
    )

    id: Mapped[int] = mapped_column(
        BigInteger,
        primary_key=True,
        autoincrement=True,
    )

    market: Mapped[str] = mapped_column(
        String(30),
        index=True,
    )

    trade_date: Mapped[date] = mapped_column(
        Date,
        index=True,
    )

    individual_buy_amount: Mapped[Decimal | None] = mapped_column(
        Numeric(30, 2),
        nullable=True,
    )

    individual_sell_amount: Mapped[Decimal | None] = mapped_column(
        Numeric(30, 2),
        nullable=True,
    )

    foreign_buy_amount: Mapped[Decimal | None] = mapped_column(
        Numeric(30, 2),
        nullable=True,
    )

    foreign_sell_amount: Mapped[Decimal | None] = mapped_column(
        Numeric(30, 2),
        nullable=True,
    )

    institution_buy_amount: Mapped[Decimal | None] = mapped_column(
        Numeric(30, 2),
        nullable=True,
    )

    institution_sell_amount: Mapped[Decimal | None] = mapped_column(
        Numeric(30, 2),
        nullable=True,
    )

    other_corporation_buy_amount: Mapped[Decimal | None] = mapped_column(
        Numeric(30, 2),
        nullable=True,
    )

    other_corporation_sell_amount: Mapped[Decimal | None] = mapped_column(
        Numeric(30, 2),
        nullable=True,
    )

    institution_breakdown_json: Mapped[dict | None] = mapped_column(
        JSONB,
        nullable=True,
    )

    source: Mapped[str] = mapped_column(
        String(40),
        default="TOSS",
    )

    raw_json: Mapped[dict | None] = mapped_column(
        JSONB,
        nullable=True,
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=datetime.utcnow,
    )

    updated_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=datetime.utcnow,
        onupdate=datetime.utcnow,
    )


class SectorIndexPrice(Base):
    __tablename__ = "sector_index_prices"
    __table_args__ = (
        UniqueConstraint(
            "market",
            "sector_code",
            "trade_date",
            name="uq_sector_index_price",
        ),
    )

    id: Mapped[int] = mapped_column(
        BigInteger,
        primary_key=True,
        autoincrement=True,
    )

    sector_code: Mapped[str] = mapped_column(
        String(60),
        index=True,
    )

    sector_name: Mapped[str] = mapped_column(
        String(160),
        index=True,
    )

    market: Mapped[str | None] = mapped_column(
        String(30),
        nullable=True,
    )

    trade_date: Mapped[date] = mapped_column(
        Date,
        index=True,
    )

    open: Mapped[float | None] = mapped_column(
        Float,
        nullable=True,
    )

    high: Mapped[float | None] = mapped_column(
        Float,
        nullable=True,
    )

    low: Mapped[float | None] = mapped_column(
        Float,
        nullable=True,
    )

    close: Mapped[float] = mapped_column(
        Float,
    )

    change_rate: Mapped[float | None] = mapped_column(
        Float,
        nullable=True,
    )

    volume: Mapped[int | None] = mapped_column(
        BigInteger,
        nullable=True,
    )

    trading_value: Mapped[Decimal | None] = mapped_column(
        Numeric(30, 2),
        nullable=True,
    )

    source: Mapped[str] = mapped_column(
        String(40),
        default="KIS",
    )

    raw_json: Mapped[dict | None] = mapped_column(
        JSONB,
        nullable=True,
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=datetime.utcnow,
    )

    updated_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=datetime.utcnow,
        onupdate=datetime.utcnow,
    )


class StockInvestorFlow(Base):
    __tablename__ = "stock_investor_flows"
    __table_args__ = (
        UniqueConstraint(
            "stock_code",
            "trade_date",
            name="uq_stock_investor_flow",
        ),
    )

    id: Mapped[int] = mapped_column(
        BigInteger,
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

    foreign_net_buy_amount: Mapped[Decimal | None] = mapped_column(
        Numeric(30, 2),
        nullable=True,
    )

    institution_net_buy_amount: Mapped[Decimal | None] = mapped_column(
        Numeric(30, 2),
        nullable=True,
    )

    individual_net_buy_amount: Mapped[Decimal | None] = mapped_column(
        Numeric(30, 2),
        nullable=True,
    )

    foreign_net_buy_volume: Mapped[int | None] = mapped_column(
        BigInteger,
        nullable=True,
    )

    institution_net_buy_volume: Mapped[int | None] = mapped_column(
        BigInteger,
        nullable=True,
    )

    individual_net_buy_volume: Mapped[int | None] = mapped_column(
        BigInteger,
        nullable=True,
    )

    foreign_holding_ratio: Mapped[float | None] = mapped_column(
        Float,
        nullable=True,
    )

    source: Mapped[str] = mapped_column(
        String(40),
        default="KIS",
    )

    raw_json: Mapped[dict | None] = mapped_column(
        JSONB,
        nullable=True,
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=datetime.utcnow,
    )

    updated_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=datetime.utcnow,
        onupdate=datetime.utcnow,
    )

class StockProgramTrade(Base):
    __tablename__ = "stock_program_trades"
    __table_args__ = (
        UniqueConstraint(
            "stock_code",
            "trade_date",
            name="uq_stock_program_trade",
        ),
    )

    id: Mapped[int] = mapped_column(
        BigInteger,
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

    arbitrage_buy_volume: Mapped[int | None] = mapped_column(
        BigInteger,
        nullable=True,
    )

    arbitrage_sell_volume: Mapped[int | None] = mapped_column(
        BigInteger,
        nullable=True,
    )

    arbitrage_net_buy_volume: Mapped[int | None] = mapped_column(
        BigInteger,
        nullable=True,
    )

    non_arbitrage_buy_volume: Mapped[int | None] = mapped_column(
        BigInteger,
        nullable=True,
    )

    non_arbitrage_sell_volume: Mapped[int | None] = mapped_column(
        BigInteger,
        nullable=True,
    )

    non_arbitrage_net_buy_volume: Mapped[int | None] = mapped_column(
        BigInteger,
        nullable=True,
    )

    source: Mapped[str] = mapped_column(
        String(40),
        default="TOSS",
    )

    raw_json: Mapped[dict | None] = mapped_column(
        JSONB,
        nullable=True,
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=datetime.utcnow,
    )

    updated_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=datetime.utcnow,
        onupdate=datetime.utcnow,
    )


class StockShortSelling(Base):
    __tablename__ = "stock_short_selling"
    __table_args__ = (
        UniqueConstraint(
            "stock_code",
            "trade_date",
            name="uq_stock_short_selling",
        ),
    )

    id: Mapped[int] = mapped_column(
        BigInteger,
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

    short_selling_volume: Mapped[int | None] = mapped_column(
        BigInteger,
        nullable=True,
    )

    short_selling_amount: Mapped[int | None] = mapped_column(
        BigInteger,
        nullable=True,
    )

    short_selling_volume_rate: Mapped[float | None] = mapped_column(
        Float,
        nullable=True,
    )

    short_selling_amount_rate: Mapped[float | None] = mapped_column(
        Float,
        nullable=True,
    )

    source: Mapped[str] = mapped_column(
        String(40),
        default="TOSS",
    )

    raw_json: Mapped[dict | None] = mapped_column(
        JSONB,
        nullable=True,
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=datetime.utcnow,
    )

    updated_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=datetime.utcnow,
        onupdate=datetime.utcnow,
    )


class StockCreditTrade(Base):
    __tablename__ = "stock_credit_trades"
    __table_args__ = (
        UniqueConstraint(
            "stock_code",
            "trade_date",
            name="uq_stock_credit_trade",
        ),
    )

    id: Mapped[int] = mapped_column(
        BigInteger,
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

    margin_loan_new_quantity: Mapped[int | None] = mapped_column(
        BigInteger,
        nullable=True,
    )

    margin_loan_return_quantity: Mapped[int | None] = mapped_column(
        BigInteger,
        nullable=True,
    )

    margin_loan_balance_quantity: Mapped[int | None] = mapped_column(
        BigInteger,
        nullable=True,
    )

    margin_loan_balance_rate: Mapped[float | None] = mapped_column(
        Float,
        nullable=True,
    )

    margin_loan_trading_rate: Mapped[float | None] = mapped_column(
        Float,
        nullable=True,
    )

    stock_loan_new_quantity: Mapped[int | None] = mapped_column(
        BigInteger,
        nullable=True,
    )

    stock_loan_return_quantity: Mapped[int | None] = mapped_column(
        BigInteger,
        nullable=True,
    )

    stock_loan_balance_quantity: Mapped[int | None] = mapped_column(
        BigInteger,
        nullable=True,
    )

    stock_loan_balance_rate: Mapped[float | None] = mapped_column(
        Float,
        nullable=True,
    )

    stock_loan_trading_rate: Mapped[float | None] = mapped_column(
        Float,
        nullable=True,
    )

    source: Mapped[str] = mapped_column(
        String(40),
        default="TOSS",
    )

    raw_json: Mapped[dict | None] = mapped_column(
        JSONB,
        nullable=True,
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=datetime.utcnow,
    )

    updated_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=datetime.utcnow,
        onupdate=datetime.utcnow,
    )


class StockSecuritiesLending(Base):
    __tablename__ = "stock_securities_lending"
    __table_args__ = (
        UniqueConstraint(
            "stock_code",
            "trade_date",
            name="uq_stock_securities_lending",
        ),
    )

    id: Mapped[int] = mapped_column(
        BigInteger,
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

    execution_quantity: Mapped[int | None] = mapped_column(
        BigInteger,
        nullable=True,
    )

    repayment_quantity: Mapped[int | None] = mapped_column(
        BigInteger,
        nullable=True,
    )

    balance_quantity: Mapped[int | None] = mapped_column(
        BigInteger,
        nullable=True,
    )

    balance_amount: Mapped[int | None] = mapped_column(
        BigInteger,
        nullable=True,
    )

    source: Mapped[str] = mapped_column(
        String(40),
        default="TOSS",
    )

    raw_json: Mapped[dict | None] = mapped_column(
        JSONB,
        nullable=True,
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=datetime.utcnow,
    )

    updated_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=datetime.utcnow,
        onupdate=datetime.utcnow,
    )

class StockValuationSnapshot(Base):
    __tablename__ = "stock_valuation_snapshots"
    __table_args__ = (
        UniqueConstraint(
            "stock_code",
            "snapshot_date",
            name="uq_stock_valuation_snapshot",
        ),
    )

    id: Mapped[int] = mapped_column(
        BigInteger,
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

    snapshot_date: Mapped[date] = mapped_column(
        Date,
        index=True,
    )

    price: Mapped[float | None] = mapped_column(
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

    dividend_yield: Mapped[float | None] = mapped_column(
        Float,
        nullable=True,
    )

    sector_code: Mapped[str | None] = mapped_column(
        String(60),
        nullable=True,
        index=True,
    )

    sector_name: Mapped[str | None] = mapped_column(
        String(160),
        nullable=True,
    )

    sector_per: Mapped[float | None] = mapped_column(
        Float,
        nullable=True,
    )

    sector_pbr: Mapped[float | None] = mapped_column(
        Float,
        nullable=True,
    )

    market_per: Mapped[float | None] = mapped_column(
        Float,
        nullable=True,
    )

    market_pbr: Mapped[float | None] = mapped_column(
        Float,
        nullable=True,
    )

    source: Mapped[str] = mapped_column(
        String(40),
        default="KIS",
    )

    raw_json: Mapped[dict | None] = mapped_column(
        JSONB,
        nullable=True,
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=datetime.utcnow,
    )

    updated_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=datetime.utcnow,
        onupdate=datetime.utcnow,
    )