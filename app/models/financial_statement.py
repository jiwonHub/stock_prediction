from datetime import datetime
from decimal import Decimal

from sqlalchemy import DateTime, ForeignKey, Numeric, String, Text, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.core.database import Base


class FinancialStatement(Base):
    __tablename__ = "financial_statements"
    __table_args__ = (
        UniqueConstraint(
            "stock_code",
            "business_year",
            "report_code",
            "fs_div",
            "sj_div",
            "account_id",
            "account_nm",
            "account_detail",
            name="uq_financial_statement_row",
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

    corp_code: Mapped[str] = mapped_column(
        String(8),
        index=True,
    )

    business_year: Mapped[str] = mapped_column(
        String(4),
        index=True,
    )

    report_code: Mapped[str] = mapped_column(
        String(5),
        index=True,
    )

    fs_div: Mapped[str] = mapped_column(
        String(10),
        default="CFS",
    )

    fs_nm: Mapped[str | None] = mapped_column(
        String(80),
        nullable=True,
    )

    sj_div: Mapped[str] = mapped_column(
        String(10),
        default="",
    )

    sj_nm: Mapped[str | None] = mapped_column(
        String(80),
        nullable=True,
    )

    account_id: Mapped[str] = mapped_column(
        String(180),
        default="",
    )

    account_nm: Mapped[str] = mapped_column(
        String(180),
        default="",
    )

    account_detail: Mapped[str] = mapped_column(
        String(500),
        default="",
    )

    currency: Mapped[str | None] = mapped_column(
        String(20),
        nullable=True,
    )

    thstrm_amount: Mapped[Decimal | None] = mapped_column(
        Numeric(30, 2),
        nullable=True,
    )

    frmtrm_amount: Mapped[Decimal | None] = mapped_column(
        Numeric(30, 2),
        nullable=True,
    )

    bfefrmtrm_amount: Mapped[Decimal | None] = mapped_column(
        Numeric(30, 2),
        nullable=True,
    )

    thstrm_nm: Mapped[str | None] = mapped_column(
        String(120),
        nullable=True,
    )

    frmtrm_nm: Mapped[str | None] = mapped_column(
        String(120),
        nullable=True,
    )

    bfefrmtrm_nm: Mapped[str | None] = mapped_column(
        String(120),
        nullable=True,
    )

    raw_json: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=False),
        default=datetime.utcnow,
    )

    stock = relationship(
        "Stock",
        back_populates="financial_statements",
    )
