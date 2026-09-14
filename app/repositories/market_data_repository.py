from datetime import date, datetime

from sqlalchemy import (
    delete,
    func,
    or_,
    select,
    update,
)
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.orm import Session
from app.models.stock import Stock

from app.models.market_data import (
    MacroIndicator,
    MarketIndexPrice,
    MarketInvestorFlow,
    SectorIndexPrice,
    StockCreditTrade,
    StockInvestorFlow,
    StockProgramTrade,
    StockSecuritiesLending,
    StockShortSelling,
    StockValuationSnapshot,
)

class MarketDataRepository:
    def __init__(self, db: Session):
        self.db = db

    def upsert_market_index_prices(
        self,
        rows: list[dict],
    ) -> int:
        if not rows:
            return 0

        stmt = insert(
            MarketIndexPrice
        ).values(rows)

        stmt = stmt.on_conflict_do_update(
            constraint="uq_market_index_price",
            set_={
                "index_name": stmt.excluded.index_name,
                "market": stmt.excluded.market,
                "open": stmt.excluded.open,
                "high": stmt.excluded.high,
                "low": stmt.excluded.low,
                "close": stmt.excluded.close,
                "change": stmt.excluded.change,
                "change_rate": stmt.excluded.change_rate,
                "volume": stmt.excluded.volume,
                "trading_value": stmt.excluded.trading_value,
                "source": stmt.excluded.source,
                "raw_json": stmt.excluded.raw_json,
                "updated_at": datetime.utcnow(),
            },
        )

        self.db.execute(stmt)
        self.db.commit()

        return len(rows)

    def upsert_market_investor_flows(
        self,
        rows: list[dict],
    ) -> int:
        if not rows:
            return 0

        stmt = insert(
            MarketInvestorFlow
        ).values(rows)

        stmt = stmt.on_conflict_do_update(
            constraint="uq_market_investor_flow",
            set_={
                "individual_buy_amount":
                    stmt.excluded.individual_buy_amount,
                "individual_sell_amount":
                    stmt.excluded.individual_sell_amount,
                "foreign_buy_amount":
                    stmt.excluded.foreign_buy_amount,
                "foreign_sell_amount":
                    stmt.excluded.foreign_sell_amount,
                "institution_buy_amount":
                    stmt.excluded.institution_buy_amount,
                "institution_sell_amount":
                    stmt.excluded.institution_sell_amount,
                "other_corporation_buy_amount":
                    stmt.excluded.other_corporation_buy_amount,
                "other_corporation_sell_amount":
                    stmt.excluded.other_corporation_sell_amount,
                "institution_breakdown_json":
                    stmt.excluded.institution_breakdown_json,
                "source":
                    stmt.excluded.source,
                "raw_json":
                    stmt.excluded.raw_json,
                "updated_at":
                    datetime.utcnow(),
            },
        )

        self.db.execute(stmt)
        self.db.commit()

        return len(rows)

    def upsert_macro_indicators(
        self,
        rows: list[dict],
    ) -> int:
        if not rows:
            return 0

        stmt = insert(
            MacroIndicator
        ).values(rows)

        stmt = stmt.on_conflict_do_update(
            constraint="uq_macro_indicator",
            set_={
                "indicator_name": stmt.excluded.indicator_name,
                "indicator_type": stmt.excluded.indicator_type,
                "frequency": stmt.excluded.frequency,
                "value": stmt.excluded.value,
                "unit": stmt.excluded.unit,
                "source": stmt.excluded.source,
                "metadata_json": stmt.excluded.metadata_json,
                "updated_at": datetime.utcnow(),
            },
        )

        self.db.execute(stmt)
        self.db.commit()

        return len(rows)

    def upsert_sector_index_prices(
        self,
        rows: list[dict],
    ) -> int:
        if not rows:
            return 0

        stmt = insert(
            SectorIndexPrice
        ).values(rows)

        stmt = stmt.on_conflict_do_update(
            constraint="uq_sector_index_price",
            set_={
                "sector_name": stmt.excluded.sector_name,
                "market": stmt.excluded.market,
                "open": stmt.excluded.open,
                "high": stmt.excluded.high,
                "low": stmt.excluded.low,
                "close": stmt.excluded.close,
                "change_rate": stmt.excluded.change_rate,
                "volume": stmt.excluded.volume,
                "trading_value": stmt.excluded.trading_value,
                "source": stmt.excluded.source,
                "raw_json": stmt.excluded.raw_json,
                "updated_at": datetime.utcnow(),
            },
        )

        self.db.execute(stmt)
        self.db.commit()

        return len(rows)

    def upsert_stock_investor_flows(
        self,
        rows: list[dict],
    ) -> int:
        if not rows:
            return 0

        stmt = insert(
            StockInvestorFlow
        ).values(rows)

        stmt = stmt.on_conflict_do_update(
            constraint="uq_stock_investor_flow",
            set_={
                "foreign_net_buy_amount":
                    stmt.excluded.foreign_net_buy_amount,
                "institution_net_buy_amount":
                    stmt.excluded.institution_net_buy_amount,
                "individual_net_buy_amount":
                    stmt.excluded.individual_net_buy_amount,

                "foreign_net_buy_volume":
                    stmt.excluded.foreign_net_buy_volume,
                "institution_net_buy_volume":
                    stmt.excluded.institution_net_buy_volume,
                "individual_net_buy_volume":
                    stmt.excluded.individual_net_buy_volume,

                "foreign_holding_ratio":
                    stmt.excluded.foreign_holding_ratio,

                "source":
                    stmt.excluded.source,
                "raw_json":
                    stmt.excluded.raw_json,

                "updated_at":
                    datetime.utcnow(),
            },
        )

        self.db.execute(stmt)
        self.db.commit()

        return len(rows)
    
    def upsert_toss_stock_investor_flows(
        self,
        rows: list[dict],
    ) -> int:
        if not rows:
            return 0

        stmt = insert(
            StockInvestorFlow
        ).values(rows)

        stmt = stmt.on_conflict_do_update(
            constraint="uq_stock_investor_flow",
            set_={
                "foreign_net_buy_volume":
                    stmt.excluded.foreign_net_buy_volume,
                "institution_net_buy_volume":
                    stmt.excluded.institution_net_buy_volume,
                "individual_net_buy_volume":
                    stmt.excluded.individual_net_buy_volume,

                "foreign_holding_ratio":
                    stmt.excluded.foreign_holding_ratio,

                "source":
                    stmt.excluded.source,
                "raw_json":
                    stmt.excluded.raw_json,

                "updated_at":
                    datetime.utcnow(),
            },
        )

        self.db.execute(stmt)
        self.db.commit()

        return len(rows)
    
    def _upsert_trading_rows(
        self,
        *,
        model,
        constraint: str,
        rows: list[dict],
        fields: tuple[str, ...],
    ) -> int:
        if not rows:
            return 0

        stmt = insert(
            model
        ).values(rows)

        update_values = {
            field:
                stmt.excluded[field]
            for field in fields
        }

        update_values.update(
            {
                "source":
                    stmt.excluded.source,
                "raw_json":
                    stmt.excluded.raw_json,
                "updated_at":
                    datetime.utcnow(),
            }
        )

        stmt = stmt.on_conflict_do_update(
            constraint=constraint,
            set_=update_values,
        )

        self.db.execute(stmt)
        self.db.commit()

        return len(rows)

    def upsert_stock_program_trades(
        self,
        rows: list[dict],
    ) -> int:
        return self._upsert_trading_rows(
            model=StockProgramTrade,
            constraint="uq_stock_program_trade",
            rows=rows,
            fields=(
                "arbitrage_buy_volume",
                "arbitrage_sell_volume",
                "arbitrage_net_buy_volume",
                "non_arbitrage_buy_volume",
                "non_arbitrage_sell_volume",
                "non_arbitrage_net_buy_volume",
            ),
        )

    def upsert_stock_short_selling(
        self,
        rows: list[dict],
    ) -> int:
        return self._upsert_trading_rows(
            model=StockShortSelling,
            constraint="uq_stock_short_selling",
            rows=rows,
            fields=(
                "short_selling_volume",
                "short_selling_amount",
                "short_selling_volume_rate",
                "short_selling_amount_rate",
            ),
        )

    def upsert_stock_credit_trades(
        self,
        rows: list[dict],
    ) -> int:
        return self._upsert_trading_rows(
            model=StockCreditTrade,
            constraint="uq_stock_credit_trade",
            rows=rows,
            fields=(
                "margin_loan_new_quantity",
                "margin_loan_return_quantity",
                "margin_loan_balance_quantity",
                "margin_loan_balance_rate",
                "margin_loan_trading_rate",
                "stock_loan_new_quantity",
                "stock_loan_return_quantity",
                "stock_loan_balance_quantity",
                "stock_loan_balance_rate",
                "stock_loan_trading_rate",
            ),
        )

    def upsert_stock_securities_lending(
        self,
        rows: list[dict],
    ) -> int:
        return self._upsert_trading_rows(
            model=StockSecuritiesLending,
            constraint="uq_stock_securities_lending",
            rows=rows,
            fields=(
                "execution_quantity",
                "repayment_quantity",
                "balance_quantity",
                "balance_amount",
            ),
        )

    def upsert_stock_valuation_snapshots(
        self,
        rows: list[dict],
    ) -> int:
        if not rows:
            return 0

        stmt = insert(
            StockValuationSnapshot
        ).values(rows)

        stmt = stmt.on_conflict_do_update(
            constraint="uq_stock_valuation_snapshot",
            set_={
                "price":
                    func.coalesce(
                        stmt.excluded.price,
                        StockValuationSnapshot.price,
                    ),
                "market_cap":
                    func.coalesce(
                        stmt.excluded.market_cap,
                        StockValuationSnapshot.market_cap,
                    ),

                "per":
                    func.coalesce(
                        stmt.excluded.per,
                        StockValuationSnapshot.per,
                    ),
                "pbr":
                    func.coalesce(
                        stmt.excluded.pbr,
                        StockValuationSnapshot.pbr,
                    ),
                "eps":
                    func.coalesce(
                        stmt.excluded.eps,
                        StockValuationSnapshot.eps,
                    ),
                "bps":
                    func.coalesce(
                        stmt.excluded.bps,
                        StockValuationSnapshot.bps,
                    ),
                "dividend_yield":
                    func.coalesce(
                        stmt.excluded.dividend_yield,
                        StockValuationSnapshot.dividend_yield,
                    ),

                "sector_code":
                    func.coalesce(
                        stmt.excluded.sector_code,
                        StockValuationSnapshot.sector_code,
                    ),
                "sector_name":
                    func.coalesce(
                        stmt.excluded.sector_name,
                        StockValuationSnapshot.sector_name,
                    ),
                "sector_per":
                    func.coalesce(
                        stmt.excluded.sector_per,
                        StockValuationSnapshot.sector_per,
                    ),
                "sector_pbr":
                    func.coalesce(
                        stmt.excluded.sector_pbr,
                        StockValuationSnapshot.sector_pbr,
                    ),

                "market_per":
                    func.coalesce(
                        stmt.excluded.market_per,
                        StockValuationSnapshot.market_per,
                    ),
                "market_pbr":
                    func.coalesce(
                        stmt.excluded.market_pbr,
                        StockValuationSnapshot.market_pbr,
                    ),

                "source":
                    stmt.excluded.source,
                "raw_json":
                    stmt.excluded.raw_json,

                "updated_at":
                    datetime.utcnow(),
            },
        )

        self.db.execute(stmt)
        self.db.commit()

        return len(rows)

    def get_market_index_prices(
        self,
        *,
        index_code: str,
        start_date: date,
        end_date: date | None = None,
    ) -> list[MarketIndexPrice]:
        stmt = (
            select(MarketIndexPrice)
            .where(
                MarketIndexPrice.index_code
                == index_code,
                MarketIndexPrice.trade_date
                >= start_date,
            )
        )

        if end_date is not None:
            stmt = stmt.where(
                MarketIndexPrice.trade_date
                <= end_date
            )

        stmt = stmt.order_by(
            MarketIndexPrice.trade_date.asc()
        )

        return list(
            self.db.scalars(
                stmt
            ).all()
        )

    def get_market_investor_flows(
        self,
        *,
        market: str,
        start_date: date,
        end_date: date | None = None,
    ) -> list[MarketInvestorFlow]:
        stmt = (
            select(
                MarketInvestorFlow
            )
            .where(
                MarketInvestorFlow.market
                == market,
                MarketInvestorFlow.trade_date
                >= start_date,
            )
        )

        if end_date is not None:
            stmt = stmt.where(
                MarketInvestorFlow.trade_date
                <= end_date
            )

        stmt = stmt.order_by(
            MarketInvestorFlow.trade_date.asc()
        )

        return list(
            self.db.scalars(
                stmt
            ).all()
        )

    def get_macro_indicators(
        self,
        *,
        indicator_code: str,
        start_date: date,
        end_date: date | None = None,
    ) -> list[MacroIndicator]:
        stmt = (
            select(MacroIndicator)
            .where(
                MacroIndicator.indicator_code
                == indicator_code,
                MacroIndicator.observed_date
                >= start_date,
            )
        )

        if end_date is not None:
            stmt = stmt.where(
                MacroIndicator.observed_date
                <= end_date
            )

        stmt = stmt.order_by(
            MacroIndicator.observed_date.asc()
        )

        return list(
            self.db.scalars(
                stmt
            ).all()
        )

    def get_sector_index_prices(
        self,
        *,
        sector_code: str,
        market: str | None = None,
        start_date: date,
        end_date: date | None = None,
    ) -> list[SectorIndexPrice]:
        stmt = (
            select(SectorIndexPrice)
            .where(
                SectorIndexPrice.sector_code
                == sector_code,
                SectorIndexPrice.trade_date
                >= start_date,
            )
        )

        if market is not None:
            stmt = stmt.where(
                SectorIndexPrice.market
                == market
            )

        if end_date is not None:
            stmt = stmt.where(
                SectorIndexPrice.trade_date
                <= end_date
            )

        stmt = stmt.order_by(
            SectorIndexPrice.trade_date.asc()
        )

        return list(
            self.db.scalars(
                stmt
            ).all()
        )

    def get_stock_investor_flows(
        self,
        *,
        stock_code: str,
        start_date: date,
        end_date: date | None = None,
    ) -> list[StockInvestorFlow]:
        stmt = (
            select(StockInvestorFlow)
            .where(
                StockInvestorFlow.stock_code
                == stock_code,
                StockInvestorFlow.trade_date
                >= start_date,
            )
        )

        if end_date is not None:
            stmt = stmt.where(
                StockInvestorFlow.trade_date
                <= end_date
            )

        stmt = stmt.order_by(
            StockInvestorFlow.trade_date.asc()
        )

        return list(
            self.db.scalars(
                stmt
            ).all()
        )
    
    def get_latest_stock_investor_flows(
        self,
        *,
        stock_code: str,
        limit: int = 60,
    ) -> list[StockInvestorFlow]:
        stmt = (
            select(
                StockInvestorFlow
            )
            .where(
                StockInvestorFlow.stock_code
                == stock_code
            )
            .order_by(
                StockInvestorFlow.trade_date.desc()
            )
            .limit(
                limit
            )
        )

        rows = list(
            self.db.scalars(
                stmt
            ).all()
        )

        rows.reverse()

        return rows
    
    def _get_latest_trading_rows(
        self,
        *,
        model,
        stock_code: str,
        limit: int,
    ) -> list:
        stmt = (
            select(
                model
            )
            .where(
                model.stock_code
                == stock_code
            )
            .order_by(
                model.trade_date.desc()
            )
            .limit(
                limit
            )
        )

        rows = list(
            self.db.scalars(
                stmt
            ).all()
        )

        rows.reverse()

        return rows

    def get_latest_stock_program_trades(
        self,
        *,
        stock_code: str,
        limit: int = 60,
    ) -> list[StockProgramTrade]:
        return self._get_latest_trading_rows(
            model=StockProgramTrade,
            stock_code=stock_code,
            limit=limit,
        )

    def get_latest_stock_short_sellings(
        self,
        *,
        stock_code: str,
        limit: int = 60,
    ) -> list[StockShortSelling]:
        return self._get_latest_trading_rows(
            model=StockShortSelling,
            stock_code=stock_code,
            limit=limit,
        )

    def get_latest_stock_credit_trades(
        self,
        *,
        stock_code: str,
        limit: int = 60,
    ) -> list[StockCreditTrade]:
        return self._get_latest_trading_rows(
            model=StockCreditTrade,
            stock_code=stock_code,
            limit=limit,
        )

    def get_latest_stock_securities_lending(
        self,
        *,
        stock_code: str,
        limit: int = 60,
    ) -> list[StockSecuritiesLending]:
        return self._get_latest_trading_rows(
            model=StockSecuritiesLending,
            stock_code=stock_code,
            limit=limit,
        )

    def get_latest_valuation(
        self,
        stock_code: str,
    ) -> StockValuationSnapshot | None:
        stmt = (
            select(
                StockValuationSnapshot
            )
            .where(
                StockValuationSnapshot.stock_code
                == stock_code
            )
            .order_by(
                StockValuationSnapshot
                .snapshot_date
                .desc()
            )
            .limit(1)
        )

        return self.db.scalar(
            stmt
        )

    def get_latest_usable_valuation(
        self,
        stock_code: str,
    ) -> StockValuationSnapshot | None:
        stmt = (
            select(
                StockValuationSnapshot
            )
            .where(
                StockValuationSnapshot.stock_code
                == stock_code,
                or_(
                    StockValuationSnapshot.per
                    .is_not(None),
                    StockValuationSnapshot.pbr
                    .is_not(None),
                    StockValuationSnapshot.eps
                    .is_not(None),
                    StockValuationSnapshot.bps
                    .is_not(None),
                ),
            )
            .order_by(
                StockValuationSnapshot
                .snapshot_date
                .desc()
            )
            .limit(1)
        )

        return self.db.scalar(
            stmt
        )
    
    def get_latest_valuation_snapshot_date(
        self,
        *,
        on_or_before: date,
    ) -> date | None:
        stmt = (
            select(
                StockValuationSnapshot.snapshot_date
            )
            .where(
                StockValuationSnapshot.snapshot_date
                <= on_or_before
            )
            .order_by(
                StockValuationSnapshot
                .snapshot_date
                .desc()
            )
            .limit(1)
        )

        return self.db.scalar(
            stmt
        )
    
    def get_valuation_rows_for_date(
        self,
        *,
        snapshot_date: date,
    ) -> list[tuple]:
        stmt = (
            select(
                StockValuationSnapshot,
                Stock.market,
            )
            .join(
                Stock,
                Stock.code
                == StockValuationSnapshot.stock_code,
            )
            .where(
                StockValuationSnapshot.snapshot_date
                == snapshot_date,
                Stock.is_active.is_(True),
                StockValuationSnapshot.price
                .is_not(None),
                StockValuationSnapshot.price > 0,
                StockValuationSnapshot.market_cap
                .is_not(None),
                StockValuationSnapshot.market_cap > 0,
                Stock.market.in_(
                    [
                        "KOSPI",
                        "KOSDAQ",
                    ]
                ),
            )
            .order_by(
                StockValuationSnapshot
                .market_cap
                .desc(),
                Stock.code.asc(),
            )
        )

        return list(
            self.db.execute(
                stmt
            ).all()
        )

    def update_valuation_benchmarks(
        self,
        rows: list[dict],
    ) -> int:
        if not rows:
            return 0

        for row in rows:
            stmt = (
                update(
                    StockValuationSnapshot
                )
                .where(
                    StockValuationSnapshot.stock_code
                    == row["stock_code"],
                    StockValuationSnapshot.snapshot_date
                    == row["snapshot_date"],
                )
                .values(
                    sector_per=row.get(
                        "sector_per"
                    ),
                    sector_pbr=row.get(
                        "sector_pbr"
                    ),
                    market_per=row.get(
                        "market_per"
                    ),
                    market_pbr=row.get(
                        "market_pbr"
                    ),
                    raw_json=row.get(
                        "raw_json"
                    ),
                    updated_at=datetime.utcnow(),
                )
            )

            self.db.execute(
                stmt
            )

        self.db.commit()

        return len(rows)
    
    def delete_valuation_snapshots_not_in(
        self,
        *,
        snapshot_date: date,
        stock_codes: list[str],
    ) -> int:
        if not stock_codes:
            return 0

        stmt = (
            delete(
                StockValuationSnapshot
            )
            .where(
                StockValuationSnapshot.snapshot_date
                == snapshot_date,
                ~StockValuationSnapshot.stock_code.in_(
                    stock_codes
                ),
            )
        )

        result = self.db.execute(
            stmt
        )

        self.db.commit()

        return int(
            result.rowcount or 0
        )