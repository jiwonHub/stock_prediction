from datetime import date, datetime
from decimal import Decimal

from sqlalchemy import and_, delete, func, or_, select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.orm import Session

from app.models.financial_metric import FinancialMetric
from app.models.financial_statement import FinancialStatement
from app.models.stock import Stock
from app.models.stock_price import StockPrice
from app.models.stock_prediction import StockPrediction


class StockRepository:
    def __init__(self, db: Session):
        self.db = db

    def get_stock(
        self,
        stock_code: str,
    ) -> Stock | None:
        return self.db.get(
            Stock,
            stock_code,
        )

    def search(
        self,
        query: str,
        *,
        limit: int = 30,
    ) -> list[Stock]:
        keyword = query.strip()

        if not keyword:
            return []

        stmt = (
            select(Stock)
            .where(
                or_(
                    Stock.code.contains(keyword),
                    Stock.name.ilike(f"%{keyword}%"),
                )
            )
            .order_by(
                Stock.market_cap.desc().nullslast(),
                Stock.name.asc(),
            )
            .limit(limit)
        )

        return list(
            self.db.scalars(stmt).all()
        )

    def upsert_stocks(
        self,
        rows: list[dict],
    ) -> int:
        if not rows:
            return 0

        stmt = insert(Stock).values(rows)

        stmt = stmt.on_conflict_do_update(
            index_elements=[Stock.code],
            set_={
                "corp_code": stmt.excluded.corp_code,
                "name": stmt.excluded.name,
                "english_name": stmt.excluded.english_name,
                "modified_at": stmt.excluded.modified_at,
                "updated_at": datetime.utcnow(),
            },
        )

        self.db.execute(stmt)
        self.db.commit()

        return len(rows)

    def update_quote(
        self,
        *,
        stock_code: str,
        current_price: float,
        change: float,
        change_rate: float,
        market_cap: float | None,
        per: float | None,
        pbr: float | None,
        eps: float | None,
        bps: float | None,
        market: str | None,
        sector_name: str | None,
    ) -> Stock:
        stock = self.get_stock(
            stock_code,
        )

        if stock is None:
            stock = Stock(
                code=stock_code,
                name=stock_code,
                market=market or "KRX",
            )
            self.db.add(stock)

        stock.current_price = current_price
        stock.change = change
        stock.change_rate = change_rate
        stock.market_cap = market_cap
        stock.per = per
        stock.pbr = pbr
        stock.eps = eps
        stock.bps = bps

        if market:
            stock.market = market

        if sector_name:
            stock.sector_name = sector_name

        stock.last_price_at = datetime.utcnow()

        self.db.commit()
        self.db.refresh(stock)

        return stock

    def upsert_daily_prices(
        self,
        *,
        stock_code: str,
        rows: list[dict],
    ) -> int:
        if not rows:
            return 0

        values = [
            {
                "stock_code": stock_code,
                "trade_date": row["trade_date"],
                "open": row["open"],
                "high": row["high"],
                "low": row["low"],
                "close": row["close"],
                "volume": row["volume"],
            }
            for row in rows
        ]

        stmt = insert(StockPrice).values(
            values
        )

        stmt = stmt.on_conflict_do_update(
            constraint="uq_stock_price_code_date",
            set_={
                "open": stmt.excluded.open,
                "high": stmt.excluded.high,
                "low": stmt.excluded.low,
                "close": stmt.excluded.close,
                "volume": stmt.excluded.volume,
            },
        )

        self.db.execute(stmt)
        self.db.commit()

        return len(values)

    def get_daily_prices(
        self,
        *,
        stock_code: str,
        start_date: date,
    ) -> list[StockPrice]:
        stmt = (
            select(StockPrice)
            .where(
                StockPrice.stock_code == stock_code,
                StockPrice.trade_date >= start_date,
            )
            .order_by(
                StockPrice.trade_date.asc()
            )
        )

        return list(
            self.db.scalars(stmt).all()
        )

    def replace_financials(
        self,
        *,
        stock_code: str,
        business_year: str,
        report_code: str,
        fs_div: str,
        rows: list[dict],
    ) -> int:
        unique_rows: dict[tuple[str, ...], dict] = {}

        for row in rows:
            key = (
                str(row.get("stock_code", stock_code)),
                str(row.get("business_year", business_year)),
                str(row.get("report_code", report_code)),
                str(row.get("fs_div", fs_div)),
                str(row.get("sj_div", "")),
                str(row.get("account_id", "")),
                str(row.get("account_nm", "")),
                str(row.get("account_detail", "")),
            )

            unique_rows[key] = row

        values = list(unique_rows.values())

        self.db.execute(
            delete(FinancialStatement).where(
                FinancialStatement.stock_code == stock_code,
                FinancialStatement.business_year == business_year,
                FinancialStatement.report_code == report_code,
                FinancialStatement.fs_div == fs_div,
            )
        )

        if values:
            stmt = insert(FinancialStatement).values(values)

            stmt = stmt.on_conflict_do_nothing(
                constraint="uq_financial_statement_row"
            )

            self.db.execute(stmt)

        self.db.commit()

        return len(values)

    def get_financial_rows(
        self,
        *,
        stock_code: str,
        business_year: str,
        report_code: str,
    ) -> list[FinancialStatement]:
        stmt = (
            select(FinancialStatement)
            .where(
                FinancialStatement.stock_code == stock_code,
                FinancialStatement.business_year == business_year,
                FinancialStatement.report_code == report_code,
            )
            .order_by(
                FinancialStatement.fs_div.asc(),
                FinancialStatement.sj_div.asc(),
                FinancialStatement.id.asc(),
            )
        )
        return list(self.db.scalars(stmt).all())

    def get_stock_codes_for_analysis(
        self,
        *,
        limit: int,
    ) -> list[str]:
        stmt = (
            select(Stock.code)
            .order_by(
                Stock.market_cap.desc().nullslast(),
                Stock.current_price.desc().nullslast(),
                Stock.code.asc(),
            )
            .limit(limit)
        )
        return list(self.db.scalars(stmt).all())

    def get_stock_codes_with_financials(
        self,
        *,
        business_year: str,
        report_code: str,
        limit: int,
    ) -> list[str]:
        stmt = (
            select(FinancialStatement.stock_code)
            .where(
                FinancialStatement.business_year == business_year,
                FinancialStatement.report_code == report_code,
            )
            .distinct()
            .order_by(FinancialStatement.stock_code.asc())
            .limit(limit)
        )
        return list(self.db.scalars(stmt).all())

    def upsert_financial_metric(
        self,
        values: dict,
    ) -> FinancialMetric:
        stmt = insert(FinancialMetric).values(values)
        update_values = {
            key: getattr(stmt.excluded, key)
            for key in values
            if key not in {"stock_code", "business_year", "report_code", "fs_div"}
        }
        update_values["updated_at"] = datetime.utcnow()

        stmt = stmt.on_conflict_do_update(
            constraint="uq_financial_metric_snapshot",
            set_=update_values,
        ).returning(FinancialMetric.id)

        metric_id = self.db.execute(stmt).scalar_one()
        self.db.commit()
        return self.db.get(FinancialMetric, metric_id)

    def get_financial_metric(
        self,
        *,
        stock_code: str,
        business_year: str,
        report_code: str,
    ) -> FinancialMetric | None:
        stmt = (
            select(FinancialMetric)
            .where(
                FinancialMetric.stock_code == stock_code,
                FinancialMetric.business_year == business_year,
                FinancialMetric.report_code == report_code,
            )
            .order_by(FinancialMetric.updated_at.desc())
            .limit(1)
        )
        return self.db.scalar(stmt)

    def get_ranking_candidates_with_metrics(
        self,
        *,
        limit: int,
    ) -> list[tuple[Stock, FinancialMetric | None]]:
        latest_year = (
            select(func.max(FinancialMetric.business_year))
            .where(
                FinancialMetric.stock_code == Stock.code,
                FinancialMetric.report_code == "11011",
            )
            .correlate(Stock)
            .scalar_subquery()
        )

        stmt = (
            select(Stock, FinancialMetric)
            .join(
                FinancialMetric,
                and_(
                    FinancialMetric.stock_code == Stock.code,
                    FinancialMetric.business_year == latest_year,
                    FinancialMetric.report_code == "11011",
                ),
            )
            .order_by(
                FinancialMetric.financial_score.desc(),
                FinancialMetric.data_completeness.desc(),
                Stock.market_cap.desc().nullslast(),
                Stock.name.asc(),
            )
            .limit(limit)
        )

        return [(stock, metric) for stock, metric in self.db.execute(stmt).all()]


    def upsert_stock_prediction(
        self,
        values: dict,
    ) -> StockPrediction:
        stmt = insert(StockPrediction).values(values)
        update_values = {
            key: getattr(stmt.excluded, key)
            for key in values
            if key not in {"stock_code", "horizon_days"}
        }
        update_values["updated_at"] = datetime.utcnow()

        stmt = stmt.on_conflict_do_update(
            constraint="uq_stock_prediction_horizon",
            set_=update_values,
        ).returning(StockPrediction.id)

        prediction_id = self.db.execute(stmt).scalar_one()
        self.db.commit()
        return self.db.get(StockPrediction, prediction_id)

    def get_stock_prediction(
        self,
        *,
        stock_code: str,
        horizon_days: int = 5,
    ) -> StockPrediction | None:
        stmt = (
            select(StockPrediction)
            .where(
                StockPrediction.stock_code == stock_code,
                StockPrediction.horizon_days == horizon_days,
            )
            .limit(1)
        )
        return self.db.scalar(stmt)
    
    def get_stock_codes_with_predictions(
        self,
        *,
        horizon_days: int = 5,
        limit: int = 100,
    ) -> list[str]:
        stmt = (
            select(
                StockPrediction.stock_code
            )
            .where(
                StockPrediction.horizon_days
                == horizon_days
            )
            .order_by(
                StockPrediction
                .ml_score
                .desc(),
                StockPrediction
                .updated_at
                .desc(),
            )
            .limit(limit)
        )

        return list(
            self.db.scalars(
                stmt
            ).all()
        )

    def get_stock_codes_for_ml(
        self,
        *,
        limit: int,
    ) -> list[str]:
        price_count = (
            select(func.count(StockPrice.id))
            .where(StockPrice.stock_code == Stock.code)
            .correlate(Stock)
            .scalar_subquery()
        )

        stmt = (
            select(Stock.code)
            .order_by(
                price_count.desc(),
                Stock.market_cap.desc().nullslast(),
                Stock.current_price.desc().nullslast(),
                Stock.code.asc(),
            )
            .limit(limit)
        )
        return list(self.db.scalars(stmt).all())

    def get_phase4_ranking_candidates(
        self,
        *,
        horizon_days: int = 5,
        scan_limit: int = 1000,
    ) -> list[tuple[Stock, FinancialMetric | None, StockPrediction | None]]:
        latest_year = (
            select(func.max(FinancialMetric.business_year))
            .where(
                FinancialMetric.stock_code == Stock.code,
                FinancialMetric.report_code == "11011",
            )
            .correlate(Stock)
            .scalar_subquery()
        )

        stmt = (
            select(Stock, FinancialMetric, StockPrediction)
            .outerjoin(
                FinancialMetric,
                and_(
                    FinancialMetric.stock_code == Stock.code,
                    FinancialMetric.business_year == latest_year,
                    FinancialMetric.report_code == "11011",
                ),
            )
            .outerjoin(
                StockPrediction,
                and_(
                    StockPrediction.stock_code == Stock.code,
                    StockPrediction.horizon_days == horizon_days,
                ),
            )
            .where(
                or_(
                    FinancialMetric.id.is_not(None),
                    StockPrediction.id.is_not(None),
                )
            )
            .order_by(
                StockPrediction.ml_score.desc().nullslast(),
                FinancialMetric.financial_score.desc().nullslast(),
                Stock.market_cap.desc().nullslast(),
                Stock.name.asc(),
            )
            .limit(scan_limit)
        )

        return [
            (stock, metric, prediction)
            for stock, metric, prediction in self.db.execute(stmt).all()
        ]

    def get_ranking_candidates(
        self,
        *,
        limit: int,
    ) -> list[Stock]:
        stmt = (
            select(Stock)
            .where(
                Stock.current_price.is_not(None)
            )
            .order_by(
                Stock.market_cap.desc().nullslast(),
                Stock.name.asc(),
            )
            .limit(limit)
        )

        return list(
            self.db.scalars(stmt).all()
        )
