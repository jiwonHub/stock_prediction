from datetime import date, datetime
from decimal import Decimal

from sqlalchemy import (
    and_,
    case,
    delete,
    func,
    or_,
    select,
    text,
)
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.orm import Session

from app.models.financial_metric import FinancialMetric
from app.models.financial_statement import FinancialStatement
from app.models.future import (
    RankingItem,
    RankingSnapshot,
)
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

    def get_latest_close_before(
        self,
        *,
        stock_code: str,
        trade_date: date,
    ) -> float | None:
        stmt = (
            select(StockPrice.close)
            .where(
                StockPrice.stock_code == stock_code,
                StockPrice.trade_date < trade_date,
            )
            .order_by(
                StockPrice.trade_date.desc()
            )
            .limit(1)
        )

        close = self.db.scalar(stmt)

        if close is None:
            return None

        return float(close)

    def get_latest_closes_before(
        self,
        *,
        stock_codes: list[str],
        trade_date: date,
    ) -> dict[str, float]:
        codes = list(
            dict.fromkeys(
                code
                for code in stock_codes
                if code
            )
        )

        if not codes:
            return {}

        ranked = (
            select(
                StockPrice.stock_code.label(
                    "stock_code"
                ),
                StockPrice.close.label(
                    "close"
                ),
                func.row_number()
                .over(
                    partition_by=(
                        StockPrice.stock_code
                    ),
                    order_by=(
                        StockPrice.trade_date.desc()
                    ),
                )
                .label("row_num"),
            )
            .where(
                StockPrice.stock_code.in_(codes),
                StockPrice.trade_date < trade_date,
            )
            .subquery()
        )

        stmt = (
            select(
                ranked.c.stock_code,
                ranked.c.close,
            )
            .where(
                ranked.c.row_num == 1
            )
        )

        return {
            str(stock_code): float(close)
            for stock_code, close
            in self.db.execute(stmt).all()
            if close is not None
        }

    def update_realtime_price(
        self,
        *,
        stock_code: str,
        current_price: float,
        change: float,
        change_rate: float,
    ) -> Stock:
        stock = self.get_stock(
            stock_code
        )

        if stock is None:
            raise ValueError(
                f"등록되지 않은 종목입니다: {stock_code}"
            )

        stock.current_price = current_price
        stock.change = change
        stock.change_rate = change_rate
        stock.last_price_at = datetime.utcnow()

        self.db.commit()
        self.db.refresh(stock)

        return stock

    def update_realtime_prices(
        self,
        rows: list[dict],
    ) -> set[str]:
        if not rows:
            return set()

        stock_codes = list(
            dict.fromkeys(
                str(row["stock_code"])
                for row in rows
            )
        )

        stmt = (
            select(Stock)
            .where(
                Stock.code.in_(stock_codes)
            )
        )

        stocks = {
            stock.code: stock
            for stock in self.db.scalars(
                stmt
            ).all()
        }

        updated_codes: set[str] = set()
        now = datetime.utcnow()

        for row in rows:
            stock_code = str(
                row["stock_code"]
            )

            stock = stocks.get(
                stock_code
            )

            if stock is None:
                continue

            stock.current_price = float(
                row["current_price"]
            )
            stock.change = float(
                row["change"]
            )
            stock.change_rate = float(
                row["change_rate"]
            )
            stock.last_price_at = now

            updated_codes.add(
                stock_code
            )

        self.db.commit()

        return updated_codes

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

        if market_cap is not None:
            stock.market_cap = market_cap

        if per is not None:
            stock.per = per

        if pbr is not None:
            stock.pbr = pbr

        if eps is not None:
            stock.eps = eps

        if bps is not None:
            stock.bps = bps

        if market:
            stock.market = market

        if sector_name:
            stock.sector_name = sector_name

        stock.last_price_at = datetime.utcnow()

        self.db.commit()
        self.db.refresh(stock)

        return stock

    def insert_missing_daily_prices(
        self,
        *,
        stock_code: str,
        rows: list[dict],
    ) -> int:
        if not rows:
            return 0

        chunk_size = 500
        inserted_total = 0

        for start in range(
            0,
            len(rows),
            chunk_size,
        ):
            chunk = rows[
                start:
                start + chunk_size
            ]

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
                for row in chunk
            ]

            stmt = insert(
                StockPrice
            ).values(
                values
            )

            stmt = (
                stmt.on_conflict_do_nothing(
                    constraint=(
                        "uq_stock_price_code_date"
                    ),
                )
            )

            stmt = stmt.returning(
                StockPrice.id
            )

            inserted_ids = (
                self.db.scalars(
                    stmt
                ).all()
            )

            inserted_total += len(
                inserted_ids
            )

        self.db.commit()

        return inserted_total

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
    
    def get_daily_price_page(
        self,
        *,
        stock_code: str,
        before: date | None = None,
        limit: int = 200,
    ) -> tuple[list[StockPrice], bool]:
        query_limit = (
            limit + 1
        )

        stmt = select(
            StockPrice
        ).where(
            StockPrice.stock_code
            == stock_code
        )

        if before is not None:
            stmt = stmt.where(
                StockPrice.trade_date
                < before
            )

        stmt = (
            stmt
            .order_by(
                StockPrice.trade_date.desc()
            )
            .limit(
                query_limit
            )
        )

        rows = list(
            self.db.scalars(
                stmt
            ).all()
        )

        has_more = (
            len(rows)
            > limit
        )

        rows = rows[
            :limit
        ]

        rows.reverse()

        return (
            rows,
            has_more,
        )
    
    def get_aggregated_price_page(
        self,
        *,
        stock_code: str,
        interval: str,
        before: date | None = None,
        limit: int = 200,
    ) -> tuple[list[dict], bool]:
        bucket_map = {
            "1w": "week",
            "1mo": "month",
            "1y": "year",
        }

        bucket = bucket_map.get(
            interval
        )

        if bucket is None:
            raise ValueError(
                f"지원하지 않는 interval입니다: "
                f"{interval}"
            )

        query_limit = (
            limit + 1
        )

        before_clause = ""

        params = {
            "stock_code": stock_code,
            "bucket": bucket,
            "query_limit": query_limit,
        }

        if before is not None:
            before_clause = (
                "WHERE bucket_start < :before"
            )

            params["before"] = before

        stmt = text(
            f"""
            WITH aggregated AS (
                SELECT
                    date_trunc(
                        :bucket,
                        trade_date
                    )::date AS bucket_start,
                    (
                        array_agg(
                            open
                            ORDER BY trade_date ASC
                        )
                    )[1] AS open,
                    MAX(high) AS high,
                    MIN(low) AS low,
                    (
                        array_agg(
                            close
                            ORDER BY trade_date DESC
                        )
                    )[1] AS close,
                    COALESCE(
                        SUM(volume),
                        0
                    ) AS volume
                FROM stock_prices
                WHERE stock_code = :stock_code
                GROUP BY
                    bucket_start
            )
            SELECT
                bucket_start,
                open,
                high,
                low,
                close,
                volume
            FROM aggregated
            {before_clause}
            ORDER BY
                bucket_start DESC
            LIMIT :query_limit
            """
        )

        result = (
            self.db.execute(
                stmt,
                params,
            )
            .mappings()
            .all()
        )

        has_more = (
            len(result)
            > limit
        )

        rows = [
            dict(row)
            for row in result[
                :limit
            ]
        ]

        rows.reverse()

        return (
            rows,
            has_more,
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

    def get_annual_financial_rows(
        self,
        *,
        stock_code: str,
        report_code: str = "11011",
    ) -> list[FinancialStatement]:
        stmt = (
            select(
                FinancialStatement
            )
            .where(
                FinancialStatement.stock_code
                == stock_code,
                FinancialStatement.report_code
                == report_code,
            )
            .order_by(
                FinancialStatement.business_year.desc(),
                FinancialStatement.fs_div.asc(),
                FinancialStatement.sj_div.asc(),
                FinancialStatement.id.asc(),
            )
        )

        return list(
            self.db.scalars(
                stmt
            ).all()
        )

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
    
    def get_stock_codes_for_market_context(
        self,
        *,
        limit: int,
    ) -> list[str]:
        stmt = (
            select(Stock.code)
            .where(
                Stock.is_active.is_(True),
                Stock.current_price.is_not(None),
                Stock.current_price > 0,
                Stock.market_cap.is_not(None),
                Stock.market_cap > 0,
                Stock.market.in_(
                    [
                        "KOSPI",
                        "KOSDAQ",
                    ]
                ),
            )
            .order_by(
                Stock.market_cap.desc(),
                Stock.current_price.desc(),
                Stock.code.asc(),
            )
            .limit(limit)
        )

        return list(
            self.db.scalars(
                stmt
            ).all()
        )
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

    def get_sector_financial_scores(
        self,
        *,
        sector_name: str,
        report_code: str = "11011",
    ) -> list[tuple[str, str, float]]:
        sector = sector_name.strip()

        if not sector:
            return []

        latest_year = (
            select(
                func.max(
                    FinancialMetric.business_year
                )
            )
            .where(
                FinancialMetric.stock_code
                == Stock.code,
                FinancialMetric.report_code
                == report_code,
            )
            .correlate(
                Stock
            )
            .scalar_subquery()
        )

        stmt = (
            select(
                Stock.code,
                Stock.name,
                FinancialMetric.financial_score,
            )
            .join(
                FinancialMetric,
                and_(
                    FinancialMetric.stock_code
                    == Stock.code,
                    FinancialMetric.business_year
                    == latest_year,
                    FinancialMetric.report_code
                    == report_code,
                ),
            )
            .where(
                Stock.sector_name
                == sector,
                Stock.is_active.is_(
                    True
                ),
            )
            .order_by(
                FinancialMetric.financial_score.desc(),
                Stock.market_cap.desc().nullslast(),
                Stock.name.asc(),
            )
        )

        return [
            (
                str(stock_code),
                str(stock_name),
                float(financial_score),
            )
            for (
                stock_code,
                stock_name,
                financial_score,
            )
            in self.db.execute(
                stmt
            ).all()
            if financial_score is not None
        ]

    def get_sector_financial_metrics(
        self,
        *,
        sector_name: str,
        business_year: str,
        report_code: str,
        fs_div: str,
    ) -> list[tuple[Stock, FinancialMetric]]:
        sector = sector_name.strip()

        if not sector:
            return []

        stmt = (
            select(
                Stock,
                FinancialMetric,
            )
            .join(
                FinancialMetric,
                FinancialMetric.stock_code
                == Stock.code,
            )
            .where(
                Stock.sector_name == sector,
                Stock.is_active.is_(True),
                FinancialMetric.business_year
                == business_year,
                FinancialMetric.report_code
                == report_code,
                FinancialMetric.fs_div
                == fs_div,
            )
            .order_by(
                Stock.market_cap.desc().nullslast(),
                Stock.name.asc(),
            )
        )

        return [
            (stock, metric)
            for stock, metric
            in self.db.execute(stmt).all()
        ]

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

    def get_latest_ranking_snapshot_items(
        self,
        *,
        ranking_version: str,
        horizon_days: int,
        universe: str,
        limit: int,
    ) -> tuple[
        RankingSnapshot | None,
        list[RankingItem],
    ]:
        snapshot = self.db.scalar(
            select(
                RankingSnapshot
            )
            .where(
                RankingSnapshot.ranking_version
                == ranking_version,
                RankingSnapshot.horizon_days
                == horizon_days,
                RankingSnapshot.universe
                == universe,
            )
            .order_by(
                RankingSnapshot.as_of_date.desc(),
                RankingSnapshot.id.desc(),
            )
            .limit(1)
        )

        if snapshot is None:
            return (
                None,
                [],
            )

        items = list(
            self.db.scalars(
                select(
                    RankingItem
                )
                .where(
                    RankingItem.snapshot_id
                    == snapshot.id
                )
                .order_by(
                    RankingItem.rank.asc()
                )
                .limit(
                    limit
                )
            ).all()
        )

        return (
            snapshot,
            items,
        )

    def get_latest_ranking_snapshot_item_by_stock_code(
        self,
        *,
        stock_code: str,
        ranking_version: str,
        horizon_days: int,
        universe: str,
    ) -> tuple[
        RankingSnapshot | None,
        RankingItem | None,
    ]:
        snapshot = self.db.scalar(
            select(
                RankingSnapshot
            )
            .where(
                RankingSnapshot.ranking_version
                == ranking_version,
                RankingSnapshot.horizon_days
                == horizon_days,
                RankingSnapshot.universe
                == universe,
            )
            .order_by(
                RankingSnapshot.as_of_date.desc(),
                RankingSnapshot.id.desc(),
            )
            .limit(1)
        )

        if snapshot is None:
            return (
                None,
                None,
            )

        item = self.db.scalar(
            select(
                RankingItem
            )
            .where(
                RankingItem.snapshot_id
                == snapshot.id,
                RankingItem.stock_code
                == stock_code,
            )
            .limit(1)
        )

        return (
            snapshot,
            item,
        )

    def get_ranking_context_by_stock_codes(
        self,
        stock_codes: list[str],
    ) -> list[
        tuple[
            Stock,
            FinancialMetric | None,
        ]
    ]:
        if not stock_codes:
            return []

        # 종목별 최신 연도를 제각각 사용하면
        # 2025년 종목과 2024년 종목을 같은 percentile에서
        # 비교하게 됩니다.
        #
        # 후보군의 70% 이상이 확보된 가장 최신 사업연도를
        # 랭킹 공통 기준연도로 사용합니다.
        minimum_coverage_count = max(
            3,
            int(
                len(
                    stock_codes
                )
                * 0.70
            ),
        )

        year_rows = list(
            self.db.execute(
                select(
                    FinancialMetric.business_year,
                    func.count(
                        func.distinct(
                            FinancialMetric.stock_code
                        )
                    ),
                )
                .where(
                    FinancialMetric.stock_code.in_(
                        stock_codes
                    ),

                    FinancialMetric.report_code
                    == "11011",
                )
                .group_by(
                    FinancialMetric.business_year
                )
                .order_by(
                    FinancialMetric.business_year.desc()
                )
            ).all()
        )

        target_business_year = None

        for (
            business_year,
            stock_count,
        ) in year_rows:
            if (
                int(
                    stock_count
                    or 0
                )
                >= minimum_coverage_count
            ):
                target_business_year = (
                    business_year
                )

                break

        # 70%를 만족하는 연도가 없으면
        # 데이터가 존재하는 가장 최신 연도를 사용합니다.
        if (
            target_business_year
            is None
            and year_rows
        ):
            target_business_year = (
                year_rows[
                    0
                ][
                    0
                ]
            )

        if target_business_year is None:
            return [
                (
                    stock,
                    None,
                )
                for stock
                in self.db.scalars(
                    select(
                        Stock
                    )
                    .where(
                        Stock.code.in_(
                            stock_codes
                        )
                    )
                ).all()
            ]

        ranked_metric_rows = (
            select(
                FinancialMetric.id.label(
                    "metric_id"
                ),

                FinancialMetric.stock_code.label(
                    "stock_code"
                ),

                func.row_number()
                .over(
                    partition_by=(
                        FinancialMetric.stock_code
                    ),
                    order_by=(
                        case(
                            (
                                FinancialMetric.fs_div
                                == "CFS",
                                0,
                            ),
                            (
                                FinancialMetric.fs_div
                                == "OFS",
                                1,
                            ),
                            else_=2,
                        ),
                        FinancialMetric.id.desc(),
                    ),
                )
                .label(
                    "row_number"
                ),
            )
            .where(
                FinancialMetric.stock_code.in_(
                    stock_codes
                ),

                FinancialMetric.business_year
                == target_business_year,

                FinancialMetric.report_code
                == "11011",
            )
            .subquery()
        )

        selected_metric = (
            select(
                ranked_metric_rows.c.metric_id,
                ranked_metric_rows.c.stock_code,
            )
            .where(
                ranked_metric_rows.c.row_number
                == 1
            )
            .subquery()
        )

        stmt = (
            select(
                Stock,
                FinancialMetric,
            )
            .outerjoin(
                selected_metric,
                selected_metric.c.stock_code
                == Stock.code,
            )
            .outerjoin(
                FinancialMetric,
                FinancialMetric.id
                == selected_metric.c.metric_id,
            )
            .where(
                Stock.code.in_(
                    stock_codes
                )
            )
        )

        return [
            (
                stock,
                metric,
            )
            for (
                stock,
                metric,
            )
            in self.db.execute(
                stmt
            ).all()
        ]

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
    
    def update_market_sector_metadata(
        self,
        rows: list[dict],
    ) -> int:
        if not rows:
            return 0

        updated = 0

        for row in rows:
            stock = self.get_stock(
                row["stock_code"]
            )

            if stock is None:
                continue

            market = row.get("market")
            sector_code = row.get(
                "sector_code"
            )

            if market:
                stock.market = market

            if sector_code:
                stock.sector_code = (
                    sector_code
                )

            updated += 1

        self.db.commit()

        return updated
