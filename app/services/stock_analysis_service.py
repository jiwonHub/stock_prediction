from __future__ import annotations

from datetime import date, timedelta

from sqlalchemy.orm import Session

from app.repositories.market_data_repository import (
    MarketDataRepository,
)
from app.repositories.stock_repository import (
    StockRepository,
)
from app.services.benchmark_interpretation_service import (
    BenchmarkInterpretationService,
)
from app.services.financial_analysis_service import (
    FinancialAnalysisService,
)
from app.services.long_term_financial_service import (
    LongTermFinancialService,
)
from app.services.investor_flow_interpretation_service import (
    InvestorFlowInterpretationService,
)
from app.services.market_context_service import (
    MarketContextService,
)
from app.services.market_data_service import (
    MarketDataService,
)
from app.services.market_regime_service import (
    MarketRegimeService,
)
from app.services.ranking_explanation_service import (
    RankingExplanationService,
)
from app.services.stock_analysis_brief_service import (
    StockAnalysisBriefService,
)
from app.services.technical_analysis_service import (
    TechnicalAnalysisService,
)
from app.services.technical_interpretation_service import (
    TechnicalInterpretationService,
)

class StockAnalysisService:
    def __init__(
        self,
        db: Session,
    ):
        self.db = db

        self.market_data_repository = (
            MarketDataRepository(
                db
            )
        )

        self.stock_repository = (
            StockRepository(
                db
            )
        )

    @staticmethod
    def _calculate_return(
        latest_close: float | None,
        base_close: float | None,
    ) -> float | None:
        if (
            latest_close is None
            or base_close is None
            or base_close <= 0
        ):
            return None

        return (
            (
                float(latest_close)
                / float(base_close)
                - 1.0
            )
            * 100.0
        )

    def _build_benchmark_performance(
        self,
        *,
        stock_code: str,
    ) -> dict:
        stock = (
            self.stock_repository
            .get_stock(
                stock_code
            )
        )

        if stock is None:
            return {
                "available": False,
                "market": None,
                "benchmark": None,
                "benchmarkIndexCode": None,
                "asOfDate": None,
                "stockClose": None,
                "benchmarkClose": None,
                "oneDay": None,
                "fiveDays": None,
                "twentyDays": None,
            }

        market = (
            stock.market
            or ""
        ).upper()

        benchmark_map = {
            "KOSPI": (
                "0001",
                "KOSPI",
            ),
            "KOSDAQ": (
                "1001",
                "KOSDAQ",
            ),
        }

        benchmark_info = (
            benchmark_map.get(
                market
            )
        )

        if benchmark_info is None:
            return {
                "available": False,
                "market": market,
                "benchmark": None,
                "benchmarkIndexCode": None,
                "asOfDate": None,
                "stockClose": None,
                "benchmarkClose": None,
                "oneDay": None,
                "fiveDays": None,
                "twentyDays": None,
            }

        (
            index_code,
            benchmark_name,
        ) = benchmark_info

        start_date = (
            date.today()
            - timedelta(
                days=90
            )
        )

        stock_rows = (
            self.stock_repository
            .get_daily_prices(
                stock_code=stock_code,
                start_date=start_date,
            )
        )

        benchmark_rows = (
            self.market_data_repository
            .get_market_index_prices(
                index_code=index_code,
                start_date=start_date,
                end_date=date.today(),
            )
        )

        stock_close_map = {
            row.trade_date:
                float(
                    row.close
                )
            for row in stock_rows
            if row.close is not None
        }

        benchmark_close_map = {
            row.trade_date:
                float(
                    row.close
                )
            for row in benchmark_rows
            if row.close is not None
        }

        common_dates = sorted(
            set(
                stock_close_map
            )
            & set(
                benchmark_close_map
            )
        )

        if not common_dates:
            return {
                "available": False,
                "market": market,
                "benchmark": benchmark_name,
                "benchmarkIndexCode": index_code,
                "asOfDate": None,
                "stockClose": None,
                "benchmarkClose": None,
                "oneDay": None,
                "fiveDays": None,
                "twentyDays": None,
            }

        latest_date = (
            common_dates[-1]
        )

        latest_stock_close = (
            stock_close_map[
                latest_date
            ]
        )

        latest_benchmark_close = (
            benchmark_close_map[
                latest_date
            ]
        )

        def build_period(
            sessions: int,
        ) -> dict:
            if len(
                common_dates
            ) <= sessions:
                return {
                    "sessionCount":
                        sessions,
                    "baseDate":
                        None,
                    "stockReturn":
                        None,
                    "benchmarkReturn":
                        None,
                    "excessReturn":
                        None,
                }

            base_date = (
                common_dates[
                    -(sessions + 1)
                ]
            )

            stock_return = (
                self._calculate_return(
                    latest_stock_close,
                    stock_close_map[
                        base_date
                    ],
                )
            )

            benchmark_return = (
                self._calculate_return(
                    latest_benchmark_close,
                    benchmark_close_map[
                        base_date
                    ],
                )
            )

            excess_return = None

            if (
                stock_return is not None
                and benchmark_return
                is not None
            ):
                excess_return = (
                    stock_return
                    - benchmark_return
                )

            return {
                "sessionCount":
                    sessions,
                "baseDate":
                    base_date.isoformat(),
                "stockReturn":
                    stock_return,
                "benchmarkReturn":
                    benchmark_return,
                "excessReturn":
                    excess_return,
            }

        return {
            "available": True,
            "market": market,
            "benchmark":
                benchmark_name,
            "benchmarkIndexCode":
                index_code,
            "asOfDate":
                latest_date.isoformat(),
            "stockClose":
                latest_stock_close,
            "benchmarkClose":
                latest_benchmark_close,
            "oneDay":
                build_period(
                    1
                ),
            "fiveDays":
                build_period(
                    5
                ),
            "twentyDays":
                build_period(
                    20
                ),
        }

    def _build_relative_strength(
        self,
        *,
        stock_code: str,
    ) -> dict:
        try:
            result = (
                MarketDataService(
                    self.db
                )
                .calculate_stock_relative_strength(
                    stock_code
                )
            )

        except ValueError:
            return {
                "available":
                    False,

                "market":
                    None,

                "sectorCode":
                    None,

                "sectorName":
                    None,

                "asOfDate":
                    None,

                "twentyDays":
                    None,

                "sixtyDays":
                    None,
            }

        def build_period(
            sessions: int,
        ) -> dict:
            suffix = (
                f"{sessions}d"
            )

            return {
                "sessionCount":
                    sessions,

                "startDate":
                    result.get(
                        f"start_date_{suffix}"
                    ),

                "stockReturn":
                    result.get(
                        f"stock_return_{suffix}_pct"
                    ),

                "sectorReturn":
                    result.get(
                        f"sector_return_{suffix}_pct"
                    ),

                "marketReturn":
                    result.get(
                        f"market_return_{suffix}_pct"
                    ),

                "sectorExcessReturn":
                    result.get(
                        f"sector_excess_{suffix}_pct"
                    ),

                "marketExcessReturn":
                    result.get(
                        f"market_excess_{suffix}_pct"
                    ),
            }

        return {
            "available":
                True,

            "market":
                result.get(
                    "market"
                ),

            "sectorCode":
                result.get(
                    "sector_code"
                ),

            "sectorName":
                result.get(
                    "sector_name"
                ),

            "asOfDate":
                result.get(
                    "as_of_date"
                ),

            "twentyDays":
                build_period(
                    20
                ),

            "sixtyDays":
                build_period(
                    60
                ),
        }

    def _build_market_regime(
        self,
        *,
        stock_code: str,
    ) -> dict:
        stock = (
            self.stock_repository
            .get_stock(
                stock_code
            )
        )

        market = (
            (
                stock.market
                if stock
                else None
            )
            or ""
        ).upper()

        if market not in (
            "KOSPI",
            "KOSDAQ",
        ):
            return {
                "available": False,
                "market": (
                    market
                    or None
                ),
                "featureDate": None,
                "state": None,
                "label": None,
                "description": (
                    "비교 가능한 시장 레짐 "
                    "정보가 없습니다."
                ),
                "trend": None,
                "volatility": None,
                "risk": None,
            }

        try:
            return (
                MarketRegimeService(
                    self.db
                )
                .get_market_state(
                    market=market
                )
            )

        except ValueError as e:
            return {
                "available": False,
                "market": market,
                "featureDate": None,
                "state": None,
                "label": None,
                "description": str(
                    e
                ),
                "trend": None,
                "volatility": None,
                "risk": None,
            }

    @staticmethod
    def _sum_flow(
        rows,
        field_name: str,
    ) -> int:
        return int(
            sum(
                int(
                    getattr(
                        row,
                        field_name,
                    )
                    or 0
                )
                for row in rows
            )
        )

    def _build_investor_flow(
        self,
        *,
        stock_code: str,
    ) -> dict:
        rows = (
            self.market_data_repository
            .get_latest_stock_investor_flows(
                stock_code=stock_code,
                limit=60,
            )
        )

        rows = sorted(
            rows,
            key=lambda row:
                row.trade_date,
        )

        rows_5d = rows[
            -5:
        ]

        rows_20d = rows[
            -20:
        ]

        rows_60d = rows[
            -60:
        ]

        def summarize(
            target_rows,
        ) -> dict:
            foreign = self._sum_flow(
                target_rows,
                "foreign_net_buy_volume",
            )

            institution = self._sum_flow(
                target_rows,
                "institution_net_buy_volume",
            )

            individual = self._sum_flow(
                target_rows,
                "individual_net_buy_volume",
            )

            smart_money = (
                foreign
                + institution
            )

            return {
                "foreignNetBuyVolume":
                    foreign,

                "institutionNetBuyVolume":
                    institution,

                "individualNetBuyVolume":
                    individual,

                "foreignInstitutionNetBuyVolume":
                    smart_money,
            }

        latest = (
            rows[
                -1
            ]
            if rows
            else None
        )

        return {
            "available":
                bool(
                    rows
                ),

            "latestDate":
                (
                    latest.trade_date.isoformat()
                    if latest
                    else None
                ),

            "foreignHoldingRatio":
                (
                    float(
                        latest.foreign_holding_ratio
                    )
                    if (
                        latest
                        and latest.foreign_holding_ratio
                        is not None
                    )
                    else None
                ),

            "fiveDays":
                summarize(
                    rows_5d
                ),

            "twentyDays":
                summarize(
                    rows_20d
                ),

            "sixtyDays":
                summarize(
                    rows_60d
                ),
        }

    @staticmethod
    def _sum_int_field(
        rows,
        field_name: str,
    ) -> int:
        return sum(
            int(
                getattr(
                    row,
                    field_name,
                )
                or 0
            )
            for row in rows
        )

    @staticmethod
    def _average_float_field(
        rows,
        field_name: str,
    ) -> float | None:
        values = [
            float(value)
            for row in rows
            if (
                value := getattr(
                    row,
                    field_name,
                )
            )
            is not None
        ]

        if not values:
            return None

        return (
            sum(values)
            / len(values)
        )

    @staticmethod
    def _balance_change(
        rows,
        field_name: str,
    ) -> int:
        if len(rows) < 2:
            return 0

        first = int(
            getattr(
                rows[0],
                field_name,
            )
            or 0
        )

        latest = int(
            getattr(
                rows[-1],
                field_name,
            )
            or 0
        )

        return (
            latest
            - first
        )

    def _build_trading_signals(
        self,
        *,
        stock_code: str,
    ) -> dict:
        program_rows = (
            self.market_data_repository
            .get_latest_stock_program_trades(
                stock_code=stock_code,
                limit=60,
            )
        )

        short_rows = (
            self.market_data_repository
            .get_latest_stock_short_sellings(
                stock_code=stock_code,
                limit=60,
            )
        )

        credit_rows = (
            self.market_data_repository
            .get_latest_stock_credit_trades(
                stock_code=stock_code,
                limit=60,
            )
        )

        lending_rows = (
            self.market_data_repository
            .get_latest_stock_securities_lending(
                stock_code=stock_code,
                limit=60,
            )
        )

        def summarize_program(
            rows,
        ) -> dict:
            arbitrage = (
                self._sum_int_field(
                    rows,
                    "arbitrage_net_buy_volume",
                )
            )

            non_arbitrage = (
                self._sum_int_field(
                    rows,
                    "non_arbitrage_net_buy_volume",
                )
            )

            return {
                "arbitrageNetBuyVolume":
                    arbitrage,

                "nonArbitrageNetBuyVolume":
                    non_arbitrage,

                "totalNetBuyVolume":
                    (
                        arbitrage
                        + non_arbitrage
                    ),
            }

        def summarize_short(
            rows,
        ) -> dict:
            latest = (
                rows[-1]
                if rows
                else None
            )

            return {
                "shortSellingVolume":
                    self._sum_int_field(
                        rows,
                        "short_selling_volume",
                    ),

                "shortSellingAmount":
                    self._sum_int_field(
                        rows,
                        "short_selling_amount",
                    ),

                "averageVolumeRate":
                    self._average_float_field(
                        rows,
                        "short_selling_volume_rate",
                    ),

                "averageAmountRate":
                    self._average_float_field(
                        rows,
                        "short_selling_amount_rate",
                    ),

                "latestVolumeRate":
                    (
                        float(
                            latest.short_selling_volume_rate
                        )
                        if (
                            latest
                            and latest.short_selling_volume_rate
                            is not None
                        )
                        else None
                    ),

                "latestAmountRate":
                    (
                        float(
                            latest.short_selling_amount_rate
                        )
                        if (
                            latest
                            and latest.short_selling_amount_rate
                            is not None
                        )
                        else None
                    ),
            }

        def summarize_credit_side(
            rows,
            *,
            prefix: str,
        ) -> dict:
            new_field = (
                f"{prefix}_new_quantity"
            )

            return_field = (
                f"{prefix}_return_quantity"
            )

            balance_field = (
                f"{prefix}_balance_quantity"
            )

            balance_rate_field = (
                f"{prefix}_balance_rate"
            )

            trading_rate_field = (
                f"{prefix}_trading_rate"
            )

            new_quantity = (
                self._sum_int_field(
                    rows,
                    new_field,
                )
            )

            return_quantity = (
                self._sum_int_field(
                    rows,
                    return_field,
                )
            )

            latest = (
                rows[-1]
                if rows
                else None
            )

            return {
                "newQuantity":
                    new_quantity,

                "returnQuantity":
                    return_quantity,

                "netNewQuantity":
                    (
                        new_quantity
                        - return_quantity
                    ),

                "latestBalanceQuantity":
                    (
                        int(
                            getattr(
                                latest,
                                balance_field,
                            )
                            or 0
                        )
                        if latest
                        else 0
                    ),

                "balanceChangeQuantity":
                    self._balance_change(
                        rows,
                        balance_field,
                    ),

                "latestBalanceRate":
                    (
                        float(
                            getattr(
                                latest,
                                balance_rate_field,
                            )
                        )
                        if (
                            latest
                            and getattr(
                                latest,
                                balance_rate_field,
                            )
                            is not None
                        )
                        else None
                    ),

                "averageTradingRate":
                    self._average_float_field(
                        rows,
                        trading_rate_field,
                    ),
            }

        def summarize_credit(
            rows,
        ) -> dict:
            return {
                "marginLoan":
                    summarize_credit_side(
                        rows,
                        prefix="margin_loan",
                    ),

                "stockLoan":
                    summarize_credit_side(
                        rows,
                        prefix="stock_loan",
                    ),
            }

        def summarize_lending(
            rows,
        ) -> dict:
            execution = (
                self._sum_int_field(
                    rows,
                    "execution_quantity",
                )
            )

            repayment = (
                self._sum_int_field(
                    rows,
                    "repayment_quantity",
                )
            )

            latest = (
                rows[-1]
                if rows
                else None
            )

            return {
                "executionQuantity":
                    execution,

                "repaymentQuantity":
                    repayment,

                "netLendingQuantity":
                    (
                        execution
                        - repayment
                    ),

                "latestBalanceQuantity":
                    (
                        int(
                            latest.balance_quantity
                            or 0
                        )
                        if latest
                        else 0
                    ),

                "balanceChangeQuantity":
                    self._balance_change(
                        rows,
                        "balance_quantity",
                    ),

                "latestBalanceAmount":
                    (
                        int(
                            latest.balance_amount
                            or 0
                        )
                        if latest
                        else 0
                    ),
            }

        def summarize_period(
            days: int,
        ) -> dict:
            return {
                "program":
                    summarize_program(
                        program_rows[
                            -days:
                        ]
                    ),

                "shortSelling":
                    summarize_short(
                        short_rows[
                            -days:
                        ]
                    ),

                "credit":
                    summarize_credit(
                        credit_rows[
                            -days:
                        ]
                    ),

                "securitiesLending":
                    summarize_lending(
                        lending_rows[
                            -days:
                        ]
                    ),
            }

        latest_dates = {
            "program":
                (
                    program_rows[
                        -1
                    ].trade_date.isoformat()
                    if program_rows
                    else None
                ),

            "shortSelling":
                (
                    short_rows[
                        -1
                    ].trade_date.isoformat()
                    if short_rows
                    else None
                ),

            "credit":
                (
                    credit_rows[
                        -1
                    ].trade_date.isoformat()
                    if credit_rows
                    else None
                ),

            "securitiesLending":
                (
                    lending_rows[
                        -1
                    ].trade_date.isoformat()
                    if lending_rows
                    else None
                ),
        }

        return {
            "available":
                bool(
                    program_rows
                    or short_rows
                    or credit_rows
                    or lending_rows
                ),

            "latestDates":
                latest_dates,

            "fiveDays":
                summarize_period(
                    5
                ),

            "twentyDays":
                summarize_period(
                    20
                ),

            "sixtyDays":
                summarize_period(
                    60
                ),
        }

    def _build_financial(
        self,
        *,
        stock_code: str,
    ) -> dict | None:
        business_year = str(
            date.today().year
            - 1
        )

        try:
            result = (
                FinancialAnalysisService(
                    self.db
                )
                .get_analysis(
                    stock_code=stock_code,
                    business_year=business_year,
                    report_code="11011",
                )
            )

        except ValueError:
            return None

        return result.model_dump()

    @staticmethod
    def _median(
        values: list[float],
    ) -> float:
        ordered = sorted(values)

        count = len(
            ordered
        )

        middle = (
            count
            // 2
        )

        if count % 2 == 1:
            return ordered[
                middle
            ]

        return (
            ordered[
                middle - 1
            ]
            + ordered[
                middle
            ]
        ) / 2.0

    @staticmethod
    def _financial_peer_state(
        top_percent: float,
    ) -> str:
        if top_percent <= 20.0:
            return "STRONG"

        if top_percent <= 40.0:
            return "GOOD"

        if top_percent <= 60.0:
            return "NEUTRAL"

        if top_percent <= 80.0:
            return "WEAK"

        return "POOR"

    @staticmethod
    def _financial_response_key(
        metric_key: str,
    ) -> str:
        mapping = {
            "revenue":
                "revenue",

            "operating_income":
                "operatingIncome",

            "net_income":
                "netIncome",

            "total_assets":
                "totalAssets",

            "total_liabilities":
                "totalLiabilities",

            "total_equity":
                "totalEquity",

            "operating_cash_flow":
                "operatingCashFlow",

            "roe":
                "roe",

            "roa":
                "roa",

            "operating_margin":
                "operatingMargin",

            "net_margin":
                "netMargin",

            "revenue_growth":
                "revenueGrowth",

            "operating_income_growth":
                "operatingIncomeGrowth",

            "net_income_growth":
                "netIncomeGrowth",

            "debt_ratio":
                "debtRatio",

            "equity_ratio":
                "equityRatio",

            "operating_cash_flow_margin":
                "operatingCashFlowMargin",

            "cash_flow_quality":
                "cashFlowQuality",
        }

        return mapping[
            metric_key
        ]

    def _build_financial_metric_comparison(
        self,
        *,
        stock_code: str,
        financial: dict | None,
    ) -> dict:
        stock = (
            self.stock_repository
            .get_stock(
                stock_code
            )
        )

        sector_name = (
            stock.sector_name.strip()
            if (
                stock
                and stock.sector_name
                and stock.sector_name.strip()
            )
            else None
        )

        business_year = (
            str(
                financial[
                    "businessYear"
                ]
            )
            if (
                financial
                and financial.get(
                    "businessYear"
                )
            )
            else None
        )

        report_code = (
            str(
                financial[
                    "reportCode"
                ]
            )
            if (
                financial
                and financial.get(
                    "reportCode"
                )
            )
            else None
        )

        fs_div = (
            str(
                financial[
                    "fsDiv"
                ]
            )
            if (
                financial
                and financial.get(
                    "fsDiv"
                )
            )
            else None
        )

        metric_rules = [
            (
                "revenue",
                "매출액",
                "원",
                True,
            ),
            (
                "operating_income",
                "영업이익",
                "원",
                True,
            ),
            (
                "net_income",
                "순이익",
                "원",
                True,
            ),
            (
                "operating_cash_flow",
                "영업현금흐름",
                "원",
                True,
            ),
            (
                "total_assets",
                "총자산",
                "원",
                True,
            ),
            (
                "total_liabilities",
                "총부채",
                "원",
                True,
            ),
            (
                "total_equity",
                "자기자본",
                "원",
                True,
            ),
            (
                "roe",
                "ROE",
                "%",
                True,
            ),
            (
                "roa",
                "ROA",
                "%",
                True,
            ),
            (
                "operating_margin",
                "영업이익률",
                "%",
                True,
            ),
            (
                "net_margin",
                "순이익률",
                "%",
                True,
            ),
            (
                "revenue_growth",
                "매출 성장률",
                "%",
                True,
            ),
            (
                "operating_income_growth",
                "영업이익 성장률",
                "%",
                True,
            ),
            (
                "net_income_growth",
                "순이익 성장률",
                "%",
                True,
            ),
            (
                "debt_ratio",
                "부채비율",
                "%",
                False,
            ),
            (
                "equity_ratio",
                "자기자본비율",
                "%",
                True,
            ),
            (
                "operating_cash_flow_margin",
                "영업현금흐름률",
                "%",
                True,
            ),
            (
                "cash_flow_quality",
                "현금흐름 품질",
                "배",
                True,
            ),
        ]

        if (
            sector_name is None
            or business_year is None
            or report_code is None
            or fs_div is None
        ):
            metrics = []

            for (
                key,
                label,
                unit,
                higher_is_better,
            ) in metric_rules:
                response_key = (
                    self._financial_response_key(
                        key
                    )
                )

                value = (
                    float(
                        financial[
                            response_key
                        ]
                    )
                    if (
                        financial
                        and financial.get(
                            response_key
                        )
                        is not None
                    )
                    else None
                )

                metrics.append(
                    {
                        "key":
                            key,

                        "label":
                            label,

                        "unit":
                            unit,

                        "higherIsBetter":
                            higher_is_better,

                        "available":
                            False,

                        "value":
                            value,

                        "peerAverage":
                            None,

                        "peerMedian":
                            None,

                        "differenceFromMedian":
                            None,

                        "peerRank":
                            None,

                        "peerCount":
                            0,

                        "topPercent":
                            None,

                        "state":
                            "UNAVAILABLE",
                    }
                )

            return {
                "available":
                    False,

                "sectorName":
                    sector_name,

                "businessYear":
                    business_year,

                "reportCode":
                    report_code,

                "fsDiv":
                    fs_div,

                "peerCount":
                    0,

                "metrics":
                    metrics,
            }

        peers = (
            self.stock_repository
            .get_sector_financial_metrics(
                sector_name=sector_name,
                business_year=business_year,
                report_code=report_code,
                fs_div=fs_div,
            )
        )

        peer_count = len(
            peers
        )

        metrics: list[
            dict
        ] = []

        for (
            key,
            label,
            unit,
            higher_is_better,
        ) in metric_rules:
            response_key = (
                self._financial_response_key(
                    key
                )
            )

            current_value = (
                float(
                    financial[
                        response_key
                    ]
                )
                if (
                    financial
                    and financial.get(
                        response_key
                    )
                    is not None
                )
                else None
            )

            peer_values = []

            for (
                _,
                metric,
            ) in peers:
                peer_value = getattr(
                    metric,
                    key,
                )

                if peer_value is None:
                    continue

                peer_values.append(
                    float(
                        peer_value
                    )
                )

            valid_peer_count = len(
                peer_values
            )

            if (
                current_value is None
                or valid_peer_count < 3
            ):
                metrics.append(
                    {
                        "key":
                            key,

                        "label":
                            label,

                        "unit":
                            unit,

                        "higherIsBetter":
                            higher_is_better,

                        "available":
                            False,

                        "value":
                            current_value,

                        "peerAverage":
                            None,

                        "peerMedian":
                            None,

                        "differenceFromMedian":
                            None,

                        "peerRank":
                            None,

                        "peerCount":
                            valid_peer_count,

                        "topPercent":
                            None,

                        "state":
                            "UNAVAILABLE",
                    }
                )

                continue

            if higher_is_better:
                better_count = sum(
                    1
                    for value
                    in peer_values
                    if value
                    > current_value
                )

            else:
                better_count = sum(
                    1
                    for value
                    in peer_values
                    if value
                    < current_value
                )

            rank = (
                1
                + better_count
            )

            top_percent = (
                rank
                / valid_peer_count
                * 100.0
            )

            average = (
                sum(
                    peer_values
                )
                / valid_peer_count
            )

            median = (
                self._median(
                    peer_values
                )
            )

            metrics.append(
                {
                    "key":
                        key,

                    "label":
                        label,

                    "unit":
                        unit,

                    "higherIsBetter":
                        higher_is_better,

                    "available":
                        True,

                    "value":
                        round(
                            current_value,
                            4,
                        ),

                    "peerAverage":
                        round(
                            average,
                            4,
                        ),

                    "peerMedian":
                        round(
                            median,
                            4,
                        ),

                    "differenceFromMedian":
                        round(
                            current_value
                            - median,
                            4,
                        ),

                    "peerRank":
                        rank,

                    "peerCount":
                        valid_peer_count,

                    "topPercent":
                        round(
                            top_percent,
                            1,
                        ),

                    "state":
                        self._financial_peer_state(
                            top_percent
                        ),
                }
            )

        return {
            "available":
                any(
                    metric[
                        "available"
                    ]
                    for metric
                    in metrics
                ),

            "sectorName":
                sector_name,

            "businessYear":
                business_year,

            "reportCode":
                report_code,

            "fsDiv":
                fs_div,

            "peerCount":
                peer_count,

            "metrics":
                metrics,
        }

    def _build_financial_peer_comparison(
        self,
        *,
        stock_code: str,
        financial: dict | None,
    ) -> dict:
        stock = (
            self.stock_repository
            .get_stock(
                stock_code
            )
        )

        sector_name = (
            stock.sector_name.strip()
            if (
                stock
                and stock.sector_name
                and stock.sector_name.strip()
            )
            else None
        )

        score = (
            float(
                financial.get(
                    "financialScore"
                )
            )
            if (
                financial
                and financial.get(
                    "financialScore"
                )
                is not None
            )
            else None
        )

        unavailable = {
            "available": False,
            "sectorName": sector_name,
            "peerCount": 0,
            "rank": None,
            "topPercent": None,
            "financialScore": score,
            "peerAverageScore": None,
            "label": "업종 비교 데이터 부족",
            "description": (
                "동일 업종 재무 비교에 필요한 "
                "데이터가 충분하지 않습니다."
            ),
        }

        if (
            sector_name is None
            or score is None
        ):
            return unavailable

        peers = (
            self.stock_repository
            .get_sector_financial_scores(
                sector_name=sector_name,
            )
        )

        if len(
            peers
        ) < 3:
            return unavailable

        peer_scores = [
            peer_score
            for (
                _,
                _,
                peer_score,
            )
            in peers
        ]

        rank = (
            1
            + sum(
                1
                for peer_score
                in peer_scores
                if peer_score
                > score
            )
        )

        peer_count = len(
            peer_scores
        )

        top_percent = (
            rank
            / peer_count
            * 100.0
        )

        average_score = (
            sum(
                peer_scores
            )
            / peer_count
        )

        if top_percent <= 20.0:
            label = "동일 업종 상위권"
        elif top_percent <= 50.0:
            label = "동일 업종 평균 이상"
        else:
            label = "동일 업종 평균 이하"

        score_gap = (
            score
            - average_score
        )

        return {
            "available": True,
            "sectorName": sector_name,
            "peerCount": peer_count,
            "rank": rank,
            "topPercent": round(
                top_percent,
                1,
            ),
            "financialScore": round(
                score,
                1,
            ),
            "peerAverageScore": round(
                average_score,
                1,
            ),
            "label": label,
            "description": (
                f"재무점수 {score:.1f}점으로 "
                f"동일 업종 {peer_count}개 중 {rank}위이며, "
                f"업종 평균보다 {score_gap:+.1f}점입니다."
            ),
        }

    @staticmethod
    def _valuation_relative_percent(
        value: float | None,
        benchmark: float | None,
    ) -> float | None:
        if (
            value is None
            or benchmark is None
            or value <= 0.0
            or benchmark <= 0.0
        ):
            return None

        return (
            (
                value
                / benchmark
                - 1.0
            )
            * 100.0
        )

    def _build_valuation(
        self,
        *,
        stock_code: str,
    ) -> dict:
        valuation = (
            self.market_data_repository
            .get_latest_valuation(
                stock_code
            )
        )

        def build_multiple(
            *,
            value: float | None,
            sector_benchmark: float | None,
            market_benchmark: float | None,
        ) -> dict:
            current = (
                float(value)
                if value is not None
                else None
            )

            sector = (
                float(
                    sector_benchmark
                )
                if sector_benchmark
                is not None
                else None
            )

            market = (
                float(
                    market_benchmark
                )
                if market_benchmark
                is not None
                else None
            )

            sector_relative = (
                self._valuation_relative_percent(
                    current,
                    sector,
                )
            )

            market_relative = (
                self._valuation_relative_percent(
                    current,
                    market,
                )
            )

            return {
                "value":
                    current,

                "comparisonAvailable":
                    (
                        sector_relative
                        is not None
                        or market_relative
                        is not None
                    ),

                "sectorBenchmark":
                    sector,

                "marketBenchmark":
                    market,

                "sectorRelativePercent":
                    (
                        round(
                            sector_relative,
                            2,
                        )
                        if sector_relative
                        is not None
                        else None
                    ),

                "marketRelativePercent":
                    (
                        round(
                            market_relative,
                            2,
                        )
                        if market_relative
                        is not None
                        else None
                    ),
            }

        if valuation is None:
            empty_multiple = {
                "value":
                    None,

                "comparisonAvailable":
                    False,

                "sectorBenchmark":
                    None,

                "marketBenchmark":
                    None,

                "sectorRelativePercent":
                    None,

                "marketRelativePercent":
                    None,
            }

            return {
                "available":
                    False,

                "snapshotDate":
                    None,

                "sectorCode":
                    None,

                "sectorName":
                    None,

                "price":
                    None,

                "marketCap":
                    None,

                "per":
                    dict(
                        empty_multiple
                    ),

                "pbr":
                    dict(
                        empty_multiple
                    ),

                "eps":
                    None,

                "bps":
                    None,

                "dividendYield":
                    None,

                "source":
                    None,
            }

        return {
            "available":
                True,

            "snapshotDate":
                valuation
                .snapshot_date
                .isoformat(),

            "sectorCode":
                valuation.sector_code,

            "sectorName":
                valuation.sector_name,

            "price":
                (
                    float(
                        valuation.price
                    )
                    if valuation.price
                    is not None
                    else None
                ),

            "marketCap":
                (
                    float(
                        valuation.market_cap
                    )
                    if valuation.market_cap
                    is not None
                    else None
                ),

            "per":
                build_multiple(
                    value=(
                        valuation.per
                    ),
                    sector_benchmark=(
                        valuation.sector_per
                    ),
                    market_benchmark=(
                        valuation.market_per
                    ),
                ),

            "pbr":
                build_multiple(
                    value=(
                        valuation.pbr
                    ),
                    sector_benchmark=(
                        valuation.sector_pbr
                    ),
                    market_benchmark=(
                        valuation.market_pbr
                    ),
                ),

            "eps":
                (
                    float(
                        valuation.eps
                    )
                    if valuation.eps
                    is not None
                    else None
                ),

            "bps":
                (
                    float(
                        valuation.bps
                    )
                    if valuation.bps
                    is not None
                    else None
                ),

            "dividendYield":
                (
                    float(
                        valuation.dividend_yield
                    )
                    if valuation.dividend_yield
                    is not None
                    else None
                ),

            "source":
                valuation.source,
        }

    def get_current_analysis(
        self,
        *,
        stock_code: str,
    ) -> dict:
        ranking = (
            RankingExplanationService(
                self.db
            )
            .get_current_explanation(
                stock_code,
                top_k=3,
            )
        )

        technical = (
            TechnicalAnalysisService(
                self.db
            )
            .inspect_stock(
                stock_code=stock_code,
            )
        )

        technical_interpretation = (
            TechnicalInterpretationService.build(
                technical=technical,
            )
        )

        financial = (
            self._build_financial(
                stock_code=stock_code,
            )
        )

        valuation = (
            self._build_valuation(
                stock_code=stock_code,
            )
        )

        long_term_financial = (
            LongTermFinancialService(
                self.db
            )
            .build(
                stock_code=stock_code,
            )
        )

        investor_flow = (
            self._build_investor_flow(
                stock_code=stock_code,
            )
        )

        investor_flow_interpretation = (
            InvestorFlowInterpretationService.build(
                investor_flow=investor_flow,
            )
        )

        trading_signals = (
            self._build_trading_signals(
                stock_code=stock_code,
            )
        )

        benchmark_performance = (
            self._build_benchmark_performance(
                stock_code=stock_code,
            )
        )

        benchmark_interpretation = (
            BenchmarkInterpretationService.build(
                benchmark_performance=(
                    benchmark_performance
                ),
            )
        )

        relative_strength = (
            self._build_relative_strength(
                stock_code=stock_code,
            )
        )

        market_regime = (
            self._build_market_regime(
                stock_code=stock_code,
            )
        )

        market_context = (
            MarketContextService(
                self.db
            )
        )

        news = (
            market_context.get_news(
                stock_code=stock_code,
                limit=5,
            )
        )

        disclosures = (
            market_context.get_disclosures(
                stock_code=stock_code,
                limit=5,
            )
        )

        financial_peer_comparison = (
            self._build_financial_peer_comparison(
                stock_code=stock_code,
                financial=financial,
            )
        )

        financial_peer_metrics = (
            self._build_financial_metric_comparison(
                stock_code=stock_code,
                financial=financial,
            )
        )

        analysis_brief = (
            StockAnalysisBriefService.build(
                ranking=ranking,

                financial=financial,

                financial_peer_comparison=(
                    financial_peer_comparison
                ),

                financial_peer_metrics=(
                    financial_peer_metrics
                ),

                valuation=valuation,

                long_term_financial=(
                    long_term_financial
                ),

                investor_flow=(
                    investor_flow
                ),

                benchmark_performance=(
                    benchmark_performance
                ),
            )
        )
        return {
            "stockCode":
                ranking[
                    "stock_code"
                ],

            "stockName":
                ranking[
                    "stock_name"
                ],

            "ranking":
                ranking,

            "technical":
                technical,

            "technicalInterpretation":
                technical_interpretation,

            "financial":
                financial,

            "valuation":
                valuation,

            "financialPeerMetrics":
                financial_peer_metrics,

            "longTermFinancial":
                long_term_financial,

            "investorFlow":
                investor_flow,

            "investorFlowInterpretation":
                investor_flow_interpretation,

            "tradingSignals":
                trading_signals,
                
            "benchmarkPerformance":
                benchmark_performance,

            "relativeStrength":
                relative_strength,

            "benchmarkInterpretation":
                benchmark_interpretation,

            "marketRegime":
                market_regime,

            "analysisBrief":
                analysis_brief,

            "news":
                news,

            "disclosures":
                disclosures,
        }