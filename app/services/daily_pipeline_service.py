from __future__ import annotations

import asyncio
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.models.future import (
    DataSyncRun,
    RankingItem,
    RankingSnapshot,
    RecommendationPerformance,
)
from app.models.stock_price import StockPrice
from app.services.financial_analysis_service import (
    FinancialAnalysisService,
)
from app.services.market_context_service import (
    MarketContextService,
)
from app.services.market_data_service import (
    MarketDataService,
)
from app.services.stock_analysis_service import (
    StockAnalysisService,
)
from app.services.stock_service import (
    StockService,
)


class DailyPipelineService:
    KST = ZoneInfo("Asia/Seoul")

    # 종합랭킹 최종 노출 개수
    RANKING_LIMIT = 100

    # CompositeRankingService가 최소 500종목을
    # 평가하므로 수급/밸류/가격/재무 데이터도
    # 동일 후보군까지 갱신합니다.
    UNIVERSE_LIMIT = 500

    TOP_CONTEXT_LIMIT = 100

    RECENT_PRICE_DAYS = 14
    BOOTSTRAP_PRICE_DAYS = 1200
    MIN_HISTORY_ROWS = 61

    _run_lock = asyncio.Lock()

    def __init__(
        self,
        db: Session,
    ):
        self.db = db

        self.stock_service = StockService(
            db
        )

        self.market_data_service = (
            MarketDataService(
                db
            )
        )

        self.market_context_service = (
            MarketContextService(
                db
            )
        )

        self.financial_service = (
            FinancialAnalysisService(
                db
            )
        )

    @classmethod
    def _today(
        cls,
    ) -> date:
        return datetime.now(
            cls.KST
        ).date()

    def _snapshot_exists(
        self,
        snapshot_date: date,
    ) -> bool:
        snapshot_id = self.db.scalar(
            select(
                RankingSnapshot.id
            )
            .where(
                RankingSnapshot.ranking_version
                == self.market_context_service.RANKING_VERSION,
                RankingSnapshot.as_of_date
                == snapshot_date,
                RankingSnapshot.horizon_days
                == self.market_context_service.PRIMARY_HORIZON_DAYS,
                RankingSnapshot.universe
                == "KRX",
            )
            .limit(1)
        )

        return snapshot_id is not None

    def _daily_refresh_completed(
        self,
        snapshot_date: date,
    ) -> bool:
        recent_runs = self.db.scalars(
            select(
                DataSyncRun
            )
            .where(
                DataSyncRun.source
                == "automation",
                DataSyncRun.sync_type
                == "daily_pipeline",
                DataSyncRun.status
                == "completed",
            )
            .order_by(
                DataSyncRun.started_at.desc()
            )
            .limit(20)
        ).all()

        target_date = (
            snapshot_date.isoformat()
        )

        return any(
            (
                metadata.get(
                    "date"
                )
                == target_date
                and metadata.get(
                    "refreshCompleted"
                ) is True
            )
            for run in recent_runs
            for metadata in [
                run.metadata_json
                or {}
            ]
        )

    def _create_run(
        self,
    ) -> int:
        run = DataSyncRun(
            source="automation",
            sync_type="daily_pipeline",
            status="running",
            metadata_json={
                "timezone": "Asia/Seoul",
                "ranking_version": (
                    self.market_context_service
                    .RANKING_VERSION
                ),
                "universe_limit": (
                    self.UNIVERSE_LIMIT
                ),

                "ranking_limit": (
                    self.RANKING_LIMIT
                ),
            },
        )

        self.db.add(
            run
        )

        self.db.commit()
        self.db.refresh(
            run
        )

        return int(
            run.id
        )

    def _finish_run(
        self,
        run_id: int,
        *,
        status: str,
        requested_count: int = 0,
        success_count: int = 0,
        failure_count: int = 0,
        error_message: str | None = None,
        metadata: dict | None = None,
    ) -> None:
        run = self.db.get(
            DataSyncRun,
            run_id,
        )

        if run is None:
            return

        run.finished_at = (
            datetime.utcnow()
        )

        run.status = status

        run.requested_count = (
            requested_count
        )

        run.success_count = (
            success_count
        )

        run.failure_count = (
            failure_count
        )

        run.error_message = (
            error_message
        )

        merged_metadata = dict(
            run.metadata_json
            or {}
        )

        if metadata:
            merged_metadata.update(
                metadata
            )

        run.metadata_json = (
            merged_metadata
        )

        self.db.commit()

    def _recover_stale_runs(
        self,
    ) -> int:
        stale_before = (
            datetime.utcnow()
            - timedelta(
                hours=2
            )
        )

        stale_runs = self.db.scalars(
            select(
                DataSyncRun
            )
            .where(
                DataSyncRun.source
                == "automation",
                DataSyncRun.sync_type
                == "daily_pipeline",
                DataSyncRun.status
                == "running",
                DataSyncRun.started_at
                < stale_before,
            )
            .order_by(
                DataSyncRun.started_at.asc()
            )
        ).all()

        if not stale_runs:
            return 0

        recovered_at = (
            datetime.utcnow()
        )

        for run in stale_runs:
            run.status = "failed"

            run.finished_at = (
                recovered_at
            )

            run.error_message = (
                "stale running 상태 자동 복구"
            )

            metadata = dict(
                run.metadata_json
                or {}
            )

            metadata[
                "staleRunRecovered"
            ] = True

            metadata[
                "staleRunRecoveredAt"
            ] = recovered_at.isoformat()

            run.metadata_json = (
                metadata
            )

        self.db.commit()

        return len(
            stale_runs
        )

    async def _sync_toss_global_market_context(
        self,
    ) -> tuple[
        dict,
        int,
    ]:
        try:
            result = await (
                self.market_data_service
                .sync_toss_market_context()
            )

            print(
                "[DAILY] "
                "toss-market-context "
                f"index={result.get('indexRows', 0)} "
                f"investor={result.get('investorRows', 0)} "
                f"fx={result.get('fxRows', 0)}",
                flush=True,
            )

            return (
                result,
                0,
            )

        except Exception as e:
            self.db.rollback()

            print(
                "[DAILY] "
                "toss-market-context "
                f"failed: {e}",
                flush=True,
            )

            return (
                {
                    "indexRows": 0,
                    "investorRows": 0,
                    "fxRows": 0,
                },
                1,
            )

    def _latest_trade_date(
        self,
        stock_codes: list[str],
        *,
        on_or_before: date,
    ) -> date | None:
        if not stock_codes:
            return None

        return self.db.scalar(
            select(
                func.max(
                    StockPrice.trade_date
                )
            )
            .where(
                StockPrice.stock_code.in_(
                    stock_codes
                ),
                StockPrice.trade_date
                <= on_or_before,
            )
        )

    def _close_map(
        self,
        stock_codes: list[str],
        *,
        trade_date: date,
    ) -> dict[str, float]:
        rows = self.db.execute(
            select(
                StockPrice.stock_code,
                StockPrice.close,
            )
            .where(
                StockPrice.stock_code.in_(
                    stock_codes
                ),
                StockPrice.trade_date
                == trade_date,
            )
        ).all()

        return {
            stock_code: float(
                close
            )
            for (
                stock_code,
                close,
            ) in rows
            if (
                close is not None
                and float(
                    close
                ) > 0.0
            )
        }
    
    async def _sync_trading_signals(
        self,
        stock_codes: list[str],
    ) -> tuple[
        int,
        int,
        int,
    ]:
        success = 0
        failed = 0
        stored_rows = 0

        total = len(
            stock_codes
        )

        for (
            index,
            stock_code,
        ) in enumerate(
            stock_codes,
            start=1,
        ):
            try:
                result = await (
                    self.market_data_service
                    .sync_stock_trading_trends(
                        stock_code,
                        count=65,
                    )
                )

                rows = sum(
                    int(
                        value
                        or 0
                    )
                    for value
                    in result.values()
                )

                stored_rows += (
                    rows
                )

                success += 1

                print(
                    "[DAILY] "
                    "trading-signals "
                    f"{index}/{total} "
                    f"{stock_code} "
                    f"{rows} rows",
                    flush=True,
                )

            except Exception as e:
                self.db.rollback()

                failed += 1

                print(
                    "[DAILY] "
                    "trading-signals "
                    f"{stock_code} "
                    f"failed: {e}",
                    flush=True,
                )

        return (
            success,
            failed,
            stored_rows,
        )

    async def _sync_daily_prices(
        self,
        stock_codes: list[str],
        *,
        today: date,
    ) -> tuple[
        int,
        int,
        int,
    ]:
        success = 0
        failed = 0
        stored_rows = 0

        history_start = (
            today
            - timedelta(
                days=(
                    self.BOOTSTRAP_PRICE_DAYS
                )
            )
        )

        for (
            index,
            stock_code,
        ) in enumerate(
            stock_codes,
            start=1,
        ):
            try:
                existing = (
                    self.stock_service
                    .repository
                    .get_daily_prices(
                        stock_code=(
                            stock_code
                        ),
                        start_date=(
                            history_start
                        ),
                    )
                )

                days = (
                    self.BOOTSTRAP_PRICE_DAYS
                    if len(
                        existing
                    ) < self.MIN_HISTORY_ROWS
                    else self.RECENT_PRICE_DAYS
                )

                count = await (
                    self.stock_service
                    .sync_daily_prices(
                        stock_code,
                        days=days,
                    )
                )

                stored_rows += (
                    count
                )

                success += 1

                print(
                    "[DAILY] "
                    f"price "
                    f"{index}/"
                    f"{len(stock_codes)} "
                    f"{stock_code} "
                    f"{count} rows",
                    flush=True,
                )

            except Exception as e:
                self.db.rollback()

                failed += 1

                print(
                    "[DAILY] "
                    f"price "
                    f"{stock_code} "
                    f"failed: {e}",
                    flush=True,
                )

        return (
            success,
            failed,
            stored_rows,
        )

    async def _ensure_financials(
        self,
        stock_codes: list[str],
        *,
        business_year: str,
    ) -> tuple[
        int,
        int,
    ]:
        created = 0
        failed = 0

        for stock_code in stock_codes:
            metric = (
                self.financial_service
                .repository
                .get_financial_metric(
                    stock_code=(
                        stock_code
                    ),
                    business_year=(
                        business_year
                    ),
                    report_code="11011",
                )
            )

            if metric is not None:
                continue

            try:
                rows = (
                    self.financial_service
                    .repository
                    .get_financial_rows(
                        stock_code=(
                            stock_code
                        ),
                        business_year=(
                            business_year
                        ),
                        report_code="11011",
                    )
                )

                if not rows:
                    await (
                        self.stock_service
                        .sync_financials(
                            stock_code,
                            business_year=(
                                business_year
                            ),
                            report_code=(
                                "11011"
                            ),
                        )
                    )

                self.financial_service.analyze(
                    stock_code=(
                        stock_code
                    ),
                    business_year=(
                        business_year
                    ),
                    report_code="11011",
                )

                created += 1

            except Exception as e:
                self.db.rollback()

                failed += 1

                print(
                    "[DAILY] "
                    "financial "
                    f"{stock_code} "
                    f"failed: {e}",
                    flush=True,
                )

        return (
            created,
            failed,
        )

    async def run(
        self,
        *,
        force: bool = False,
    ) -> dict:
        if self._run_lock.locked():
            return {
                "status": (
                    "already_running"
                ),
                "message": (
                    "일일 자동화가 "
                    "이미 실행 중입니다."
                ),
            }

        async with self._run_lock:
            today = self._today()

            stale_runs_recovered = (
                self._recover_stale_runs()
            )

            if stale_runs_recovered:
                print(
                    "[DAILY] "
                    "stale runs recovered: "
                    f"{stale_runs_recovered}",
                    flush=True,
                )

            run_id = (
                self._create_run()
            )

            try:
                if (
                    today.weekday() >= 5
                    and not force
                ):
                    result = {
                        "status": "skipped",
                        "reason": "weekend",
                        "date": (
                            today.isoformat()
                        ),
                    }

                    self._finish_run(
                        run_id,
                        status="skipped",
                        metadata=result,
                    )

                    return result

                if (
                    self._daily_refresh_completed(
                        today
                    )
                    and not force
                ):
                    self.market_context_service.evaluate_performance()

                    result = {
                        "status": (
                            "already_completed"
                        ),
                        "date": (
                            today.isoformat()
                        ),
                        "rankingVersion": (
                            self.market_context_service
                            .RANKING_VERSION
                        ),
                    }

                    self._finish_run(
                        run_id,
                        status="completed",
                        metadata=result,
                    )

                    return result

                (
                    toss_market_context,
                    toss_market_context_failures,
                ) = await (
                    self
                    ._sync_toss_global_market_context()
                )

                (
                    market_success,
                    investor_rows,
                    valuation_rows,
                    investor_failures,
                    valuation_failures,
                ) = await (
                    self.market_data_service
                    .sync_ranked_stock_context(
                        limit=(
                            self.UNIVERSE_LIMIT
                        )
                    )
                )

                (
                    valuation_benchmark_rows,
                    valuation_sector_count,
                    valuation_market_count,
                ) = (
                    self.market_data_service
                    .refresh_valuation_benchmarks(
                        snapshot_date=(
                            today
                        ),
                    )
                )

                universe = (
                    self.market_data_service
                    .get_current_universe_stock_codes(
                        limit=(
                            self.UNIVERSE_LIMIT
                        )
                    )
                )

                if not universe:
                    raise RuntimeError(
                        "현재 랭킹 후보 "
                        "Universe가 없습니다."
                    )

                (
                    trading_signal_success,
                    trading_signal_failures,
                    trading_signal_rows,
                ) = await (
                    self._sync_trading_signals(
                        universe
                    )
                )

                (
                    price_success,
                    price_failures,
                    price_rows,
                ) = await (
                    self._sync_daily_prices(
                        universe,
                        today=today,
                    )
                )

                # 일봉이 들어온 뒤
                # 기존 추천 성과를 먼저 평가합니다.
                self.market_context_service.evaluate_performance()

                market_date = (
                    self._latest_trade_date(
                        universe,
                        on_or_before=(
                            today
                        ),
                    )
                )

                # 토/일뿐 아니라
                # 평일 공휴일도 여기서 걸러집니다.
                if (
                    market_date is None
                    or market_date
                    != today
                ):
                    result = {
                        "status": (
                            "skipped"
                        ),
                        "reason": (
                            "non_trading_day_"
                            "or_eod_not_ready"
                        ),
                        "date": (
                            today.isoformat()
                        ),
                        "latestTradeDate": (
                            market_date.isoformat()
                            if market_date
                            else None
                        ),
                        "universeCount": (
                            len(
                                universe
                            )
                        ),
                    }

                    self._finish_run(
                        run_id,
                        status="skipped",
                        requested_count=(
                            len(
                                universe
                            )
                        ),
                        success_count=(
                            price_success
                        ),
                        failure_count=(
                            price_failures
                        ),
                        metadata=result,
                    )

                    return result

                (
                    current_price_success,
                    current_price_failures,
                ) = await (
                    self.stock_service
                    .sync_current_prices(
                        universe
                    )
                )

                disclosure_rows = 0
                disclosure_failures = 0

                try:
                    disclosure_rows = await (
                        self.market_context_service
                        .sync_disclosures(
                            stock_code=None,
                            limit=100,
                        )
                    )
                except Exception as e:
                    self.db.rollback()

                    disclosure_failures = 1

                    print(
                        "[DAILY] "
                        "disclosures failed: "
                        f"{e}",
                        flush=True,
                    )

                global_news_rows = 0
                global_news_failures = 0

                try:
                    global_news_rows = await (
                        self.market_context_service
                        .sync_news(
                            stock_code=None,
                            limit=50,
                        )
                    )
                except Exception as e:
                    self.db.rollback()

                    global_news_failures = 1

                    print(
                        "[DAILY] "
                        "global news failed: "
                        f"{e}",
                        flush=True,
                    )

                business_year = str(
                    today.year - 1
                )

                financial_created = 0
                financial_failures = 0

                # 종합 랭킹이 재무 데이터를 실제로 사용하므로
                # 랭킹 계산 전에 Universe 전체의 재무 데이터를 보장합니다.
                # 연차보고서가 아직 없는 1~3월에는
                # 매일 재요청하지 않습니다.
                if today.month >= 4:
                    (
                        financial_created,
                        financial_failures,
                    ) = await (
                        self._ensure_financials(
                            universe,
                            business_year=(
                                business_year
                            ),
                        )
                    )

                rankings = await (
                    self.stock_service
                    .get_rankings(
                        limit=(
                            self.RANKING_LIMIT
                        ),
                        force_recompute=True,
                    )
                )

                if len(
                    rankings
                ) < 10:
                    raise RuntimeError(
                        "랭킹 결과가 "
                        "10종목 미만입니다."
                    )

                # 성과 진입가는
                # 장 마감 일봉 종가를 사용합니다.
                close_map = (
                    self._close_map(
                        [
                            row.stockCode
                            for row
                            in rankings
                        ],
                        trade_date=(
                            market_date
                        ),
                    )
                )

                for row in rankings:
                    close = (
                        close_map.get(
                            row.stockCode
                        )
                    )

                    if close is not None:
                        row.currentPrice = (
                            close
                        )

                top_news_rows = 0
                top_news_failures = 0
                top_disclosure_rows = 0
                top_disclosure_failures = 0

                for row in rankings[
                    :self.TOP_CONTEXT_LIMIT
                ]:
                    try:
                        top_news_rows += await (
                            self.market_context_service
                            .sync_news(
                                stock_code=(
                                    row.stockCode
                                ),
                                limit=20,
                            )
                        )

                    except Exception as e:
                        self.db.rollback()

                        top_news_failures += 1

                        print(
                            "[DAILY] "
                            "stock news "
                            f"{row.stockCode} "
                            f"failed: {e}",
                            flush=True,
                        )

                    try:
                        top_disclosure_rows += await (
                            self.market_context_service
                            .sync_disclosures(
                                stock_code=(
                                    row.stockCode
                                ),
                                limit=20,
                            )
                        )

                    except Exception as e:
                        self.db.rollback()

                        top_disclosure_failures += 1

                        print(
                            "[DAILY] "
                            "stock disclosures "
                            f"{row.stockCode} "
                            f"failed: {e}",
                            flush=True,
                        )

                self.market_context_service.record_rankings(
                    rankings,
                    as_of_date=(
                        market_date
                    ),
                    replace_existing=True,
                )

                analysis_snapshot_success = 0
                analysis_snapshot_failures = 0

                analysis_service = StockAnalysisService(
                    self.db
                )

                for row in rankings:
                    try:
                        analysis_service.save_daily_snapshot(
                            stock_code=(
                                row.stockCode
                            ),
                            snapshot_date=(
                                market_date
                            ),
                        )

                        analysis_snapshot_success += 1

                    except Exception as e:
                        self.db.rollback()

                        analysis_snapshot_failures += 1

                        print(
                            "[DAILY] "
                            "analysis snapshot "
                            f"{row.stockCode} "
                            f"failed: {e}",
                            flush=True,
                        )

                self.market_context_service.evaluate_performance()

                total_failures = (
                    toss_market_context_failures
                    + investor_failures
                    + valuation_failures
                    + trading_signal_failures
                    + price_failures
                    + current_price_failures
                    + disclosure_failures
                    + global_news_failures
                    + top_news_failures
                    + financial_failures
                )

                result = {
                    "status": "completed",
                    "refreshCompleted": True,
                    "date": (
                        today.isoformat()
                    ),
                    "marketDate": (
                        market_date.isoformat()
                    ),
                    "rankingVersion": (
                        self.market_context_service
                        .RANKING_VERSION
                    ),
                    "universeCount": (
                        len(
                            universe
                        )
                    ),
                    "rankingCount": (
                        len(
                            rankings
                        )
                    ),
                    "top10": [
                        {
                            "rank": (
                                row.rank
                            ),
                            "stockCode": (
                                row.stockCode
                            ),
                            "stockName": (
                                row.stockName
                            ),
                            "aiScore": (
                                row.totalScore
                            ),
                        }
                        for row
                        in rankings[:10]
                    ],
                    "tossMarketContext": {
                        "indexRows": int(
                            toss_market_context.get(
                                "indexRows",
                                0,
                            )
                        ),
                        "investorRows": int(
                            toss_market_context.get(
                                "investorRows",
                                0,
                            )
                        ),
                        "fxRows": int(
                            toss_market_context.get(
                                "fxRows",
                                0,
                            )
                        ),
                        "failures": (
                            toss_market_context_failures
                        ),
                    },
                    "marketContext": {
                        "successStocks": (
                            market_success
                        ),

                        "investorRows": (
                            investor_rows
                        ),

                        "valuationRows": (
                            valuation_rows
                        ),

                        "valuationBenchmarkRows": (
                            valuation_benchmark_rows
                        ),

                        "valuationSectorCount": (
                            valuation_sector_count
                        ),

                        "valuationMarketCount": (
                            valuation_market_count
                        ),
                    },
                    "tradingSignals": {
                        "successStocks": (
                            trading_signal_success
                        ),
                        "failedStocks": (
                            trading_signal_failures
                        ),
                        "storedRows": (
                            trading_signal_rows
                        ),
                    },
                    "dailyPrices": {
                        "successStocks": (
                            price_success
                        ),
                        "failedStocks": (
                            price_failures
                        ),
                        "storedRows": (
                            price_rows
                        ),
                    },
                    "currentPrices": {
                        "successStocks": (
                            current_price_success
                        ),
                        "failedStocks": (
                            current_price_failures
                        ),
                    },
                    "news": {
                        "globalSaved": (
                            global_news_rows
                        ),
                        "top10Saved": (
                            top_news_rows
                        ),
                        "failures": (
                            global_news_failures
                            + top_news_failures
                        ),
                    },
                    "disclosures": {
                        "saved": (
                            disclosure_rows
                        ),
                        "failures": (
                            disclosure_failures
                        ),
                    },
                    "financials": {
                        "businessYear": (
                            business_year
                        ),
                        "created": (
                            financial_created
                        ),
                        "failed": (
                            financial_failures
                        ),
                    },
                }

                self._finish_run(
                    run_id,
                    status="completed",
                    requested_count=(
                        len(
                            universe
                        )
                    ),
                    success_count=(
                        price_success
                    ),
                    failure_count=(
                        total_failures
                    ),
                    metadata=result,
                )

                return result

            except Exception as e:
                self.db.rollback()

                self._finish_run(
                    run_id,
                    status="failed",
                    error_message=str(
                        e
                    ),
                    metadata={
                        "date": (
                            today.isoformat()
                        ),
                        "exceptionType": (
                            type(
                                e
                            ).__name__
                        ),
                    },
                )

                raise

    def get_status(
        self,
    ) -> dict:
        today = self._today()

        latest_run = self.db.scalar(
            select(
                DataSyncRun
            )
            .where(
                DataSyncRun.source
                == "automation",
                DataSyncRun.sync_type
                == "daily_pipeline",
            )
            .order_by(
                DataSyncRun.started_at.desc()
            )
            .limit(1)
        )

        latest_snapshot = self.db.scalar(
            select(
                RankingSnapshot
            )
            .where(
                RankingSnapshot.ranking_version
                == self.market_context_service.RANKING_VERSION
            )
            .order_by(
                RankingSnapshot.as_of_date.desc()
            )
            .limit(1)
        )

        latest_snapshot_item_count = 0
        latest_snapshot_performance_count = 0

        if latest_snapshot is not None:
            latest_snapshot_item_count = int(
                self.db.scalar(
                    select(
                        func.count(
                            RankingItem.id
                        )
                    )
                    .where(
                        RankingItem.snapshot_id
                        == latest_snapshot.id
                    )
                )
                or 0
            )

            latest_snapshot_performance_count = int(
                self.db.scalar(
                    select(
                        func.count(
                            RecommendationPerformance.id
                        )
                    )
                    .join(
                        RankingItem,
                        RankingItem.id
                        == RecommendationPerformance.ranking_item_id,
                    )
                    .where(
                        RankingItem.snapshot_id
                        == latest_snapshot.id
                    )
                )
                or 0
            )

        latest_market_date = (
            self.db.scalar(
                select(
                    func.max(
                        StockPrice.trade_date
                    )
                )
            )
        )

        counts = self.db.execute(
            select(
                func.count(
                    RecommendationPerformance.id
                ),
                func.count(
                    RecommendationPerformance.evaluated_at
                ),
            )
            .join(
                RankingItem,
                RankingItem.id
                == RecommendationPerformance.ranking_item_id,
            )
            .join(
                RankingSnapshot,
                RankingSnapshot.id
                == RankingItem.snapshot_id,
            )
            .where(
                RankingSnapshot.ranking_version
                == self.market_context_service.RANKING_VERSION
            )
        ).one()

        total = int(
            counts[0]
            or 0
        )

        evaluated = int(
            counts[1]
            or 0
        )

        return {
            "enabled": True,
            "timezone": (
                "Asia/Seoul"
            ),
            "scheduleKst": (
                "평일 16:40"
            ),
            "rankingVersion": (
                self.market_context_service
                .RANKING_VERSION
            ),
            "today": (
                today.isoformat()
            ),
            "todayCompleted": (
                latest_snapshot
                is not None
                and latest_snapshot.as_of_date
                == today
            ),
            "latestMarketDate": (
                latest_market_date.isoformat()
                if latest_market_date
                else None
            ),
            "latestRankingDate": (
                latest_snapshot.as_of_date.isoformat()
                if latest_snapshot
                else None
            ),
            "latestRankingSnapshot": (
                {
                    "id": int(
                        latest_snapshot.id
                    ),
                    "itemCount": (
                        latest_snapshot_item_count
                    ),
                    "performanceCount": (
                        latest_snapshot_performance_count
                    ),
                }
                if latest_snapshot
                else None
            ),
            "performance": {
                "total": total,
                "evaluated": (
                    evaluated
                ),
                "pending": (
                    total
                    - evaluated
                ),
            },
            "latestRun": (
                {
                    "id": (
                        int(
                            latest_run.id
                        )
                    ),
                    "status": (
                        latest_run.status
                    ),
                    "startedAt": (
                        latest_run.started_at.isoformat()
                        if latest_run.started_at
                        else None
                    ),
                    "finishedAt": (
                        latest_run.finished_at.isoformat()
                        if latest_run.finished_at
                        else None
                    ),
                    "requestedCount": (
                        latest_run.requested_count
                    ),
                    "successCount": (
                        latest_run.success_count
                    ),
                    "failureCount": (
                        latest_run.failure_count
                    ),
                    "errorMessage": (
                        latest_run.error_message
                    ),
                    "metadata": (
                        latest_run.metadata_json
                        or {}
                    ),
                }
                if latest_run
                else None
            ),
        }