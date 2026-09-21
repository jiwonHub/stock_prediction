from __future__ import annotations

import asyncio
from functools import wraps
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

from sqlalchemy import func, select, text
from sqlalchemy.orm import Session

from app.core.database import SessionLocal, engine
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


_DAILY_PIPELINE_ADVISORY_LOCK_KEY = 2026092101


def _with_daily_pipeline_advisory_lock(
    method,
):
    @wraps(method)
    async def wrapper(
        self,
        *args,
        **kwargs,
    ):
        if (
            engine.dialect.name
            != "postgresql"
        ):
            return await method(
                self,
                *args,
                **kwargs,
            )

        connection = (
            engine.connect()
        )

        acquired = False

        try:
            acquired = bool(
                connection.scalar(
                    text(
                        """
                        SELECT pg_try_advisory_lock(
                            :lock_key
                        )
                        """
                    ),
                    {
                        "lock_key": (
                            _DAILY_PIPELINE_ADVISORY_LOCK_KEY
                        ),
                    },
                )
            )

            # session-level advisory lock은
            # commit 후에도 유지됩니다.
            # 장시간 idle transaction 방지용입니다.
            connection.commit()

            if not acquired:
                return {
                    "status": (
                        "already_running"
                    ),
                    "message": (
                        "다른 프로세스에서 "
                        "일일 자동화가 "
                        "이미 실행 중입니다."
                    ),
                }

            return await method(
                self,
                *args,
                **kwargs,
            )

        finally:
            if acquired:
                try:
                    connection.scalar(
                        text(
                            """
                            SELECT pg_advisory_unlock(
                                :lock_key
                            )
                            """
                        ),
                        {
                            "lock_key": (
                                _DAILY_PIPELINE_ADVISORY_LOCK_KEY
                            ),
                        },
                    )

                    connection.commit()

                except Exception as e:
                    print(
                        "[DAILY][ADVISORY-LOCK] "
                        "unlock failed: "
                        f"{type(e).__name__}: {e}",
                        flush=True,
                    )

            connection.close()

    return wrapper


class DailyPipelineService:
    KST = ZoneInfo("Asia/Seoul")

    # 종합랭킹 최종 노출 개수
    RANKING_LIMIT = 100

    # CompositeRankingService가 최소 500종목을
    # 평가하므로 수급/밸류/가격/재무 데이터도
    # 동일 후보군까지 갱신합니다.
    UNIVERSE_LIMIT = 500

    TOP_CONTEXT_LIMIT = 100

    DAILY_RUN_HOUR = 16
    DAILY_RUN_MINUTE = 40

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

    def _update_run_progress(
        self,
        run_id: int,
        *,
        stage: str,
        current: int,
        total: int,
        stock_code: str | None = None,
        force: bool = False,
    ) -> None:
        if (
            not force
            and current not in (0, total)
            and current % 10 != 0
        ):
            return

        progress_db = SessionLocal()

        try:
            run = progress_db.get(
                DataSyncRun,
                run_id,
            )

            if (
                run is None
                or run.status != "running"
            ):
                return

            metadata = dict(
                run.metadata_json
                or {}
            )

            percent = (
                round(
                    current
                    / total
                    * 100.0,
                    1,
                )
                if total > 0
                else 0.0
            )

            metadata[
                "progress"
            ] = {
                "stage": stage,
                "current": current,
                "total": total,
                "percent": percent,
                "stockCode": stock_code,
                "updatedAt": (
                    datetime.utcnow()
                    .isoformat()
                ),
            }

            run.metadata_json = (
                metadata
            )

            progress_db.commit()

        except Exception as e:
            progress_db.rollback()

            print(
                "[DAILY][PROGRESS] "
                f"update failed: "
                f"{type(e).__name__}: {e}",
                flush=True,
            )

        finally:
            progress_db.close()

    def _recover_stale_runs(
        self,
    ) -> int:
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
                "orphaned running 상태 자동 복구"
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

            metadata[
                "recoveryReason"
            ] = (
                "advisory_lock_acquired_"
                "with_existing_running_run"
            )

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
        *,
        run_id: int,
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

        self._update_run_progress(
            run_id,
            stage="trading_signals",
            current=0,
            total=total,
        )

        for (
            index,
            stock_code,
        ) in enumerate(
            stock_codes,
            start=1,
        ):
            self._update_run_progress(
                run_id,
                stage="trading_signals",
                current=index,
                total=total,
                stock_code=stock_code,
            )

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
        run_id: int,
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

        history_start = (
            today
            - timedelta(
                days=(
                    self.BOOTSTRAP_PRICE_DAYS
                )
            )
        )

        self._update_run_progress(
            run_id,
            stage="daily_prices",
            current=0,
            total=total,
        )

        for (
            index,
            stock_code,
        ) in enumerate(
            stock_codes,
            start=1,
        ):
            self._update_run_progress(
                run_id,
                stage="daily_prices",
                current=index,
                total=total,
                stock_code=stock_code,
            )

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
        run_id: int,
    ) -> tuple[
        int,
        int,
    ]:
        created = 0
        failed = 0

        total = len(
            stock_codes
        )

        self._update_run_progress(
            run_id,
            stage="financials",
            current=0,
            total=total,
        )

        for (
            index,
            stock_code,
        ) in enumerate(
            stock_codes,
            start=1,
        ):
            self._update_run_progress(
                run_id,
                stage="financials",
                current=index,
                total=total,
                stock_code=stock_code,
            )

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

    @_with_daily_pipeline_advisory_lock
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
            now_kst = datetime.now(
                self.KST
            )

            today = now_kst.date()

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
                    not force
                    and (
                        now_kst.hour
                        < self.DAILY_RUN_HOUR
                        or (
                            now_kst.hour
                            == self.DAILY_RUN_HOUR
                            and now_kst.minute
                            < self.DAILY_RUN_MINUTE
                        )
                    )
                ):
                    result = {
                        "status": "skipped",
                        "reason": (
                            "before_scheduled_time"
                        ),
                        "date": (
                            today.isoformat()
                        ),
                        "currentTimeKst": (
                            now_kst.strftime(
                                "%H:%M"
                            )
                        ),
                        "scheduledTimeKst": (
                            f"{self.DAILY_RUN_HOUR:02d}:"
                            f"{self.DAILY_RUN_MINUTE:02d}"
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

                self._update_run_progress(
                    run_id,
                    stage="toss_market_context",
                    current=0,
                    total=1,
                    force=True,
                )

                (
                    toss_market_context,
                    toss_market_context_failures,
                ) = await (
                    self
                    ._sync_toss_global_market_context()
                )

                self._update_run_progress(
                    run_id,
                    stage="toss_market_context",
                    current=1,
                    total=1,
                    force=True,
                )

                self._update_run_progress(
                    run_id,
                    stage="market_context",
                    current=0,
                    total=1,
                    force=True,
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
                        ),
                        progress_callback=(
                            lambda (
                                current,
                                total,
                                stock_code,
                            ): (
                                self._update_run_progress(
                                    run_id,
                                    stage="market_context",
                                    current=current,
                                    total=total,
                                    stock_code=stock_code,
                                )
                            )
                        ),
                    )
                )

                self._update_run_progress(
                    run_id,
                    stage="market_context",
                    current=1,
                    total=1,
                    force=True,
                )

                self._update_run_progress(
                    run_id,
                    stage="valuation_benchmarks",
                    current=0,
                    total=1,
                    force=True,
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

                self._update_run_progress(
                    run_id,
                    stage="valuation_benchmarks",
                    current=1,
                    total=1,
                    force=True,
                )

                self._update_run_progress(
                    run_id,
                    stage="universe",
                    current=0,
                    total=1,
                    force=True,
                )

                universe = (
                    self.market_data_service
                    .get_current_universe_stock_codes(
                        limit=(
                            self.UNIVERSE_LIMIT
                        )
                    )
                )

                self._update_run_progress(
                    run_id,
                    stage="universe",
                    current=1,
                    total=1,
                    force=True,
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
                        universe,
                        run_id=run_id,
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
                        run_id=run_id,
                    )
                )

                # 일봉이 들어온 뒤
                # 기존 추천 성과를 먼저 평가합니다.
                self._update_run_progress(
                    run_id,
                    stage="evaluate_performance",
                    current=0,
                    total=1,
                    force=True,
                )

                self.market_context_service.evaluate_performance()

                self._update_run_progress(
                    run_id,
                    stage="evaluate_performance",
                    current=1,
                    total=1,
                    force=True,
                )

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

                self._update_run_progress(
                    run_id,
                    stage="current_prices",
                    current=0,
                    total=1,
                    force=True,
                )

                (
                    current_price_success,
                    current_price_failures,
                ) = await (
                    self.stock_service
                    .sync_current_prices(
                        universe
                    )
                )

                self._update_run_progress(
                    run_id,
                    stage="current_prices",
                    current=1,
                    total=1,
                    force=True,
                )

                disclosure_rows = 0
                disclosure_failures = 0

                self._update_run_progress(
                    run_id,
                    stage="global_disclosures",
                    current=0,
                    total=1,
                    force=True,
                )

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

                self._update_run_progress(
                    run_id,
                    stage="global_disclosures",
                    current=1,
                    total=1,
                    force=True,
                )

                global_news_rows = 0
                global_news_failures = 0

                self._update_run_progress(
                    run_id,
                    stage="global_news",
                    current=0,
                    total=1,
                    force=True,
                )

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

                self._update_run_progress(
                    run_id,
                    stage="global_news",
                    current=1,
                    total=1,
                    force=True,
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
                            run_id=run_id,
                        )
                    )

                self._update_run_progress(
                    run_id,
                    stage="ranking",
                    current=0,
                    total=1,
                    force=True,
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

                self._update_run_progress(
                    run_id,
                    stage="ranking",
                    current=1,
                    total=1,
                    force=True,
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

                top_context_rows = rankings[
                    :self.TOP_CONTEXT_LIMIT
                ]

                top_context_total = len(
                    top_context_rows
                )

                self._update_run_progress(
                    run_id,
                    stage="top_context",
                    current=0,
                    total=top_context_total,
                )

                for (
                    index,
                    row,
                ) in enumerate(
                    top_context_rows,
                    start=1,
                ):
                    self._update_run_progress(
                        run_id,
                        stage="top_context",
                        current=index,
                        total=top_context_total,
                        stock_code=(
                            row.stockCode
                        ),
                    )

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

                analysis_total = len(
                    rankings
                )

                self._update_run_progress(
                    run_id,
                    stage="analysis_snapshots",
                    current=0,
                    total=analysis_total,
                )

                for (
                    index,
                    row,
                ) in enumerate(
                    rankings,
                    start=1,
                ):
                    self._update_run_progress(
                        run_id,
                        stage="analysis_snapshots",
                        current=index,
                        total=analysis_total,
                        stock_code=(
                            row.stockCode
                        ),
                    )

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
                    + top_disclosure_failures
                    + global_news_failures
                    + top_news_failures
                    + financial_failures
                    + analysis_snapshot_failures
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
                            + top_disclosure_rows
                        ),
                        "failures": (
                            disclosure_failures
                            + top_disclosure_failures
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
                    "analysisSnapshots": {
                        "success": (
                            analysis_snapshot_success
                        ),
                        "failed": (
                            analysis_snapshot_failures
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

        today_start_kst = datetime.combine(
            today,
            datetime.min.time(),
            tzinfo=self.KST,
        )

        tomorrow_start_kst = (
            today_start_kst
            + timedelta(
                days=1
            )
        )

        utc = ZoneInfo("UTC")

        today_start_utc = (
            today_start_kst
            .astimezone(
                utc
            )
            .replace(
                tzinfo=None
            )
        )

        tomorrow_start_utc = (
            tomorrow_start_kst
            .astimezone(
                utc
            )
            .replace(
                tzinfo=None
            )
        )

        today_run_rows = self.db.execute(
            select(
                DataSyncRun.status,
                func.count(
                    DataSyncRun.id
                ),
            )
            .where(
                DataSyncRun.source
                == "automation",
                DataSyncRun.sync_type
                == "daily_pipeline",
                DataSyncRun.started_at
                >= today_start_utc,
                DataSyncRun.started_at
                < tomorrow_start_utc,
            )
            .group_by(
                DataSyncRun.status
            )
        ).all()

        today_run_counts = {
            str(status): int(
                count
                or 0
            )
            for status, count
            in today_run_rows
        }

        stale_before = (
            datetime.utcnow()
            - timedelta(
                hours=2
            )
        )

        running_runs = self.db.scalars(
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
            )
        ).all()

        stale_running_count = 0

        for running_run in running_runs:
            metadata = (
                running_run.metadata_json
                or {}
            )

            progress = (
                metadata.get(
                    "progress"
                )
                or {}
            )

            last_activity_at = (
                running_run.started_at
            )

            progress_updated_at = (
                progress.get(
                    "updatedAt"
                )
            )

            if progress_updated_at:
                try:
                    last_activity_at = (
                        datetime.fromisoformat(
                            progress_updated_at
                        )
                    )
                except (
                    TypeError,
                    ValueError,
                ):
                    pass

            if (
                last_activity_at
                is not None
                and last_activity_at
                < stale_before
            ):
                stale_running_count += 1

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
                and self._daily_refresh_completed(
                    today
                )
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
            "automationRuns": {
                "today": {
                    "total": sum(
                        today_run_counts.values()
                    ),
                    "running": (
                        today_run_counts.get(
                            "running",
                            0,
                        )
                    ),
                    "completed": (
                        today_run_counts.get(
                            "completed",
                            0,
                        )
                    ),
                    "failed": (
                        today_run_counts.get(
                            "failed",
                            0,
                        )
                    ),
                    "skipped": (
                        today_run_counts.get(
                            "skipped",
                            0,
                        )
                    ),
                },
                "staleRunningCount": (
                    stale_running_count
                ),
            },
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
            "progress": (
                (
                    latest_run.metadata_json
                    or {}
                ).get(
                    "progress"
                )
                if (
                    latest_run
                    and latest_run.status
                    == "running"
                )
                else None
            ),
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