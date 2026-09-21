import asyncio
import secrets
from datetime import date

from fastapi import (
    APIRouter,
    Depends,
    Header,
    HTTPException,
    Query,
)
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.config import (
    settings,
)
from app.core.database import (
    SessionLocal,
    get_db,
)
from app.models.future import (
    RankingSnapshot,
)
from app.models.stock import (
    Stock,
)
from app.services.daily_pipeline_service import (
    DailyPipelineService,
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
router = APIRouter(
    prefix="/automation",
    tags=["automation"],
)


_background_tasks: set[asyncio.Task[None]] = set()


async def _run_daily_pipeline_background(
    *,
    force: bool,
) -> None:
    db = SessionLocal()

    try:
        print(
            "[AUTOMATION][BACKGROUND] "
            f"start force={force}",
            flush=True,
        )

        result = await (
            DailyPipelineService(
                db
            )
            .run(
                force=force
            )
        )

        print(
            "[AUTOMATION][BACKGROUND] "
            "completed "
            f"status={result.get('status')}",
            flush=True,
        )

    except Exception as e:
        db.rollback()

        print(
            "[AUTOMATION][BACKGROUND] "
            "failed "
            f"{type(e).__name__}: {e}",
            flush=True,
        )

    finally:
        db.close()


def _start_daily_pipeline_background(
    *,
    force: bool,
) -> None:
    task = asyncio.create_task(
        _run_daily_pipeline_background(
            force=force,
        )
    )

    _background_tasks.add(
        task
    )

    task.add_done_callback(
        _background_tasks.discard
    )


def _verify_secret(
    provided: str | None,
) -> None:
    expected = (
        settings
        .automation_cron_secret
        .strip()
    )

    if not expected:
        raise HTTPException(
            status_code=503,
            detail=(
                "AUTOMATION_CRON_SECRET가 "
                "설정되지 않았습니다."
            ),
        )

    if (
        provided is None
        or not secrets.compare_digest(
            provided.encode("utf-8"),
            expected.encode("utf-8"),
        )
    ):
        raise HTTPException(
            status_code=401,
            detail=(
                "자동화 인증에 "
                "실패했습니다."
            ),
        )


@router.post(
    "/daily"
)
async def run_daily_pipeline(
    force: bool = Query(
        default=False
    ),
    wait: bool = Query(
        default=False
    ),
    automation_secret: str | None = Header(
        default=None,
        alias="X-Automation-Secret",
    ),
    db: Session = Depends(
        get_db
    ),
):
    _verify_secret(
        automation_secret
    )

    if not wait:
        _start_daily_pipeline_background(
            force=force,
        )

        print(
            "[AUTOMATION][TRIGGER] "
            f"accepted force={force}",
            flush=True,
        )

        return {
            "status": "accepted",
            "mode": "background",
            "force": force,
        }

    try:
        return await (
            DailyPipelineService(
                db
            )
            .run(
                force=force
            )
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=(
                "일일 자동화 실패: "
                f"{e}"
            ),
        ) from e


@router.post(
    "/dev/analysis-snapshots"
)
async def rebuild_analysis_snapshots_for_dev(
    stock_codes: str = Query(
        min_length=6,
        max_length=80,
    ),
    automation_secret: str | None = Header(
        default=None,
        alias="X-Automation-Secret",
    ),
    db: Session = Depends(
        get_db
    ),
):
    _verify_secret(
        automation_secret
    )

    codes = list(
        dict.fromkeys(
            code.strip().upper()
            for code in stock_codes.split(",")
            if code.strip()
        )
    )

    if not codes:
        raise HTTPException(
            status_code=400,
            detail=(
                "stock_codes가 비어 있습니다."
            ),
        )

    if len(codes) > 10:
        raise HTTPException(
            status_code=400,
            detail=(
                "개발용 재분석은 최대 10종목까지 "
                "실행할 수 있습니다."
            ),
        )

    invalid_codes = [
        code
        for code in codes
        if (
            len(code) != 6
            or not code.isalnum()
        )
    ]

    if invalid_codes:
        raise HTTPException(
            status_code=400,
            detail=(
                "잘못된 종목코드: "
                + ", ".join(
                    invalid_codes
                )
            ),
        )

    ranking_snapshot = db.scalar(
        select(
            RankingSnapshot
        )
        .where(
            RankingSnapshot.ranking_version
            == MarketContextService.RANKING_VERSION
        )
        .order_by(
            RankingSnapshot.as_of_date.desc(),
            RankingSnapshot.id.desc(),
        )
        .limit(1)
    )

    if ranking_snapshot is None:
        raise HTTPException(
            status_code=409,
            detail=(
                "기준 RankingSnapshot이 없습니다."
            ),
        )

    target_date = date.today()

    market_data_service = MarketDataService(
        db
    )
    analysis_service = StockAnalysisService(
        db
    )

    def get_sector_sample_state(
        sector_key: str,
    ) -> tuple[
        int,
        int,
        set[str],
    ]:
        rows = (
            market_data_service
            .repository
            .get_valuation_rows_for_date(
                snapshot_date=target_date
            )
        )

        per_count = 0
        pbr_count = 0
        fully_valid_codes: set[str] = set()

        for valuation, _market in rows:
            row_sector_key = (
                valuation.sector_code
                or valuation.sector_name
            )

            if row_sector_key != sector_key:
                continue

            has_per = (
                valuation.per is not None
                and float(valuation.per) > 0.0
            )
            has_pbr = (
                valuation.pbr is not None
                and float(valuation.pbr) > 0.0
            )

            if has_per:
                per_count += 1

            if has_pbr:
                pbr_count += 1

            if has_per and has_pbr:
                fully_valid_codes.add(
                    valuation.stock_code
                )

        return (
            per_count,
            pbr_count,
            fully_valid_codes,
        )

    completed: list[str] = []
    failed: list[dict[str, str]] = []
    sector_benchmarks: list[dict] = []

    for stock_code in codes:
        try:
            stock = db.get(
                Stock,
                stock_code,
            )

            if stock is None:
                raise ValueError(
                    "등록되지 않은 종목입니다: "
                    f"{stock_code}"
                )

            sector_key = (
                stock.sector_code
                or stock.sector_name
            )

            if not sector_key:
                raise ValueError(
                    "업종 정보가 없습니다: "
                    f"{stock_code}"
                )

            (
                sector_member_count,
                minimum_samples,
            ) = (
                market_data_service
                .get_sector_valuation_sample_requirement(
                    sector_key=sector_key
                )
            )

            (
                per_sample_count,
                pbr_sample_count,
                fully_valid_codes,
            ) = get_sector_sample_state(
                sector_key
            )

            if stock_code not in fully_valid_codes:
                await (
                    market_data_service
                    .sync_stock_valuation(
                        stock_code,
                        market_override=stock.market,
                        market_cap_override=(
                            stock.market_cap
                        ),
                    )
                )

                (
                    per_sample_count,
                    pbr_sample_count,
                    fully_valid_codes,
                ) = get_sector_sample_state(
                    sector_key
                )

            peer_filters = [
                Stock.is_active.is_(True),
                Stock.market.in_(
                    [
                        "KOSPI",
                        "KOSDAQ",
                    ]
                ),
                Stock.code != stock_code,
            ]

            if stock.sector_code:
                peer_filters.append(
                    Stock.sector_code
                    == stock.sector_code
                )
            else:
                peer_filters.append(
                    Stock.sector_name
                    == stock.sector_name
                )

            peers = list(
                db.scalars(
                    select(Stock)
                    .where(
                        *peer_filters
                    )
                    .order_by(
                        Stock.market_cap
                        .desc()
                        .nullslast(),
                        Stock.code.asc(),
                    )
                ).all()
            )

            peer_attempts = 0
            peer_failures: list[
                dict[str, str]
            ] = []

            for peer in peers:
                if (
                    per_sample_count
                    >= minimum_samples
                    and pbr_sample_count
                    >= minimum_samples
                ):
                    break

                if peer_attempts >= 20:
                    break

                if peer.code in fully_valid_codes:
                    continue

                peer_attempts += 1

                try:
                    await (
                        market_data_service
                        .sync_stock_valuation(
                            peer.code,
                            market_override=peer.market,
                            market_cap_override=(
                                peer.market_cap
                            ),
                        )
                    )

                except Exception as peer_error:
                    db.rollback()

                    peer_failures.append(
                        {
                            "stockCode": peer.code,
                            "error": str(
                                peer_error
                            ),
                        }
                    )

                (
                    per_sample_count,
                    pbr_sample_count,
                    fully_valid_codes,
                ) = get_sector_sample_state(
                    sector_key
                )

            (
                market_data_service
                .refresh_valuation_benchmarks(
                    snapshot_date=target_date
                )
            )

            (
                per_sample_count,
                pbr_sample_count,
                _fully_valid_codes,
            ) = get_sector_sample_state(
                sector_key
            )

            benchmark_ready = (
                per_sample_count
                >= minimum_samples
                and pbr_sample_count
                >= minimum_samples
            )
            sector_benchmarks.append(
                {
                    "stockCode": stock_code,
                    "sectorKey": sector_key,
                    "perSamples":
                        per_sample_count,
                    "pbrSamples":
                        pbr_sample_count,
                    "sectorMemberCount":
                        sector_member_count,
                    "minimumSamples":
                        minimum_samples,
                    "peerAttempts":
                        peer_attempts,
                    "peerFailures":
                        peer_failures,
                    "status": (
                        "ready"
                        if benchmark_ready
                        else "insufficient"
                    ),
                }
            )

            if not benchmark_ready:
                raise ValueError(
                    "업종 밸류에이션 표본 부족: "
                    f"{sector_key} "
                    f"PER={per_sample_count}/"
                    f"{minimum_samples}, "
                    f"PBR={pbr_sample_count}/"
                    f"{minimum_samples}"
                )

            analysis_service.save_daily_snapshot(
                stock_code=stock_code,
                snapshot_date=target_date,
            )

            completed.append(
                stock_code
            )

        except Exception as e:
            db.rollback()

            failed.append(
                {
                    "stockCode": stock_code,
                    "error": str(e),
                }
            )

    return {
        "status": (
            "completed"
            if not failed
            else "partial"
        ),
        "mode": "dev-analysis-only",
        "snapshotDate": (
            target_date.isoformat()
        ),
        "rankingSnapshotDate": (
            ranking_snapshot
            .as_of_date
            .isoformat()
        ),
        "requested": len(codes),
        "completed": completed,
        "failed": failed,
        "sectorBenchmarks": (
            sector_benchmarks
        ),
    }

@router.get(
    "/runs"
)
def get_automation_runs(
    limit: int = Query(
        default=20,
        ge=1,
        le=50,
    ),
    db: Session = Depends(
        get_db
    ),
):
    return (
        DailyPipelineService(
            db
        )
        .get_run_history(
            limit=limit
        )
    )


@router.get(
    "/status"
)
def get_automation_status(
    db: Session = Depends(
        get_db
    ),
):
    return (
        DailyPipelineService(
            db
        )
        .get_status()
    )