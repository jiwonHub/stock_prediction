import asyncio
import secrets

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
from app.services.daily_pipeline_service import (
    DailyPipelineService,
)
from app.services.market_context_service import (
    MarketContextService,
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
def rebuild_analysis_snapshots_for_dev(
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

    analysis_service = StockAnalysisService(
        db
    )

    completed: list[str] = []
    failed: list[dict[str, str]] = []

    for stock_code in codes:
        try:
            analysis_service.save_daily_snapshot(
                stock_code=stock_code,
                snapshot_date=(
                    ranking_snapshot.as_of_date
                ),
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
            ranking_snapshot
            .as_of_date
            .isoformat()
        ),
        "requested": len(codes),
        "completed": completed,
        "failed": failed,
    }


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