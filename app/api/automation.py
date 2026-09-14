import secrets

from fastapi import (
    APIRouter,
    BackgroundTasks,
    Depends,
    Header,
    HTTPException,
    Query,
)
from sqlalchemy.orm import Session

from app.core.config import (
    settings,
)
from app.core.database import (
    SessionLocal,
    get_db,
)
from app.services.daily_pipeline_service import (
    DailyPipelineService,
)


router = APIRouter(
    prefix="/automation",
    tags=["automation"],
)


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
            provided,
            expected,
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
    background_tasks: BackgroundTasks,
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
        background_tasks.add_task(
            _run_daily_pipeline_background,
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