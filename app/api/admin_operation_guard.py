from collections.abc import AsyncIterator

from fastapi import HTTPException

from app.services.daily_pipeline_service import (
    DailyPipelineService,
)


async def ensure_daily_pipeline_not_running(
) -> AsyncIterator[None]:
    async with (
        DailyPipelineService
        .manual_operation_guard()
    ) as acquired:
        if not acquired:
            raise HTTPException(
                status_code=409,
                detail=(
                    "일일 자동화 실행 중에는 "
                    "수동 데이터 갱신을 "
                    "실행할 수 없습니다."
                ),
            )

        yield