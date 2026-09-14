from datetime import date
from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import Response
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.core.exceptions import ConfigurationError, ExternalApiError
from app.schemas.stock import (
    ChartPageResponse,
    ChartPointResponse,
    StockAnalysisResponse,
    StockResponse,
)
from app.services.stock_analysis_service import (
    StockAnalysisService,
)
from app.services.stock_service import StockService
from app.services.stock_logo_service import (
    StockLogoService,
)


router = APIRouter(
    prefix="/stocks",
    tags=["stocks"],
)


@router.get(
    "/search",
    response_model=list[StockResponse],
)
async def search_stocks(
    q: str = Query(
        min_length=1,
        max_length=100,
    ),
    db: Session = Depends(get_db),
):
    service = StockService(db)

    return await service.search_stocks(
        q
    )

@router.get(
    "/{stock_code}/logo",
)
async def get_stock_logo(
    stock_code: str,
    db: Session = Depends(
        get_db
    ),
):
    try:
        result = (
            await StockLogoService(
                db
            ).get_logo(
                stock_code=(
                    stock_code
                ),
            )
        )

        if result is None:
            raise HTTPException(
                status_code=404,
                detail=(
                    "회사 공식 로고를 찾을 수 없습니다."
                ),
            )

        return Response(
            content=result.body,
            media_type=(
                result.content_type
            ),
            headers={
                "Cache-Control": (
                    "no-store"
                ),
                "X-Logo-Source": (
                    result.source
                ),
                "X-Logo-Asset": (
                    result.asset_url
                ),
            },
        )

    except ValueError as e:
        raise HTTPException(
            status_code=404,
            detail=str(e),
        ) from e

    except ConfigurationError as e:
        raise HTTPException(
            status_code=500,
            detail=str(e),
        ) from e

    except ExternalApiError as e:
        raise HTTPException(
            status_code=502,
            detail=str(e),
        ) from e

@router.get(
    "/{stock_code}/chart/page",
    response_model=ChartPageResponse,
)
def get_chart_page(
    stock_code: str,
    interval: str = Query(
        default="1d",
    ),
    count: int = Query(
        default=200,
        ge=1,
        le=200,
    ),
    before: date | None = Query(
        default=None,
    ),
    db: Session = Depends(
        get_db
    ),
):
    service = StockService(
        db
    )

    try:
        return service.get_chart_page(
            stock_code,
            interval=interval,
            before=before,
            count=count,
        )

    except ValueError as e:
        raise HTTPException(
            status_code=404,
            detail=str(e),
        ) from e

@router.get(
    "/{stock_code}/chart",
    response_model=list[ChartPointResponse],
)
async def get_chart(
    stock_code: str,
    period: str = Query(
        default="1M",
        pattern="^(1D|1W|1M|3M|1Y)$",
    ),
    refresh: bool = False,
    db: Session = Depends(get_db),
):
    service = StockService(db)

    try:
        return await service.get_chart(
            stock_code,
            period=period,
            refresh=refresh,
        )
    except ValueError as e:
        raise HTTPException(
            status_code=404,
            detail=str(e),
        ) from e
    except ConfigurationError as e:
        raise HTTPException(
            status_code=500,
            detail=str(e),
        ) from e
    except ExternalApiError as e:
        raise HTTPException(
            status_code=502,
            detail=str(e),
        ) from e


@router.get(
    "/{stock_code}/analysis",
    response_model=StockAnalysisResponse,
)
def get_stock_analysis(
    stock_code: str,
    db: Session = Depends(get_db),
):
    try:
        return StockAnalysisService(
            db
        ).get_current_analysis(
            stock_code=stock_code,
        )

    except ValueError as e:
        raise HTTPException(
            status_code=404,
            detail=str(e),
        ) from e


@router.get(
    "/{stock_code}",
    response_model=StockResponse,
)
async def get_stock(
    stock_code: str,
    refresh: bool = False,
    db: Session = Depends(get_db),
):
    service = StockService(db)

    try:
        return await service.get_stock(
            stock_code,
            refresh=refresh,
        )
    except ValueError as e:
        raise HTTPException(
            status_code=404,
            detail=str(e),
        ) from e
    except ConfigurationError as e:
        raise HTTPException(
            status_code=500,
            detail=str(e),
        ) from e
    except ExternalApiError as e:
        raise HTTPException(
            status_code=502,
            detail=str(e),
        ) from e
