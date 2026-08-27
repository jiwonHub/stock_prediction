from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.core.exceptions import ConfigurationError, ExternalApiError
from app.schemas.stock import StockResponse, SyncResponse
from app.services.stock_service import StockService


router = APIRouter(
    prefix="/admin/sync",
    tags=["admin-sync"],
)


@router.post(
    "/stocks",
    response_model=SyncResponse,
)
async def sync_stocks(
    db: Session = Depends(get_db),
):
    service = StockService(db)

    try:
        count = await service.sync_stocks_from_dart()
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

    return SyncResponse(
        count=count,
        message=f"DART 상장 종목 {count}건 저장 완료",
    )

@router.post(
    "/prices",
    response_model=SyncResponse,
)
async def sync_prices(
    limit: int = Query(
        default=100,
        ge=1,
        le=100,
    ),
    db: Session = Depends(get_db),
):
    service = StockService(db)

    try:
        success, skipped = (
            await service.sync_ranked_current_prices(
                limit=limit,
            )
        )
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

    return SyncResponse(
        count=success,
        message=(
            f"TOP {limit} 현재가 "
            f"{success}건 갱신, "
            f"{skipped}건 건너뜀"
        ),
    )

@router.post(
    "/price/{stock_code}",
    response_model=StockResponse,
)
async def sync_price(
    stock_code: str,
    db: Session = Depends(get_db),
):
    service = StockService(db)

    try:
        return await service.sync_current_price(
            stock_code
        )
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


@router.post(
    "/chart/{stock_code}",
    response_model=SyncResponse,
)
async def sync_chart(
    stock_code: str,
    days: int = Query(
        default=365,
        ge=7,
        le=3650,
    ),
    db: Session = Depends(get_db),
):
    service = StockService(db)

    try:
        count = await service.sync_daily_prices(
            stock_code,
            days=days,
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

    return SyncResponse(
        count=count,
        message=f"{stock_code} 일봉 {count}건 저장 완료",
    )


@router.post(
    "/financials/{stock_code}",
    response_model=SyncResponse,
)
async def sync_financials(
    stock_code: str,
    year: int = Query(
        ge=2015,
        le=2100,
    ),
    report_code: str = Query(
        default="11011",
        pattern="^(11011|11012|11013|11014)$",
    ),
    db: Session = Depends(get_db),
):
    service = StockService(db)

    try:
        count, fs_div = await service.sync_financials(
            stock_code,
            business_year=str(year),
            report_code=report_code,
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

    return SyncResponse(
        count=count,
        message=(
            f"{stock_code} {year} {report_code} "
            f"{fs_div} 재무제표 {count}건 저장 완료"
        ),
    )
