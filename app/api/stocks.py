from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.core.exceptions import ConfigurationError, ExternalApiError
from app.schemas.stock import ChartPointResponse, StockResponse
from app.services.stock_service import StockService


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
    "/{stock_code}/chart",
    response_model=list[ChartPointResponse],
)
async def get_chart(
    stock_code: str,
    period: str = Query(
        default="1M",
        pattern="^(1D|1W|1M|3M|1Y)$",
    ),
    refresh: bool = True,
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
    "/{stock_code}",
    response_model=StockResponse,
)
async def get_stock(
    stock_code: str,
    refresh: bool = True,
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
