from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.schemas.ranking import RankingResponse
from app.services.market_context_service import (
    MarketContextService,
)
from app.services.stock_service import StockService


router = APIRouter(
    prefix="/rankings",
    tags=["rankings"],
)


@router.get(
    "",
    response_model=list[RankingResponse],
)
async def get_rankings(
    limit: int = Query(
        default=100,
        ge=1,
        le=100,
    ),
    db: Session = Depends(get_db),
):
    service = StockService(db)

    rankings = await service.get_rankings(
        limit=limit,
    )

    missing_codes = [
        item.stockCode
        for item in rankings
        if item.currentPrice <= 0.0
    ]

    if missing_codes:
        await service.sync_current_prices(
            missing_codes
        )

        rankings = await service.get_rankings(
            limit=limit,
        )

    MarketContextService(
        db
    ).record_rankings(
        rankings
    )

    return rankings