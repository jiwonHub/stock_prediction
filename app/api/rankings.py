from fastapi import (
    APIRouter,
    Depends,
    HTTPException,
    Query,
)
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.schemas.ranking import (
    RankingExplanationResponse,
    RankingResponse,
)
from app.services.ranking_explanation_service import (
    RankingExplanationService,
)
from app.services.stock_service import StockService


router = APIRouter(
    prefix="/rankings",
    tags=["rankings"],
)


@router.get(
    "/{stock_code}/explanation",
    response_model=RankingExplanationResponse,
)
def get_ranking_explanation(
    stock_code: str,
    top_k: int = Query(
        default=3,
        ge=1,
        le=5,
    ),
    db: Session = Depends(get_db),
):
    try:
        return RankingExplanationService(
            db
        ).get_current_explanation(
            stock_code,
            top_k=top_k,
        )

    except ValueError as e:
        raise HTTPException(
            status_code=404,
            detail=str(e),
        ) from e


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

    if rankings:
        from app.services.market_context_service import (
            MarketContextService,
        )

        MarketContextService(
            db
        ).record_rankings(
            rankings
        )

    return rankings