from fastapi import (
    APIRouter,
    Depends,
    HTTPException,
    Query,
)
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.models.future import (
    RankingItem,
    RankingSnapshot,
)
from app.models.stock_price import StockPrice
from app.schemas.ranking import (
    RankingExplanationResponse,
    RankingResponse,
    RankingSnapshotResponse,
)
from app.services.market_context_service import (
    MarketContextService,
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
    "/snapshot",
    response_model=RankingSnapshotResponse,
)
async def get_ranking_snapshot(
    limit: int = Query(
        default=100,
        ge=1,
        le=100,
    ),
    db: Session = Depends(get_db),
):
    snapshot = db.scalar(
        select(
            RankingSnapshot
        )
        .where(
            RankingSnapshot.ranking_version
            == MarketContextService.RANKING_VERSION,
            RankingSnapshot.horizon_days
            == MarketContextService.PRIMARY_HORIZON_DAYS,
            RankingSnapshot.universe
            == "KRX",
        )
        .order_by(
            RankingSnapshot.as_of_date.desc(),
            RankingSnapshot.id.desc(),
        )
        .limit(1)
    )

    if snapshot is None:
        raise HTTPException(
            status_code=503,
            detail=(
                "사용 가능한 랭킹 스냅샷이 없습니다."
            ),
        )

    item_count = int(
        db.scalar(
            select(
                func.count(
                    RankingItem.id
                )
            )
            .where(
                RankingItem.snapshot_id
                == snapshot.id
            )
        )
        or 0
    )

    latest_market_date = db.scalar(
        select(
            func.max(
                StockPrice.trade_date
            )
        )
    )

    rankings = await StockService(
        db
    ).get_rankings(
        limit=limit,
    )

    integrity_ok = (
        item_count == 100
        and len(rankings) == limit
    )

    if not integrity_ok:
        raise HTTPException(
            status_code=503,
            detail=(
                "랭킹 스냅샷이 불완전합니다. "
                f"items={item_count}, "
                f"requested={limit}, "
                f"loaded={len(rankings)}"
            ),
        )

    is_stale = (
        latest_market_date is not None
        and snapshot.as_of_date
        < latest_market_date
    )

    return RankingSnapshotResponse(
        asOfDate=(
            snapshot.as_of_date
        ),
        latestMarketDate=(
            latest_market_date
        ),
        rankingVersion=(
            snapshot.ranking_version
        ),
        horizonDays=int(
            snapshot.horizon_days
        ),
        universe=(
            snapshot.universe
        ),
        itemCount=item_count,
        requestedCount=limit,
        integrityOk=True,
        isStale=is_stale,
        items=rankings,
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

    return rankings