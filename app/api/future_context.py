from xml.etree import ElementTree

import httpx
from fastapi import (
    APIRouter,
    Depends,
    HTTPException,
    Query,
)
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.core.exceptions import (
    ConfigurationError,
    ExternalApiError,
)
from app.services.market_context_service import (
    MarketContextService,
)


router = APIRouter(
    tags=["market-context"]
)


@router.get("/news")
async def get_news(
    stock_code: str | None = Query(
        default=None
    ),
    limit: int = Query(
        default=100,
        ge=1,
        le=500,
    ),
    db: Session = Depends(get_db),
):
    service = MarketContextService(
        db
    )

    try:
        current = service.get_news(
            stock_code=stock_code,
            limit=limit,
        )

        if len(current) < min(
            10,
            limit,
        ):
            await service.sync_news(
                stock_code=stock_code,
                limit=limit,
            )

            current = service.get_news(
                stock_code=stock_code,
                limit=limit,
            )

        return current

    except (
        httpx.HTTPError,
        ElementTree.ParseError,
    ) as e:
        raise HTTPException(
            status_code=502,
            detail=(
                f"뉴스 수집 실패: {e}"
            ),
        ) from e


@router.get("/disclosures")
async def get_disclosures(
    stock_code: str | None = Query(
        default=None
    ),
    limit: int = Query(
        default=100,
        ge=1,
        le=500,
    ),
    db: Session = Depends(get_db),
):
    service = MarketContextService(
        db
    )

    try:
        current = (
            service.get_disclosures(
                stock_code=stock_code,
                limit=limit,
            )
        )

        if len(current) < min(
            10,
            limit,
        ):
            await service.sync_disclosures(
                stock_code=stock_code,
                limit=limit,
            )

            current = (
                service.get_disclosures(
                    stock_code=stock_code,
                    limit=limit,
                )
            )

        return current

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
    "/recommendations/performance"
)
def get_recommendation_performance(
    limit: int = Query(
        default=100,
        ge=1,
        le=500,
    ),
    db: Session = Depends(get_db),
):
    return MarketContextService(
        db
    ).get_performance(
        limit=limit
    )