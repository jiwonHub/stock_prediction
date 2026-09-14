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
from app.services.market_brief_service import (
    MarketBriefService,
)
from app.services.market_data_service import (
    MarketDataService,
)
from app.services.market_regime_service import (
    MarketRegimeService,
)
from app.services.ranking_backtest_service import (
    RankingBacktestService,
)


router = APIRouter(
    tags=["market-context"]
)


@router.get(
    "/market-context/summary"
)
def get_market_context_summary(
    days: int = Query(
        default=30,
        ge=1,
        le=365,
    ),
    db: Session = Depends(
        get_db
    ),
):
    result = (
        MarketDataService(
            db
        )
        .get_toss_market_context_summary(
            days=days
        )
    )

    regime_service = (
        MarketRegimeService(
            db
        )
    )

    regimes = {}

    for market in (
        "KOSPI",
        "KOSDAQ",
    ):
        try:
            regimes[
                market
            ] = (
                regime_service
                .get_market_state(
                    market=market
                )
            )

        except ValueError as e:
            regimes[
                market
            ] = {
                "available": False,
                "market": market,
                "featureDate": None,
                "state": None,
                "label": None,
                "description": str(
                    e
                ),
                "trend": None,
                "volatility": None,
                "risk": None,
            }

    summary = (
        result.setdefault(
            "summary",
            {},
        )
    )

    summary[
        "regimes"
    ] = regimes

    summary[
        "brief"
    ] = (
        MarketBriefService.build(
            summary
        )
    )

    return result


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

@router.get(
    "/recommendations/backtest"
)
def get_recommendation_backtest(
    db: Session = Depends(
        get_db
    ),
):
    return RankingBacktestService(
        db
    ).build_report()