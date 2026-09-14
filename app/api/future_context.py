from fastapi import (
    APIRouter,
    Depends,
    HTTPException,
    Query,
)
from sqlalchemy.orm import Session

from app.core.database import get_db

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
def get_news(
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
    return MarketContextService(
        db
    ).get_news(
        stock_code=stock_code,
        limit=limit,
    )

@router.get("/disclosures")
def get_disclosures(
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
    return MarketContextService(
        db
    ).get_disclosures(
        stock_code=stock_code,
        limit=limit,
    )


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