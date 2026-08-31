from __future__ import annotations

from datetime import date, timedelta

from sqlalchemy.orm import Session

from app.repositories.market_data_repository import (
    MarketDataRepository,
)
from app.utils.market_regime_features import (
    MARKET_REGIME_FEATURE_NAMES,
    build_latest_market_regime_features,
)


class MarketRegimeService:
    MARKET_INDEX_CODES = {
        "KOSPI":
            "0001",

        "KOSDAQ":
            "1001",
    }

    def __init__(
        self,
        db: Session,
    ):
        self.db = db

        self.market_repository = (
            MarketDataRepository(
                db
            )
        )

    @staticmethod
    def _trend_label(
        score: float,
    ) -> str:
        if score > 0.0:
            return "BULL"

        if score < 0.0:
            return "BEAR"

        return "SIDEWAYS"

    @staticmethod
    def _volatility_label(
        score: float,
    ) -> str:
        if score > 0.0:
            return "HIGH"

        if score < 0.0:
            return "LOW"

        return "NORMAL"

    @staticmethod
    def _risk_label(
        score: float,
    ) -> str:
        if score > 0.0:
            return "RISK_ON"

        if score < 0.0:
            return "RISK_OFF"

        return "NEUTRAL"

    def inspect_market(
        self,
        *,
        market: str,
    ) -> dict:
        normalized_market = (
            market
            .strip()
            .upper()
        )

        index_code = (
            self.MARKET_INDEX_CODES
            .get(
                normalized_market
            )
        )

        if index_code is None:
            raise ValueError(
                "market은 KOSPI 또는 "
                "KOSDAQ이어야 합니다."
            )

        rows = (
            self.market_repository
            .get_market_index_prices(
                index_code=(
                    index_code
                ),
                start_date=(
                    date.today()
                    - timedelta(
                        days=600
                    )
                ),
            )
        )

        rows = sorted(
            rows,
            key=lambda row:
                row.trade_date,
        )

        if len(
            rows
        ) < 61:
            raise ValueError(
                "시장지수 데이터가 부족합니다: "
                f"{normalized_market} "
                f"{len(rows)}건"
            )

        features = (
            build_latest_market_regime_features(
                rows
            )
        )

        trend_score = (
            features[
                "market_trend_regime_score"
            ]
        )

        volatility_score = (
            features[
                "market_volatility_regime_score"
            ]
        )

        risk_score = (
            features[
                "market_risk_regime_score"
            ]
        )

        return {
            "status":
                "pass",

            "market":
                normalized_market,

            "index_code":
                index_code,

            "feature_date":
                rows[
                    -1
                ]
                .trade_date
                .isoformat(),

            "price_rows":
                len(
                    rows
                ),

            "feature_count":
                len(
                    features
                ),

            "expected_feature_count":
                len(
                    MARKET_REGIME_FEATURE_NAMES
                ),

            "regime": {
                "trend":
                    self._trend_label(
                        trend_score
                    ),

                "volatility":
                    self._volatility_label(
                        volatility_score
                    ),

                "risk":
                    self._risk_label(
                        risk_score
                    ),
            },

            "features":
                features,
        }