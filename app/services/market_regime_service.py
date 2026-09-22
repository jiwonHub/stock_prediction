from __future__ import annotations

from sqlalchemy.orm import Session

from app.repositories.market_data_repository import (
    MarketDataRepository,
)
from app.services.market_regime_guidance_service import (
    MarketRegimeGuidanceService,
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

    @staticmethod
    def _build_interpretation(
        *,
        trend: str,
        volatility: str,
        risk: str,
    ) -> dict:
        if risk == "RISK_ON":
            state = "FAVORABLE"
            label = "우호"

            if volatility == "LOW":
                description = (
                    "상승 추세가 유지되고 변동성도 "
                    "낮아 시장 환경이 우호적입니다."
                )
            else:
                description = (
                    "상승 추세가 유지되고 변동성이 "
                    "과도하지 않아 시장 환경이 "
                    "우호적입니다."
                )

        elif risk == "RISK_OFF":
            state = "RISKY"
            label = "위험"

            if trend == "BEAR":
                description = (
                    "하락 추세 신호가 나타나 시장 "
                    "위험이 높은 구간입니다."
                )
            else:
                description = (
                    "변동성과 낙폭이 확대되어 시장 "
                    "위험이 높은 구간입니다."
                )

        else:
            state = "NEUTRAL"
            label = "중립"

            if volatility == "HIGH":
                description = (
                    "추세 방향은 뚜렷하지 않지만 "
                    "변동성이 높아 보수적인 대응이 "
                    "필요합니다."
                )
            elif trend == "BULL":
                description = (
                    "상승 흐름은 있으나 위험 선호 "
                    "신호가 충분히 강하지 않아 "
                    "중립 구간으로 판단합니다."
                )
            elif trend == "BEAR":
                description = (
                    "약한 하락 흐름이 있으나 위험 "
                    "회피 신호가 확정적이지 않아 "
                    "중립 구간으로 판단합니다."
                )
            else:
                description = (
                    "추세와 변동성 신호가 혼재되어 "
                    "방향성이 뚜렷하지 않은 "
                    "중립 구간입니다."
                )

        return {
            "state": state,
            "label": label,
            "description": description,
        }

    def get_market_state(
        self,
        *,
        market: str,
    ) -> dict:
        inspected = (
            self.inspect_market(
                market=market
            )
        )

        regime = inspected[
            "regime"
        ]

        interpretation = (
            self._build_interpretation(
                trend=regime[
                    "trend"
                ],
                volatility=regime[
                    "volatility"
                ],
                risk=regime[
                    "risk"
                ],
            )
        )

        guidance = (
            MarketRegimeGuidanceService.build(
                trend=regime[
                    "trend"
                ],
                volatility=regime[
                    "volatility"
                ],
                risk=regime[
                    "risk"
                ],
            )
        )

        return {
            "available": True,
            "market": inspected[
                "market"
            ],
            "featureDate": inspected[
                "feature_date"
            ],
            "state": interpretation[
                "state"
            ],
            "label": interpretation[
                "label"
            ],
            "description": interpretation[
                "description"
            ],
            "trend": regime[
                "trend"
            ],
            "volatility": regime[
                "volatility"
            ],
            "risk": regime[
                "risk"
            ],
            "guidance":
                guidance,
        }

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
            .get_latest_market_index_prices(
                index_code=(
                    index_code
                ),
                limit=260,
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