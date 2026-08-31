from __future__ import annotations

from datetime import date, timedelta

from sqlalchemy.orm import Session

from app.repositories.market_data_repository import (
    MarketDataRepository,
)
from app.repositories.stock_repository import (
    StockRepository,
)
from app.utils.sector_regime_features import (
    SECTOR_CONTEXT_FEATURE_NAMES,
    build_latest_sector_regime_features,
)


class SectorRegimeService:
    MARKET_INDEX_CODES = {
        "KOSPI": "0001",
        "KOSDAQ": "1001",
    }

    PERIODS = (
        5,
        20,
        60,
    )

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

        self.stock_repository = (
            StockRepository(
                db
            )
        )

    @staticmethod
    def _close_on_or_before(
        rows: list,
        target_date: date,
    ) -> float | None:
        for row in reversed(
            rows
        ):
            if row.trade_date > target_date:
                continue

            close = float(
                row.close
            )

            if close > 0.0:
                return close

        return None

    @classmethod
    def _return_between_dates(
        cls,
        rows: list,
        *,
        start_date: date,
        end_date: date,
    ) -> float | None:
        start_close = (
            cls._close_on_or_before(
                rows,
                start_date,
            )
        )

        end_close = (
            cls._close_on_or_before(
                rows,
                end_date,
            )
        )

        if (
            start_close is None
            or end_close is None
            or start_close <= 0.0
        ):
            return None

        return (
            end_close
            / start_close
            - 1.0
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

    def inspect_stock(
        self,
        *,
        stock_code: str,
    ) -> dict:
        stock = (
            self.stock_repository
            .get_stock(
                stock_code
            )
        )

        if stock is None:
            raise ValueError(
                "등록되지 않은 종목입니다: "
                f"{stock_code}"
            )

        market = str(
            stock.market
            or ""
        ).strip()

        sector_code = str(
            stock.sector_code
            or ""
        ).strip()

        sector_name = str(
            stock.sector_name
            or ""
        ).strip()

        market_index_code = (
            self.MARKET_INDEX_CODES
            .get(
                market
            )
        )

        if market_index_code is None:
            raise ValueError(
                "시장 구분이 올바르지 않습니다: "
                f"{stock_code} {market}"
            )

        if not sector_code:
            raise ValueError(
                "업종 코드가 없습니다: "
                f"{stock_code}"
            )

        start_date = (
            date.today()
            - timedelta(
                days=600
            )
        )

        sector_rows = (
            self.market_repository
            .get_sector_index_prices(
                sector_code=(
                    sector_code
                ),
                market=market,
                start_date=(
                    start_date
                ),
            )
        )

        market_rows = (
            self.market_repository
            .get_market_index_prices(
                index_code=(
                    market_index_code
                ),
                start_date=(
                    start_date
                ),
            )
        )

        if len(sector_rows) < 61:
            raise ValueError(
                "업종지수 61거래일 부족: "
                f"{stock_code} "
                f"{len(sector_rows)}건"
            )

        if len(market_rows) < 61:
            raise ValueError(
                "시장지수 61거래일 부족: "
                f"{stock_code} "
                f"{len(market_rows)}건"
            )

        as_of_date = min(
            sector_rows[-1].trade_date,
            market_rows[-1].trade_date,
        )

        sector_rows = [
            row
            for row
            in sector_rows
            if row.trade_date
            <= as_of_date
        ]

        market_rows = [
            row
            for row
            in market_rows
            if row.trade_date
            <= as_of_date
        ]

        sector_regime = (
            build_latest_sector_regime_features(
                sector_rows
            )
        )

        relative_momentum: dict[
            str,
            float,
        ] = {}

        for period in self.PERIODS:
            period_start_date = (
                market_rows[
                    -(period + 1)
                ].trade_date
            )

            sector_return = (
                self._return_between_dates(
                    sector_rows,
                    start_date=(
                        period_start_date
                    ),
                    end_date=(
                        as_of_date
                    ),
                )
            )

            market_return = (
                self._return_between_dates(
                    market_rows,
                    start_date=(
                        period_start_date
                    ),
                    end_date=(
                        as_of_date
                    ),
                )
            )

            if (
                sector_return is None
                or market_return is None
            ):
                raise ValueError(
                    "업종 상대 모멘텀 계산 데이터 부족: "
                    f"{stock_code} "
                    f"{period}거래일"
                )

            relative_momentum[
                f"sector_market_excess_{period}d"
            ] = (
                sector_return
                - market_return
            )

        features = {
            **sector_regime,
            **relative_momentum,
        }

        trend_score = (
            features[
                "sector_trend_regime_score"
            ]
        )

        volatility_score = (
            features[
                "sector_volatility_regime_score"
            ]
        )

        risk_score = (
            features[
                "sector_risk_regime_score"
            ]
        )

        return {
            "status": "pass",
            "stock_code": stock_code,
            "market": market,
            "sector_code": sector_code,
            "sector_name": sector_name,
            "feature_date": (
                as_of_date.isoformat()
            ),
            "sector_price_rows": len(
                sector_rows
            ),
            "market_price_rows": len(
                market_rows
            ),
            "feature_count": len(
                features
            ),
            "expected_feature_count": len(
                SECTOR_CONTEXT_FEATURE_NAMES
            ),
            "regime": {
                "trend": (
                    self._trend_label(
                        trend_score
                    )
                ),
                "volatility": (
                    self._volatility_label(
                        volatility_score
                    )
                ),
                "risk": (
                    self._risk_label(
                        risk_score
                    )
                ),
            },
            "features": features,
        }