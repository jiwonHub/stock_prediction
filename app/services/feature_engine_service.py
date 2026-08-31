from __future__ import annotations

import math
from datetime import date, timedelta

from sqlalchemy.orm import Session

from app.repositories.feature_repository import (
    FeatureRepository,
)
from app.repositories.market_data_repository import (
    MarketDataRepository,
)
from app.repositories.stock_repository import (
    StockRepository,
)
from app.services.market_data_service import (
    MarketDataService,
)
from app.utils.ml_features import (
    build_latest_feature_dict,
)
from app.utils.advanced_technical_features import (
    build_advanced_technical_features,
)
from app.utils.market_regime_features import (
    build_latest_market_regime_features,
)
from app.utils.sector_regime_features import (
    build_latest_sector_regime_features,
)

class FeatureEngineService:
    FEATURE_VERSION = "v4-sector-1"

    MACRO_CODES = (
        "BOK_BASE_RATE",
        "USD_KRW",
        "KTB_3Y",
        "KTB_10Y",
    )

    def __init__(
        self,
        db: Session,
    ):
        self.db = db

        self.feature_repository = (
            FeatureRepository(
                db
            )
        )

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

        self.market_service = (
            MarketDataService(
                db
            )
        )

        self._macro_cache: dict[
            tuple[str, date],
            list,
        ] = {}

    @staticmethod
    def _float_or_none(
        value,
    ) -> float | None:
        if value is None:
            return None

        try:
            return float(
                value
            )
        except (
            TypeError,
            ValueError,
        ):
            return None

    @classmethod
    def _percent_to_ratio(
        cls,
        value,
    ) -> float | None:
        parsed = (
            cls._float_or_none(
                value
            )
        )

        if parsed is None:
            return None

        return (
            parsed
            / 100.0
        )

    @classmethod
    def _relative_ratio(
        cls,
        value,
        benchmark,
    ) -> float | None:
        left = (
            cls._float_or_none(
                value
            )
        )

        right = (
            cls._float_or_none(
                benchmark
            )
        )

        if (
            left is None
            or right is None
            or left <= 0.0
            or right <= 0.0
        ):
            return None

        return (
            left
            / right
            - 1.0
        )

    @staticmethod
    def _latest_macro_value(
        rows: list,
        *,
        target_date: date,
    ) -> float | None:
        for row in reversed(
            rows
        ):
            if (
                row.observed_date
                <= target_date
            ):
                return float(
                    row.value
                )

        return None

    @classmethod
    def _macro_change(
        cls,
        rows: list,
        *,
        end_date: date,
        calendar_days: int,
        percent: bool,
    ) -> float | None:
        current = (
            cls._latest_macro_value(
                rows,
                target_date=end_date,
            )
        )

        previous = (
            cls._latest_macro_value(
                rows,
                target_date=(
                    end_date
                    - timedelta(
                        days=calendar_days
                    )
                ),
            )
        )

        if (
            current is None
            or previous is None
        ):
            return None

        if percent:
            if previous == 0.0:
                return None

            return (
                current
                / previous
                - 1.0
            )

        return (
            current
            - previous
        )

    def _get_macro_rows(
        self,
        *,
        indicator_code: str,
        feature_date: date,
    ) -> list:
        key = (
            indicator_code,
            feature_date,
        )

        cached = (
            self._macro_cache
            .get(
                key
            )
        )

        if cached is not None:
            return cached

        rows = (
            self.market_repository
            .get_macro_indicators(
                indicator_code=(
                    indicator_code
                ),
                start_date=(
                    feature_date
                    - timedelta(
                        days=400
                    )
                ),
                end_date=(
                    feature_date
                ),
            )
        )

        self._macro_cache[
            key
        ] = rows

        return rows

    @staticmethod
    def _flow_ratio(
        *,
        flow_rows: list,
        price_rows: list,
        field_name: str,
        periods: int,
        feature_date: date,
    ) -> float | None:
        price_volume = {
            row.trade_date:
                float(
                    row.volume
                    or 0
                )
            for row
            in price_rows
            if (
                row.trade_date
                <= feature_date
            )
        }

        eligible = [
            row
            for row
            in flow_rows
            if (
                row.trade_date
                <= feature_date
                and row.trade_date
                in price_volume
            )
        ]

        selected = (
            eligible[
                -periods:
            ]
        )

        if not selected:
            return None

        denominator = sum(
            price_volume[
                row.trade_date
            ]
            for row
            in selected
        )

        if denominator <= 0.0:
            return None

        numerator = sum(
            float(
                getattr(
                    row,
                    field_name,
                )
                or 0
            )
            for row
            in selected
        )

        return (
            numerator
            / denominator
        )

    def get_current_universe_stock_codes(
        self,
        *,
        limit: int = 100,
    ) -> list[str]:
        return (
            self.market_service
            .get_current_universe_stock_codes(
                limit=limit,
            )
        )

    def build_current_snapshot_row(
        self,
        stock_code: str,
        *,
        feature_version:
            str | None = None,
    ) -> tuple[
        dict,
        list[str],
    ]:
        relative = (
            self.market_service
            .calculate_stock_relative_strength(
                stock_code
            )
        )

        feature_date = (
            date.fromisoformat(
                relative[
                    "as_of_date"
                ]
            )
        )

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

        if market == "KOSPI":
            market_index_code = "0001"

        elif market == "KOSDAQ":
            market_index_code = "1001"

        else:
            raise ValueError(
                "시장 구분이 올바르지 않습니다: "
                f"{stock_code} {market}"
            )

        if not sector_code:
            raise ValueError(
                "업종 코드가 없습니다: "
                f"{stock_code}"
            )

        regime_rows = (
            self.market_repository
            .get_market_index_prices(
                index_code=(
                    market_index_code
                ),
                start_date=(
                    feature_date
                    - timedelta(
                        days=600
                    )
                ),
            )
        )

        regime_rows = sorted(
            (
                row
                for row
                in regime_rows
                if row.trade_date
                <= feature_date
            ),
            key=lambda row:
                row.trade_date,
        )

        market_regime = (
            build_latest_market_regime_features(
                regime_rows
            )
        )

        sector_regime_rows = (
            self.market_repository
            .get_sector_index_prices(
                sector_code=(
                    sector_code
                ),
                market=market,
                start_date=(
                    feature_date
                    - timedelta(
                        days=600
                    )
                ),
                end_date=(
                    feature_date
                ),
            )
        )

        sector_regime = (
            build_latest_sector_regime_features(
                sector_regime_rows
            )
        )

        price_rows = (
            self.stock_repository
            .get_daily_prices(
                stock_code=(
                    stock_code
                ),
                start_date=(
                    feature_date
                    - timedelta(
                        days=240
                    )
                ),
            )
        )

        price_rows = [
            row
            for row
            in price_rows
            if (
                row.trade_date
                <= feature_date
            )
        ]

        technical = (
            build_latest_feature_dict(
                price_rows
            )
        )

        advanced_technical = (
            build_advanced_technical_features(
                price_rows
            )
        )

        flow_rows = (
            self.market_repository
            .get_stock_investor_flows(
                stock_code=(
                    stock_code
                ),
                start_date=(
                    feature_date
                    - timedelta(
                        days=120
                    )
                ),
                end_date=(
                    feature_date
                ),
            )
        )

        valuation = (
            self.market_repository
            .get_latest_valuation(
                stock_code
            )
        )

        macro_rows = {
            code:
                self._get_macro_rows(
                    indicator_code=(
                        code
                    ),
                    feature_date=(
                        feature_date
                    ),
                )
            for code
            in self.MACRO_CODES
        }

        features: dict[
            str,
            float | None,
        ] = {
            **technical,
            **advanced_technical,
            **market_regime,
            **sector_regime,
        }

        for period in (
            5,
            20,
            60,
        ):
            sector_return = (
                self._percent_to_ratio(
                    relative.get(
                        f"sector_return_{period}d_pct"
                    )
                )
            )

            market_return = (
                self._percent_to_ratio(
                    relative.get(
                        f"market_return_{period}d_pct"
                    )
                )
            )

            features[
                f"sector_return_{period}d"
            ] = sector_return

            features[
                f"market_return_{period}d"
            ] = market_return

            features[
                f"sector_excess_{period}d"
            ] = (
                self._percent_to_ratio(
                    relative.get(
                        f"sector_excess_{period}d_pct"
                    )
                )
            )

            features[
                f"market_excess_{period}d"
            ] = (
                self._percent_to_ratio(
                    relative.get(
                        f"market_excess_{period}d_pct"
                    )
                )
            )

            features[
                f"sector_market_excess_{period}d"
            ] = (
                sector_return
                - market_return
                if (
                    sector_return is not None
                    and market_return is not None
                )
                else None
            )

        flow_definitions = (
            (
                "foreign",
                "foreign_net_buy_volume",
            ),
            (
                "institution",
                "institution_net_buy_volume",
            ),
            (
                "individual",
                "individual_net_buy_volume",
            ),
        )

        for (
            prefix,
            field_name,
        ) in flow_definitions:
            for period in (
                5,
                20,
            ):
                features[
                    f"{prefix}_flow_{period}d_ratio"
                ] = (
                    self._flow_ratio(
                        flow_rows=(
                            flow_rows
                        ),
                        price_rows=(
                            price_rows
                        ),
                        field_name=(
                            field_name
                        ),
                        periods=(
                            period
                        ),
                        feature_date=(
                            feature_date
                        ),
                    )
                )

        features.update(
            {
                "per":
                    (
                        self._float_or_none(
                            valuation.per
                        )
                        if valuation
                        is not None
                        else None
                    ),

                "pbr":
                    (
                        self._float_or_none(
                            valuation.pbr
                        )
                        if valuation
                        is not None
                        else None
                    ),

                "sector_per_relative":
                    (
                        self._relative_ratio(
                            valuation.per,
                            valuation.sector_per,
                        )
                        if valuation
                        is not None
                        else None
                    ),

                "sector_pbr_relative":
                    (
                        self._relative_ratio(
                            valuation.pbr,
                            valuation.sector_pbr,
                        )
                        if valuation
                        is not None
                        else None
                    ),

                "market_per_relative":
                    (
                        self._relative_ratio(
                            valuation.per,
                            valuation.market_per,
                        )
                        if valuation
                        is not None
                        else None
                    ),

                "market_pbr_relative":
                    (
                        self._relative_ratio(
                            valuation.pbr,
                            valuation.market_pbr,
                        )
                        if valuation
                        is not None
                        else None
                    ),

                "log_market_cap":
                    (
                        math.log1p(
                            max(
                                0.0,
                                float(
                                    valuation
                                    .market_cap
                                ),
                            )
                        )
                        if (
                            valuation
                            is not None
                            and valuation
                            .market_cap
                            is not None
                        )
                        else None
                    ),
            }
        )

        base_rate = (
            self._latest_macro_value(
                macro_rows[
                    "BOK_BASE_RATE"
                ],
                target_date=(
                    feature_date
                ),
            )
        )

        usdkrw = (
            self._latest_macro_value(
                macro_rows[
                    "USD_KRW"
                ],
                target_date=(
                    feature_date
                ),
            )
        )

        ktb_3y = (
            self._latest_macro_value(
                macro_rows[
                    "KTB_3Y"
                ],
                target_date=(
                    feature_date
                ),
            )
        )

        ktb_10y = (
            self._latest_macro_value(
                macro_rows[
                    "KTB_10Y"
                ],
                target_date=(
                    feature_date
                ),
            )
        )

        ktb_3y_change = (
            self._macro_change(
                macro_rows[
                    "KTB_3Y"
                ],
                end_date=(
                    feature_date
                ),
                calendar_days=20,
                percent=False,
            )
        )

        ktb_10y_change = (
            self._macro_change(
                macro_rows[
                    "KTB_10Y"
                ],
                end_date=(
                    feature_date
                ),
                calendar_days=20,
                percent=False,
            )
        )

        features.update(
            {
                "base_rate_pct":
                    base_rate,

                "usdkrw":
                    usdkrw,

                "usdkrw_change_20d":
                    self._macro_change(
                        macro_rows[
                            "USD_KRW"
                        ],
                        end_date=(
                            feature_date
                        ),
                        calendar_days=20,
                        percent=True,
                    ),

                "ktb_3y_pct":
                    ktb_3y,

                "ktb_10y_pct":
                    ktb_10y,

                "ktb_3y_change_20d_bp":
                    (
                        ktb_3y_change
                        * 100.0
                        if ktb_3y_change
                        is not None
                        else None
                    ),

                "ktb_10y_change_20d_bp":
                    (
                        ktb_10y_change
                        * 100.0
                        if ktb_10y_change
                        is not None
                        else None
                    ),

                "yield_curve_10y_3y_bp":
                    (
                        (
                            ktb_10y
                            - ktb_3y
                        )
                        * 100.0
                        if (
                            ktb_10y
                            is not None
                            and ktb_3y
                            is not None
                        )
                        else None
                    ),
            }
        )

        missing = sorted(
            key
            for key, value
            in features.items()
            if value is None
        )

        return (
            {
                "stock_code":
                    stock_code,

                "feature_date":
                    feature_date,

                "feature_version":
                    (
                        feature_version
                        or self
                        .FEATURE_VERSION
                    ),

                "features":
                    features,
            },
            missing,
        )