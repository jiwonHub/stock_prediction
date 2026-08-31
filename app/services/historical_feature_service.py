from __future__ import annotations

from bisect import bisect_right
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
    FEATURE_NAMES,
    build_feature_dict_at_index,
)
from app.utils.advanced_technical_features import (
    ADVANCED_TECHNICAL_FEATURE_NAMES,
    build_advanced_technical_feature_series,
)
from app.utils.market_regime_features import (
    MARKET_REGIME_FEATURE_NAMES,
    build_market_regime_feature_series,
)
from app.utils.sector_regime_features import (
    SECTOR_CONTEXT_FEATURE_NAMES,
    build_sector_regime_feature_series,
)
from app.services.disclosure_feature_service import (
    DISCLOSURE_FEATURE_NAMES,
    DisclosureFeatureService,
)

class HistoricalFeatureService:
    FEATURE_VERSION = "v5-historical-disclosure-1"

    HISTORY_CALENDAR_DAYS = 1200

    PERIODS = (
        5,
        20,
        60,
    )

    MACRO_CODES = (
        "BOK_BASE_RATE",
        "USD_KRW",
        "KTB_3Y",
        "KTB_10Y",
    )

    HISTORICAL_FEATURE_NAMES = (
        *FEATURE_NAMES,
        "return_60d",

        *ADVANCED_TECHNICAL_FEATURE_NAMES,

        *MARKET_REGIME_FEATURE_NAMES,

    *SECTOR_CONTEXT_FEATURE_NAMES,

    *DISCLOSURE_FEATURE_NAMES,

        "sector_return_5d",
        "sector_return_20d",
        "sector_return_60d",

        "market_return_5d",
        "market_return_20d",
        "market_return_60d",

        "sector_excess_5d",
        "sector_excess_20d",
        "sector_excess_60d",

        "market_excess_5d",
        "market_excess_20d",
        "market_excess_60d",

        "base_rate_pct",
        "usdkrw",
        "usdkrw_change_20d",

        "ktb_3y_pct",
        "ktb_10y_pct",

        "ktb_3y_change_20d_bp",
        "ktb_10y_change_20d_bp",

        "yield_curve_10y_3y_bp",
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

        self.disclosure_feature_service = (
    DisclosureFeatureService(
        db
    )
)

        self._market_cache: dict[
            str,
            tuple[
                list,
                list[date],
                list[float],
                list[
                    dict[str, float]
                    | None
                ],
            ],
        ] = {}

        self._sector_cache: dict[
            tuple[str, str],
            tuple[
                list[date],
                list[float],
                list[
                    dict[str, float]
                    | None
                ],
            ],
        ] = {}

        self._macro_cache: dict[
            str,
            tuple[
                list[date],
                list[float],
            ],
        ] = {}

    @staticmethod
    def _value_on_or_before(
        dates: list[date],
        values: list[float],
        target_date: date,
    ) -> float | None:
        index = (
            bisect_right(
                dates,
                target_date,
            )
            - 1
        )

        if index < 0:
            return None

        return values[
            index
        ]

    @classmethod
    def _return_between(
        cls,
        dates: list[date],
        closes: list[float],
        *,
        start_date: date,
        end_date: date,
    ) -> float | None:
        start_close = (
            cls._value_on_or_before(
                dates,
                closes,
                start_date,
            )
        )

        end_close = (
            cls._value_on_or_before(
                dates,
                closes,
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

    @classmethod
    def _macro_change(
        cls,
        dates: list[date],
        values: list[float],
        *,
        end_date: date,
        calendar_days: int,
        percent: bool,
    ) -> float | None:
        current = (
            cls._value_on_or_before(
                dates,
                values,
                end_date,
            )
        )

        previous = (
            cls._value_on_or_before(
                dates,
                values,
                end_date
                - timedelta(
                    days=calendar_days
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

    def _get_market_series(
        self,
        *,
        market: str,
        start_date: date,
    ) -> tuple[
        list,
        list[date],
        list[float],
        list[
            dict[str, float]
            | None
        ],
    ]:
        index_code = (
            "0001"
            if market == "KOSPI"
            else "1001"
        )

        cached = (
            self._market_cache
            .get(
                index_code
            )
        )

        if cached is not None:
            return cached

        rows = (
            self.market_repository
            .get_market_index_prices(
                index_code=(
                    index_code
                ),
                start_date=(
                    start_date
                ),
            )
        )

        rows = sorted(
            rows,
            key=lambda row:
                row.trade_date,
        )

        regime_series = (
            build_market_regime_feature_series(
                rows
            )
        )

        result = (
            rows,
            [
                row.trade_date
                for row
                in rows
            ],
            [
                float(
                    row.close
                )
                for row
                in rows
            ],
            regime_series,
        )

        self._market_cache[
            index_code
        ] = result

        return result

    def _get_sector_series(
        self,
        *,
        market: str,
        sector_code: str,
        start_date: date,
    ) -> tuple[
        list[date],
        list[float],
        list[
            dict[str, float]
            | None
        ],
    ]:
        key = (
            market,
            sector_code,
        )

        cached = (
            self._sector_cache
            .get(
                key
            )
        )

        if cached is not None:
            return cached

        rows = (
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

        regime_series = (
            build_sector_regime_feature_series(
                rows
            )
        )

        result = (
            [
                row.trade_date
                for row
                in rows
            ],
            [
                float(
                    row.close
                )
                for row
                in rows
            ],
            regime_series,
        )

        self._sector_cache[
            key
        ] = result

        return result

    def _get_macro_series(
        self,
        *,
        indicator_code: str,
        start_date: date,
    ) -> tuple[
        list[date],
        list[float],
    ]:
        cached = (
            self._macro_cache
            .get(
                indicator_code
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
                    start_date
                ),
            )
        )

        result = (
            [
                row.observed_date
                for row
                in rows
            ],
            [
                float(
                    row.value
                )
                for row
                in rows
            ],
        )

        self._macro_cache[
            indicator_code
        ] = result

        return result

    def build_stock_rows(
        self,
        stock_code: str,
        *,
        feature_version:
            str | None = None,
    ) -> tuple[
        list[dict],
        dict,
    ]:
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

        if market not in {
            "KOSPI",
            "KOSDAQ",
        }:
            raise ValueError(
                "시장 구분이 올바르지 않습니다: "
                f"{stock_code} {market}"
            )

        if not sector_code:
            raise ValueError(
                "업종 코드가 없습니다: "
                f"{stock_code}"
            )

        history_start = (
            date.today()
            - timedelta(
                days=(
                    self
                    .HISTORY_CALENDAR_DAYS
                )
            )
        )

        stock_rows = (
            self.stock_repository
            .get_daily_prices(
                stock_code=(
                    stock_code
                ),
                start_date=(
                    history_start
                ),
            )
        )

        (
            market_rows,
            market_dates,
            market_closes,
            market_regime_series,
        ) = (
            self._get_market_series(
                market=market,
                start_date=(
                    history_start
                ),
            )
        )

        (
            sector_dates,
            sector_closes,
            sector_regime_series,
        ) = (
            self._get_sector_series(
                market=market,
                sector_code=(
                    sector_code
                ),
                start_date=(
                    history_start
                ),
            )
        )

        macro_series = {
            code:
                self._get_macro_series(
                    indicator_code=(
                        code
                    ),
                    start_date=(
                        history_start
                    ),
                )
            for code
            in self.MACRO_CODES
        }

        if len(
            stock_rows
        ) < 61:
            raise ValueError(
                "종목 일봉 61거래일 부족: "
                f"{stock_code}"
            )

        if len(
            market_rows
        ) < 61:
            raise ValueError(
                "시장지수 61거래일 부족: "
                f"{stock_code}"
            )

        if len(
            sector_dates
        ) < 61:
            raise ValueError(
                "업종지수 61거래일 부족: "
                f"{stock_code}"
            )

        stock_dates = [
            row.trade_date
            for row
            in stock_rows
        ]

        stock_closes = [
            float(
                row.close
            )
            for row
            in stock_rows
        ]

        advanced_series = (
            build_advanced_technical_feature_series(
                stock_rows
            )
        )

        market_index_by_date = {
            trade_date:
                index
            for index, trade_date
            in enumerate(
                market_dates
            )
        }

        built_rows: list[
            dict
        ] = []

        skipped = 0

        first_feature_date: (
            date | None
        ) = None

        last_feature_date: (
            date | None
        ) = None

        expected_feature_set = set(
            self.HISTORICAL_FEATURE_NAMES
        )

        for stock_index in range(
            60,
            len(
                stock_rows
            ),
        ):
            feature_date = (
                stock_rows[
                    stock_index
                ].trade_date
            )

            market_index = (
                market_index_by_date
                .get(
                    feature_date
                )
            )

            if (
                market_index is None
                or market_index < 60
            ):
                skipped += 1
                continue

            sector_index = (
                bisect_right(
                    sector_dates,
                    feature_date,
                )
                - 1
            )

            if sector_index < 60:
                skipped += 1
                continue

            market_regime = (
                market_regime_series[
                    market_index
                ]
            )

            if market_regime is None:
                skipped += 1
                continue

            sector_regime = (
                sector_regime_series[
                    sector_index
                ]
            )

            if sector_regime is None:
                skipped += 1
                continue

            features = (
                build_feature_dict_at_index(
                    stock_rows,
                    stock_index,
                )
            )

            advanced_features = (
                advanced_series[
                    stock_index
                ]
            )

            if advanced_features is None:
                skipped += 1
                continue

            features.update(
                advanced_features
            )

            features.update(
                market_regime
            )

            features.update(
                sector_regime
            )

            disclosure_features = (
                self.disclosure_feature_service
                .build_features(
                    stock_code=(
                        stock_code
                    ),
                    as_of_date=(
                        feature_date
                    ),
                )
            )

            features.update(
                disclosure_features
            )

            relative_ok = True

            for period in (
                self.PERIODS
            ):
                period_start_date = (
                    market_dates[
                        market_index
                        - period
                    ]
                )

                stock_return = (
                    self._return_between(
                        stock_dates,
                        stock_closes,
                        start_date=(
                            period_start_date
                        ),
                        end_date=(
                            feature_date
                        ),
                    )
                )

                sector_return = (
                    self._return_between(
                        sector_dates,
                        sector_closes,
                        start_date=(
                            period_start_date
                        ),
                        end_date=(
                            feature_date
                        ),
                    )
                )

                market_return = (
                    self._return_between(
                        market_dates,
                        market_closes,
                        start_date=(
                            period_start_date
                        ),
                        end_date=(
                            feature_date
                        ),
                    )
                )

                if (
                    stock_return is None
                    or sector_return is None
                    or market_return is None
                ):
                    relative_ok = False
                    break

                features[
                    f"sector_return_{period}d"
                ] = sector_return

                features[
                    f"market_return_{period}d"
                ] = market_return

                features[
                    f"sector_excess_{period}d"
                ] = (
                    stock_return
                    - sector_return
                )

                features[
                    f"market_excess_{period}d"
                ] = (
                    stock_return
                    - market_return
                )

                features[
                    f"sector_market_excess_{period}d"
                ] = (
                    sector_return
                    - market_return
                )

            if not relative_ok:
                skipped += 1
                continue

            (
                base_dates,
                base_values,
            ) = macro_series[
                "BOK_BASE_RATE"
            ]

            (
                usd_dates,
                usd_values,
            ) = macro_series[
                "USD_KRW"
            ]

            (
                ktb3_dates,
                ktb3_values,
            ) = macro_series[
                "KTB_3Y"
            ]

            (
                ktb10_dates,
                ktb10_values,
            ) = macro_series[
                "KTB_10Y"
            ]

            base_rate = (
                self._value_on_or_before(
                    base_dates,
                    base_values,
                    feature_date,
                )
            )

            usdkrw = (
                self._value_on_or_before(
                    usd_dates,
                    usd_values,
                    feature_date,
                )
            )

            ktb_3y = (
                self._value_on_or_before(
                    ktb3_dates,
                    ktb3_values,
                    feature_date,
                )
            )

            ktb_10y = (
                self._value_on_or_before(
                    ktb10_dates,
                    ktb10_values,
                    feature_date,
                )
            )

            usdkrw_change = (
                self._macro_change(
                    usd_dates,
                    usd_values,
                    end_date=(
                        feature_date
                    ),
                    calendar_days=20,
                    percent=True,
                )
            )

            ktb_3y_change = (
                self._macro_change(
                    ktb3_dates,
                    ktb3_values,
                    end_date=(
                        feature_date
                    ),
                    calendar_days=20,
                    percent=False,
                )
            )

            ktb_10y_change = (
                self._macro_change(
                    ktb10_dates,
                    ktb10_values,
                    end_date=(
                        feature_date
                    ),
                    calendar_days=20,
                    percent=False,
                )
            )

            if any(
                value is None
                for value
                in (
                    base_rate,
                    usdkrw,
                    ktb_3y,
                    ktb_10y,
                    usdkrw_change,
                    ktb_3y_change,
                    ktb_10y_change,
                )
            ):
                skipped += 1
                continue

            features.update(
                {
                    "base_rate_pct":
                        float(
                            base_rate
                        ),

                    "usdkrw":
                        float(
                            usdkrw
                        ),

                    "usdkrw_change_20d":
                        float(
                            usdkrw_change
                        ),

                    "ktb_3y_pct":
                        float(
                            ktb_3y
                        ),

                    "ktb_10y_pct":
                        float(
                            ktb_10y
                        ),

                    "ktb_3y_change_20d_bp":
                        float(
                            ktb_3y_change
                        )
                        * 100.0,

                    "ktb_10y_change_20d_bp":
                        float(
                            ktb_10y_change
                        )
                        * 100.0,

                    "yield_curve_10y_3y_bp":
                        (
                            float(
                                ktb_10y
                            )
                            - float(
                                ktb_3y
                            )
                        )
                        * 100.0,
                }
            )

            actual_feature_set = set(
                features.keys()
            )

            if (
                actual_feature_set
                != expected_feature_set
            ):
                missing = sorted(
                    expected_feature_set
                    - actual_feature_set
                )

                extra = sorted(
                    actual_feature_set
                    - expected_feature_set
                )

                raise ValueError(
                    "Historical Feature 구조 불일치: "
                    f"{stock_code} "
                    f"{feature_date} "
                    f"missing={missing}, "
                    f"extra={extra}"
                )

            built_rows.append(
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
                }
            )

            if first_feature_date is None:
                first_feature_date = (
                    feature_date
                )

            last_feature_date = (
                feature_date
            )

        return (
            built_rows,
            {
                "stock_code":
                    stock_code,

                "built":
                    len(
                        built_rows
                    ),

                "skipped":
                    skipped,

                "first_feature_date":
                    (
                        first_feature_date
                        .isoformat()
                        if first_feature_date
                        is not None
                        else None
                    ),

                "last_feature_date":
                    (
                        last_feature_date
                        .isoformat()
                        if last_feature_date
                        is not None
                        else None
                    ),
            },
        )

    def save_rows(
        self,
        rows: list[dict],
        *,
        batch_size: int = 500,
    ) -> int:
        saved = 0

        for start in range(
            0,
            len(rows),
            batch_size,
        ):
            batch = rows[
                start:
                start + batch_size
            ]

            saved += (
                self.feature_repository
                .upsert_snapshots(
                    batch
                )
            )

        return saved