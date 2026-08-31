from __future__ import annotations

import math
from statistics import fmean, pstdev


MARKET_REGIME_FEATURE_NAMES = (
    "market_ma20_gap",
    "market_ma60_gap",
    "market_ma20_slope_5d",
    "market_volatility_20d",
    "market_volatility_60d",
    "market_drawdown_60d",
    "market_trend_strength_20d",
    "market_volatility_percentile",
    "market_trend_regime_score",
    "market_volatility_regime_score",
    "market_risk_regime_score",
)


def _annualized_volatility(
    daily_returns: list[float],
) -> float:
    if len(daily_returns) < 2:
        return 0.0

    return (
        pstdev(
            daily_returns
        )
        * math.sqrt(
            252.0
        )
    )


def _percentile_rank(
    values: list[float],
    current: float,
) -> float:
    if not values:
        return 0.5

    less_or_equal = sum(
        1
        for value in values
        if value <= current
    )

    return (
        less_or_equal
        / len(values)
    )


def build_market_regime_feature_series(
    rows: list,
) -> list[dict[str, float] | None]:
    if not rows:
        return []

    rows = sorted(
        rows,
        key=lambda row:
            row.trade_date,
    )

    closes = [
        float(
            row.close
        )
        for row
        in rows
    ]

    daily_returns = [
        0.0
        for _ in rows
    ]

    for index in range(
        1,
        len(
            rows
        ),
    ):
        previous_close = closes[
            index - 1
        ]

        current_close = closes[
            index
        ]

        if (
            previous_close <= 0.0
            or current_close <= 0.0
        ):
            continue

        daily_returns[
            index
        ] = math.log(
            current_close
            / previous_close
        )

    ma20_series: list[
        float | None
    ] = [
        None
        for _ in rows
    ]

    ma60_series: list[
        float | None
    ] = [
        None
        for _ in rows
    ]

    vol20_series: list[
        float | None
    ] = [
        None
        for _ in rows
    ]

    vol60_series: list[
        float | None
    ] = [
        None
        for _ in rows
    ]

    for index in range(
        len(
            rows
        )
    ):
        if index >= 19:
            ma20_series[
                index
            ] = fmean(
                closes[
                    index - 19:
                    index + 1
                ]
            )

            vol20_series[
                index
            ] = (
                _annualized_volatility(
                    daily_returns[
                        index - 19:
                        index + 1
                    ]
                )
            )

        if index >= 59:
            ma60_series[
                index
            ] = fmean(
                closes[
                    index - 59:
                    index + 1
                ]
            )

            vol60_series[
                index
            ] = (
                _annualized_volatility(
                    daily_returns[
                        index - 59:
                        index + 1
                    ]
                )
            )

    result: list[
        dict[str, float] | None
    ] = [
        None
        for _ in rows
    ]

    for index in range(
        60,
        len(
            rows
        ),
    ):
        close = closes[
            index
        ]

        ma20 = ma20_series[
            index
        ]

        ma60 = ma60_series[
            index
        ]

        previous_ma20 = (
            ma20_series[
                index - 5
            ]
        )

        volatility_20 = (
            vol20_series[
                index
            ]
        )

        volatility_60 = (
            vol60_series[
                index
            ]
        )

        if any(
            value is None
            for value
            in (
                ma20,
                ma60,
                previous_ma20,
                volatility_20,
                volatility_60,
            )
        ):
            continue

        if (
            close <= 0.0
            or ma20 <= 0.0
            or ma60 <= 0.0
            or previous_ma20 <= 0.0
        ):
            continue

        return_20d = (
            close
            / closes[
                index - 20
            ]
            - 1.0
        )

        ma20_slope_5d = (
            ma20
            / previous_ma20
            - 1.0
        )

        high_60 = max(
            closes[
                index - 59:
                index + 1
            ]
        )

        drawdown_60 = (
            close
            / high_60
            - 1.0
            if high_60 > 0.0
            else 0.0
        )

        daily_volatility_20 = (
            volatility_20
            / math.sqrt(
                252.0
            )
        )

        trend_denominator = (
            daily_volatility_20
            * math.sqrt(
                20.0
            )
        )

        trend_strength_20d = (
            return_20d
            / trend_denominator
            if trend_denominator > 0.0
            else 0.0
        )

        percentile_start = max(
            19,
            index - 251,
        )

        historical_volatilities = [
            value
            for value
            in vol20_series[
                percentile_start:
                index + 1
            ]
            if value is not None
        ]

        volatility_percentile = (
            _percentile_rank(
                historical_volatilities,
                volatility_20,
            )
        )

        if (
            close > ma20
            and ma20 > ma60
            and ma20_slope_5d > 0.0
        ):
            trend_score = 1.0

        elif (
            close < ma20
            and ma20 < ma60
            and ma20_slope_5d < 0.0
        ):
            trend_score = -1.0

        else:
            trend_score = 0.0

        if volatility_percentile >= 0.67:
            volatility_score = 1.0

        elif volatility_percentile <= 0.33:
            volatility_score = -1.0

        else:
            volatility_score = 0.0

        if (
            trend_score > 0.0
            and volatility_score < 1.0
        ):
            risk_score = 1.0

        elif (
            trend_score < 0.0
            or (
                volatility_score > 0.0
                and drawdown_60 <= -0.05
            )
        ):
            risk_score = -1.0

        else:
            risk_score = 0.0

        features = {
            "market_ma20_gap":
                close
                / ma20
                - 1.0,

            "market_ma60_gap":
                close
                / ma60
                - 1.0,

            "market_ma20_slope_5d":
                ma20_slope_5d,

            "market_volatility_20d":
                volatility_20,

            "market_volatility_60d":
                volatility_60,

            "market_drawdown_60d":
                drawdown_60,

            "market_trend_strength_20d":
                trend_strength_20d,

            "market_volatility_percentile":
                volatility_percentile,

            "market_trend_regime_score":
                trend_score,

            "market_volatility_regime_score":
                volatility_score,

            "market_risk_regime_score":
                risk_score,
        }

        if not all(
            math.isfinite(
                value
            )
            for value
            in features.values()
        ):
            continue

        result[
            index
        ] = features

    return result


def build_latest_market_regime_features(
    rows: list,
) -> dict[str, float]:
    series = (
        build_market_regime_feature_series(
            rows
        )
    )

    if not series:
        raise ValueError(
            "시장 Regime 계산 데이터가 없습니다."
        )

    latest = series[
        -1
    ]

    if latest is None:
        raise ValueError(
            "최신 시장 Regime을 계산할 수 없습니다."
        )

    return latest