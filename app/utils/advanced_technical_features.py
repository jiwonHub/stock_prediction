from __future__ import annotations

import math
from statistics import fmean, pstdev


ADVANCED_TECHNICAL_FEATURE_NAMES = (
    "ema_12_gap",
    "ema_26_gap",

    "macd_line_ratio",
    "macd_signal_ratio",
    "macd_hist_ratio",

    "bollinger_position_20",
    "bollinger_width_20",

    "atr_14_ratio",

    "stochastic_k_14",
    "stochastic_d_3",

    "adx_14",

    "volume_zscore_20",
    "volume_trend_5_20",

    "price_vs_high_60",
    "price_vs_low_60",
)


def _ema(
    values: list[float],
    span: int,
) -> float:
    if not values:
        raise ValueError(
            "EMA 계산 데이터가 없습니다."
        )

    alpha = (
        2.0
        / (
            span
            + 1.0
        )
    )

    result = float(
        values[0]
    )

    for value in values[1:]:
        result = (
            alpha
            * float(
                value
            )
            + (
                1.0
                - alpha
            )
            * result
        )

    return result


def _true_range(
    rows: list,
    index: int,
) -> float:
    high = float(
        rows[
            index
        ].high
    )

    low = float(
        rows[
            index
        ].low
    )

    if index == 0:
        return (
            high
            - low
        )

    previous_close = float(
        rows[
            index - 1
        ].close
    )

    return max(
        high - low,
        abs(
            high
            - previous_close
        ),
        abs(
            low
            - previous_close
        ),
    )


def _stochastic_k(
    rows: list,
    index: int,
    period: int = 14,
) -> float:
    start = (
        index
        - period
        + 1
    )

    window = rows[
        start:
        index + 1
    ]

    highest = max(
        float(
            row.high
        )
        for row
        in window
    )

    lowest = min(
        float(
            row.low
        )
        for row
        in window
    )

    close = float(
        rows[
            index
        ].close
    )

    denominator = (
        highest
        - lowest
    )

    if denominator <= 0.0:
        return 0.5

    return (
        close
        - lowest
    ) / denominator


def _dx_at(
    rows: list,
    index: int,
    period: int = 14,
) -> float:
    start = (
        index
        - period
        + 1
    )

    plus_dm = 0.0
    minus_dm = 0.0
    true_range = 0.0

    for current_index in range(
        start,
        index + 1,
    ):
        if current_index <= 0:
            continue

        current_high = float(
            rows[
                current_index
            ].high
        )

        current_low = float(
            rows[
                current_index
            ].low
        )

        previous_high = float(
            rows[
                current_index - 1
            ].high
        )

        previous_low = float(
            rows[
                current_index - 1
            ].low
        )

        up_move = (
            current_high
            - previous_high
        )

        down_move = (
            previous_low
            - current_low
        )

        if (
            up_move
            > down_move
            and up_move
            > 0.0
        ):
            plus_dm += (
                up_move
            )

        if (
            down_move
            > up_move
            and down_move
            > 0.0
        ):
            minus_dm += (
                down_move
            )

        true_range += (
            _true_range(
                rows,
                current_index,
            )
        )

    if true_range <= 0.0:
        return 0.0

    plus_di = (
        plus_dm
        / true_range
    )

    minus_di = (
        minus_dm
        / true_range
    )

    denominator = (
        plus_di
        + minus_di
    )

    if denominator <= 0.0:
        return 0.0

    return abs(
        plus_di
        - minus_di
    ) / denominator


def build_advanced_technical_features(
    rows: list,
    index: int | None = None,
) -> dict[str, float]:
    if len(
        rows
    ) < 61:
        raise ValueError(
            "고급 기술지표 계산에는 "
            "최소 61거래일이 필요합니다."
        )

    if index is None:
        index = (
            len(rows)
            - 1
        )

    if (
        index < 60
        or index >= len(
            rows
        )
    ):
        raise ValueError(
            "고급 기술지표 index가 "
            "올바르지 않습니다."
        )

    closes = [
        float(
            row.close
        )
        for row
        in rows[
            :index + 1
        ]
    ]

    close = closes[
        -1
    ]

    if close <= 0.0:
        raise ValueError(
            "종가가 0 이하입니다."
        )

    ema_12 = _ema(
        closes,
        12,
    )

    ema_26 = _ema(
        closes,
        26,
    )

    macd_series = []

    for series_index in range(
        25,
        len(
            closes
        ),
    ):
        history = closes[
            :series_index + 1
        ]

        macd_series.append(
            _ema(
                history,
                12,
            )
            - _ema(
                history,
                26,
            )
        )

    macd_line = (
        ema_12
        - ema_26
    )

    macd_signal = _ema(
        macd_series,
        9,
    )

    macd_hist = (
        macd_line
        - macd_signal
    )

    close_20 = closes[
        -20:
    ]

    bb_middle = fmean(
        close_20
    )

    bb_std = pstdev(
        close_20
    )

    bb_upper = (
        bb_middle
        + 2.0
        * bb_std
    )

    bb_lower = (
        bb_middle
        - 2.0
        * bb_std
    )

    bb_range = (
        bb_upper
        - bb_lower
    )

    bollinger_position = (
        (
            close
            - bb_lower
        )
        / bb_range
        if bb_range
        > 0.0
        else 0.5
    )

    bollinger_width = (
        bb_range
        / bb_middle
        if bb_middle
        > 0.0
        else 0.0
    )

    true_ranges = [
        _true_range(
            rows,
            current_index,
        )
        for current_index
        in range(
            index - 13,
            index + 1,
        )
    ]

    atr_14 = fmean(
        true_ranges
    )

    stochastic_values = [
        _stochastic_k(
            rows,
            current_index,
            14,
        )
        for current_index
        in range(
            index - 2,
            index + 1,
        )
    ]

    stochastic_k = (
        stochastic_values[
            -1
        ]
    )

    stochastic_d = fmean(
        stochastic_values
    )

    dx_values = [
        _dx_at(
            rows,
            current_index,
            14,
        )
        for current_index
        in range(
            index - 13,
            index + 1,
        )
    ]

    adx_14 = fmean(
        dx_values
    )

    volumes_20 = [
        float(
            row.volume
            or 0
        )
        for row
        in rows[
            index - 19:
            index + 1
        ]
    ]

    volume_mean_20 = fmean(
        volumes_20
    )

    volume_std_20 = pstdev(
        volumes_20
    )

    current_volume = (
        volumes_20[
            -1
        ]
    )

    volume_zscore = (
        (
            current_volume
            - volume_mean_20
        )
        / volume_std_20
        if volume_std_20
        > 0.0
        else 0.0
    )

    volume_mean_5 = fmean(
        volumes_20[
            -5:
        ]
    )

    volume_trend = (
        volume_mean_5
        / volume_mean_20
        - 1.0
        if volume_mean_20
        > 0.0
        else 0.0
    )

    rows_60 = rows[
        index - 59:
        index + 1
    ]

    high_60 = max(
        float(
            row.high
        )
        for row
        in rows_60
    )

    low_60 = min(
        float(
            row.low
        )
        for row
        in rows_60
    )

    features = {
        "ema_12_gap":
            close
            / ema_12
            - 1.0,

        "ema_26_gap":
            close
            / ema_26
            - 1.0,

        "macd_line_ratio":
            macd_line
            / close,

        "macd_signal_ratio":
            macd_signal
            / close,

        "macd_hist_ratio":
            macd_hist
            / close,

        "bollinger_position_20":
            bollinger_position,

        "bollinger_width_20":
            bollinger_width,

        "atr_14_ratio":
            atr_14
            / close,

        "stochastic_k_14":
            stochastic_k,

        "stochastic_d_3":
            stochastic_d,

        "adx_14":
            adx_14,

        "volume_zscore_20":
            volume_zscore,

        "volume_trend_5_20":
            volume_trend,

        "price_vs_high_60":
            close
            / high_60
            - 1.0,

        "price_vs_low_60":
            close
            / low_60
            - 1.0,
    }

    for (
        name,
        value,
    ) in features.items():
        if not math.isfinite(
            value
        ):
            raise ValueError(
                "고급 기술 Feature "
                "NaN/Infinity: "
                f"{name}"
            )

    return features

def build_advanced_technical_feature_series(
    rows: list,
) -> list[dict[str, float] | None]:
    if not rows:
        return []

    closes = [
        float(
            row.close
        )
        for row
        in rows
    ]

    ema_12_series: list[float] = []
    ema_26_series: list[float] = []
    macd_series: list[float] = []
    signal_series: list[float | None] = []

    alpha_12 = 2.0 / 13.0
    alpha_26 = 2.0 / 27.0
    alpha_9 = 2.0 / 10.0

    ema_12 = closes[0]
    ema_26 = closes[0]

    signal: float | None = None

    for index, close in enumerate(
        closes
    ):
        if index > 0:
            ema_12 = (
                alpha_12
                * close
                + (
                    1.0
                    - alpha_12
                )
                * ema_12
            )

            ema_26 = (
                alpha_26
                * close
                + (
                    1.0
                    - alpha_26
                )
                * ema_26
            )

        ema_12_series.append(
            ema_12
        )

        ema_26_series.append(
            ema_26
        )

        macd = (
            ema_12
            - ema_26
        )

        macd_series.append(
            macd
        )

        if index < 25:
            signal_series.append(
                None
            )
            continue

        if signal is None:
            signal = macd
        else:
            signal = (
                alpha_9
                * macd
                + (
                    1.0
                    - alpha_9
                )
                * signal
            )

        signal_series.append(
            signal
        )

    result: list[
        dict[str, float] | None
    ] = [
        None
        for _ in rows
    ]

    for index in range(
        60,
        len(rows),
    ):
        close = closes[
            index
        ]

        if close <= 0.0:
            continue

        ema_12 = (
            ema_12_series[
                index
            ]
        )

        ema_26 = (
            ema_26_series[
                index
            ]
        )

        macd_line = (
            macd_series[
                index
            ]
        )

        macd_signal = (
            signal_series[
                index
            ]
        )

        if macd_signal is None:
            continue

        macd_hist = (
            macd_line
            - macd_signal
        )

        close_20 = closes[
            index - 19:
            index + 1
        ]

        bb_middle = fmean(
            close_20
        )

        bb_std = pstdev(
            close_20
        )

        bb_upper = (
            bb_middle
            + 2.0
            * bb_std
        )

        bb_lower = (
            bb_middle
            - 2.0
            * bb_std
        )

        bb_range = (
            bb_upper
            - bb_lower
        )

        bollinger_position = (
            (
                close
                - bb_lower
            )
            / bb_range
            if bb_range > 0.0
            else 0.5
        )

        bollinger_width = (
            bb_range
            / bb_middle
            if bb_middle > 0.0
            else 0.0
        )

        true_ranges = [
            _true_range(
                rows,
                current_index,
            )
            for current_index
            in range(
                index - 13,
                index + 1,
            )
        ]

        atr_14 = fmean(
            true_ranges
        )

        stochastic_values = [
            _stochastic_k(
                rows,
                current_index,
                14,
            )
            for current_index
            in range(
                index - 2,
                index + 1,
            )
        ]

        stochastic_k = (
            stochastic_values[
                -1
            ]
        )

        stochastic_d = fmean(
            stochastic_values
        )

        dx_values = [
            _dx_at(
                rows,
                current_index,
                14,
            )
            for current_index
            in range(
                index - 13,
                index + 1,
            )
        ]

        adx_14 = fmean(
            dx_values
        )

        volumes_20 = [
            float(
                row.volume
                or 0
            )
            for row
            in rows[
                index - 19:
                index + 1
            ]
        ]

        volume_mean_20 = fmean(
            volumes_20
        )

        volume_std_20 = pstdev(
            volumes_20
        )

        current_volume = (
            volumes_20[
                -1
            ]
        )

        volume_zscore = (
            (
                current_volume
                - volume_mean_20
            )
            / volume_std_20
            if volume_std_20 > 0.0
            else 0.0
        )

        volume_mean_5 = fmean(
            volumes_20[
                -5:
            ]
        )

        volume_trend = (
            volume_mean_5
            / volume_mean_20
            - 1.0
            if volume_mean_20 > 0.0
            else 0.0
        )

        rows_60 = rows[
            index - 59:
            index + 1
        ]

        high_60 = max(
            float(
                row.high
            )
            for row
            in rows_60
        )

        low_60 = min(
            float(
                row.low
            )
            for row
            in rows_60
        )

        features = {
            "ema_12_gap":
                close
                / ema_12
                - 1.0,

            "ema_26_gap":
                close
                / ema_26
                - 1.0,

            "macd_line_ratio":
                macd_line
                / close,

            "macd_signal_ratio":
                macd_signal
                / close,

            "macd_hist_ratio":
                macd_hist
                / close,

            "bollinger_position_20":
                bollinger_position,

            "bollinger_width_20":
                bollinger_width,

            "atr_14_ratio":
                atr_14
                / close,

            "stochastic_k_14":
                stochastic_k,

            "stochastic_d_3":
                stochastic_d,

            "adx_14":
                adx_14,

            "volume_zscore_20":
                volume_zscore,

            "volume_trend_5_20":
                volume_trend,

            "price_vs_high_60":
                close
                / high_60
                - 1.0,

            "price_vs_low_60":
                close
                / low_60
                - 1.0,
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