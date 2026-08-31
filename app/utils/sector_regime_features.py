from __future__ import annotations

from app.utils.market_regime_features import (
    MARKET_REGIME_FEATURE_NAMES,
    build_market_regime_feature_series,
)


SECTOR_REGIME_FEATURE_NAMES = tuple(
    name.replace(
        "market_",
        "sector_",
        1,
    )
    for name
    in MARKET_REGIME_FEATURE_NAMES
)

SECTOR_RELATIVE_MOMENTUM_FEATURE_NAMES = (
    "sector_market_excess_5d",
    "sector_market_excess_20d",
    "sector_market_excess_60d",
)

SECTOR_CONTEXT_FEATURE_NAMES = (
    *SECTOR_REGIME_FEATURE_NAMES,
    *SECTOR_RELATIVE_MOMENTUM_FEATURE_NAMES,
)


def _rename_market_features(
    features: dict[str, float],
) -> dict[str, float]:
    return {
        name.replace(
            "market_",
            "sector_",
            1,
        ): float(value)
        for name, value
        in features.items()
    }


def build_sector_regime_feature_series(
    rows: list,
) -> list[dict[str, float] | None]:
    market_style_series = (
        build_market_regime_feature_series(
            rows
        )
    )

    return [
        (
            None
            if features is None
            else _rename_market_features(
                features
            )
        )
        for features
        in market_style_series
    ]


def build_latest_sector_regime_features(
    rows: list,
) -> dict[str, float]:
    series = (
        build_sector_regime_feature_series(
            rows
        )
    )

    if not series:
        raise ValueError(
            "업종 Regime 계산 데이터가 없습니다."
        )

    latest = series[-1]

    if latest is None:
        raise ValueError(
            "최신 업종 Regime을 계산할 수 없습니다."
        )

    return latest