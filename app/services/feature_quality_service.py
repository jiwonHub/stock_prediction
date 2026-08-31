from __future__ import annotations

import math
from statistics import fmean, pstdev

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.models.future import StockFeatureSnapshot
from app.utils.ml_features import FEATURE_NAMES
from app.utils.advanced_technical_features import (
    ADVANCED_TECHNICAL_FEATURE_NAMES,
)
from app.utils.market_regime_features import (
    MARKET_REGIME_FEATURE_NAMES,
)
from app.utils.sector_regime_features import (
    SECTOR_CONTEXT_FEATURE_NAMES,
)


class FeatureQualityService:
    EXPECTED_FEATURES = (
        *FEATURE_NAMES,
        "return_60d",

        *ADVANCED_TECHNICAL_FEATURE_NAMES,

        *MARKET_REGIME_FEATURE_NAMES,

        *SECTOR_CONTEXT_FEATURE_NAMES,

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

        "foreign_flow_5d_ratio",
        "foreign_flow_20d_ratio",

        "institution_flow_5d_ratio",
        "institution_flow_20d_ratio",

        "individual_flow_5d_ratio",
        "individual_flow_20d_ratio",

        "per",
        "pbr",

        "sector_per_relative",
        "sector_pbr_relative",

        "market_per_relative",
        "market_pbr_relative",

        "log_market_cap",

        "base_rate_pct",
        "usdkrw",
        "usdkrw_change_20d",

        "ktb_3y_pct",
        "ktb_10y_pct",

        "ktb_3y_change_20d_bp",
        "ktb_10y_change_20d_bp",

        "yield_curve_10y_3y_bp",
    )

    RANGE_RULES = {
        "rsi_14": (
            0.0,
            1.0,
        ),
        "close_position": (
            0.0,
            1.0,
        ),
        "volatility_20d": (
            0.0,
            None,
        ),
        "intraday_range": (
            0.0,
            None,
        ),
        "log_market_cap": (
            0.0,
            None,
        ),

        "atr_14_ratio": (
            0.0,
            None,
        ),

        "stochastic_k_14": (
            0.0,
            1.0,
        ),

        "stochastic_d_3": (
            0.0,
            1.0,
        ),

        "adx_14": (
            0.0,
            1.0,
        ),

        "bollinger_width_20": (
            0.0,
            None,
        ),

        "price_vs_high_60": (
            None,
            0.0,
        ),

        "price_vs_low_60": (
            0.0,
            None,
        ),

        "market_volatility_20d": (
            0.0,
            None,
        ),

        "market_volatility_60d": (
            0.0,
            None,
        ),

        "market_drawdown_60d": (
            None,
            0.0,
        ),

        "market_volatility_percentile": (
            0.0,
            1.0,
        ),

        "market_trend_regime_score": (
            -1.0,
            1.0,
        ),

        "market_volatility_regime_score": (
            -1.0,
            1.0,
        ),

        "market_risk_regime_score": (
            -1.0,
            1.0,
        ),

        "sector_volatility_20d": (
            0.0,
            None,
        ),

        "sector_volatility_60d": (
            0.0,
            None,
        ),

        "sector_drawdown_60d": (
            None,
            0.0,
        ),

        "sector_volatility_percentile": (
            0.0,
            1.0,
        ),

        "sector_trend_regime_score": (
            -1.0,
            1.0,
        ),

        "sector_volatility_regime_score": (
            -1.0,
            1.0,
        ),

        "sector_risk_regime_score": (
            -1.0,
            1.0,
        ),
    }

    SHARED_MACRO_FEATURES = {
        "base_rate_pct",
        "usdkrw",
        "usdkrw_change_20d",
        "ktb_3y_pct",
        "ktb_10y_pct",
        "ktb_3y_change_20d_bp",
        "ktb_10y_change_20d_bp",
        "yield_curve_10y_3y_bp",
    }

    def __init__(
        self,
        db: Session,
    ):
        self.db = db

    @staticmethod
    def _percentile(
        values: list[float],
        q: float,
    ) -> float | None:
        if not values:
            return None

        ordered = sorted(
            values
        )

        if len(ordered) == 1:
            return ordered[0]

        position = (
            len(ordered) - 1
        ) * q

        lower = math.floor(
            position
        )

        upper = math.ceil(
            position
        )

        if lower == upper:
            return ordered[
                lower
            ]

        weight = (
            position
            - lower
        )

        return (
            ordered[lower]
            * (1.0 - weight)
            + ordered[upper]
            * weight
        )

    def inspect_latest(
        self,
        *,
        feature_version: str,
        expected_snapshots: int = 100,
        include_all_features: bool = False,
    ) -> dict:
        latest_date = (
            self.db.scalar(
                select(
                    func.max(
                        StockFeatureSnapshot
                        .feature_date
                    )
                )
                .where(
                    StockFeatureSnapshot
                    .feature_version
                    == feature_version
                )
            )
        )

        if latest_date is None:
            return {
                "status": "fail",
                "feature_version":
                    feature_version,
                "feature_date":
                    None,
                "snapshot_count":
                    0,
                "expected_snapshot_count":
                    expected_snapshots,
                "expected_feature_count":
                    len(
                        self.EXPECTED_FEATURES
                    ),
                "issues": [
                    "Feature Snapshot이 없습니다."
                ],
                "warnings": [],
                "problem_stocks": [],
                "problem_features": [],
            }

        rows = list(
            self.db.scalars(
                select(
                    StockFeatureSnapshot
                )
                .where(
                    StockFeatureSnapshot
                    .feature_version
                    == feature_version,
                    StockFeatureSnapshot
                    .feature_date
                    == latest_date,
                )
                .order_by(
                    StockFeatureSnapshot
                    .stock_code
                    .asc()
                )
            ).all()
        )

        expected_set = set(
            self.EXPECTED_FEATURES
        )

        feature_values = {
            name: []
            for name
            in self.EXPECTED_FEATURES
        }

        missing_counts = {
            name: 0
            for name
            in self.EXPECTED_FEATURES
        }

        invalid_counts = {
            name: 0
            for name
            in self.EXPECTED_FEATURES
        }

        range_counts = {
            name: 0
            for name
            in self.EXPECTED_FEATURES
        }

        problem_stocks = []
        structural_issue_count = 0

        for row in rows:
            features = (
                row.features
                or {}
            )

            actual_set = set(
                features.keys()
            )

            missing_keys = sorted(
                expected_set
                - actual_set
            )

            extra_keys = sorted(
                actual_set
                - expected_set
            )

            null_keys = []
            invalid_keys = []
            range_keys = []

            if (
                missing_keys
                or extra_keys
            ):
                structural_issue_count += 1

            for name in self.EXPECTED_FEATURES:
                if name not in features:
                    missing_counts[
                        name
                    ] += 1
                    continue

                value = features.get(
                    name
                )

                if value is None:
                    missing_counts[
                        name
                    ] += 1

                    null_keys.append(
                        name
                    )

                    continue

                try:
                    number = float(
                        value
                    )

                except (
                    TypeError,
                    ValueError,
                ):
                    invalid_counts[
                        name
                    ] += 1

                    invalid_keys.append(
                        name
                    )

                    continue

                if not math.isfinite(
                    number
                ):
                    invalid_counts[
                        name
                    ] += 1

                    invalid_keys.append(
                        name
                    )

                    continue

                feature_values[
                    name
                ].append(
                    number
                )

                rule = (
                    self.RANGE_RULES
                    .get(
                        name
                    )
                )

                if rule is None:
                    continue

                minimum, maximum = (
                    rule
                )

                violated = (
                    minimum is not None
                    and number < minimum
                ) or (
                    maximum is not None
                    and number > maximum
                )

                if violated:
                    range_counts[
                        name
                    ] += 1

                    range_keys.append(
                        name
                    )

            if (
                missing_keys
                or extra_keys
                or null_keys
                or invalid_keys
                or range_keys
            ):
                problem_stocks.append(
                    {
                        "stock_code":
                            row.stock_code,
                        "missing_keys":
                            missing_keys,
                        "extra_keys":
                            extra_keys,
                        "null_features":
                            null_keys,
                        "invalid_features":
                            invalid_keys,
                        "range_violations":
                            range_keys,
                    }
                )

        feature_stats = []
        problem_features = []

        total_missing = 0
        total_invalid = 0
        total_range = 0
        total_outliers = 0

        zero_variance = []

        for name in self.EXPECTED_FEATURES:
            values = (
                feature_values[
                    name
                ]
            )

            missing = (
                missing_counts[
                    name
                ]
            )

            invalid = (
                invalid_counts[
                    name
                ]
            )

            range_violations = (
                range_counts[
                    name
                ]
            )

            total_missing += (
                missing
            )

            total_invalid += (
                invalid
            )

            total_range += (
                range_violations
            )

            mean = (
                fmean(values)
                if values
                else None
            )

            std = (
                pstdev(values)
                if len(values) >= 2
                else 0.0
                if values
                else None
            )

            q01 = (
                self._percentile(
                    values,
                    0.01,
                )
            )

            q25 = (
                self._percentile(
                    values,
                    0.25,
                )
            )

            q50 = (
                self._percentile(
                    values,
                    0.50,
                )
            )

            q75 = (
                self._percentile(
                    values,
                    0.75,
                )
            )

            q99 = (
                self._percentile(
                    values,
                    0.99,
                )
            )

            outlier_count = 0

            if (
                q25 is not None
                and q75 is not None
            ):
                iqr = (
                    q75
                    - q25
                )

                if iqr > 0.0:
                    lower = (
                        q25
                        - 3.0
                        * iqr
                    )

                    upper = (
                        q75
                        + 3.0
                        * iqr
                    )

                    outlier_count = sum(
                        1
                        for value
                        in values
                        if (
                            value < lower
                            or value > upper
                        )
                    )

            total_outliers += (
                outlier_count
            )

            if (
                std == 0.0
                and values
                and name
                not in self
                .SHARED_MACRO_FEATURES
            ):
                zero_variance.append(
                    name
                )

            stat = {
                "feature":
                    name,

                "non_null_count":
                    len(values),

                "missing_count":
                    missing,

                "missing_rate_pct":
                    round(
                        missing
                        / len(rows)
                        * 100.0,
                        4,
                    )
                    if rows
                    else 0.0,

                "invalid_count":
                    invalid,

                "range_violation_count":
                    range_violations,

                "outlier_count":
                    outlier_count,

                "min":
                    min(values)
                    if values
                    else None,

                "p01":
                    q01,

                "median":
                    q50,

                "p99":
                    q99,

                "max":
                    max(values)
                    if values
                    else None,

                "mean":
                    mean,

                "std":
                    std,
            }

            feature_stats.append(
                stat
            )

            if (
                missing > 0
                or invalid > 0
                or range_violations > 0
                or outlier_count > 0
            ):
                problem_features.append(
                    stat
                )

        total_cells = (
            len(rows)
            * len(
                self.EXPECTED_FEATURES
            )
        )

        issues = []
        warnings = []

        if (
            len(rows)
            != expected_snapshots
        ):
            issues.append(
                "Snapshot 개수 불일치: "
                f"{len(rows)}/"
                f"{expected_snapshots}"
            )

        if structural_issue_count:
            issues.append(
                "Feature 구조 불일치 종목: "
                f"{structural_issue_count}개"
            )

        if total_invalid:
            issues.append(
                "NaN/Infinity/비숫자: "
                f"{total_invalid}개"
            )

        if total_range:
            issues.append(
                "기본 범위 규칙 위반: "
                f"{total_range}개"
            )

        if total_missing:
            warnings.append(
                "결측 Feature: "
                f"{total_missing}/"
                f"{total_cells} "
                "("
                f"{total_missing / total_cells * 100.0:.4f}%"
                ")"
            )

        if total_outliers:
            warnings.append(
                "3×IQR 극단치 후보: "
                f"{total_outliers}개"
            )

        if zero_variance:
            warnings.append(
                "횡단면 분산 0 Feature: "
                + ", ".join(
                    zero_variance
                )
            )

        status = (
            "fail"
            if issues
            else "warning"
            if warnings
            else "pass"
        )

        result = {
            "status":
                status,

            "feature_version":
                feature_version,

            "feature_date":
                latest_date
                .isoformat(),

            "snapshot_count":
                len(rows),

            "expected_snapshot_count":
                expected_snapshots,

            "expected_feature_count":
                len(
                    self.EXPECTED_FEATURES
                ),

            "total_feature_cells":
                total_cells,

            "missing_cells":
                total_missing,

            "missing_rate_pct":
                round(
                    total_missing
                    / total_cells
                    * 100.0,
                    4,
                )
                if total_cells
                else 0.0,

            "invalid_cells":
                total_invalid,

            "range_violation_cells":
                total_range,

            "outlier_candidates":
                total_outliers,

            "structural_issue_stocks":
                structural_issue_count,

            "issues":
                issues,

            "warnings":
                warnings,

            "problem_stocks":
                problem_stocks,

            "problem_features":
                problem_features,
        }

        if include_all_features:
            result[
                "feature_stats"
            ] = feature_stats

        return result