from __future__ import annotations

from collections import defaultdict
from math import ceil, sqrt
from statistics import mean, median

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models.future import (
    RankingItem,
    RankingSnapshot,
    RecommendationPerformance,
)
from app.services.composite_ranking_service import (
    CompositeRankingService,
)
from app.services.market_context_service import (
    MarketContextService,
)


class RankingBacktestService:
    HORIZONS = (
        20,
        60,
        120,
        240,
    )

    CUTOFFS = (
        10,
        20,
    )

    FACTORS = (
        "quality",
        "growth",
        "value",
        "financial_health",
        "relative_strength",
        "flow",
        "ml",
    )

    HORIZON_WEIGHTS = {
        20: 0.10,
        60: 0.20,
        120: 0.30,
        240: 0.40,
    }

    MIN_PORTFOLIO_COVERAGE = 0.80

    MIN_FACTOR_PAIRS = 20
    MIN_FACTOR_SNAPSHOTS = 5

    MIN_TUNING_LONG_HORIZON_SNAPSHOTS = 20

    def __init__(
        self,
        db: Session,
    ):
        self.db = db

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

    @staticmethod
    def _round(
        value: float | None,
        digits: int = 4,
    ) -> float | None:
        if value is None:
            return None

        return round(
            float(
                value
            ),
            digits,
        )

    @staticmethod
    def _pearson(
        pairs: list[
            tuple[
                float,
                float,
            ]
        ],
    ) -> float | None:
        if len(
            pairs
        ) < 2:
            return None

        xs = [
            item[
                0
            ]
            for item
            in pairs
        ]

        ys = [
            item[
                1
            ]
            for item
            in pairs
        ]

        x_mean = mean(
            xs
        )

        y_mean = mean(
            ys
        )

        numerator = sum(
            (
                x
                - x_mean
            )
            * (
                y
                - y_mean
            )
            for x, y
            in pairs
        )

        x_var = sum(
            (
                x
                - x_mean
            )
            ** 2
            for x
            in xs
        )

        y_var = sum(
            (
                y
                - y_mean
            )
            ** 2
            for y
            in ys
        )

        denominator = sqrt(
            x_var
            * y_var
        )

        if denominator <= 0.0:
            return None

        return (
            numerator
            / denominator
        )

    def _load_rows(
        self,
    ):
        stmt = (
            select(
                RecommendationPerformance,
                RankingItem,
                RankingSnapshot,
            )
            .join(
                RankingItem,
                RankingItem.id
                == RecommendationPerformance
                .ranking_item_id,
            )
            .join(
                RankingSnapshot,
                RankingSnapshot.id
                == RankingItem.snapshot_id,
            )
            .where(
                RankingSnapshot
                .ranking_version
                == MarketContextService
                .RANKING_VERSION,

                RecommendationPerformance
                .horizon_days
                .in_(
                    self.HORIZONS
                ),

                RecommendationPerformance
                .actual_return
                .is_not(
                    None
                ),

                RecommendationPerformance
                .excess_return
                .is_not(
                    None
                ),

                RankingItem.rank
                <= MarketContextService
                .PERFORMANCE_RANK_LIMIT,
            )
            .order_by(
                RankingSnapshot
                .as_of_date
                .asc(),

                RecommendationPerformance
                .horizon_days
                .asc(),

                RankingItem.rank.asc(),
            )
        )

        return list(
            self.db.execute(
                stmt
            ).all()
        )

    def _portfolio_summary(
        self,
        rows,
        *,
        horizon: int,
        cutoff: int,
    ) -> dict:
        grouped = defaultdict(
            list
        )

        for (
            performance,
            item,
            snapshot,
        ) in rows:
            if (
                performance
                .horizon_days
                != horizon
            ):
                continue

            if item.rank > cutoff:
                continue

            grouped[
                int(
                    snapshot.id
                )
            ].append(
                (
                    performance,
                    item,
                    snapshot,
                )
            )

        minimum_count = max(
            1,
            ceil(
                cutoff
                * self.MIN_PORTFOLIO_COVERAGE
            ),
        )

        portfolios = []

        observation_count = 0

        for snapshot_rows in (
            grouped.values()
        ):
            if (
                len(
                    snapshot_rows
                )
                < minimum_count
            ):
                continue

            actual_returns = [
                float(
                    performance
                    .actual_return
                )
                for (
                    performance,
                    _,
                    _,
                )
                in snapshot_rows
                if performance
                .actual_return
                is not None
            ]

            benchmark_returns = [
                float(
                    performance
                    .benchmark_return
                )
                for (
                    performance,
                    _,
                    _,
                )
                in snapshot_rows
                if performance
                .benchmark_return
                is not None
            ]

            excess_returns = [
                float(
                    performance
                    .excess_return
                )
                for (
                    performance,
                    _,
                    _,
                )
                in snapshot_rows
                if performance
                .excess_return
                is not None
            ]

            if (
                len(
                    excess_returns
                )
                < minimum_count
            ):
                continue

            observation_count += len(
                excess_returns
            )

            portfolios.append(
                {
                    "actual":
                        mean(
                            actual_returns
                        ),

                    "benchmark":
                        (
                            mean(
                                benchmark_returns
                            )
                            if benchmark_returns
                            else None
                        ),

                    "excess":
                        mean(
                            excess_returns
                        ),
                }
            )

        if not portfolios:
            return {
                "cutoff":
                    cutoff,

                "snapshotCount":
                    0,

                "observationCount":
                    0,

                "averageReturn":
                    None,

                "averageBenchmarkReturn":
                    None,

                "averageExcessReturn":
                    None,

                "medianExcessReturn":
                    None,

                "excessHitRate":
                    None,

                "winRate":
                    None,
            }

        actual_values = [
            row[
                "actual"
            ]
            for row
            in portfolios
        ]

        benchmark_values = [
            row[
                "benchmark"
            ]
            for row
            in portfolios
            if row[
                "benchmark"
            ]
            is not None
        ]

        excess_values = [
            row[
                "excess"
            ]
            for row
            in portfolios
        ]

        return {
            "cutoff":
                cutoff,

            "snapshotCount":
                len(
                    portfolios
                ),

            "observationCount":
                observation_count,

            "averageReturn":
                self._round(
                    mean(
                        actual_values
                    ),
                    3,
                ),

            "averageBenchmarkReturn":
                (
                    self._round(
                        mean(
                            benchmark_values
                        ),
                        3,
                    )
                    if benchmark_values
                    else None
                ),

            "averageExcessReturn":
                self._round(
                    mean(
                        excess_values
                    ),
                    3,
                ),

            "medianExcessReturn":
                self._round(
                    median(
                        excess_values
                    ),
                    3,
                ),

            "excessHitRate":
                self._round(
                    sum(
                        1
                        for value
                        in excess_values
                        if value > 0.0
                    )
                    / len(
                        excess_values
                    )
                    * 100.0,
                    1,
                ),

            "winRate":
                self._round(
                    sum(
                        1
                        for value
                        in actual_values
                        if value > 0.0
                    )
                    / len(
                        actual_values
                    )
                    * 100.0,
                    1,
                ),
        }

    def _factor_diagnostic(
        self,
        rows,
        *,
        factor: str,
        horizon: int,
    ) -> dict:
        grouped = defaultdict(
            list
        )

        for (
            performance,
            item,
            snapshot,
        ) in rows:
            if (
                performance
                .horizon_days
                != horizon
            ):
                continue

            components = dict(
                item.score_components_json
                or {}
            )

            factor_score = (
                self._float_or_none(
                    components.get(
                        factor
                    )
                )
            )

            excess_return = (
                self._float_or_none(
                    performance
                    .excess_return
                )
            )

            if (
                factor_score is None
                or excess_return is None
            ):
                continue

            grouped[
                int(
                    snapshot.id
                )
            ].append(
                (
                    factor_score,
                    excess_return,
                )
            )

        ic_values = []
        spreads = []

        pair_count = 0

        for pairs in (
            grouped.values()
        ):
            if (
                len(
                    pairs
                )
                < self.MIN_FACTOR_PAIRS
            ):
                continue

            ic = self._pearson(
                pairs
            )

            if ic is not None:
                ic_values.append(
                    ic
                )

            ordered = sorted(
                pairs,
                key=lambda pair:
                    pair[
                        0
                    ],
            )

            bucket_size = max(
                3,
                len(
                    ordered
                )
                // 4,
            )

            low = ordered[
                :bucket_size
            ]

            high = ordered[
                -bucket_size:
            ]

            low_excess = mean(
                pair[
                    1
                ]
                for pair
                in low
            )

            high_excess = mean(
                pair[
                    1
                ]
                for pair
                in high
            )

            spreads.append(
                high_excess
                - low_excess
            )

            pair_count += len(
                pairs
            )

        return {
            "horizonTradingDays":
                horizon,

            "snapshotCount":
                len(
                    ic_values
                ),

            "pairCount":
                pair_count,

            "meanIc":
                (
                    self._round(
                        mean(
                            ic_values
                        ),
                        4,
                    )
                    if ic_values
                    else None
                ),

            "icPositiveRate":
                (
                    self._round(
                        sum(
                            1
                            for value
                            in ic_values
                            if value > 0.0
                        )
                        / len(
                            ic_values
                        )
                        * 100.0,
                        1,
                    )
                    if ic_values
                    else None
                ),

            "meanTopBottomExcessSpread":
                (
                    self._round(
                        mean(
                            spreads
                        ),
                        3,
                    )
                    if spreads
                    else None
                ),
        }

    def _weight_review(
        self,
        factor_diagnostics: dict,
        rows,
    ) -> dict:
        current_weights = dict(
            CompositeRankingService
            .WEIGHTS
        )

        long_horizon_dates = {
            snapshot.as_of_date
            for (
                performance,
                _,
                snapshot,
            )
            in rows
            if performance
            .horizon_days
            >= 60
        }

        ready = (
            len(
                long_horizon_dates
            )
            >= (
                self
                .MIN_TUNING_LONG_HORIZON_SNAPSHOTS
            )
        )

        if not ready:
            return {
                "status":
                    "insufficient_data",

                "autoApply":
                    False,

                "longHorizonSnapshotCount":
                    len(
                        long_horizon_dates
                    ),

                "minimumLongHorizonSnapshots":
                    (
                        self
                        .MIN_TUNING_LONG_HORIZON_SNAPSHOTS
                    ),

                "currentWeights":
                    current_weights,

                "suggestedWeights":
                    current_weights,

                "changes": {
                    factor:
                        0.0
                    for factor
                    in current_weights
                },

                "message": (
                    "60거래일 이상 성과가 충분히 "
                    "누적되기 전에는 가중치 변경을 "
                    "제안하지 않습니다."
                ),
            }

        raw_weights = {}

        for (
            factor,
            base_weight,
        ) in (
            current_weights.items()
        ):
            diagnostics = (
                factor_diagnostics.get(
                    factor,
                    {}
                )
            )

            weighted_ic_sum = 0.0
            weighted_spread_sum = 0.0
            available_weight = 0.0

            for horizon in (
                self.HORIZONS
            ):
                item = (
                    diagnostics.get(
                        str(
                            horizon
                        ),
                        {},
                    )
                )

                if (
                    int(
                        item.get(
                            "snapshotCount"
                        )
                        or 0
                    )
                    < self.MIN_FACTOR_SNAPSHOTS
                ):
                    continue

                ic = (
                    self._float_or_none(
                        item.get(
                            "meanIc"
                        )
                    )
                )

                spread = (
                    self._float_or_none(
                        item.get(
                            "meanTopBottomExcessSpread"
                        )
                    )
                )

                if (
                    ic is None
                    or spread is None
                ):
                    continue

                horizon_weight = (
                    self.HORIZON_WEIGHTS[
                        horizon
                    ]
                )

                weighted_ic_sum += (
                    ic
                    * horizon_weight
                )

                weighted_spread_sum += (
                    spread
                    * horizon_weight
                )

                available_weight += (
                    horizon_weight
                )

            if available_weight <= 0.0:
                raw_weights[
                    factor
                ] = base_weight

                continue

            weighted_ic = (
                weighted_ic_sum
                / available_weight
            )

            weighted_spread = (
                weighted_spread_sum
                / available_weight
            )

            ic_adjustment = max(
                -0.25,
                min(
                    0.25,
                    weighted_ic
                    * 2.0,
                ),
            )

            spread_adjustment = max(
                -0.15,
                min(
                    0.15,
                    weighted_spread
                    / 10.0,
                ),
            )

            multiplier = max(
                0.65,
                min(
                    1.35,
                    1.0
                    + ic_adjustment
                    + spread_adjustment,
                ),
            )

            raw_weights[
                factor
            ] = (
                base_weight
                * multiplier
            )

        total = sum(
            raw_weights.values()
        )

        suggested = {
            factor:
                round(
                    value
                    / total,
                    4,
                )
            for (
                factor,
                value,
            )
            in raw_weights.items()
        }

        changes = {
            factor:
                round(
                    suggested[
                        factor
                    ]
                    - current_weights[
                        factor
                    ],
                    4,
                )
            for factor
            in current_weights
        }

        return {
            "status":
                "review_ready",

            "autoApply":
                False,

            "longHorizonSnapshotCount":
                len(
                    long_horizon_dates
                ),

            "minimumLongHorizonSnapshots":
                (
                    self
                    .MIN_TUNING_LONG_HORIZON_SNAPSHOTS
                ),

            "currentWeights":
                current_weights,

            "suggestedWeights":
                suggested,

            "changes":
                changes,

            "message": (
                "제안 가중치는 자동 적용하지 않습니다. "
                "시간순 검증 후 production 가중치를 "
                "변경해야 합니다."
            ),
        }

    def build_report(
        self,
    ) -> dict:
        MarketContextService(
            self.db
        ).evaluate_performance()

        rows = self._load_rows()

        horizon_reports = []

        month_map = {
            20: 1,
            60: 3,
            120: 6,
            240: 12,
        }

        for horizon in (
            self.HORIZONS
        ):
            horizon_reports.append(
                {
                    "tradingDays":
                        horizon,

                    "approxMonths":
                        month_map[
                            horizon
                        ],

                    "top10":
                        self._portfolio_summary(
                            rows,
                            horizon=horizon,
                            cutoff=10,
                        ),

                    "top20":
                        self._portfolio_summary(
                            rows,
                            horizon=horizon,
                            cutoff=20,
                        ),
                }
            )

        factor_diagnostics = {}

        for factor in (
            self.FACTORS
        ):
            factor_diagnostics[
                factor
            ] = {
                str(
                    horizon
                ):
                    self._factor_diagnostic(
                        rows,
                        factor=factor,
                        horizon=horizon,
                    )
                for horizon
                in self.HORIZONS
            }

        snapshot_dates = sorted(
            {
                snapshot.as_of_date
                for (
                    _,
                    _,
                    snapshot,
                )
                in rows
            }
        )

        return {
            "rankingVersion":
                MarketContextService
                .RANKING_VERSION,

            "trackedRankLimit":
                MarketContextService
                .PERFORMANCE_RANK_LIMIT,

            "horizons":
                horizon_reports,

            "factorDiagnostics":
                factor_diagnostics,

            "weightReview":
                self._weight_review(
                    factor_diagnostics,
                    rows,
                ),

            "evaluatedSnapshotCount":
                len(
                    snapshot_dates
                ),

            "firstEvaluatedSnapshotDate":
                (
                    snapshot_dates[
                        0
                    ].isoformat()
                    if snapshot_dates
                    else None
                ),

            "lastEvaluatedSnapshotDate":
                (
                    snapshot_dates[
                        -1
                    ].isoformat()
                    if snapshot_dates
                    else None
                ),

            "methodology": {
                "portfolio": (
                    "각 랭킹 스냅샷의 TOP10/TOP20 "
                    "동일가중 평균 수익률"
                ),

                "benchmark": (
                    "종목 시장에 대응하는 "
                    "KOSPI/KOSDAQ 지수 수익률"
                ),

                "factorIc": (
                    "동일 스냅샷 내 Factor 점수와 "
                    "미래 초과수익률의 횡단면 IC"
                ),

                "factorSpread": (
                    "Factor 상위 25%와 하위 25%의 "
                    "평균 초과수익률 차이"
                ),

                "limitations": [
                    (
                        "새 종합랭킹 버전이 생성된 "
                        "시점 이후 데이터만 사용합니다."
                    ),
                    (
                        "현재 TOP100 내부에서 "
                        "Factor를 검증합니다."
                    ),
                    (
                        "가중치 제안은 자동으로 "
                        "production에 적용하지 않습니다."
                    ),
                ],
            },
        }