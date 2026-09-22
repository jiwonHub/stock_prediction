from __future__ import annotations

from collections import defaultdict
from math import ceil, sqrt
from statistics import mean, median, pstdev

from sqlalchemy import func, select
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
    MIN_RELIABLE_FACTORS = 5

    RECENT_PERFORMANCE_MIN_SNAPSHOTS = 10
    RECENT_EXCESS_DROP_WARNING_PCT = 1.0

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
        benchmark_observation_count = 0
        benchmark_snapshot_count = 0

        for snapshot_rows in (
            grouped.values()
        ):
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

            if (
                len(
                    actual_returns
                )
                < minimum_count
            ):
                continue

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

            observation_count += len(
                actual_returns
            )

            benchmark_observation_count += len(
                excess_returns
            )

            benchmark_comparable = (
                len(
                    excess_returns
                )
                >= minimum_count
            )

            if benchmark_comparable:
                benchmark_snapshot_count += 1

            portfolios.append(
                {
                    "date":
                        snapshot_rows[
                            0
                        ][
                            2
                        ].as_of_date,

                    "actual":
                        mean(
                            actual_returns
                        ),

                    "benchmark":
                        (
                            mean(
                                benchmark_returns
                            )
                            if (
                                benchmark_comparable
                                and benchmark_returns
                            )
                            else None
                        ),

                    "excess":
                        (
                            mean(
                                excess_returns
                            )
                            if benchmark_comparable
                            else None
                        ),
                }
            )

        if not portfolios:
            return {
                "cutoff":
                    cutoff,

                "minimumRequiredObservations":
                    minimum_count,

                "snapshotCount":
                    0,

                "benchmarkSnapshotCount":
                    0,

                "observationCount":
                    0,

                "benchmarkObservationCount":
                    0,

                "actualCoveragePercent":
                    None,

                "benchmarkCoveragePercent":
                    None,

                "firstSnapshotDate":
                    None,

                "lastSnapshotDate":
                    None,

                "averageReturn":
                    None,

                "medianReturn":
                    None,

                "returnStdDev":
                    None,

                "worstReturn":
                    None,

                "bestReturn":
                    None,

                "averageBenchmarkReturn":
                    None,

                "averageExcessReturn":
                    None,

                "medianExcessReturn":
                    None,

                "excessStdDev":
                    None,

                "worstExcessReturn":
                    None,

                "bestExcessReturn":
                    None,

                "excessHitRate":
                    None,

                "winRate":
                    None,

                "recentSnapshotCount":
                    0,

                "recentAverageReturn":
                    None,

                "recentAverageExcessReturn":
                    None,

                "recentReturnDelta":
                    None,

                "recentExcessDelta":
                    None,

                "recentUnderperformance":
                    False,
            }

        portfolios.sort(
            key=lambda row:
                row[
                    "date"
                ]
        )

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
            if row[
                "excess"
            ]
            is not None
        ]

        recent_portfolios = (
            portfolios[
                -20:
            ]
        )

        recent_actual_values = [
            row[
                "actual"
            ]
            for row
            in recent_portfolios
        ]

        recent_excess_values = [
            row[
                "excess"
            ]
            for row
            in recent_portfolios
            if row[
                "excess"
            ]
            is not None
        ]

        average_return = mean(
            actual_values
        )

        average_excess_return = (
            mean(
                excess_values
            )
            if excess_values
            else None
        )

        recent_average_return = (
            mean(
                recent_actual_values
            )
            if recent_actual_values
            else None
        )

        recent_average_excess_return = (
            mean(
                recent_excess_values
            )
            if recent_excess_values
            else None
        )

        recent_return_delta = (
            recent_average_return
            - average_return
            if recent_average_return
            is not None
            else None
        )

        recent_excess_delta = (
            recent_average_excess_return
            - average_excess_return
            if (
                recent_average_excess_return
                is not None
                and average_excess_return
                is not None
            )
            else None
        )

        recent_underperformance = (
            len(
                recent_portfolios
            )
            >= self.RECENT_PERFORMANCE_MIN_SNAPSHOTS
            and recent_average_excess_return
            is not None
            and recent_excess_delta
            is not None
            and recent_average_excess_return
            < 0.0
            and recent_excess_delta
            <= (
                -self
                .RECENT_EXCESS_DROP_WARNING_PCT
            )
        )

        expected_observation_count = (
            len(
                portfolios
            )
            * cutoff
        )

        actual_coverage_percent = (
            observation_count
            / expected_observation_count
            * 100.0
            if expected_observation_count > 0
            else None
        )

        benchmark_coverage_percent = (
            benchmark_observation_count
            / observation_count
            * 100.0
            if observation_count > 0
            else None
        )

        return {
            "cutoff":
                cutoff,

            "minimumRequiredObservations":
                minimum_count,

            "snapshotCount":
                len(
                    portfolios
                ),

            "benchmarkSnapshotCount":
                benchmark_snapshot_count,

            "observationCount":
                observation_count,

            "benchmarkObservationCount":
                benchmark_observation_count,

            "actualCoveragePercent":
                self._round(
                    actual_coverage_percent,
                    1,
                ),

            "benchmarkCoveragePercent":
                self._round(
                    benchmark_coverage_percent,
                    1,
                ),

            "firstSnapshotDate":
                portfolios[
                    0
                ][
                    "date"
                ].isoformat(),

            "lastSnapshotDate":
                portfolios[
                    -1
                ][
                    "date"
                ].isoformat(),

            "averageReturn":
                self._round(
                    average_return,
                    3,
                ),

            "medianReturn":
                self._round(
                    median(
                        actual_values
                    ),
                    3,
                ),

            "returnStdDev":
                (
                    self._round(
                        pstdev(
                            actual_values
                        ),
                        3,
                    )
                    if len(
                        actual_values
                    ) >= 2
                    else None
                ),

            "worstReturn":
                self._round(
                    min(
                        actual_values
                    ),
                    3,
                ),

            "bestReturn":
                self._round(
                    max(
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
                    average_excess_return,
                    3,
                ),

            "medianExcessReturn":
                (
                    self._round(
                        median(
                            excess_values
                        ),
                        3,
                    )
                    if excess_values
                    else None
                ),

            "excessStdDev":
                (
                    self._round(
                        pstdev(
                            excess_values
                        ),
                        3,
                    )
                    if len(
                        excess_values
                    ) >= 2
                    else None
                ),

            "worstExcessReturn":
                (
                    self._round(
                        min(
                            excess_values
                        ),
                        3,
                    )
                    if excess_values
                    else None
                ),

            "bestExcessReturn":
                (
                    self._round(
                        max(
                            excess_values
                        ),
                        3,
                    )
                    if excess_values
                    else None
                ),

            "excessHitRate":
                (
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
                    )
                    if excess_values
                    else None
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

            "recentSnapshotCount":
                len(
                    recent_portfolios
                ),

            "recentAverageReturn":
                self._round(
                    recent_average_return,
                    3,
                ),

            "recentAverageExcessReturn":
                self._round(
                    recent_average_excess_return,
                    3,
                ),

            "recentReturnDelta":
                self._round(
                    recent_return_delta,
                    3,
                ),

            "recentExcessDelta":
                self._round(
                    recent_excess_delta,
                    3,
                ),

            "recentUnderperformance":
                recent_underperformance,
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

        sufficient_data = (
            len(
                ic_values
            )
            >= self.MIN_FACTOR_SNAPSHOTS
            and len(
                spreads
            )
            >= self.MIN_FACTOR_SNAPSHOTS
            and pair_count
            >= (
                self.MIN_FACTOR_PAIRS
                * self.MIN_FACTOR_SNAPSHOTS
            )
        )

        return {
            "horizonTradingDays":
                horizon,

            "snapshotCount":
                len(
                    ic_values
                ),

            "spreadSnapshotCount":
                len(
                    spreads
                ),

            "pairCount":
                pair_count,

            "sufficientData":
                sufficient_data,

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

            "icStdDev":
                (
                    self._round(
                        pstdev(
                            ic_values
                        ),
                        4,
                    )
                    if len(
                        ic_values
                    ) >= 2
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

            "spreadStdDev":
                (
                    self._round(
                        pstdev(
                            spreads
                        ),
                        3,
                    )
                    if len(
                        spreads
                    ) >= 2
                    else None
                ),

            "spreadPositiveRate":
                (
                    self._round(
                        sum(
                            1
                            for value
                            in spreads
                            if value > 0.0
                        )
                        / len(
                            spreads
                        )
                        * 100.0,
                        1,
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
            if (
                performance
                .horizon_days
                >= 60
                and performance
                .excess_return
                is not None
            )
        }

        reliable_factors = []

        for factor in (
            current_weights
        ):
            diagnostics = (
                factor_diagnostics.get(
                    factor,
                    {}
                )
            )

            has_reliable_long_horizon = any(
                bool(
                    diagnostics.get(
                        str(
                            horizon
                        ),
                        {},
                    ).get(
                        "sufficientData",
                        False,
                    )
                )
                for horizon in (
                    60,
                    120,
                    240,
                )
            )

            if has_reliable_long_horizon:
                reliable_factors.append(
                    factor
                )

        ready = (
            len(
                long_horizon_dates
            )
            >= (
                self
                .MIN_TUNING_LONG_HORIZON_SNAPSHOTS
            )
            and len(
                reliable_factors
            )
            >= self.MIN_RELIABLE_FACTORS
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

                "reliableFactorCount":
                    len(
                        reliable_factors
                    ),

                "minimumReliableFactors":
                    self.MIN_RELIABLE_FACTORS,

                "reliableFactors":
                    reliable_factors,

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
                    "장기 성과와 Factor 진단 신뢰도가 "
                    "충분히 누적되기 전에는 가중치 변경을 "
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

                if not bool(
                    item.get(
                        "sufficientData",
                        False,
                    )
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

            "reliableFactorCount":
                len(
                    reliable_factors
                ),

            "minimumReliableFactors":
                self.MIN_RELIABLE_FACTORS,

            "reliableFactors":
                reliable_factors,

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

    def _evaluation_progress(
        self,
    ) -> list[dict]:
        result = []

        for horizon in (
            self.HORIZONS
        ):
            base_filters = (
                RankingSnapshot
                .ranking_version
                == MarketContextService
                .RANKING_VERSION,

                RecommendationPerformance
                .horizon_days
                == horizon,

                RankingItem.rank
                <= MarketContextService
                .PERFORMANCE_RANK_LIMIT,
            )

            def load_counts(
                *extra_filters,
            ) -> tuple[int, int]:
                row = self.db.execute(
                    select(
                        func.count(
                            RecommendationPerformance.id
                        ),
                        func.count(
                            func.distinct(
                                RankingSnapshot.id
                            )
                        ),
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
                        *base_filters,
                        *extra_filters,
                    )
                ).one()

                return (
                    int(
                        row[0]
                        or 0
                    ),
                    int(
                        row[1]
                        or 0
                    ),
                )

            (
                total_observation_count,
                total_snapshot_count,
            ) = load_counts()

            (
                evaluated_observation_count,
                evaluated_snapshot_count,
            ) = load_counts(
                RecommendationPerformance
                .actual_return
                .is_not(
                    None
                ),
            )

            (
                benchmark_observation_count,
                benchmark_snapshot_count,
            ) = load_counts(
                RecommendationPerformance
                .excess_return
                .is_not(
                    None
                ),
            )

            pending_observation_count = max(
                0,
                total_observation_count
                - evaluated_observation_count,
            )

            evaluation_percent = (
                evaluated_observation_count
                / total_observation_count
                * 100.0
                if total_observation_count > 0
                else 0.0
            )

            benchmark_coverage_percent = (
                benchmark_observation_count
                / evaluated_observation_count
                * 100.0
                if evaluated_observation_count > 0
                else None
            )

            result.append(
                {
                    "tradingDays":
                        horizon,

                    "totalObservationCount":
                        total_observation_count,

                    "evaluatedObservationCount":
                        evaluated_observation_count,

                    "benchmarkObservationCount":
                        benchmark_observation_count,

                    "pendingObservationCount":
                        pending_observation_count,

                    "evaluationPercent":
                        self._round(
                            evaluation_percent,
                            1,
                        ),

                    "benchmarkCoveragePercent":
                        self._round(
                            benchmark_coverage_percent,
                            1,
                        ),

                    "totalSnapshotCount":
                        total_snapshot_count,

                    "evaluatedSnapshotCount":
                        evaluated_snapshot_count,

                    "benchmarkSnapshotCount":
                        benchmark_snapshot_count,
                }
            )

        return result

    def build_report(
        self,
    ) -> dict:
        MarketContextService(
            self.db
        ).evaluate_performance()

        rows = self._load_rows()

        evaluation_progress = (
            self._evaluation_progress()
        )

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

            "evaluationProgress":
                evaluation_progress,

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

                "overlap": (
                    "랭킹 스냅샷별 보유기간이 서로 겹칠 수 있어 "
                    "일별 스냅샷 수익률을 단순 복리 누적하지 않습니다."
                ),

                "recentWindow": (
                    "최근 성과는 최신 평가 스냅샷 "
                    "최대 20개 기준입니다."
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