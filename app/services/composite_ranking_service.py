from __future__ import annotations

from datetime import date, timedelta

from sqlalchemy.orm import Session

from app.repositories.market_data_repository import (
    MarketDataRepository,
)
from app.repositories.stock_repository import (
    StockRepository,
)
from app.services.market_data_service import (
    MarketDataService,
)
from app.services.ml_ranking_inference_service import (
    MlRankingInferenceService,
)


class CompositeRankingService:
    RANKING_VERSION = "phase13-long-term-investment-v3"

    WEIGHTS = {
        "quality": 0.25,
        "growth": 0.20,
        "value": 0.20,
        "financial_health": 0.15,
        "relative_strength": 0.10,
        "flow": 0.05,
        "ml": 0.05,
    }

    FINANCIAL_METRIC_DIRECTIONS = {
        "roe": True,
        "roa": True,
        "operating_margin": True,
        "net_margin": True,
        "revenue_growth": True,
        "operating_income_growth": True,
        "net_income_growth": True,
        "debt_ratio": False,
        "equity_ratio": True,
        "operating_cash_flow_margin": True,
        "cash_flow_quality": True,
    }

    FINANCIAL_FACTOR_COMPONENTS = {
        "quality": {
            "roe": 0.35,
            "roa": 0.15,
            "operating_margin": 0.30,
            "net_margin": 0.20,
        },
        "growth": {
            "revenue_growth": 0.35,
            "operating_income_growth": 0.40,
            "net_income_growth": 0.25,
        },
        "financial_health": {
            "debt_ratio": 0.35,
            "equity_ratio": 0.25,
            "operating_cash_flow_margin": 0.25,
            "cash_flow_quality": 0.15,
        },
    }

    MIN_CANDIDATE_UNIVERSE = 500
    CANDIDATE_MULTIPLIER = 5
    MIN_DATA_COVERAGE = 0.70
    FLOW_LOOKBACK_DAYS = 180

    def __init__(
        self,
        db: Session,
    ):
        self.db = db

        self.stock_repository = (
            StockRepository(
                db
            )
        )

        self.market_repository = (
            MarketDataRepository(
                db
            )
        )

        self.market_service = (
            MarketDataService(
                db
            )
        )

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
    def _relative_ratio(
        cls,
        value,
        benchmark,
    ) -> float | None:
        left = cls._float_or_none(
            value
        )

        right = cls._float_or_none(
            benchmark
        )

        if (
            left is None
            or right is None
            or left <= 0.0
            or right <= 0.0
        ):
            return None

        return (
            left / right
            - 1.0
        )

    @staticmethod
    def _percentile_scores(
        values: dict[
            str,
            float | None,
        ],
        *,
        higher_is_better: bool,
    ) -> dict[
        str,
        float | None,
    ]:
        result: dict[
            str,
            float | None,
        ] = {
            code: None
            for code
            in values
        }

        valid = [
            (
                code,
                float(value),
            )
            for code, value
            in values.items()
            if value is not None
        ]

        if not valid:
            return result

        if len(valid) == 1:
            result[
                valid[0][0]
            ] = 50.0

            return result

        valid.sort(
            key=lambda item: (
                item[1],
                item[0],
            )
        )

        index = 0
        denominator = (
            len(valid)
            - 1
        )

        while index < len(valid):
            end = index + 1

            current_value = (
                valid[
                    index
                ][1]
            )

            while (
                end < len(valid)
                and valid[end][1]
                == current_value
            ):
                end += 1

            average_rank = (
                index
                + end
                - 1
            ) / 2.0

            score = (
                average_rank
                / denominator
                * 100.0
            )

            if not higher_is_better:
                score = (
                    100.0
                    - score
                )

            for position in range(
                index,
                end,
            ):
                result[
                    valid[
                        position
                    ][0]
                ] = score

            index = end

        return result

    @staticmethod
    def _weighted_average(
        values: list[
            tuple[
                float | None,
                float,
            ]
        ],
    ) -> float | None:
        available = [
            (
                float(value),
                float(weight),
            )
            for value, weight
            in values
            if value is not None
        ]

        if not available:
            return None

        denominator = sum(
            weight
            for _, weight
            in available
        )

        if denominator <= 0.0:
            return None

        return sum(
            value * weight
            for value, weight
            in available
        ) / denominator

    @classmethod
    def _group_percentile_scores(
        cls,
        values: dict[
            str,
            float | None,
        ],
        groups: dict[
            str,
            str | None,
        ],
        *,
        higher_is_better: bool,
        min_group_size: int = 3,
    ) -> dict[
        str,
        float | None,
    ]:
        result = cls._percentile_scores(
            values,
            higher_is_better=(
                higher_is_better
            ),
        )

        grouped_codes: dict[
            str,
            list[str],
        ] = {}

        for stock_code, group in groups.items():
            group_name = str(
                group or ""
            ).strip()

            if not group_name:
                continue

            if values.get(
                stock_code
            ) is None:
                continue

            grouped_codes.setdefault(
                group_name,
                [],
            ).append(
                stock_code
            )

        for stock_codes in grouped_codes.values():
            if len(
                stock_codes
            ) < min_group_size:
                continue

            group_values = {
                stock_code: values[
                    stock_code
                ]
                for stock_code
                in stock_codes
            }

            result.update(
                cls._percentile_scores(
                    group_values,
                    higher_is_better=(
                        higher_is_better
                    ),
                )
            )

        return result

    @classmethod
    def _build_financial_factor_scores(
        cls,
        *,
        financial_raw: dict[
            str,
            dict[
                str,
                float | None,
            ],
        ],
        sector_map: dict[
            str,
            str | None,
        ],
    ) -> dict[
        str,
        dict[
            str,
            float | None,
        ],
    ]:
        metric_scores = {
            metric_key:
                cls._group_percentile_scores(
                    values,
                    sector_map,
                    higher_is_better=(
                        cls.FINANCIAL_METRIC_DIRECTIONS[
                            metric_key
                        ]
                    ),
                )
            for metric_key, values
            in financial_raw.items()
        }

        stock_codes = set()

        for values in financial_raw.values():
            stock_codes.update(
                values.keys()
            )

        factor_scores: dict[
            str,
            dict[
                str,
                float | None,
            ],
        ] = {}

        for stock_code in stock_codes:
            factor_scores[
                stock_code
            ] = {
                factor_name:
                    cls._weighted_average(
                        [
                            (
                                metric_scores[
                                    metric_key
                                ].get(
                                    stock_code
                                ),
                                weight,
                            )
                            for metric_key, weight
                            in components.items()
                        ]
                    )
                for factor_name, components
                in cls.FINANCIAL_FACTOR_COMPONENTS.items()
            }

        return factor_scores

    @staticmethod
    def _smart_money_flow_ratio(
        *,
        flow_rows: list,
        price_rows: list,
        periods: int,
    ) -> float | None:
        price_volume = {
            row.trade_date:
                float(
                    row.volume
                    or 0
                )
            for row
            in price_rows
        }

        eligible = [
            row
            for row
            in flow_rows
            if row.trade_date
            in price_volume
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
                row.foreign_net_buy_volume
                or 0
            )
            + float(
                row.institution_net_buy_volume
                or 0
            )
            for row
            in selected
        )

        return (
            numerator
            / denominator
        )

    def score_current_universe(
        self,
        *,
        limit: int = 100,
    ) -> dict:
        candidate_limit = max(
            self.MIN_CANDIDATE_UNIVERSE,
            limit
            * self.CANDIDATE_MULTIPLIER,
        )

        stock_codes = (
            self.market_service
            .get_current_universe_stock_codes(
                limit=candidate_limit,
            )
        )

        ml_result = (
            MlRankingInferenceService(
                self.db
            )
            .score_current_universe(
                limit=candidate_limit,

                # Composite Ranking은
                # ML feature contribution 설명이 아니라
                # 상대점수만 필요합니다.
                include_explanations=False,
            )
        )

        ml_map = {
            str(row["stock_code"]): row
            for row
            in ml_result["scores"]
        }

        contexts = (
            self.stock_repository
            .get_ranking_context_by_stock_codes(
                stock_codes
            )
        )

        context_map = {
            stock.code: (
                stock,
                metric,
            )
            for stock, metric
            in contexts
        }

        sector_map: dict[
            str,
            str | None,
        ] = {}

        financial_raw: dict[
            str,
            dict[
                str,
                float | None,
            ],
        ] = {
            metric_key: {}
            for metric_key
            in self.FINANCIAL_METRIC_DIRECTIONS
        }

        per_relative_raw: dict[
            str,
            float | None,
        ] = {}

        pbr_relative_raw: dict[
            str,
            float | None,
        ] = {}

        per_absolute_raw: dict[
            str,
            float | None,
        ] = {}

        pbr_absolute_raw: dict[
            str,
            float | None,
        ] = {}

        relative_strength_raw: dict[
            str,
            float | None,
        ] = {}

        flow_20d_raw: dict[
            str,
            float | None,
        ] = {}

        flow_60d_raw: dict[
            str,
            float | None,
        ] = {}

        raw_rows: list[dict] = []

        start_date = (
            date.today()
            - timedelta(
                days=(
                    self.FLOW_LOOKBACK_DAYS
                )
            )
        )

        for stock_code in stock_codes:
            context = (
                context_map.get(
                    stock_code
                )
            )

            if context is None:
                continue

            stock, metric = context

            sector_map[
                stock_code
            ] = stock.sector_name

            for metric_key in (
                financial_raw
            ):
                financial_raw[
                    metric_key
                ][
                    stock_code
                ] = (
                    self._float_or_none(
                        getattr(
                            metric,
                            metric_key,
                            None,
                        )
                    )
                    if metric
                    is not None
                    else None
                )

            financial_score = (
                float(
                    metric.financial_score
                )
                if metric
                is not None
                else 0.0
            )

            valuation = (
                self.market_repository
                .get_latest_valuation(
                    stock_code
                )
            )

            per_benchmark = None
            pbr_benchmark = None

            if valuation is not None:
                per_benchmark = (
                    valuation.sector_per
                    if (
                        valuation.sector_per
                        is not None
                    )
                    else valuation.market_per
                )

                pbr_benchmark = (
                    valuation.sector_pbr
                    if (
                        valuation.sector_pbr
                        is not None
                    )
                    else valuation.market_pbr
                )

            per_value = (
                self._float_or_none(
                    valuation.per
                )
                if valuation
                is not None
                else None
            )

            pbr_value = (
                self._float_or_none(
                    valuation.pbr
                )
                if valuation
                is not None
                else None
            )

            if (
                per_value is not None
                and per_value <= 0.0
            ):
                per_value = None

            if (
                pbr_value is not None
                and pbr_value <= 0.0
            ):
                pbr_value = None

            per_relative = (
                self._relative_ratio(
                    per_value,
                    per_benchmark,
                )
            )

            pbr_relative = (
                self._relative_ratio(
                    pbr_value,
                    pbr_benchmark,
                )
            )

            price_rows = (
                self.stock_repository
                .get_daily_prices(
                    stock_code=(
                        stock_code
                    ),
                    start_date=(
                        start_date
                    ),
                )
            )

            flow_rows = (
                self.market_repository
                .get_stock_investor_flows(
                    stock_code=(
                        stock_code
                    ),
                    start_date=(
                        start_date
                    ),
                )
            )

            flow_20d = (
                self._smart_money_flow_ratio(
                    flow_rows=(
                        flow_rows
                    ),
                    price_rows=(
                        price_rows
                    ),
                    periods=20,
                )
            )

            flow_60d = (
                self._smart_money_flow_ratio(
                    flow_rows=(
                        flow_rows
                    ),
                    price_rows=(
                        price_rows
                    ),
                    periods=60,
                )
            )

            relative_strength_value = None

            try:
                relative_strength = (
                    self.market_service
                    .calculate_stock_relative_strength(
                        stock_code
                    )
                )

                relative_strength_value = (
                    self._weighted_average(
                        [
                            (
                                self._float_or_none(
                                    relative_strength.get(
                                        "sector_excess_20d_pct"
                                    )
                                ),
                                0.20,
                            ),
                            (
                                self._float_or_none(
                                    relative_strength.get(
                                        "sector_excess_60d_pct"
                                    )
                                ),
                                0.40,
                            ),
                            (
                                self._float_or_none(
                                    relative_strength.get(
                                        "market_excess_20d_pct"
                                    )
                                ),
                                0.15,
                            ),
                            (
                                self._float_or_none(
                                    relative_strength.get(
                                        "market_excess_60d_pct"
                                    )
                                ),
                                0.25,
                            ),
                        ]
                    )
                )

            except ValueError:
                relative_strength_value = None

            ml_row = (
                ml_map.get(
                    stock_code
                )
            )

            ml_score = (
                self._float_or_none(
                    ml_row.get(
                        "ml_score"
                    )
                )
                if ml_row
                is not None
                else None
            )

            per_relative_raw[
                stock_code
            ] = per_relative

            pbr_relative_raw[
                stock_code
            ] = pbr_relative

            per_absolute_raw[
                stock_code
            ] = per_value

            pbr_absolute_raw[
                stock_code
            ] = pbr_value

            relative_strength_raw[
                stock_code
            ] = relative_strength_value

            flow_20d_raw[
                stock_code
            ] = flow_20d

            flow_60d_raw[
                stock_code
            ] = flow_60d

            raw_rows.append(
                {
                    "stock_code":
                        stock_code,

                    "stock_name":
                        stock.name,

                    "sector_name":
                        (
                            stock.sector_name.strip()
                            if (
                                stock.sector_name
                                and stock.sector_name.strip()
                            )
                            else None
                        ),

                    "market_cap":
                        float(
                            stock.market_cap
                            or 0.0
                        ),

                    "financial_score":
                        financial_score,

                    "ml_score":
                        ml_score,
                }
            )

        financial_metric_counts = {
            metric_key: sum(
                1
                for value in values.values()
                if value is not None
            )
            for metric_key, values
            in financial_raw.items()
        }

        print(
            "[RANKING][INPUT] "
            f"candidate={len(stock_codes)} "
            f"contexts={len(context_map)} "
            f"rawRows={len(raw_rows)} "
            f"financialMetrics={financial_metric_counts} "
            f"PER={sum(1 for value in per_absolute_raw.values() if value is not None)} "
            f"PBR={sum(1 for value in pbr_absolute_raw.values() if value is not None)} "
            f"PER_REL={sum(1 for value in per_relative_raw.values() if value is not None)} "
            f"PBR_REL={sum(1 for value in pbr_relative_raw.values() if value is not None)}",
            flush=True,
        )

        financial_factor_scores = (
            self._build_financial_factor_scores(
                financial_raw=(
                    financial_raw
                ),
                sector_map=(
                    sector_map
                ),
            )
        )

        financial_factor_counts = {
            factor_name: sum(
                1
                for scores in (
                    financial_factor_scores.values()
                )
                if scores.get(
                    factor_name
                ) is not None
            )
            for factor_name
            in self.FINANCIAL_FACTOR_COMPONENTS
        }

        print(
            "[RANKING][FINANCIAL-SCORE] "
            f"counts={financial_factor_counts}",
            flush=True,
        )

        financial_factor_coverage: dict[
            str,
            dict[
                str,
                float,
            ],
        ] = {}

        for stock_code in stock_codes:
            stock_factor_coverage = {}

            for (
                factor_name,
                components,
            ) in (
                self.FINANCIAL_FACTOR_COMPONENTS
                .items()
            ):
                available_component_weight = sum(
                    float(
                        component_weight
                    )
                    for (
                        metric_key,
                        component_weight,
                    ) in components.items()
                    if (
                        financial_raw[
                            metric_key
                        ].get(
                            stock_code
                        )
                        is not None
                    )
                )

                stock_factor_coverage[
                    factor_name
                ] = min(
                    1.0,
                    max(
                        0.0,
                        available_component_weight,
                    ),
                )

            financial_factor_coverage[
                stock_code
            ] = (
                stock_factor_coverage
            )

        per_relative_scores = (
            self._percentile_scores(
                per_relative_raw,
                higher_is_better=False,
            )
        )

        pbr_relative_scores = (
            self._percentile_scores(
                pbr_relative_raw,
                higher_is_better=False,
            )
        )

        per_absolute_scores = (
            self._percentile_scores(
                per_absolute_raw,
                higher_is_better=False,
            )
        )

        pbr_absolute_scores = (
            self._percentile_scores(
                pbr_absolute_raw,
                higher_is_better=False,
            )
        )

        relative_strength_scores = (
            self._percentile_scores(
                relative_strength_raw,
                higher_is_better=True,
            )
        )

        flow_20d_scores = (
            self._percentile_scores(
                flow_20d_raw,
                higher_is_better=True,
            )
        )

        flow_60d_scores = (
            self._percentile_scores(
                flow_60d_raw,
                higher_is_better=True,
            )
        )

        scored_rows: list[dict] = []

        for row in raw_rows:
            stock_code = (
                row[
                    "stock_code"
                ]
            )

            financial_factors = (
                financial_factor_scores.get(
                    stock_code,
                    {},
                )
            )

            quality_score = (
                financial_factors.get(
                    "quality"
                )
            )

            growth_score = (
                financial_factors.get(
                    "growth"
                )
            )

            financial_health_score = (
                financial_factors.get(
                    "financial_health"
                )
            )

            per_score = (
                per_relative_scores.get(
                    stock_code
                )
                if per_relative_raw.get(
                    stock_code
                ) is not None
                else per_absolute_scores.get(
                    stock_code
                )
            )

            pbr_score = (
                pbr_relative_scores.get(
                    stock_code
                )
                if pbr_relative_raw.get(
                    stock_code
                ) is not None
                else pbr_absolute_scores.get(
                    stock_code
                )
            )

            value_score = (
                self._weighted_average(
                    [
                        (
                            per_score,
                            0.50,
                        ),
                        (
                            pbr_score,
                            0.50,
                        ),
                    ]
                )
            )

            flow_score = (
                self._weighted_average(
                    [
                        (
                            flow_20d_scores.get(
                                stock_code
                            ),
                            0.40,
                        ),
                        (
                            flow_60d_scores.get(
                                stock_code
                            ),
                            0.60,
                        ),
                    ]
                )
            )

            relative_strength_score = (
                relative_strength_scores.get(
                    stock_code
                )
            )

            factor_scores = {
                "quality":
                    quality_score,

                "growth":
                    growth_score,

                "value":
                    value_score,

                "financial_health":
                    financial_health_score,

                "relative_strength":
                    relative_strength_score,

                "flow":
                    flow_score,

                "ml":
                    row[
                        "ml_score"
                    ],
            }

            financial_coverage = (
                financial_factor_coverage.get(
                    stock_code,
                    {},
                )
            )

            value_coverage = (
                (
                    0.50
                    if per_absolute_raw.get(
                        stock_code
                    )
                    is not None
                    else 0.0
                )
                +
                (
                    0.50
                    if pbr_absolute_raw.get(
                        stock_code
                    )
                    is not None
                    else 0.0
                )
            )

            flow_coverage = (
                (
                    0.40
                    if flow_20d_raw.get(
                        stock_code
                    )
                    is not None
                    else 0.0
                )
                +
                (
                    0.60
                    if flow_60d_raw.get(
                        stock_code
                    )
                    is not None
                    else 0.0
                )
            )

            factor_coverage = {
                "quality":
                    float(
                        financial_coverage.get(
                            "quality",
                            0.0,
                        )
                    ),

                "growth":
                    float(
                        financial_coverage.get(
                            "growth",
                            0.0,
                        )
                    ),

                "value":
                    value_coverage,

                "financial_health":
                    float(
                        financial_coverage.get(
                            "financial_health",
                            0.0,
                        )
                    ),

                "relative_strength":
                    (
                        1.0
                        if relative_strength_score
                        is not None
                        else 0.0
                    ),

                "flow":
                    flow_coverage,

                "ml":
                    (
                        1.0
                        if row[
                            "ml_score"
                        ]
                        is not None
                        else 0.0
                    ),
            }

            available_weight = sum(
                self.WEIGHTS[
                    factor_name
                ]
                * factor_coverage[
                    factor_name
                ]
                for factor_name
                in self.WEIGHTS
            )

            if (
                available_weight
                < self.MIN_DATA_COVERAGE
            ):
                continue

            weighted_score = sum(
                float(
                    score
                )
                * self.WEIGHTS[
                    factor_name
                ]
                * factor_coverage[
                    factor_name
                ]
                for (
                    factor_name,
                    score,
                ) in factor_scores.items()
                if (
                    score is not None
                    and factor_coverage[
                        factor_name
                    ]
                    > 0.0
                )
            )

            total_score = (
                weighted_score
                / available_weight
            )

            scored_rows.append(
                {
                    **row,

                    "quality_score":
                        (
                            round(
                                quality_score,
                                4,
                            )
                            if quality_score
                            is not None
                            else None
                        ),

                    "growth_score":
                        (
                            round(
                                growth_score,
                                4,
                            )
                            if growth_score
                            is not None
                            else None
                        ),

                    "value_score":
                        (
                            round(
                                value_score,
                                4,
                            )
                            if value_score
                            is not None
                            else None
                        ),

                    "financial_health_score":
                        (
                            round(
                                financial_health_score,
                                4,
                            )
                            if financial_health_score
                            is not None
                            else None
                        ),

                    "relative_strength_score":
                        (
                            round(
                                relative_strength_scores[
                                    stock_code
                                ],
                                4,
                            )
                            if relative_strength_scores.get(
                                stock_code
                            )
                            is not None
                            else None
                        ),

                    "flow_score":
                        (
                            round(
                                flow_score,
                                4,
                            )
                            if flow_score
                            is not None
                            else None
                        ),

                    "factor_coverage": {
                        factor_name:
                            round(
                                coverage
                                * 100.0,
                                2,
                            )
                        for (
                            factor_name,
                            coverage,
                        ) in (
                            factor_coverage.items()
                        )
                    },

                    "data_coverage":
                        round(
                            available_weight
                            * 100.0,
                            2,
                        ),

                    "total_score":
                        round(
                            total_score,
                            4,
                        ),
                }
            )

        score_debug_samples = [
            {
                "stock": row.get(
                    "stock_code"
                ),
                "quality": row.get(
                    "quality_score"
                ),
                "value": row.get(
                    "value_score"
                ),
                "coverage": row.get(
                    "data_coverage"
                ),
            }
            for row in scored_rows
            if (
                row.get(
                    "quality_score"
                ) is None
                or row.get(
                    "value_score"
                ) is None
            )
        ][
            :10
        ]

        print(
            "[RANKING][SCORE] "
            f"rawRows={len(raw_rows)} "
            f"eligible={len(scored_rows)} "
            f"quality={sum(1 for row in scored_rows if row.get('quality_score') is not None)} "
            f"value={sum(1 for row in scored_rows if row.get('value_score') is not None)} "
            f"missingSamples={score_debug_samples}",
            flush=True,
        )

        scored_rows.sort(
            key=lambda row: (
                row["total_score"],
                row["data_coverage"],
                row["market_cap"],
                row["stock_code"],
            ),
            reverse=True,
        )

        sector_rows: dict[
            str,
            list[dict],
        ] = {}

        for row in scored_rows:
            sector_name = str(
                row.get(
                    "sector_name"
                )
                or ""
            ).strip()

            if not sector_name:
                continue

            sector_rows.setdefault(
                sector_name,
                [],
            ).append(
                row
            )

        for rows in sector_rows.values():
            peer_count = len(
                rows
            )

            for sector_rank, row in enumerate(
                rows,
                start=1,
            ):
                row[
                    "sector_rank"
                ] = sector_rank

                row[
                    "sector_peer_count"
                ] = peer_count

        for row in scored_rows:
            row.setdefault(
                "sector_rank",
                None,
            )

            row.setdefault(
                "sector_peer_count",
                0,
            )

        selected_rows = (
            scored_rows[
                :limit
            ]
        )

        for index, row in enumerate(
            selected_rows,
            start=1,
        ):
            row[
                "rank"
            ] = index

        return {
            "status":
                "pass",

            "strategy":
                self.RANKING_VERSION,

            "weights":
                dict(
                    self.WEIGHTS
                ),

            "candidate_count":
                len(
                    stock_codes
                ),

            "eligible_count":
                len(
                    scored_rows
                ),

            "universe_count":
                len(
                    selected_rows
                ),

            "minimum_data_coverage":
                round(
                    self.MIN_DATA_COVERAGE
                    * 100.0,
                    2,
                ),

            "coverage": {
                "financial":
                    sum(
                        row[
                            "financial_score"
                        ]
                        > 0.0
                        for row
                        in raw_rows
                    ),

                "per_relative":
                    sum(
                        value
                        is not None
                        for value
                        in per_relative_raw.values()
                    ),

                "pbr_relative":
                    sum(
                        value
                        is not None
                        for value
                        in pbr_relative_raw.values()
                    ),

                "relative_strength":
                    sum(
                        value
                        is not None
                        for value
                        in relative_strength_raw.values()
                    ),

                "flow_20d":
                    sum(
                        value
                        is not None
                        for value
                        in flow_20d_raw.values()
                    ),

                "flow_60d":
                    sum(
                        value
                        is not None
                        for value
                        in flow_60d_raw.values()
                    ),

                "ml":
                    sum(
                        row[
                            "ml_score"
                        ]
                        is not None
                        for row
                        in raw_rows
                    ),
            },

            "scores":
                selected_rows,
        }