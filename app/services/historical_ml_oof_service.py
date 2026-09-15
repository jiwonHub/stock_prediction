from __future__ import annotations

from bisect import bisect_right
from collections import defaultdict
from datetime import timedelta

import json
import math
import numpy as np

from scipy.stats import rankdata, spearmanr
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models.financial_metric import FinancialMetric
from app.models.financial_statement import FinancialStatement
from app.models.future import Disclosure
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from app.repositories.market_data_repository import (
    MarketDataRepository,
)
from app.repositories.stock_repository import (
    StockRepository,
)
from app.services.historical_dataset_service import (
    HistoricalDatasetService,
)
from app.services.historical_ml_calibration_service import (
    HistoricalMlCalibrationService,
)
from app.services.historical_ml_training_service import (
    HistoricalMlTrainingService,
)
from app.services.historical_ml_walk_forward_service import (
    HistoricalMlWalkForwardService,
)


class HistoricalMlOofService:
    FINAL_PARAMS = {
        "n_estimators": 600,
        "max_depth": 3,
        "learning_rate": 0.03,
        "min_child_weight": 15,
        "subsample": 0.70,
        "colsample_bytree": 0.70,
        "reg_alpha": 0.20,
        "reg_lambda": 2.50,
    }

    RANDOM_STATE = 42

    def __init__(
        self,
        db: Session,
    ):
        self.db = db

        self.dataset_service = (
            HistoricalDatasetService(
                db
            )
        )

        self.training_service = (
            HistoricalMlTrainingService(
                db
            )
        )

        self.calibration_service = (
            HistoricalMlCalibrationService(
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

    @classmethod
    def _make_classifier(
        cls,
    ):
        try:
            from xgboost import (
                XGBClassifier,
            )

        except ImportError as e:
            raise RuntimeError(
                "xgboost가 설치되어 있지 않습니다."
            ) from e

        return XGBClassifier(
            objective="binary:logistic",
            eval_metric="logloss",
            tree_method="hist",
            random_state=(
                cls.RANDOM_STATE
            ),
            n_jobs=4,
            **cls.FINAL_PARAMS,
        )

    @staticmethod
    def _threshold_metrics(
        *,
        y_true: np.ndarray,
        probability: np.ndarray,
        threshold: float,
    ) -> dict:
        prediction = (
            probability
            >= threshold
        ).astype(
            np.int32
        )

        return {
            "threshold":
                float(
                    threshold
                ),

            "accuracy_pct":
                float(
                    accuracy_score(
                        y_true,
                        prediction,
                    )
                    * 100.0
                ),

            "balanced_accuracy_pct":
                float(
                    balanced_accuracy_score(
                        y_true,
                        prediction,
                    )
                    * 100.0
                ),

            "precision_pct":
                float(
                    precision_score(
                        y_true,
                        prediction,
                        zero_division=0,
                    )
                    * 100.0
                ),

            "recall_pct":
                float(
                    recall_score(
                        y_true,
                        prediction,
                        zero_division=0,
                    )
                    * 100.0
                ),

            "predicted_positive_rate_pct":
                float(
                    np.mean(
                        prediction
                    )
                    * 100.0
                ),
        }

    @staticmethod
    def _bootstrap_mean_ci(
        values: np.ndarray,
        *,
        iterations: int = 10000,
        confidence: float = 0.95,
        random_state: int = 42,
    ) -> dict:
        values = np.asarray(
            values,
            dtype=np.float64,
        )

        if len(values) < 2:
            raise ValueError(
                "Bootstrap에 필요한 표본이 부족합니다."
            )

        rng = np.random.default_rng(
            random_state
        )

        sampled_indexes = rng.integers(
            0,
            len(values),
            size=(
                iterations,
                len(values),
            ),
        )

        sampled_means = np.mean(
            values[
                sampled_indexes
            ],
            axis=1,
        )

        alpha = (
            1.0
            - confidence
        ) / 2.0

        lower = float(
            np.quantile(
                sampled_means,
                alpha,
            )
        )

        upper = float(
            np.quantile(
                sampled_means,
                1.0 - alpha,
            )
        )

        return {
            "samples":
                int(
                    len(values)
                ),

            "mean_pct":
                float(
                    np.mean(
                        values
                    )
                    * 100.0
                ),

            "median_pct":
                float(
                    np.median(
                        values
                    )
                    * 100.0
                ),

            "ci_lower_pct":
                lower
                * 100.0,

            "ci_upper_pct":
                upper
                * 100.0,

            "bootstrap_probability_mean_above_zero_pct":
                float(
                    np.mean(
                        sampled_means > 0.0
                    )
                    * 100.0
                ),
        }

    @staticmethod
    def _max_drawdown_pct(
        returns: list[float],
    ) -> float:
        if not returns:
            return 0.0

        equity = np.concatenate([
            np.asarray(
                [1.0],
                dtype=np.float64,
            ),
            np.cumprod(
                1.0
                + np.asarray(
                    returns,
                    dtype=np.float64,
                )
            ),
        ])

        peak = np.maximum.accumulate(
            equity
        )

        drawdown = (
            equity / peak
        ) - 1.0

        return float(
            np.min(drawdown)
            * 100.0
        )

    @staticmethod
    def _annualized_return_pct(
        returns: list[float],
        *,
        rebalance_step: int,
    ) -> float:
        if not returns:
            return 0.0

        final_equity = float(
            np.prod(
                1.0
                + np.asarray(
                    returns,
                    dtype=np.float64,
                )
            )
        )

        if final_equity <= 0.0:
            return -100.0

        periods_per_year = (
            252.0
            / rebalance_step
        )

        annualized = (
            final_equity
            ** (
                periods_per_year
                / len(returns)
            )
        ) - 1.0

        return float(
            annualized
            * 100.0
        )

    @staticmethod
    def _annualized_sharpe(
        returns: list[float],
        *,
        rebalance_step: int,
    ) -> float | None:
        if len(returns) < 2:
            return None

        values = np.asarray(
            returns,
            dtype=np.float64,
        )

        std = float(
            np.std(
                values,
                ddof=1,
            )
        )

        if std <= 0.0:
            return None

        return float(
            np.mean(values)
            / std
            * math.sqrt(
                252.0
                / rebalance_step
            )
        )
    
    @staticmethod
    def _percentile_array(
        values: np.ndarray,
    ) -> np.ndarray:
        values = np.asarray(
            values,
            dtype=np.float64,
        )

        if len(values) <= 1:
            return np.full(
                len(values),
                50.0,
                dtype=np.float64,
            )

        ranks = rankdata(
            values,
            method="average",
        )

        return (
            (ranks - 1.0)
            / (len(values) - 1.0)
            * 100.0
        )

    def _build_historical_flow_scores(
        self,
        *,
        stock_codes: list[str],
        feature_dates: list,
    ) -> tuple[np.ndarray, dict]:
        if (
            len(stock_codes)
            != len(feature_dates)
        ):
            raise ValueError(
                "Flow 평가용 stock/date 행 수가 "
                "일치하지 않습니다."
            )

        flow_5d = np.full(
            len(stock_codes),
            np.nan,
            dtype=np.float64,
        )

        flow_20d = np.full(
            len(stock_codes),
            np.nan,
            dtype=np.float64,
        )

        indexes_by_stock = {}

        for index, stock_code in enumerate(
            stock_codes
        ):
            indexes_by_stock.setdefault(
                stock_code,
                [],
            ).append(
                index
            )

        for stock_code, indexes in (
            indexes_by_stock.items()
        ):
            stock_dates = [
                feature_dates[index]
                for index
                in indexes
            ]

            load_start_date = (
                min(stock_dates)
                - timedelta(days=60)
            )

            load_end_date = max(
                stock_dates
            )

            flow_rows = (
                self.market_repository
                .get_stock_investor_flows(
                    stock_code=stock_code,
                    start_date=load_start_date,
                    end_date=load_end_date,
                )
            )

            price_rows = (
                self.stock_repository
                .get_daily_prices(
                    stock_code=stock_code,
                    start_date=load_start_date,
                )
            )

            volume_by_date = {
                row.trade_date:
                    float(
                        row.volume
                        or 0.0
                    )
                for row
                in price_rows
                if (
                    row.trade_date
                    <= load_end_date
                )
            }

            flow_dates = []
            net_values = []
            volume_values = []

            for row in flow_rows:
                price_volume = (
                    volume_by_date.get(
                        row.trade_date
                    )
                )

                if (
                    price_volume is None
                    or price_volume <= 0.0
                ):
                    continue

                flow_dates.append(
                    row.trade_date
                )

                net_values.append(
                    float(
                        row.foreign_net_buy_volume
                        or 0.0
                    )
                    + float(
                        row.institution_net_buy_volume
                        or 0.0
                    )
                )

                volume_values.append(
                    price_volume
                )

            prefix_net = np.concatenate([
                np.asarray(
                    [0.0],
                    dtype=np.float64,
                ),
                np.cumsum(
                    np.asarray(
                        net_values,
                        dtype=np.float64,
                    )
                ),
            ])

            prefix_volume = np.concatenate([
                np.asarray(
                    [0.0],
                    dtype=np.float64,
                ),
                np.cumsum(
                    np.asarray(
                        volume_values,
                        dtype=np.float64,
                    )
                ),
            ])

            for index in indexes:
                end = bisect_right(
                    flow_dates,
                    feature_dates[index],
                )

                for periods, target in (
                    (5, flow_5d),
                    (20, flow_20d),
                ):
                    if end < periods:
                        continue

                    start = (
                        end
                        - periods
                    )

                    denominator = float(
                        prefix_volume[end]
                        - prefix_volume[start]
                    )

                    if denominator <= 0.0:
                        continue

                    target[index] = (
                        float(
                            prefix_net[end]
                            - prefix_net[start]
                        )
                        / denominator
                    )

        flow_scores = np.full(
            len(stock_codes),
            50.0,
            dtype=np.float64,
        )

        indexes_by_date = {}

        for index, feature_date in enumerate(
            feature_dates
        ):
            indexes_by_date.setdefault(
                feature_date,
                [],
            ).append(
                index
            )

        for indexes in indexes_by_date.values():
            index_array = np.asarray(
                indexes,
                dtype=np.int64,
            )

            score_5d = np.full(
                len(index_array),
                50.0,
                dtype=np.float64,
            )

            score_20d = np.full(
                len(index_array),
                50.0,
                dtype=np.float64,
            )

            daily_5d = flow_5d[
                index_array
            ]

            daily_20d = flow_20d[
                index_array
            ]

            mask_5d = np.isfinite(
                daily_5d
            )

            mask_20d = np.isfinite(
                daily_20d
            )

            if np.any(mask_5d):
                score_5d[mask_5d] = (
                    self._percentile_array(
                        daily_5d[
                            mask_5d
                        ]
                    )
                )

            if np.any(mask_20d):
                score_20d[mask_20d] = (
                    self._percentile_array(
                        daily_20d[
                            mask_20d
                        ]
                    )
                )

            flow_scores[index_array] = (
                score_5d
                * 0.40
                + score_20d
                * 0.60
            )

        return (
            flow_scores,
            {
                "rows":
                    len(stock_codes),

                "flow_5d_rows":
                    int(
                        np.sum(
                            np.isfinite(
                                flow_5d
                            )
                        )
                    ),

                "flow_20d_rows":
                    int(
                        np.sum(
                            np.isfinite(
                                flow_20d
                            )
                        )
                    ),
            },
        )

    @classmethod
    def _flow_weight_sweep(
        cls,
        *,
        probabilities: np.ndarray,
        flow_scores: np.ndarray,
        future_returns: np.ndarray,
        stock_codes: list[str],
        feature_dates: list,
        rebalance_step: int,
    ) -> list[dict]:
        ml_scores = np.zeros(
            len(probabilities),
            dtype=np.float64,
        )

        indexes_by_date = {}

        for index, feature_date in enumerate(
            feature_dates
        ):
            indexes_by_date.setdefault(
                feature_date,
                [],
            ).append(
                index
            )

        for indexes in indexes_by_date.values():
            index_array = np.asarray(
                indexes,
                dtype=np.int64,
            )

            ml_scores[index_array] = (
                cls._percentile_array(
                    probabilities[
                        index_array
                    ]
                )
            )

        results = []

        for flow_weight in (
            0.05,
            0.10,
            0.15,
            0.20,
            0.30,
        ):
            composite_scores = (
                ml_scores
                * (1.0 - flow_weight)
                + flow_scores
                * flow_weight
            )

            metrics = cls._ranking_metrics(
                probabilities=composite_scores,
                future_returns=future_returns,
                stock_codes=stock_codes,
                feature_dates=feature_dates,
                rebalance_step=rebalance_step,
            )

            target = next(
                scenario
                for scenario
                in metrics[
                    "turnover_buffer_backtest"
                ][
                    "scenarios"
                ]
                if (
                    scenario[
                        "portfolio_size"
                    ]
                    == 10
                    and scenario[
                        "exit_rank"
                    ]
                    == 20
                    and scenario[
                        "transaction_cost_bps"
                    ]
                    == 20.0
                )
            )

            results.append(
                {
                    "flow_weight":
                        flow_weight,

                    "ml_weight":
                        1.0
                        - flow_weight,

                    "spearman_ic_mean":
                        metrics[
                            "spearman_ic_mean"
                        ],

                    "top10_excess_mean_pct":
                        metrics[
                            "top10_excess_mean_pct"
                        ],

                    "buffer_top10_exit20_cost20":
                        target[
                            "summary"
                        ],
                }
            )

        return results

    def _build_historical_financial_scores(
        self,
        *,
        stock_codes: list[str],
        feature_dates: list,
    ) -> tuple[np.ndarray, dict]:
        if len(stock_codes) != len(feature_dates):
            raise ValueError(
                "Financial 평가용 stock/date 행 수가 "
                "일치하지 않습니다."
            )

        unique_stock_codes = sorted(
            set(stock_codes)
        )

        target_years = (
            "2022",
            "2023",
            "2024",
            "2025",
        )

        metrics = self.db.scalars(
            select(FinancialMetric)
            .where(
                FinancialMetric.stock_code.in_(
                    unique_stock_codes
                ),
                FinancialMetric.report_code
                == "11011",
                FinancialMetric.business_year.in_(
                    target_years
                ),
            )
        ).all()

        statements = self.db.scalars(
            select(FinancialStatement)
            .where(
                FinancialStatement.stock_code.in_(
                    unique_stock_codes
                ),
                FinancialStatement.report_code
                == "11011",
                FinancialStatement.business_year.in_(
                    target_years
                ),
            )
        ).all()

        receipts_by_key = defaultdict(
            set
        )

        for row in statements:
            if not row.raw_json:
                continue

            payload = json.loads(
                row.raw_json
            )

            receipt_no = payload.get(
                "rcept_no"
            )

            if not receipt_no:
                continue

            receipts_by_key[
                (
                    row.stock_code,
                    row.business_year,
                    row.fs_div,
                )
            ].add(
                str(receipt_no)
            )

        receipt_nos = {
            receipt_no
            for values
            in receipts_by_key.values()
            for receipt_no
            in values
        }

        disclosures = self.db.scalars(
            select(Disclosure)
            .where(
                Disclosure.receipt_no.in_(
                    receipt_nos
                )
            )
        ).all()

        disclosure_date_by_receipt = {
            disclosure.receipt_no:
                disclosure.receipt_date
            for disclosure
            in disclosures
            if disclosure.receipt_date
            is not None
        }

        versions_by_stock = defaultdict(
            list
        )

        invalid_metric_versions = 0

        for metric in metrics:
            key = (
                metric.stock_code,
                metric.business_year,
                metric.fs_div,
            )

            metric_receipts = (
                receipts_by_key.get(
                    key,
                    set(),
                )
            )

            if len(metric_receipts) != 1:
                invalid_metric_versions += 1
                continue

            receipt_no = next(
                iter(
                    metric_receipts
                )
            )

            available_date = (
                disclosure_date_by_receipt.get(
                    receipt_no
                )
            )

            if available_date is None:
                invalid_metric_versions += 1
                continue

            versions_by_stock[
                metric.stock_code
            ].append(
                (
                    int(
                        metric.business_year
                    ),
                    available_date,
                    float(
                        metric.financial_score
                    ),
                )
            )

        for stock_code in versions_by_stock:
            versions_by_stock[
                stock_code
            ].sort(
                key=lambda row: (
                    row[0],
                    row[1],
                )
            )

        financial_scores = np.full(
            len(stock_codes),
            50.0,
            dtype=np.float64,
        )

        covered_rows = 0

        selected_year_counts = defaultdict(
            int
        )

        for index, (
            stock_code,
            feature_date,
        ) in enumerate(
            zip(
                stock_codes,
                feature_dates,
            )
        ):
            available = [
                version
                for version
                in versions_by_stock.get(
                    stock_code,
                    []
                )
                if (
                    version[1]
                    < feature_date
                )
            ]

            if not available:
                continue

            selected = max(
                available,
                key=lambda row: (
                    row[0],
                    row[1],
                ),
            )

            financial_scores[
                index
            ] = selected[2]

            covered_rows += 1

            selected_year_counts[
                selected[0]
            ] += 1

        total_rows = len(
            stock_codes
        )

        return (
            financial_scores,
            {
                "rows":
                    total_rows,

                "covered_rows":
                    covered_rows,

                "neutral_rows":
                    (
                        total_rows
                        - covered_rows
                    ),

                "coverage_pct":
                    float(
                        covered_rows
                        / total_rows
                        * 100.0
                    )
                    if total_rows
                    else 0.0,

                "invalid_metric_versions":
                    invalid_metric_versions,

                "selected_business_years": {
                    str(year):
                        count
                    for year, count
                    in sorted(
                        selected_year_counts.items()
                    )
                },
            },
        )

    @classmethod
    def _financial_weight_sweep(
        cls,
        *,
        probabilities: np.ndarray,
        flow_scores: np.ndarray,
        financial_scores: np.ndarray,
        future_returns: np.ndarray,
        stock_codes: list[str],
        feature_dates: list,
        rebalance_step: int,
    ) -> list[dict]:
        ml_scores = np.zeros(
            len(probabilities),
            dtype=np.float64,
        )

        indexes_by_date = {}

        for index, feature_date in enumerate(
            feature_dates
        ):
            indexes_by_date.setdefault(
                feature_date,
                [],
            ).append(
                index
            )

        for indexes in indexes_by_date.values():
            index_array = np.asarray(
                indexes,
                dtype=np.int64,
            )

            ml_scores[
                index_array
            ] = (
                cls._percentile_array(
                    probabilities[
                        index_array
                    ]
                )
            )

        flow_weight = 0.20

        results = []

        for financial_weight in (
            0.10,
            0.20,
            0.30,
            0.40,
            0.45,
        ):
            ml_weight = (
                1.0
                - flow_weight
                - financial_weight
            )

            composite_scores = (
                ml_scores
                * ml_weight
                + flow_scores
                * flow_weight
                + financial_scores
                * financial_weight
            )

            metrics = cls._ranking_metrics(
                probabilities=composite_scores,
                future_returns=future_returns,
                stock_codes=stock_codes,
                feature_dates=feature_dates,
                rebalance_step=rebalance_step,
            )

            target = next(
                scenario
                for scenario
                in metrics[
                    "turnover_buffer_backtest"
                ][
                    "scenarios"
                ]
                if (
                    scenario[
                        "portfolio_size"
                    ]
                    == 10
                    and scenario[
                        "exit_rank"
                    ]
                    == 20
                    and scenario[
                        "transaction_cost_bps"
                    ]
                    == 20.0
                )
            )

            results.append(
                {
                    "financial_weight":
                        financial_weight,

                    "flow_weight":
                        flow_weight,

                    "ml_weight":
                        ml_weight,

                    "spearman_ic_mean":
                        metrics[
                            "spearman_ic_mean"
                        ],

                    "top10_excess_mean_pct":
                        metrics[
                            "top10_excess_mean_pct"
                        ],

                    "buffer_top10_exit20_cost20":
                        target[
                            "summary"
                        ],
                }
            )

        return results

    @classmethod
    def _ml_weight_sweep(
        cls,
        *,
        probabilities: np.ndarray,
        flow_scores: np.ndarray,
        financial_scores: np.ndarray,
        future_returns: np.ndarray,
        stock_codes: list[str],
        feature_dates: list,
        rebalance_step: int,
    ) -> dict:
        ml_scores = np.zeros(
            len(probabilities),
            dtype=np.float64,
        )

        financial_rank_scores = np.full(
            len(probabilities),
            50.0,
            dtype=np.float64,
        )

        indexes_by_date = {}

        for index, feature_date in enumerate(
            feature_dates
        ):
            indexes_by_date.setdefault(
                feature_date,
                [],
            ).append(
                index
            )

        for indexes in indexes_by_date.values():
            index_array = np.asarray(
                indexes,
                dtype=np.int64,
            )

            ml_scores[
                index_array
            ] = (
                cls._percentile_array(
                    probabilities[
                        index_array
                    ]
                )
            )

            financial_rank_scores[
                index_array
            ] = (
                cls._percentile_array(
                    financial_scores[
                        index_array
                    ]
                )
            )

        from app.services.composite_ranking_service import (
            CompositeRankingService,
        )

        production_weights = (
            CompositeRankingService
            .WEIGHTS
        )

        financial_source_weight = (
            production_weights[
                "quality"
            ]
            + production_weights[
                "growth"
            ]
            + production_weights[
                "financial_health"
            ]
        )

        flow_source_weight = (
            production_weights[
                "flow"
            ]
        )

        proxy_source_weight = (
            financial_source_weight
            + flow_source_weight
        )

        if proxy_source_weight <= 0.0:
            raise ValueError(
                "Historical Composite Proxy "
                "가중치가 올바르지 않습니다."
            )

        proxy_financial_weight = (
            financial_source_weight
            / proxy_source_weight
        )

        proxy_flow_weight = (
            flow_source_weight
            / proxy_source_weight
        )

        baseline_scores = (
            financial_rank_scores
            * proxy_financial_weight
            + flow_scores
            * proxy_flow_weight
        )

        results = []

        for ml_weight in (
            0.0,
            0.025,
            0.05,
            0.075,
            0.10,
        ):
            composite_scores = (
                baseline_scores
                * (
                    1.0
                    - ml_weight
                )
                + ml_scores
                * ml_weight
            )

            metrics = cls._ranking_metrics(
                probabilities=(
                    composite_scores
                ),
                future_returns=(
                    future_returns
                ),
                stock_codes=(
                    stock_codes
                ),
                feature_dates=(
                    feature_dates
                ),
                rebalance_step=(
                    rebalance_step
                ),
            )

            top10_cost20 = next(
                scenario
                for scenario
                in metrics[
                    "portfolio_backtest"
                ][
                    "scenarios"
                ]
                if (
                    scenario[
                        "portfolio_size"
                    ]
                    == 10
                    and scenario[
                        "transaction_cost_bps"
                    ]
                    == 20.0
                )
            )

            buffer_top10 = next(
                scenario
                for scenario
                in metrics[
                    "turnover_buffer_backtest"
                ][
                    "scenarios"
                ]
                if (
                    scenario[
                        "portfolio_size"
                    ]
                    == 10
                    and scenario[
                        "exit_rank"
                    ]
                    == 20
                    and scenario[
                        "transaction_cost_bps"
                    ]
                    == 20.0
                )
            )

            results.append(
                {
                    "ml_weight":
                        ml_weight,

                    "ml_weight_pct":
                        ml_weight
                        * 100.0,

                    "spearman_ic_mean":
                        metrics[
                            "spearman_ic_mean"
                        ],

                    "spearman_ic_positive_rate_pct":
                        metrics[
                            "spearman_ic_positive_rate_pct"
                        ],

                    "top10_excess_mean_pct":
                        metrics[
                            "top10_excess_mean_pct"
                        ],

                    "top10_excess_positive_rate_pct":
                        metrics[
                            "top10_excess_positive_rate_pct"
                        ],

                    "top20_excess_mean_pct":
                        metrics[
                            "top20_excess_mean_pct"
                        ],

                    "long_short_10_mean_pct":
                        metrics[
                            "long_short_10_mean_pct"
                        ],

                    "top10_cost20":
                        top10_cost20[
                            "summary"
                        ],

                    "buffer_top10_exit20_cost20":
                        buffer_top10[
                            "summary"
                        ],
                }
            )

        return {
            "proxy_definition": {
                "is_full_production_composite":
                    False,

                "financial_source_weight":
                    financial_source_weight,

                "flow_source_weight":
                    flow_source_weight,

                "normalized_financial_weight":
                    proxy_financial_weight,

                "normalized_flow_weight":
                    proxy_flow_weight,

                "excluded_production_factors": [
                    "value",
                    "relative_strength",
                ],

                "note": (
                    "과거 시점 재현 가능한 "
                    "financial_score와 flow만으로 "
                    "구성한 Historical Proxy에 "
                    "ML을 증분 적용한 검증"
                ),
            },

            "results":
                results,
        }

    @classmethod
    def _portfolio_backtest(
        cls,
        *,
        daily_results: list[dict],
        rebalance_step: int,
    ) -> dict:
        scenarios = []

        for portfolio_size in (
            10,
            20,
        ):
            holdings_key = (
                f"top{portfolio_size}_codes"
            )

            return_key = (
                f"top{portfolio_size}_return"
            )

            for transaction_cost_bps in (
                0.0,
                10.0,
                20.0,
            ):
                offsets = []

                for offset in range(
                    rebalance_step
                ):
                    cohort = daily_results[
                        offset::rebalance_step
                    ]

                    if not cohort:
                        continue

                    previous_holdings = None

                    net_returns = []
                    benchmark_returns = []
                    turnovers = []

                    for row in cohort:
                        holdings = set(
                            row[
                                holdings_key
                            ]
                        )

                        if previous_holdings is None:
                            turnover = 1.0

                        else:
                            overlap = len(
                                previous_holdings
                                & holdings
                            )

                            turnover = (
                                1.0
                                - (
                                    overlap
                                    / portfolio_size
                                )
                            )

                        gross_return = float(
                            row[
                                return_key
                            ]
                        )

                        transaction_cost = (
                            turnover
                            * transaction_cost_bps
                            / 10000.0
                        )

                        net_return = (
                            gross_return
                            - transaction_cost
                        )

                        net_returns.append(
                            net_return
                        )

                        benchmark_returns.append(
                            float(
                                row[
                                    "universe_return"
                                ]
                            )
                        )

                        turnovers.append(
                            turnover
                        )

                        previous_holdings = (
                            holdings
                        )

                    net_excess = (
                        np.asarray(
                            net_returns,
                            dtype=np.float64,
                        )
                        - np.asarray(
                            benchmark_returns,
                            dtype=np.float64,
                        )
                    )

                    offsets.append(
                        {
                            "offset":
                                offset,

                            "periods":
                                len(cohort),

                            "first_date":
                                cohort[
                                    0
                                ][
                                    "date"
                                ].isoformat(),

                            "last_date":
                                cohort[
                                    -1
                                ][
                                    "date"
                                ].isoformat(),

                            "net_return_mean_pct":
                                float(
                                    np.mean(
                                        net_returns
                                    )
                                    * 100.0
                                ),

                            "benchmark_return_mean_pct":
                                float(
                                    np.mean(
                                        benchmark_returns
                                    )
                                    * 100.0
                                ),

                            "net_excess_mean_pct":
                                float(
                                    np.mean(
                                        net_excess
                                    )
                                    * 100.0
                                ),

                            "net_excess_positive_rate_pct":
                                float(
                                    np.mean(
                                        net_excess
                                        > 0.0
                                    )
                                    * 100.0
                                ),

                            "annualized_return_pct":
                                cls._annualized_return_pct(
                                    net_returns,
                                    rebalance_step=(
                                        rebalance_step
                                    ),
                                ),

                            "benchmark_annualized_return_pct":
                                cls._annualized_return_pct(
                                    benchmark_returns,
                                    rebalance_step=(
                                        rebalance_step
                                    ),
                                ),

                            "annualized_sharpe":
                                cls._annualized_sharpe(
                                    net_returns,
                                    rebalance_step=(
                                        rebalance_step
                                    ),
                                ),

                            "max_drawdown_pct":
                                cls._max_drawdown_pct(
                                    net_returns
                                ),

                            "benchmark_max_drawdown_pct":
                                cls._max_drawdown_pct(
                                    benchmark_returns
                                ),

                            "turnover_mean_pct":
                                float(
                                    np.mean(
                                        turnovers
                                    )
                                    * 100.0
                                ),
                        }
                    )

                excess_values = [
                    row[
                        "net_excess_mean_pct"
                    ]
                    for row
                    in offsets
                ]

                drawdown_values = [
                    row[
                        "max_drawdown_pct"
                    ]
                    for row
                    in offsets
                ]

                scenarios.append(
                    {
                        "portfolio_size":
                            portfolio_size,

                        "transaction_cost_bps":
                            transaction_cost_bps,

                        "offsets":
                            offsets,

                        "summary": {
                            "positive_excess_offsets":
                                sum(
                                    value > 0.0
                                    for value
                                    in excess_values
                                ),

                            "total_offsets":
                                len(offsets),

                            "net_excess_offset_mean_pct":
                                float(
                                    np.mean(
                                        excess_values
                                    )
                                ),

                            "net_excess_offset_min_pct":
                                float(
                                    np.min(
                                        excess_values
                                    )
                                ),

                            "net_excess_offset_max_pct":
                                float(
                                    np.max(
                                        excess_values
                                    )
                                ),

                            "annualized_return_offset_mean_pct":
                                float(
                                    np.mean(
                                        [
                                            row[
                                                "annualized_return_pct"
                                            ]
                                            for row
                                            in offsets
                                        ]
                                    )
                                ),

                            "max_drawdown_offset_worst_pct":
                                float(
                                    np.min(
                                        drawdown_values
                                    )
                                ),

                            "turnover_offset_mean_pct":
                                float(
                                    np.mean(
                                        [
                                            row[
                                                "turnover_mean_pct"
                                            ]
                                            for row
                                            in offsets
                                        ]
                                    )
                                ),
                        },
                    }
                )

        return {
            "rebalance_step":
                rebalance_step,

            "portfolio_weighting":
                "equal_weight",

            "benchmark":
                (
                    "same-date OOF universe "
                    "equal-weight mean return"
                ),

            "turnover_definition":
                (
                    "1 - holdings overlap ratio; "
                    "initial entry = 100%"
                ),

            "transaction_cost_rule":
                (
                    "net = gross - "
                    "turnover * cost_bps / 10000"
                ),

            "scenarios":
                scenarios,
        }
    
    @classmethod
    def _turnover_buffer_backtest(
        cls,
        *,
        daily_results: list[dict],
        rebalance_step: int,
    ) -> dict:
        scenarios = []

        configurations = (
            (10, 15),
            (10, 20),
            (20, 30),
            (20, 40),
        )

        for (
            portfolio_size,
            exit_rank,
        ) in configurations:
            for transaction_cost_bps in (
                10.0,
                20.0,
            ):
                offsets = []

                for offset in range(
                    rebalance_step
                ):
                    cohort = daily_results[
                        offset::rebalance_step
                    ]

                    if not cohort:
                        continue

                    previous_holdings = None

                    net_returns = []
                    benchmark_returns = []
                    turnovers = []

                    for row in cohort:
                        ranked_codes = row[
                            "ranked_codes"
                        ]

                        ranked_returns = row[
                            "ranked_returns"
                        ]

                        rank_by_code = {
                            code:
                                index + 1
                            for (
                                index,
                                code,
                            )
                            in enumerate(
                                ranked_codes
                            )
                        }

                        return_by_code = {
                            code:
                                float(
                                    future_return
                                )
                            for (
                                code,
                                future_return,
                            )
                            in zip(
                                ranked_codes,
                                ranked_returns,
                            )
                        }

                        if previous_holdings is None:
                            holdings = list(
                                ranked_codes[
                                    :portfolio_size
                                ]
                            )

                            turnover = 1.0

                        else:
                            survivors = [
                                code
                                for code
                                in previous_holdings
                                if (
                                    rank_by_code.get(
                                        code,
                                        10**9,
                                    )
                                    <= exit_rank
                                )
                            ]

                            survivors.sort(
                                key=lambda code:
                                    rank_by_code[
                                        code
                                    ]
                            )

                            holdings = list(
                                survivors
                            )

                            for code in ranked_codes:
                                if (
                                    code
                                    not in holdings
                                ):
                                    holdings.append(
                                        code
                                    )

                                if (
                                    len(
                                        holdings
                                    )
                                    >= portfolio_size
                                ):
                                    break

                            holdings = holdings[
                                :portfolio_size
                            ]

                            overlap = len(
                                set(
                                    previous_holdings
                                )
                                & set(
                                    holdings
                                )
                            )

                            turnover = (
                                1.0
                                - (
                                    overlap
                                    / portfolio_size
                                )
                            )

                        gross_return = float(
                            np.mean(
                                [
                                    return_by_code[
                                        code
                                    ]
                                    for code
                                    in holdings
                                ]
                            )
                        )

                        transaction_cost = (
                            turnover
                            * transaction_cost_bps
                            / 10000.0
                        )

                        net_return = (
                            gross_return
                            - transaction_cost
                        )

                        net_returns.append(
                            net_return
                        )

                        benchmark_returns.append(
                            float(
                                row[
                                    "universe_return"
                                ]
                            )
                        )

                        turnovers.append(
                            turnover
                        )

                        previous_holdings = list(
                            holdings
                        )

                    net_array = np.asarray(
                        net_returns,
                        dtype=np.float64,
                    )

                    benchmark_array = np.asarray(
                        benchmark_returns,
                        dtype=np.float64,
                    )

                    excess_array = (
                        net_array
                        - benchmark_array
                    )

                    cumulative_return = (
                        float(
                            np.prod(
                                1.0
                                + net_array
                            )
                        )
                        - 1.0
                    )

                    benchmark_cumulative_return = (
                        float(
                            np.prod(
                                1.0
                                + benchmark_array
                            )
                        )
                        - 1.0
                    )

                    offsets.append(
                        {
                            "offset":
                                offset,

                            "periods":
                                len(
                                    net_returns
                                ),

                            "net_excess_mean_pct":
                                float(
                                    np.mean(
                                        excess_array
                                    )
                                    * 100.0
                                ),

                            "cumulative_return_pct":
                                cumulative_return
                                * 100.0,

                            "benchmark_cumulative_return_pct":
                                (
                                    benchmark_cumulative_return
                                    * 100.0
                                ),

                            "cumulative_excess_pct":
                                (
                                    cumulative_return
                                    - benchmark_cumulative_return
                                )
                                * 100.0,

                            "annualized_return_pct":
                                cls._annualized_return_pct(
                                    net_returns,
                                    rebalance_step=(
                                        rebalance_step
                                    ),
                                ),

                            "annualized_sharpe":
                                cls._annualized_sharpe(
                                    net_returns,
                                    rebalance_step=(
                                        rebalance_step
                                    ),
                                ),

                            "max_drawdown_pct":
                                cls._max_drawdown_pct(
                                    net_returns
                                ),

                            "turnover_mean_pct":
                                float(
                                    np.mean(
                                        turnovers
                                    )
                                    * 100.0
                                ),
                        }
                    )

                excess_values = [
                    row[
                        "net_excess_mean_pct"
                    ]
                    for row
                    in offsets
                ]

                cumulative_excess_values = [
                    row[
                        "cumulative_excess_pct"
                    ]
                    for row
                    in offsets
                ]

                scenarios.append(
                    {
                        "portfolio_size":
                            portfolio_size,

                        "exit_rank":
                            exit_rank,

                        "transaction_cost_bps":
                            transaction_cost_bps,

                        "offsets":
                            offsets,

                        "summary": {
                            "positive_excess_offsets":
                                sum(
                                    value > 0.0
                                    for value
                                    in excess_values
                                ),

                            "total_offsets":
                                len(
                                    offsets
                                ),

                            "net_excess_offset_mean_pct":
                                float(
                                    np.mean(
                                        excess_values
                                    )
                                ),

                            "net_excess_offset_min_pct":
                                float(
                                    np.min(
                                        excess_values
                                    )
                                ),

                            "net_excess_offset_max_pct":
                                float(
                                    np.max(
                                        excess_values
                                    )
                                ),

                            "cumulative_excess_offset_mean_pct":
                                float(
                                    np.mean(
                                        cumulative_excess_values
                                    )
                                ),

                            "annualized_return_offset_mean_pct":
                                float(
                                    np.mean(
                                        [
                                            row[
                                                "annualized_return_pct"
                                            ]
                                            for row
                                            in offsets
                                        ]
                                    )
                                ),

                            "max_drawdown_offset_worst_pct":
                                float(
                                    np.min(
                                        [
                                            row[
                                                "max_drawdown_pct"
                                            ]
                                            for row
                                            in offsets
                                        ]
                                    )
                                ),

                            "turnover_offset_mean_pct":
                                float(
                                    np.mean(
                                        [
                                            row[
                                                "turnover_mean_pct"
                                            ]
                                            for row
                                            in offsets
                                        ]
                                    )
                                ),
                        },
                    }
                )

        return {
            "strategy":
                "rank_buffer",

            "description":
                (
                    "신규 종목은 목표 TOP N에서 진입하고, "
                    "기존 보유 종목은 exit_rank 밖으로 "
                    "밀릴 때까지 유지"
                ),

            "transaction_cost_convention":
                (
                    "transaction_cost_bps는 "
                    "교체 turnover에 적용하는 "
                    "round-trip 비용으로 취급"
                ),

            "scenarios":
                scenarios,
        }

    @staticmethod
    def _ranking_metrics(
        *,
        probabilities: np.ndarray,
        future_returns: np.ndarray,
        stock_codes: list[str],
        feature_dates: list,
        rebalance_step: int = 5,
    ) -> dict:
        grouped_indexes = {}

        for (
            index,
            feature_date,
        ) in enumerate(
            feature_dates
        ):
            grouped_indexes.setdefault(
                feature_date,
                [],
            ).append(
                index
            )

        daily_results = []

        for feature_date in sorted(
            grouped_indexes
        ):
            indexes = np.asarray(
                grouped_indexes[
                    feature_date
                ],
                dtype=np.int64,
            )

            if len(
                indexes
            ) < 20:
                continue

            daily_probability = (
                probabilities[
                    indexes
                ]
            )

            daily_return = (
                future_returns[
                    indexes
                ]
            )

            daily_stock_codes = [
                stock_codes[index]
                for index
                in indexes
            ]

            order = np.argsort(
                daily_probability
            )[::-1]

            top10_count = min(
                10,
                len(
                    order
                ),
            )

            top20_count = min(
                20,
                len(
                    order
                ),
            )

            bottom10_count = min(
                10,
                len(
                    order
                ),
            )

            top10_indexes = order[
                :top10_count
            ]

            top20_indexes = order[
                :top20_count
            ]

            bottom10_indexes = order[
                -bottom10_count:
            ]

            universe_return = float(
                np.mean(
                    daily_return
                )
            )

            top10_return = float(
                np.mean(
                    daily_return[
                        top10_indexes
                    ]
                )
            )

            top20_return = float(
                np.mean(
                    daily_return[
                        top20_indexes
                    ]
                )
            )

            bottom10_return = float(
                np.mean(
                    daily_return[
                        bottom10_indexes
                    ]
                )
            )

            ic = None

            if (
                np.std(
                    daily_probability
                )
                > 0.0
                and np.std(
                    daily_return
                )
                > 0.0
            ):
                ic_result = spearmanr(
                    daily_probability,
                    daily_return,
                )

                if np.isfinite(
                    ic_result.statistic
                ):
                    ic = float(
                        ic_result.statistic
                    )

            daily_results.append(
                {
                    "date":
                        feature_date,

                    "stock_count":
                        len(
                            indexes
                        ),

                    "ic":
                        ic,

                    "universe_return":
                        universe_return,

                    "top10_return":
                        top10_return,

                    "top20_return":
                        top20_return,

                    "bottom10_return":
                        bottom10_return,

                    "top10_excess":
                        (
                            top10_return
                            - universe_return
                        ),

                    "top20_excess":
                        (
                            top20_return
                            - universe_return
                        ),

                    "long_short_10":
                        (
                            top10_return
                            - bottom10_return
                        ),

                    "top10_codes": [
                        daily_stock_codes[
                            index
                        ]
                        for index
                        in top10_indexes
                    ],

                    "top20_codes": [
                        daily_stock_codes[
                            index
                        ]
                        for index
                        in top20_indexes
                    ],

                    "ranked_codes": [
                        daily_stock_codes[
                            index
                        ]
                        for index
                        in order
                    ],

                    "ranked_returns": [
                        float(
                            daily_return[
                                index
                            ]
                        )
                        for index
                        in order
                    ],
                }
            )

        if not daily_results:
            raise ValueError(
                "Ranking 평가 가능한 "
                "OOF 날짜가 없습니다."
            )

        ic_values = np.asarray(
            [
                row["ic"]
                for row in daily_results
                if row["ic"] is not None
            ],
            dtype=np.float64,
        )

        top10_excess = np.asarray(
            [
                row[
                    "top10_excess"
                ]
                for row
                in daily_results
            ],
            dtype=np.float64,
        )

        top20_excess = np.asarray(
            [
                row[
                    "top20_excess"
                ]
                for row
                in daily_results
            ],
            dtype=np.float64,
        )

        long_short = np.asarray(
            [
                row[
                    "long_short_10"
                ]
                for row
                in daily_results
            ],
            dtype=np.float64,
        )

        universe_returns = np.asarray(
            [
                row[
                    "universe_return"
                ]
                for row
                in daily_results
            ],
            dtype=np.float64,
        )

        top10_returns = np.asarray(
            [
                row[
                    "top10_return"
                ]
                for row
                in daily_results
            ],
            dtype=np.float64,
        )

        top20_returns = np.asarray(
            [
                row[
                    "top20_return"
                ]
                for row
                in daily_results
            ],
            dtype=np.float64,
        )

        non_overlapping_offsets = []

        for offset in range(
            rebalance_step
        ):
            cohort = (
                daily_results[
                    offset::rebalance_step
                ]
            )

            if not cohort:
                continue

            cohort_top10_excess = (
                np.asarray(
                    [
                        row[
                            "top10_excess"
                        ]
                        for row
                        in cohort
                    ],
                    dtype=np.float64,
                )
            )

            cohort_top20_excess = (
                np.asarray(
                    [
                        row[
                            "top20_excess"
                        ]
                        for row
                        in cohort
                    ],
                    dtype=np.float64,
                )
            )

            cohort_long_short = (
                np.asarray(
                    [
                        row[
                            "long_short_10"
                        ]
                        for row
                        in cohort
                    ],
                    dtype=np.float64,
                )
            )

            cohort_ic = (
                np.asarray(
                    [
                        row[
                            "ic"
                        ]
                        for row
                        in cohort
                        if row[
                            "ic"
                        ]
                        is not None
                    ],
                    dtype=np.float64,
                )
            )

            top10_bootstrap = (
                HistoricalMlOofService
                ._bootstrap_mean_ci(
                    cohort_top10_excess,
                    iterations=10000,
                    confidence=0.95,
                    random_state=(
                        42
                        + offset
                    ),
                )
            )

            long_short_bootstrap = (
                HistoricalMlOofService
                ._bootstrap_mean_ci(
                    cohort_long_short,
                    iterations=10000,
                    confidence=0.95,
                    random_state=(
                        142
                        + offset
                    ),
                )
            )

            non_overlapping_offsets.append(
                {
                    "offset":
                        offset,

                    "periods":
                        len(
                            cohort
                        ),

                    "top10_bootstrap":
                        top10_bootstrap,

                    "long_short_bootstrap":
                        long_short_bootstrap,

                    "first_date":
                        cohort[
                            0
                        ][
                            "date"
                        ].isoformat(),

                    "last_date":
                        cohort[
                            -1
                        ][
                            "date"
                        ].isoformat(),

                    "ic_mean":
                        float(
                            np.mean(
                                cohort_ic
                            )
                        ),

                    "top10_excess_mean_pct":
                        float(
                            np.mean(
                                cohort_top10_excess
                            )
                            * 100.0
                        ),

                    "top10_excess_median_pct":
                        float(
                            np.median(
                                cohort_top10_excess
                            )
                            * 100.0
                        ),

                    "top10_excess_positive_rate_pct":
                        float(
                            np.mean(
                                cohort_top10_excess
                                > 0.0
                            )
                            * 100.0
                        ),

                    "top20_excess_mean_pct":
                        float(
                            np.mean(
                                cohort_top20_excess
                            )
                            * 100.0
                        ),

                    "long_short_10_mean_pct":
                        float(
                            np.mean(
                                cohort_long_short
                            )
                            * 100.0
                        ),

                    "long_short_10_positive_rate_pct":
                        float(
                            np.mean(
                                cohort_long_short
                                > 0.0
                            )
                            * 100.0
                        ),
                }
            )

        portfolio_backtest = (
            HistoricalMlOofService
            ._portfolio_backtest(
                daily_results=(
                    daily_results
                ),
                rebalance_step=(
                    rebalance_step
                ),
            )
        )

        turnover_buffer_backtest = (
            HistoricalMlOofService
            ._turnover_buffer_backtest(
                daily_results=(
                    daily_results
                ),
                rebalance_step=(
                    rebalance_step
                ),
            )
        )

        return {
            "ranking_dates":
                len(
                    daily_results
                ),

            "average_stocks_per_date":
                float(
                    np.mean(
                        [
                            row[
                                "stock_count"
                            ]
                            for row
                            in daily_results
                        ]
                    )
                ),

            "spearman_ic_mean":
                float(
                    np.mean(
                        ic_values
                    )
                ),

            "spearman_ic_std":
                float(
                    np.std(
                        ic_values
                    )
                ),

            "spearman_ic_positive_rate_pct":
                float(
                    np.mean(
                        ic_values > 0.0
                    )
                    * 100.0
                ),

            "universe_return_mean_pct":
                float(
                    np.mean(
                        universe_returns
                    )
                    * 100.0
                ),

            "top10_return_mean_pct":
                float(
                    np.mean(
                        top10_returns
                    )
                    * 100.0
                ),

            "top20_return_mean_pct":
                float(
                    np.mean(
                        top20_returns
                    )
                    * 100.0
                ),

            "top10_excess_mean_pct":
                float(
                    np.mean(
                        top10_excess
                    )
                    * 100.0
                ),

            "top10_excess_median_pct":
                float(
                    np.median(
                        top10_excess
                    )
                    * 100.0
                ),

            "top10_excess_positive_rate_pct":
                float(
                    np.mean(
                        top10_excess > 0.0
                    )
                    * 100.0
                ),

            "top20_excess_mean_pct":
                float(
                    np.mean(
                        top20_excess
                    )
                    * 100.0
                ),

            "top20_excess_positive_rate_pct":
                float(
                    np.mean(
                        top20_excess > 0.0
                    )
                    * 100.0
                ),

            "long_short_10_mean_pct":
                float(
                    np.mean(
                        long_short
                    )
                    * 100.0
                ),

            "long_short_10_positive_rate_pct":
                float(
                    np.mean(
                        long_short > 0.0
                    )
                    * 100.0
                ),

            "non_overlapping_rebalance_step":
                rebalance_step,

            "portfolio_backtest":
                portfolio_backtest,

            "turnover_buffer_backtest":
                turnover_buffer_backtest,

            "non_overlapping_offsets":
                non_overlapping_offsets,

            "non_overlapping_summary": {
                "top10_positive_offsets":
                    sum(
                        row[
                            "top10_excess_mean_pct"
                        ] > 0.0
                        for row
                        in non_overlapping_offsets
                    ),

                "top20_positive_offsets":
                    sum(
                        row[
                            "top20_excess_mean_pct"
                        ] > 0.0
                        for row
                        in non_overlapping_offsets
                    ),

                "long_short_positive_offsets":
                    sum(
                        row[
                            "long_short_10_mean_pct"
                        ] > 0.0
                        for row
                        in non_overlapping_offsets
                    ),

                "top10_excess_offset_mean_pct":
                    float(
                        np.mean(
                            [
                                row[
                                    "top10_excess_mean_pct"
                                ]
                                for row
                                in non_overlapping_offsets
                            ]
                        )
                    ),

                "top10_excess_offset_std_pct":
                    float(
                        np.std(
                            [
                                row[
                                    "top10_excess_mean_pct"
                                ]
                                for row
                                in non_overlapping_offsets
                            ]
                        )
                    ),

                "top10_excess_offset_min_pct":
                    float(
                        np.min(
                            [
                                row[
                                    "top10_excess_mean_pct"
                                ]
                                for row
                                in non_overlapping_offsets
                            ]
                        )
                    ),

                "top10_excess_offset_max_pct":
                    float(
                        np.max(
                            [
                                row[
                                    "top10_excess_mean_pct"
                                ]
                                for row
                                in non_overlapping_offsets
                            ]
                        )
                    ),
            },
        }
    
    def run(
        self,
        *,
        feature_version: str,
        horizon: int = 5,
        folds: int = 4,
        initial_train_ratio: float = 0.55,
    ) -> dict:
        dataset = (
            self.dataset_service
            .build_dataset(
                feature_version=(
                    feature_version
                ),
                horizon=horizon,
                feature_strategy=(
                    "stock_internal_only"
                ),
            )
        )

        temporal_split = (
            self.training_service
            .build_temporal_split(
                feature_version=(
                    feature_version
                ),
                horizon=horizon,
                feature_strategy=(
                    "stock_internal_only"
                ),
                dataset=dataset,
            )
        )

        walk_forward_last_date = max(
            temporal_split.valid_dates
        )

        locked_test_first_date = min(
            temporal_split.test_dates
        )

        selected_indexes = (
            HistoricalMlWalkForwardService
            ._selected_indexes(
                feature_names=(
                    dataset.feature_names
                ),
                feature_strategy=(
                    "stock_internal_only"
                ),
            )
        )

        selected_feature_names = [
            dataset.feature_names[
                index
            ]
            for index
            in selected_indexes
        ]

        eligible_dates = sorted(
            {
                feature_date
                for feature_date
                in dataset.feature_dates
                if feature_date
                <= walk_forward_last_date
            }
        )

        initial_train_count = int(
            len(
                eligible_dates
            )
            * initial_train_ratio
        )

        remaining_count = (
            len(
                eligible_dates
            )
            - initial_train_count
        )

        base_fold_size = (
            remaining_count
            // folds
        )

        x_all = np.asarray(
            dataset.x[
                :,
                selected_indexes
            ],
            dtype=np.float32,
        )

        y_all = np.asarray(
            dataset.y,
            dtype=np.float32,
        )

        all_probabilities = []
        all_targets = []
        all_future_returns = []
        all_stock_codes = []
        all_feature_dates = []
        all_fold_ids = []

        fold_results = []

        for fold_index in range(
            folds
        ):
            validation_start_index = (
                initial_train_count
                + (
                    fold_index
                    * base_fold_size
                )
            )

            if fold_index == (
                folds - 1
            ):
                validation_end_index = (
                    len(
                        eligible_dates
                    )
                )
            else:
                validation_end_index = (
                    validation_start_index
                    + base_fold_size
                )

            train_end_index = (
                validation_start_index
                - horizon
            )

            train_dates = set(
                eligible_dates[
                    :train_end_index
                ]
            )

            validation_dates = set(
                eligible_dates[
                    validation_start_index:
                    validation_end_index
                ]
            )

            train_indexes = (
                HistoricalMlWalkForwardService
                ._indexes_for_dates(
                    feature_dates=(
                        dataset.feature_dates
                    ),
                    allowed_dates=(
                        train_dates
                    ),
                )
            )

            validation_indexes = (
                HistoricalMlWalkForwardService
                ._indexes_for_dates(
                    feature_dates=(
                        dataset.feature_dates
                    ),
                    allowed_dates=(
                        validation_dates
                    ),
                )
            )

            x_train = x_all[
                train_indexes
            ]

            y_train = (
                y_all[
                    train_indexes
                ]
                > 0.0
            ).astype(
                np.int32
            )

            x_validation = x_all[
                validation_indexes
            ]

            y_validation = (
                y_all[
                    validation_indexes
                ]
                > 0.0
            ).astype(
                np.int32
            )

            classifier = (
                self._make_classifier()
            )

            classifier.fit(
                x_train,
                y_train,
            )

            probability = (
                classifier
                .predict_proba(
                    x_validation
                )[:, 1]
            )

            auc = float(
                roc_auc_score(
                    y_validation,
                    probability,
                )
            )

            raw_probability_metrics = (
                self.calibration_service
                ._metrics(
                    y_true=(
                        y_validation
                    ),
                    probability=(
                        probability
                    ),
                )
            )

            all_probabilities.append(
                probability
            )

            all_targets.append(
                y_validation
            )

            all_future_returns.append(
                y_all[
                    validation_indexes
                ]
            )

            all_stock_codes.extend(
                [
                    dataset.stock_codes[
                        index
                    ]
                    for index
                    in validation_indexes
                ]
            )

            all_feature_dates.extend(
                [
                    dataset.feature_dates[
                        index
                    ]
                    for index
                    in validation_indexes
                ]
            )

            all_fold_ids.append(
                np.full(
                    len(
                        y_validation
                    ),
                    fold_index + 1,
                    dtype=np.int32,
                )
            )

            fold_results.append(
                {
                    "fold":
                        fold_index + 1,

                    "rows":
                        len(
                            y_validation
                        ),

                    "first_date":
                        min(
                            validation_dates
                        ).isoformat(),

                    "last_date":
                        max(
                            validation_dates
                        ).isoformat(),

                    "roc_auc":
                        auc,

                    "log_loss":
                        raw_probability_metrics[
                            "log_loss"
                        ],

                    "brier_score":
                        raw_probability_metrics[
                            "brier_score"
                        ],

                    "ece_10":
                        raw_probability_metrics[
                            "ece_10"
                        ],

                    "probability_mean":
                        raw_probability_metrics[
                            "probability_mean"
                        ],

                    "actual_positive_rate":
                        raw_probability_metrics[
                            "actual_positive_rate"
                        ],
                }
            )

        oof_probability = np.concatenate(
            all_probabilities
        )

        oof_y = np.concatenate(
            all_targets
        )

        oof_future_return = np.concatenate(
            all_future_returns
        )

        fold_ids = np.concatenate(
            all_fold_ids
        )

        ranking_metrics = (
            self._ranking_metrics(
                probabilities=(
                    oof_probability
                ),
                future_returns=(
                    oof_future_return
                ),
                stock_codes=(
                    all_stock_codes
                ),
                feature_dates=(
                    all_feature_dates
                ),
                rebalance_step=horizon,
            )
        )

        (
            historical_flow_scores,
            historical_flow_coverage,
        ) = (
            self._build_historical_flow_scores(
                stock_codes=(
                    all_stock_codes
                ),
                feature_dates=(
                    all_feature_dates
                ),
            )
        )

        flow_weight_sweep = (
            self._flow_weight_sweep(
                probabilities=(
                    oof_probability
                ),
                flow_scores=(
                    historical_flow_scores
                ),
                future_returns=(
                    oof_future_return
                ),
                stock_codes=(
                    all_stock_codes
                ),
                feature_dates=(
                    all_feature_dates
                ),
                rebalance_step=horizon,
            )
        )

        (
            historical_financial_scores,
            historical_financial_coverage,
        ) = (
            self._build_historical_financial_scores(
                stock_codes=(
                    all_stock_codes
                ),
                feature_dates=(
                    all_feature_dates
                ),
            )
        )

        financial_weight_sweep = (
            self._financial_weight_sweep(
                probabilities=(
                    oof_probability
                ),
                flow_scores=(
                    historical_flow_scores
                ),
                financial_scores=(
                    historical_financial_scores
                ),
                future_returns=(
                    oof_future_return
                ),
                stock_codes=(
                    all_stock_codes
                ),
                feature_dates=(
                    all_feature_dates
                ),
                rebalance_step=horizon,
            )
        )

        ml_weight_sweep = (
            self._ml_weight_sweep(
                probabilities=(
                    oof_probability
                ),
                flow_scores=(
                    historical_flow_scores
                ),
                financial_scores=(
                    historical_financial_scores
                ),
                future_returns=(
                    oof_future_return
                ),
                stock_codes=(
                    all_stock_codes
                ),
                feature_dates=(
                    all_feature_dates
                ),
                rebalance_step=horizon,
            )
        )

        oof_metrics = (
            self.calibration_service
            ._metrics(
                y_true=(
                    oof_y
                ),
                probability=(
                    oof_probability
                ),
            )
        )

        threshold_candidates = (
            np.arange(
                0.40,
                0.601,
                0.01,
            )
        )

        threshold_results = []

        for threshold in (
            threshold_candidates
        ):
            pooled = (
                self._threshold_metrics(
                    y_true=(
                        oof_y
                    ),
                    probability=(
                        oof_probability
                    ),
                    threshold=float(
                        threshold
                    ),
                )
            )

            per_fold_balanced = []

            for fold_number in range(
                1,
                folds + 1,
            ):
                mask = (
                    fold_ids
                    == fold_number
                )

                fold_metrics = (
                    self._threshold_metrics(
                        y_true=(
                            oof_y[
                                mask
                            ]
                        ),
                        probability=(
                            oof_probability[
                                mask
                            ]
                        ),
                        threshold=float(
                            threshold
                        ),
                    )
                )

                per_fold_balanced.append(
                    fold_metrics[
                        "balanced_accuracy_pct"
                    ]
                )

            threshold_results.append(
                {
                    **pooled,

                    "fold_balanced_accuracy_pct":
                        [
                            float(
                                value
                            )
                            for value
                            in per_fold_balanced
                        ],

                    "fold_balanced_mean_pct":
                        float(
                            np.mean(
                                per_fold_balanced
                            )
                        ),

                    "fold_balanced_std_pct":
                        float(
                            np.std(
                                per_fold_balanced
                            )
                        ),

                    "fold_balanced_min_pct":
                        float(
                            np.min(
                                per_fold_balanced
                            )
                        ),
                }
            )

        threshold_results.sort(
            key=lambda row: (
                row[
                    "fold_balanced_min_pct"
                ],
                row[
                    "fold_balanced_mean_pct"
                ],
                -row[
                    "fold_balanced_std_pct"
                ],
            ),
            reverse=True,
        )

        best_threshold = (
            threshold_results[
                0
            ]
        )

        return {
            "status":
                "pass",

            "feature_version":
                feature_version,

            "target_horizon":
                horizon,

            "feature_strategy":
                "stock_internal_only",

            "feature_count":
                len(
                    selected_feature_names
                ),

            "model_candidate":
                "lower_sampling",

            "model_params":
                dict(
                    self.FINAL_PARAMS
                ),

            "calibration":
                "raw",

            "fold_count":
                folds,

            "test_dataset_used":
                False,

            "locked_test_first_date":
                locked_test_first_date
                .isoformat(),

            "oof_rows":
                int(
                    len(
                        oof_y
                    )
                ),

            "oof_probability_metrics":
                oof_metrics,

            "oof_ranking_metrics":
                ranking_metrics,

            "historical_flow_coverage":
                historical_flow_coverage,

            "flow_weight_sweep":
                flow_weight_sweep,

            "historical_financial_coverage":
                historical_financial_coverage,

            "financial_weight_sweep":
                financial_weight_sweep,

            "ml_weight_sweep":
                ml_weight_sweep,

            "folds":
                fold_results,

            "threshold_selection_rule":
                (
                    "OOF Fold별 Balanced Accuracy의 "
                    "최저값 우선, 그다음 평균, "
                    "낮은 표준편차 순서"
                ),

            "best_threshold":
                best_threshold,

            "threshold_results":
                threshold_results,
        }