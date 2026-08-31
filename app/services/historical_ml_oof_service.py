from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from scipy.stats import spearmanr
from sqlalchemy.orm import Session

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
        temporal_split = (
            self.training_service
            .build_temporal_split(
                feature_version=(
                    feature_version
                ),
                horizon=horizon,
            )
        )

        dataset = (
            self.dataset_service
            .build_dataset(
                feature_version=(
                    feature_version
                ),
                horizon=horizon,
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