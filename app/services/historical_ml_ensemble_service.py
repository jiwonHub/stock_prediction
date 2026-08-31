from __future__ import annotations

import numpy as np
from sqlalchemy.orm import Session

from app.services.historical_dataset_service import (
    HistoricalDatasetService,
)
from app.services.historical_ml_oof_service import (
    HistoricalMlOofService,
)
from app.services.historical_ml_training_service import (
    HistoricalMlTrainingService,
)
from app.services.historical_ml_walk_forward_service import (
    HistoricalMlWalkForwardService,
)
from app.services.historical_ml_walk_forward_tuning_service import (
    HistoricalMlWalkForwardTuningService,
)


class HistoricalMlEnsembleService:
    INTERNAL_PARAMS = {
        "n_estimators": 600,
        "max_depth": 3,
        "learning_rate": 0.03,
        "min_child_weight": 15,
        "subsample": 0.70,
        "colsample_bytree": 0.70,
        "reg_alpha": 0.20,
        "reg_lambda": 2.50,
    }

    CONTEXT_PARAMS = {
        "n_estimators": 600,
        "max_depth": 3,
        "learning_rate": 0.03,
        "min_child_weight": 20,
        "subsample": 0.85,
        "colsample_bytree": 0.75,
        "reg_alpha": 0.25,
        "reg_lambda": 3.00,
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

    @classmethod
    def _make_classifier(
        cls,
        params: dict,
    ):
        return (
            HistoricalMlWalkForwardTuningService
            ._make_classifier(
                params
            )
        )

    @staticmethod
    def _rank_normalize_by_date(
        *,
        scores: np.ndarray,
        feature_dates: list,
    ) -> np.ndarray:
        normalized = np.zeros(
            len(
                scores
            ),
            dtype=np.float64,
        )

        grouped = {}

        for (
            index,
            feature_date,
        ) in enumerate(
            feature_dates
        ):
            grouped.setdefault(
                feature_date,
                [],
            ).append(
                index
            )

        for indexes in grouped.values():
            index_array = np.asarray(
                indexes,
                dtype=np.int64,
            )

            values = scores[
                index_array
            ]

            order = np.argsort(
                np.argsort(
                    values
                )
            )

            if len(
                values
            ) <= 1:
                normalized[
                    index_array
                ] = 0.5

                continue

            normalized[
                index_array
            ] = (
                order.astype(
                    np.float64
                )
                / (
                    len(
                        values
                    )
                    - 1
                )
            )

        return normalized

    def run_walk_forward(
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
            )
        )

        split = (
            self.training_service
            .build_temporal_split(
                feature_version=(
                    feature_version
                ),
                horizon=horizon,
            )
        )

        walk_forward_last_date = max(
            split.valid_dates
        )

        locked_test_first_date = min(
            split.test_dates
        )

        internal_indexes = (
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

        context_indexes = (
            HistoricalMlWalkForwardService
            ._selected_indexes(
                feature_names=(
                    dataset.feature_names
                ),
                feature_strategy=(
                    "no_macro"
                ),
            )
        )

        eligible_indexes = [
            index
            for (
                index,
                feature_date,
            )
            in enumerate(
                dataset.feature_dates
            )
            if feature_date
            <= walk_forward_last_date
        ]

        feature_dates = [
            dataset.feature_dates[
                index
            ]
            for index
            in eligible_indexes
        ]

        stock_codes = [
            dataset.stock_codes[
                index
            ]
            for index
            in eligible_indexes
        ]

        future_returns = np.asarray(
            dataset.y[
                eligible_indexes
            ],
            dtype=np.float32,
        )

        x_internal = np.asarray(
            dataset.x[
                eligible_indexes
            ][
                :,
                internal_indexes
            ],
            dtype=np.float32,
        )

        x_context = np.asarray(
            dataset.x[
                eligible_indexes
            ][
                :,
                context_indexes
            ],
            dtype=np.float32,
        )

        unique_dates = sorted(
            set(
                feature_dates
            )
        )

        initial_train_count = int(
            len(
                unique_dates
            )
            * initial_train_ratio
        )

        remaining_count = (
            len(
                unique_dates
            )
            - initial_train_count
        )

        fold_size = (
            remaining_count
            // folds
        )

        weights = [
            0.00,
            0.20,
            0.40,
            0.60,
            0.80,
            1.00,
        ]

        weight_outputs = {
            weight: {
                "scores": [],
                "returns": [],
                "stock_codes": [],
                "feature_dates": [],
            }
            for weight
            in weights
        }

        fold_results = []

        for fold_index in range(
            folds
        ):
            validation_start = (
                initial_train_count
                + (
                    fold_index
                    * fold_size
                )
            )

            if fold_index == (
                folds - 1
            ):
                validation_end = len(
                    unique_dates
                )
            else:
                validation_end = (
                    validation_start
                    + fold_size
                )

            train_end = (
                validation_start
                - horizon
            )

            train_dates = set(
                unique_dates[
                    :train_end
                ]
            )

            validation_dates = set(
                unique_dates[
                    validation_start:
                    validation_end
                ]
            )

            train_indexes = (
                HistoricalMlWalkForwardService
                ._indexes_for_dates(
                    feature_dates=(
                        feature_dates
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
                        feature_dates
                    ),
                    allowed_dates=(
                        validation_dates
                    ),
                )
            )

            y_train = (
                future_returns[
                    train_indexes
                ]
                > 0.0
            ).astype(
                np.int32
            )

            internal_model = (
                self._make_classifier(
                    self.INTERNAL_PARAMS
                )
            )

            context_model = (
                self._make_classifier(
                    self.CONTEXT_PARAMS
                )
            )

            internal_model.fit(
                x_internal[
                    train_indexes
                ],
                y_train,
            )

            context_model.fit(
                x_context[
                    train_indexes
                ],
                y_train,
            )

            internal_probability = (
                internal_model
                .predict_proba(
                    x_internal[
                        validation_indexes
                    ]
                )[:, 1]
            )

            context_probability = (
                context_model
                .predict_proba(
                    x_context[
                        validation_indexes
                    ]
                )[:, 1]
            )

            validation_dates_list = [
                feature_dates[
                    index
                ]
                for index
                in validation_indexes
            ]

            validation_stock_codes = [
                stock_codes[
                    index
                ]
                for index
                in validation_indexes
            ]

            validation_returns = (
                future_returns[
                    validation_indexes
                ]
            )

            internal_rank = (
                self._rank_normalize_by_date(
                    scores=(
                        internal_probability
                    ),
                    feature_dates=(
                        validation_dates_list
                    ),
                )
            )

            context_rank = (
                self._rank_normalize_by_date(
                    scores=(
                        context_probability
                    ),
                    feature_dates=(
                        validation_dates_list
                    ),
                )
            )

            current_fold = {
                "fold":
                    fold_index + 1,

                "weights":
                    [],
            }

            for internal_weight in (
                weights
            ):
                context_weight = (
                    1.0
                    - internal_weight
                )

                ensemble_score = (
                    internal_rank
                    * internal_weight
                    + context_rank
                    * context_weight
                )

                ranking = (
                    HistoricalMlOofService
                    ._ranking_metrics(
                        probabilities=(
                            ensemble_score
                        ),
                        future_returns=(
                            validation_returns
                        ),
                        stock_codes=(
                            validation_stock_codes
                        ),
                        feature_dates=(
                            validation_dates_list
                        ),
                        rebalance_step=horizon,
                    )
                )

                current_fold[
                    "weights"
                ].append(
                    {
                        "internal_weight":
                            internal_weight,

                        "context_weight":
                            context_weight,

                        "ic_mean":
                            ranking[
                                "spearman_ic_mean"
                            ],

                        "top10_excess_mean_pct":
                            ranking[
                                "top10_excess_mean_pct"
                            ],
                    }
                )

                output = (
                    weight_outputs[
                        internal_weight
                    ]
                )

                output[
                    "scores"
                ].append(
                    ensemble_score
                )

                output[
                    "returns"
                ].append(
                    validation_returns
                )

                output[
                    "stock_codes"
                ].extend(
                    validation_stock_codes
                )

                output[
                    "feature_dates"
                ].extend(
                    validation_dates_list
                )

            fold_results.append(
                current_fold
            )

        results = []

        for internal_weight in (
            weights
        ):
            output = (
                weight_outputs[
                    internal_weight
                ]
            )

            scores = np.concatenate(
                output[
                    "scores"
                ]
            )

            returns = np.concatenate(
                output[
                    "returns"
                ]
            )

            ranking = (
                HistoricalMlOofService
                ._ranking_metrics(
                    probabilities=scores,
                    future_returns=returns,
                    stock_codes=(
                        output[
                            "stock_codes"
                        ]
                    ),
                    feature_dates=(
                        output[
                            "feature_dates"
                        ]
                    ),
                    rebalance_step=horizon,
                )
            )

            results.append(
                {
                    "internal_weight":
                        internal_weight,

                    "context_weight":
                        (
                            1.0
                            - internal_weight
                        ),

                    "ic_mean":
                        ranking[
                            "spearman_ic_mean"
                        ],

                    "ic_positive_rate_pct":
                        ranking[
                            "spearman_ic_positive_rate_pct"
                        ],

                    "top10_excess_mean_pct":
                        ranking[
                            "top10_excess_mean_pct"
                        ],

                    "top10_excess_positive_rate_pct":
                        ranking[
                            "top10_excess_positive_rate_pct"
                        ],

                    "top20_excess_mean_pct":
                        ranking[
                            "top20_excess_mean_pct"
                        ],

                    "long_short_10_mean_pct":
                        ranking[
                            "long_short_10_mean_pct"
                        ],

                    "positive_offsets":
                        ranking[
                            "non_overlapping_summary"
                        ][
                            "top10_positive_offsets"
                        ],

                    "offset_min_pct":
                        ranking[
                            "non_overlapping_summary"
                        ][
                            "top10_excess_offset_min_pct"
                        ],
                }
            )

        results.sort(
            key=lambda row: (
                row[
                    "positive_offsets"
                ],
                row[
                    "top10_excess_mean_pct"
                ],
                row[
                    "ic_mean"
                ],
                row[
                    "offset_min_pct"
                ],
            ),
            reverse=True,
        )

        return {
            "status":
                "pass",

            "target_horizon":
                horizon,

            "test_dataset_used":
                False,

            "locked_test_first_date":
                locked_test_first_date
                .isoformat(),

            "internal_feature_count":
                len(
                    internal_indexes
                ),

            "context_feature_count":
                len(
                    context_indexes
                ),

            "folds":
                fold_results,

            "best":
                results[
                    0
                ],

            "results":
                results,
        }