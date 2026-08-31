from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
)
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


class HistoricalMlWalkForwardService:
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

    @staticmethod
    def _indexes_for_dates(
        *,
        feature_dates: list,
        allowed_dates: set,
    ) -> list[int]:
        return [
            index
            for (
                index,
                feature_date,
            )
            in enumerate(
                feature_dates
            )
            if feature_date
            in allowed_dates
        ]

    @staticmethod
    def _mean(
        values: list[float],
    ) -> float:
        return float(
            np.mean(
                np.asarray(
                    values,
                    dtype=np.float64,
                )
            )
        )

    @staticmethod
    def _std(
        values: list[float],
    ) -> float:
        return float(
            np.std(
                np.asarray(
                    values,
                    dtype=np.float64,
                )
            )
        )

    @staticmethod
    def _selected_indexes(
        *,
        feature_names: list[str],
        feature_strategy: str,
    ) -> list[int]:
        supported = {
            "full",
            "no_market",
            "no_sector",
            "no_macro",
            "no_disclosure",
            "stock_internal_only",
        }

        if feature_strategy not in supported:
            raise ValueError(
                "지원하지 않는 Feature Strategy입니다: "
                f"{feature_strategy}"
            )

        def is_macro(
            feature_name: str,
        ) -> bool:
            return feature_name.startswith(
                (
                    "base_rate",
                    "usdkrw",
                    "ktb_",
                    "yield_curve_",
                )
            )

        indexes = []

        for (
            index,
            feature_name,
        ) in enumerate(
            feature_names
        ):
            include = True

            if feature_strategy == "no_market":
                include = not feature_name.startswith(
                    "market_"
                )

            elif feature_strategy == "no_sector":
                include = not feature_name.startswith(
                    "sector_"
                )

            elif feature_strategy == "no_macro":
                include = not is_macro(
                    feature_name
                )

            elif feature_strategy == "no_disclosure":
                include = not feature_name.startswith(
                    "disclosure_"
                )

            elif feature_strategy == "stock_internal_only":
                include = not (
                    feature_name.startswith(
                        "market_"
                    )
                    or feature_name.startswith(
                        "sector_"
                    )
                    or feature_name.startswith(
                        "disclosure_"
                    )
                    or is_macro(
                        feature_name
                    )
                )

            if include:
                indexes.append(
                    index
                )

        if not indexes:
            raise ValueError(
                "선택된 Feature가 없습니다."
            )

        return indexes

    def run(
        self,
        *,
        feature_version: str,
        horizon: int = 5,
        folds: int = 4,
        initial_train_ratio: float = 0.55,
        feature_strategy: str = "no_market",
    ) -> dict:
        if folds < 3 or folds > 8:
            raise ValueError(
                "folds는 3~8 범위여야 합니다."
            )

        if not (
            0.40
            <= initial_train_ratio
            <= 0.70
        ):
            raise ValueError(
                "initial_train_ratio는 "
                "0.40~0.70 범위여야 합니다."
            )

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
            self._selected_indexes(
                feature_names=(
                    dataset.feature_names
                ),
                feature_strategy=(
                    feature_strategy
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

        eligible_row_indexes = [
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

        if not eligible_row_indexes:
            raise ValueError(
                "Walk-forward에 사용할 "
                "Historical Row가 없습니다."
            )

        eligible_dates = sorted(
            {
                dataset.feature_dates[
                    index
                ]
                for index
                in eligible_row_indexes
            }
        )

        if len(
            eligible_dates
        ) < 150:
            raise ValueError(
                "Walk-forward에 사용할 "
                "거래일이 부족합니다."
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

        if remaining_count < (
            folds * 20
        ):
            raise ValueError(
                "Fold별 Validation 기간이 "
                "너무 짧습니다."
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

            if train_end_index <= 0:
                raise ValueError(
                    "Purge 적용 후 "
                    "Train 기간이 없습니다."
                )

            train_dates = set(
                eligible_dates[
                    :train_end_index
                ]
            )

            purge_dates = (
                eligible_dates[
                    train_end_index:
                    validation_start_index
                ]
            )

            validation_dates = set(
                eligible_dates[
                    validation_start_index:
                    validation_end_index
                ]
            )

            train_indexes = (
                self._indexes_for_dates(
                    feature_dates=(
                        dataset.feature_dates
                    ),
                    allowed_dates=(
                        train_dates
                    ),
                )
            )

            validation_indexes = (
                self._indexes_for_dates(
                    feature_dates=(
                        dataset.feature_dates
                    ),
                    allowed_dates=(
                        validation_dates
                    ),
                )
            )

            if (
                not train_indexes
                or not validation_indexes
            ):
                raise ValueError(
                    f"Fold {fold_index + 1}: "
                    "Train 또는 Validation이 "
                    "비어 있습니다."
                )

            x_train = x_all[
                train_indexes
            ]

            y_train_return = y_all[
                train_indexes
            ]

            x_validation = x_all[
                validation_indexes
            ]

            y_validation_return = y_all[
                validation_indexes
            ]

            y_train = (
                y_train_return
                > 0.0
            ).astype(
                np.int32
            )

            y_validation = (
                y_validation_return
                > 0.0
            ).astype(
                np.int32
            )

            if len(
                np.unique(
                    y_train
                )
            ) < 2:
                raise ValueError(
                    f"Fold {fold_index + 1}: "
                    "Train에 두 클래스가 "
                    "모두 존재하지 않습니다."
                )

            classifier = (
                self.calibration_service
                ._make_classifier()
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

            probability_metrics = (
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

            prediction = (
                probability
                >= 0.50
            ).astype(
                np.int32
            )

            accuracy = float(
                accuracy_score(
                    y_validation,
                    prediction,
                )
                * 100.0
            )

            balanced_accuracy = float(
                balanced_accuracy_score(
                    y_validation,
                    prediction,
                )
                * 100.0
            )

            positive_rate = float(
                np.mean(
                    y_validation
                )
                * 100.0
            )

            majority_accuracy = float(
                max(
                    np.mean(
                        y_validation
                    ),
                    1.0
                    - np.mean(
                        y_validation
                    ),
                )
                * 100.0
            )

            fold_results.append(
                {
                    "fold":
                        fold_index + 1,

                    "train_rows":
                        len(
                            train_indexes
                        ),

                    "validation_rows":
                        len(
                            validation_indexes
                        ),

                    "train_unique_dates":
                        len(
                            train_dates
                        ),

                    "validation_unique_dates":
                        len(
                            validation_dates
                        ),

                    "train_period": {
                        "first":
                            min(
                                train_dates
                            ).isoformat(),

                        "last":
                            max(
                                train_dates
                            ).isoformat(),
                    },

                    "purge_period": {
                        "dates":
                            len(
                                purge_dates
                            ),

                        "first":
                            (
                                purge_dates[
                                    0
                                ].isoformat()
                                if purge_dates
                                else None
                            ),

                        "last":
                            (
                                purge_dates[
                                    -1
                                ].isoformat()
                                if purge_dates
                                else None
                            ),
                    },

                    "validation_period": {
                        "first":
                            min(
                                validation_dates
                            ).isoformat(),

                        "last":
                            max(
                                validation_dates
                            ).isoformat(),
                    },

                    "roc_auc":
                        probability_metrics[
                            "roc_auc"
                        ],

                    "accuracy_pct":
                        accuracy,

                    "balanced_accuracy_pct":
                        balanced_accuracy,

                    "majority_accuracy_pct":
                        majority_accuracy,

                    "accuracy_edge_vs_majority_pct":
                        accuracy
                        - majority_accuracy,

                    "log_loss":
                        probability_metrics[
                            "log_loss"
                        ],

                    "brier_score":
                        probability_metrics[
                            "brier_score"
                        ],

                    "ece_10":
                        probability_metrics[
                            "ece_10"
                        ],

                    "probability_mean":
                        probability_metrics[
                            "probability_mean"
                        ],

                    "actual_positive_rate_pct":
                        positive_rate,
                }
            )

        auc_values = [
            float(
                row[
                    "roc_auc"
                ]
            )
            for row
            in fold_results
            if row[
                "roc_auc"
            ]
            is not None
        ]

        balanced_values = [
            float(
                row[
                    "balanced_accuracy_pct"
                ]
            )
            for row
            in fold_results
        ]

        log_loss_values = [
            float(
                row[
                    "log_loss"
                ]
            )
            for row
            in fold_results
        ]

        brier_values = [
            float(
                row[
                    "brier_score"
                ]
            )
            for row
            in fold_results
        ]

        ece_values = [
            float(
                row[
                    "ece_10"
                ]
            )
            for row
            in fold_results
        ]

        return {
            "status":
                "pass",

            "feature_version":
                feature_version,

            "target_horizon":
                horizon,

            "feature_strategy":
                feature_strategy,

            "feature_count":
                len(
                    selected_feature_names
                ),

            "model_candidate":
                "strong_regularization",

            "calibration":
                "raw",

            "fold_count":
                folds,

            "initial_train_ratio":
                initial_train_ratio,

            "walk_forward_first_date":
                min(
                    eligible_dates
                ).isoformat(),

            "walk_forward_last_date":
                walk_forward_last_date
                .isoformat(),

            "locked_test_first_date":
                locked_test_first_date
                .isoformat(),

            "test_dataset_used":
                False,

            "purge_rule":
                (
                    f"각 Fold Validation 직전 "
                    f"{horizon}개 거래일 제거"
                ),

            "folds":
                fold_results,

            "summary": {
                "roc_auc_mean":
                    self._mean(
                        auc_values
                    ),

                "roc_auc_std":
                    self._std(
                        auc_values
                    ),

                "roc_auc_min":
                    min(
                        auc_values
                    ),

                "roc_auc_max":
                    max(
                        auc_values
                    ),

                "balanced_accuracy_mean_pct":
                    self._mean(
                        balanced_values
                    ),

                "balanced_accuracy_std_pct":
                    self._std(
                        balanced_values
                    ),

                "log_loss_mean":
                    self._mean(
                        log_loss_values
                    ),

                "brier_score_mean":
                    self._mean(
                        brier_values
                    ),

                "ece_10_mean":
                    self._mean(
                        ece_values
                    ),

                "folds_auc_above_0_50":
                    sum(
                        value > 0.50
                        for value
                        in auc_values
                    ),

                "folds_auc_above_0_55":
                    sum(
                        value > 0.55
                        for value
                        in auc_values
                    ),
            },
        }