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
from app.services.historical_ml_walk_forward_service import (
    HistoricalMlWalkForwardService,
)


class HistoricalMlWalkForwardTuningService:
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

    @staticmethod
    def _candidate_params() -> list[dict]:
        return [
            {
                "name": "baseline",
                "n_estimators": 500,
                "max_depth": 4,
                "learning_rate": 0.03,
                "min_child_weight": 5,
                "subsample": 0.85,
                "colsample_bytree": 0.80,
                "reg_alpha": 0.05,
                "reg_lambda": 1.50,
            },
            {
                "name": "shallow_regularized",
                "n_estimators": 600,
                "max_depth": 3,
                "learning_rate": 0.03,
                "min_child_weight": 10,
                "subsample": 0.90,
                "colsample_bytree": 0.80,
                "reg_alpha": 0.10,
                "reg_lambda": 2.00,
            },
            {
                "name": "strong_regularization",
                "n_estimators": 600,
                "max_depth": 3,
                "learning_rate": 0.03,
                "min_child_weight": 20,
                "subsample": 0.85,
                "colsample_bytree": 0.75,
                "reg_alpha": 0.25,
                "reg_lambda": 3.00,
            },
            {
                "name": "very_shallow",
                "n_estimators": 700,
                "max_depth": 2,
                "learning_rate": 0.03,
                "min_child_weight": 10,
                "subsample": 0.90,
                "colsample_bytree": 0.80,
                "reg_alpha": 0.10,
                "reg_lambda": 2.00,
            },
            {
                "name": "very_shallow_strong",
                "n_estimators": 700,
                "max_depth": 2,
                "learning_rate": 0.025,
                "min_child_weight": 20,
                "subsample": 0.85,
                "colsample_bytree": 0.75,
                "reg_alpha": 0.25,
                "reg_lambda": 3.00,
            },
            {
                "name": "shallow_slow",
                "n_estimators": 800,
                "max_depth": 3,
                "learning_rate": 0.02,
                "min_child_weight": 15,
                "subsample": 0.90,
                "colsample_bytree": 0.75,
                "reg_alpha": 0.20,
                "reg_lambda": 2.50,
            },
            {
                "name": "lower_sampling",
                "n_estimators": 600,
                "max_depth": 3,
                "learning_rate": 0.03,
                "min_child_weight": 15,
                "subsample": 0.70,
                "colsample_bytree": 0.70,
                "reg_alpha": 0.20,
                "reg_lambda": 2.50,
            },
        ]

    @classmethod
    def _make_classifier(
        cls,
        params: dict,
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
            **params,
        )

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

        candidate_results = []

        for candidate in (
            self._candidate_params()
        ):
            candidate_name = (
                candidate[
                    "name"
                ]
            )

            params = {
                key:
                    value
                for (
                    key,
                    value,
                )
                in candidate.items()
                if key != "name"
            }

            folds_result = []

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
                            dataset
                            .feature_dates
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
                            dataset
                            .feature_dates
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
                    self._make_classifier(
                        params
                    )
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

                metrics = (
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

                balanced_accuracy = (
                    float(
                        balanced_accuracy_score(
                            y_validation,
                            prediction,
                        )
                        * 100.0
                    )
                )

                folds_result.append(
                    {
                        "fold":
                            fold_index + 1,

                        "roc_auc":
                            metrics[
                                "roc_auc"
                            ],

                        "balanced_accuracy_pct":
                            balanced_accuracy,

                        "accuracy_pct":
                            accuracy,

                        "log_loss":
                            metrics[
                                "log_loss"
                            ],

                        "brier_score":
                            metrics[
                                "brier_score"
                            ],

                        "validation_first":
                            min(
                                validation_dates
                            ).isoformat(),

                        "validation_last":
                            max(
                                validation_dates
                            ).isoformat(),
                    }
                )

            auc_values = [
                float(
                    fold[
                        "roc_auc"
                    ]
                )
                for fold
                in folds_result
            ]

            balanced_values = [
                float(
                    fold[
                        "balanced_accuracy_pct"
                    ]
                )
                for fold
                in folds_result
            ]

            logloss_values = [
                float(
                    fold[
                        "log_loss"
                    ]
                )
                for fold
                in folds_result
            ]

            brier_values = [
                float(
                    fold[
                        "brier_score"
                    ]
                )
                for fold
                in folds_result
            ]

            candidate_results.append(
                {
                    "name":
                        candidate_name,

                    "params":
                        params,

                    "folds":
                        folds_result,

                    "fold_auc_values":
                        auc_values,

                    "folds_auc_above_0_50":
                        sum(
                            auc > 0.50
                            for auc
                            in auc_values
                        ),

                    "folds_auc_above_0_52":
                        sum(
                            auc > 0.52
                            for auc
                            in auc_values
                        ),

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

                    "log_loss_mean":
                        self._mean(
                            logloss_values
                        ),

                    "brier_score_mean":
                        self._mean(
                            brier_values
                        ),
                }
            )

        candidate_results.sort(
            key=lambda row: (
                row[
                    "folds_auc_above_0_50"
                ],
                row[
                    "roc_auc_mean"
                ],
                -row[
                    "roc_auc_std"
                ],
                row[
                    "roc_auc_min"
                ],
                -row[
                    "log_loss_mean"
                ],
            ),
            reverse=True,
        )

        best = (
            candidate_results[
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

            "fold_count":
                folds,

            "test_dataset_used":
                False,

            "locked_test_first_date":
                locked_test_first_date
                .isoformat(),

            "selection_rule":
                (
                    "Fold AUC > 0.50 개수 우선, "
                    "그다음 평균 AUC, 낮은 표준편차, "
                    "최저 AUC, LogLoss 순서"
                ),

            "best_candidate":
                best,

            "results":
                candidate_results,
        }