from __future__ import annotations

import numpy as np
from sqlalchemy.orm import Session

from app.services.historical_ml_baseline_service import (
    HistoricalMlBaselineService,
)
from app.services.historical_ml_training_service import (
    HistoricalMlTrainingService,
)


class HistoricalMlTuningService:
    RANDOM_STATE = 42

    def __init__(
        self,
        db: Session,
    ):
        self.db = db

        self.training_service = (
            HistoricalMlTrainingService(
                db
            )
        )

        self.baseline_service = (
            HistoricalMlBaselineService(
                db
            )
        )

    @staticmethod
    def _make_classifier(
        params: dict,
    ):
        try:
            from xgboost import XGBClassifier

        except ImportError as e:
            raise RuntimeError(
                "xgboost가 설치되어 있지 않습니다."
            ) from e

        return XGBClassifier(
            objective="binary:logistic",
            eval_metric="logloss",
            tree_method="hist",
            random_state=(
                HistoricalMlTuningService
                .RANDOM_STATE
            ),
            n_jobs=4,
            **params,
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
                "name": "shallow_slow",
                "n_estimators": 800,
                "max_depth": 3,
                "learning_rate": 0.02,
                "min_child_weight": 10,
                "subsample": 0.90,
                "colsample_bytree": 0.75,
                "reg_alpha": 0.10,
                "reg_lambda": 2.00,
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
                "name": "lower_sampling",
                "n_estimators": 600,
                "max_depth": 4,
                "learning_rate": 0.03,
                "min_child_weight": 10,
                "subsample": 0.70,
                "colsample_bytree": 0.70,
                "reg_alpha": 0.20,
                "reg_lambda": 2.00,
            },
            {
                "name": "faster_shallow",
                "n_estimators": 350,
                "max_depth": 3,
                "learning_rate": 0.05,
                "min_child_weight": 10,
                "subsample": 0.85,
                "colsample_bytree": 0.80,
                "reg_alpha": 0.10,
                "reg_lambda": 2.00,
            },
            {
                "name": "deep_regularized",
                "n_estimators": 500,
                "max_depth": 5,
                "learning_rate": 0.02,
                "min_child_weight": 15,
                "subsample": 0.80,
                "colsample_bytree": 0.70,
                "reg_alpha": 0.25,
                "reg_lambda": 3.00,
            },
        ]

    def tune_no_market(
        self,
        *,
        feature_version: str,
        horizon: int = 5,
    ) -> dict:
        split = (
            self.training_service
            .build_temporal_split(
                feature_version=(
                    feature_version
                ),
                horizon=horizon,
            )
        )

        selected_indexes = [
            index
            for (
                index,
                feature_name,
            )
            in enumerate(
                split.feature_names
            )
            if not feature_name.startswith(
                "market_"
            )
        ]

        selected_feature_names = [
            split.feature_names[
                index
            ]
            for index
            in selected_indexes
        ]

        x_train = np.asarray(
            split.x_train[
                :,
                selected_indexes
            ],
            dtype=np.float32,
        )

        y_train = np.asarray(
            split.y_train,
            dtype=np.float32,
        )

        x_valid = np.asarray(
            split.x_valid[
                :,
                selected_indexes
            ],
            dtype=np.float32,
        )

        y_valid = np.asarray(
            split.y_valid,
            dtype=np.float32,
        )

        y_train_class = (
            y_train > 0.0
        ).astype(
            np.int32
        )

        results = []

        for candidate in (
            self._candidate_params()
        ):
            candidate_name = (
                candidate["name"]
            )

            params = {
                key: value
                for key, value
                in candidate.items()
                if key != "name"
            }

            classifier = (
                self._make_classifier(
                    params
                )
            )

            classifier.fit(
                x_train,
                y_train_class,
            )

            probability = (
                classifier
                .predict_proba(
                    x_valid
                )[:, 1]
            )

            metrics = (
                self.baseline_service
                ._classification_metrics(
                    y_true_return=(
                        y_valid
                    ),
                    probability=(
                        probability
                    ),
                )
            )

            results.append(
                {
                    "name":
                        candidate_name,

                    "params":
                        params,

                    "roc_auc":
                        metrics[
                            "roc_auc"
                        ],

                    "balanced_accuracy_pct":
                        metrics[
                            "balanced_accuracy_pct"
                        ],

                    "accuracy_pct":
                        metrics[
                            "accuracy_pct"
                        ],

                    "accuracy_edge_vs_majority_pct":
                        metrics[
                            "accuracy_edge_vs_majority_pct"
                        ],

                    "log_loss":
                        metrics[
                            "log_loss"
                        ],

                    "probability_mean":
                        metrics[
                            "probability_mean"
                        ],
                }
            )

        results.sort(
            key=lambda row: (
                -(
                    row[
                        "roc_auc"
                    ]
                    if row[
                        "roc_auc"
                    ]
                    is not None
                    else -1.0
                ),
                row[
                    "log_loss"
                ],
            )
        )

        best = results[
            0
        ]

        return {
            "status":
                "pass",

            "feature_version":
                feature_version,

            "target_horizon":
                horizon,

            "feature_strategy":
                "no_market",

            "feature_count":
                len(
                    selected_feature_names
                ),

            "train_rows":
                int(
                    x_train.shape[
                        0
                    ]
                ),

            "validation_rows":
                int(
                    x_valid.shape[
                        0
                    ]
                ),

            "selection_rule":
                (
                    "Validation ROC-AUC 우선, "
                    "동률 시 LogLoss 사용. "
                    "Test Dataset 미사용."
                ),

            "best_candidate":
                best,

            "results":
                results,
        }