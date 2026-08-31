from __future__ import annotations

import numpy as np
from sqlalchemy.orm import Session

from app.services.historical_ml_baseline_service import (
    HistoricalMlBaselineService,
)
from app.services.historical_ml_training_service import (
    HistoricalMlTrainingService,
)


class HistoricalMlAblationService:
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
    def _is_disclosure_feature(
        feature_name: str,
    ) -> bool:
        return feature_name.startswith(
            "disclosure_"
        )

    @staticmethod
    def _is_market_feature(
        feature_name: str,
    ) -> bool:
        return feature_name.startswith(
            "market_"
        )

    @staticmethod
    def _is_sector_feature(
        feature_name: str,
    ) -> bool:
        return feature_name.startswith(
            "sector_"
        )

    @staticmethod
    def _is_macro_feature(
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

    def _feature_groups(
        self,
        feature_names: list[str],
    ) -> dict[str, set[str]]:
        disclosure = {
            name
            for name in feature_names
            if self._is_disclosure_feature(
                name
            )
        }

        market = {
            name
            for name in feature_names
            if self._is_market_feature(
                name
            )
        }

        sector = {
            name
            for name in feature_names
            if self._is_sector_feature(
                name
            )
        }

        macro = {
            name
            for name in feature_names
            if self._is_macro_feature(
                name
            )
        }

        external_context = (
            disclosure
            | market
            | sector
            | macro
        )

        stock_internal = (
            set(
                feature_names
            )
            - external_context
        )

        return {
            "disclosure":
                disclosure,

            "market":
                market,

            "sector":
                sector,

            "macro":
                macro,

            "external_context":
                external_context,

            "stock_internal":
                stock_internal,
        }

    @staticmethod
    def _column_indexes(
        *,
        all_feature_names: list[str],
        selected_feature_names: set[str],
    ) -> list[int]:
        return [
            index
            for (
                index,
                feature_name,
            )
            in enumerate(
                all_feature_names
            )
            if feature_name
            in selected_feature_names
        ]

    def run_validation_ablation(
        self,
        *,
        feature_version: str,
        horizon: int,
        train_ratio: float = 0.70,
        valid_ratio: float = 0.15,
    ) -> dict:
        split = (
            self.training_service
            .build_temporal_split(
                feature_version=(
                    feature_version
                ),
                horizon=horizon,
                train_ratio=(
                    train_ratio
                ),
                valid_ratio=(
                    valid_ratio
                ),
            )
        )

        all_features = list(
            split.feature_names
        )

        all_feature_set = set(
            all_features
        )

        groups = (
            self._feature_groups(
                all_features
            )
        )

        experiments = {
            "full":
                all_feature_set,

            "no_disclosure":
                all_feature_set
                - groups[
                    "disclosure"
                ],

            "no_market":
                all_feature_set
                - groups[
                    "market"
                ],

            "no_sector":
                all_feature_set
                - groups[
                    "sector"
                ],

            "no_macro":
                all_feature_set
                - groups[
                    "macro"
                ],

            "stock_internal_only":
                groups[
                    "stock_internal"
                ],
        }

        y_train_class = (
            np.asarray(
                split.y_train
            )
            > 0.0
        ).astype(
            np.int32
        )

        results = []

        for (
            experiment_name,
            selected_features,
        ) in experiments.items():
            indexes = (
                self._column_indexes(
                    all_feature_names=(
                        all_features
                    ),
                    selected_feature_names=(
                        selected_features
                    ),
                )
            )

            if not indexes:
                raise ValueError(
                    f"{experiment_name}: "
                    "선택된 Feature가 없습니다."
                )

            x_train = np.asarray(
                split.x_train[
                    :,
                    indexes
                ],
                dtype=np.float32,
            )

            x_valid = np.asarray(
                split.x_valid[
                    :,
                    indexes
                ],
                dtype=np.float32,
            )

            _, classifier = (
                self.baseline_service
                ._make_models()
            )

            classifier.fit(
                x_train,
                y_train_class,
            )

            valid_probability = (
                classifier
                .predict_proba(
                    x_valid
                )[:, 1]
            )

            metrics = (
                self.baseline_service
                ._classification_metrics(
                    y_true_return=(
                        np.asarray(
                            split.y_valid
                        )
                    ),
                    probability=(
                        valid_probability
                    ),
                )
            )

            results.append(
                {
                    "experiment":
                        experiment_name,

                    "feature_count":
                        len(
                            indexes
                        ),

                    "removed_feature_count":
                        len(
                            all_features
                        )
                        - len(
                            indexes
                        ),

                    "roc_auc":
                        metrics[
                            "roc_auc"
                        ],

                    "accuracy_pct":
                        metrics[
                            "accuracy_pct"
                        ],

                    "balanced_accuracy_pct":
                        metrics[
                            "balanced_accuracy_pct"
                        ],

                    "majority_baseline_accuracy_pct":
                        metrics[
                            "majority_baseline_accuracy_pct"
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
                row[
                    "roc_auc"
                ]
                if row[
                    "roc_auc"
                ]
                is not None
                else -1.0
            ),
            reverse=True,
        )

        return {
            "status":
                "pass",

            "feature_version":
                feature_version,

            "target_horizon":
                horizon,

            "selection_rule":
                (
                    "Feature 조합 선택은 "
                    "Validation ROC-AUC 기준이며 "
                    "Test Dataset은 사용하지 않음"
                ),

            "train_rows":
                int(
                    split.x_train.shape[
                        0
                    ]
                ),

            "validation_rows":
                int(
                    split.x_valid.shape[
                        0
                    ]
                ),

            "group_counts": {
                name:
                    len(
                        feature_group
                    )
                for (
                    name,
                    feature_group,
                )
                in groups.items()
            },

            "results":
                results,

            "best_validation_experiment":
                results[
                    0
                ][
                    "experiment"
                ],

            "best_validation_roc_auc":
                results[
                    0
                ][
                    "roc_auc"
                ],
        }