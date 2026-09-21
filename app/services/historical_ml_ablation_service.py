from __future__ import annotations

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score
from sqlalchemy.orm import Session

from app.services.historical_ml_baseline_service import (
    HistoricalMlBaselineService,
)
from app.services.historical_ml_training_service import (
    HistoricalMlTrainingService,
)
from app.services.historical_dataset_service import (
    HistoricalDatasetService,
)
from app.services.historical_ml_oof_service import (
    HistoricalMlOofService,
)
from app.services.historical_ml_walk_forward_service import (
    HistoricalMlWalkForwardService,
)
from app.services.historical_ml_walk_forward_tuning_service import (
    HistoricalMlWalkForwardTuningService,
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
    
    @staticmethod
    def _ranking_signal_metrics(
        *,
        probabilities: np.ndarray,
        future_returns: np.ndarray,
        feature_dates: list,
    ) -> dict:
        grouped_indexes = {}

        for index, feature_date in enumerate(
            feature_dates
        ):
            grouped_indexes.setdefault(
                feature_date,
                [],
            ).append(index)

        ic_values = []
        top10_excess_values = []
        top20_excess_values = []
        long_short_values = []

        for feature_date in sorted(
            grouped_indexes
        ):
            indexes = np.asarray(
                grouped_indexes[
                    feature_date
                ],
                dtype=np.int64,
            )

            if len(indexes) < 20:
                continue

            daily_probability = probabilities[
                indexes
            ]

            daily_return = future_returns[
                indexes
            ]

            universe_return = float(
                np.mean(
                    daily_return
                )
            )

            order = np.argsort(
                daily_probability
            )[::-1]

            top10_indexes = order[:10]
            top20_indexes = order[:20]
            bottom10_indexes = order[-10:]

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

            if (
                np.std(
                    daily_probability
                ) > 0.0
                and np.std(
                    daily_return
                ) > 0.0
            ):
                ic_result = spearmanr(
                    daily_probability,
                    daily_return,
                )

                if np.isfinite(
                    ic_result.statistic
                ):
                    ic_values.append(
                        float(
                            ic_result.statistic
                        )
                    )

            top10_excess_values.append(
                top10_return
                - universe_return
            )

            top20_excess_values.append(
                top20_return
                - universe_return
            )

            long_short_values.append(
                top10_return
                - bottom10_return
            )

        return {
            "ranking_dates":
                len(
                    top10_excess_values
                ),

            "spearman_ic_mean": (
                float(
                    np.mean(
                        ic_values
                    )
                )
                if ic_values
                else None
            ),

            "spearman_ic_positive_rate_pct": (
                float(
                    np.mean(
                        np.asarray(
                            ic_values,
                            dtype=np.float64,
                        ) > 0.0
                    )
                    * 100.0
                )
                if ic_values
                else None
            ),

            "top10_excess_mean_pct":
                float(
                    np.mean(
                        top10_excess_values
                    )
                    * 100.0
                ),

            "top20_excess_mean_pct":
                float(
                    np.mean(
                        top20_excess_values
                    )
                    * 100.0
                ),

            "long_short_10_mean_pct":
                float(
                    np.mean(
                        long_short_values
                    )
                    * 100.0
                ),
        }

    @staticmethod
    def _permute_feature_within_dates(
        *,
        x_validation: np.ndarray,
        validation_dates: list,
        feature_index: int,
        random_state: int,
    ) -> np.ndarray:
        permuted = np.array(
            x_validation,
            copy=True,
        )

        indexes_by_date = {}

        for row_index, feature_date in enumerate(
            validation_dates
        ):
            indexes_by_date.setdefault(
                feature_date,
                [],
            ).append(row_index)

        rng = np.random.default_rng(
            random_state
        )

        for indexes in indexes_by_date.values():
            if len(indexes) <= 1:
                continue

            row_indexes = np.asarray(
                indexes,
                dtype=np.int64,
            )

            source_values = np.array(
                permuted[
                    row_indexes,
                    feature_index,
                ],
                copy=True,
            )

            rng.shuffle(
                source_values
            )

            permuted[
                row_indexes,
                feature_index,
            ] = source_values

        return permuted

    def run_oof_feature_permutation(
        self,
        *,
        feature_version: str,
        horizon: int = 5,
        folds: int = 4,
        initial_train_ratio: float = 0.55,
    ) -> dict:
        dataset_service = HistoricalDatasetService(
            self.db
        )

        dataset = dataset_service.build_dataset(
            feature_version=feature_version,
            horizon=horizon,
            feature_strategy=(
                "stock_internal_only"
            ),
        )

        temporal_split = (
            self.training_service
            .build_temporal_split(
                feature_version=feature_version,
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
            dataset.feature_names[index]
            for index in selected_indexes
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
            len(eligible_dates)
            * initial_train_ratio
        )

        remaining_count = (
            len(eligible_dates)
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

        baseline_probabilities = []

        permuted_probabilities = {
            feature_name: []
            for feature_name
            in selected_feature_names
        }

        all_targets = []
        all_future_returns = []
        all_feature_dates = []

        for fold_index in range(
            folds
        ):
            validation_start_index = (
                initial_train_count
                + fold_index
                * base_fold_size
            )

            if fold_index == folds - 1:
                validation_end_index = (
                    len(eligible_dates)
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

            validation_feature_dates = [
                dataset.feature_dates[index]
                for index
                in validation_indexes
            ]

            classifier = (
                HistoricalMlOofService
                ._make_classifier()
            )

            classifier.fit(
                x_train,
                y_train,
            )

            baseline_probability = (
                classifier
                .predict_proba(
                    x_validation
                )[:, 1]
            )

            baseline_probabilities.append(
                baseline_probability
            )

            for (
                feature_index,
                feature_name,
            ) in enumerate(
                selected_feature_names
            ):
                permuted_x = (
                    self
                    ._permute_feature_within_dates(
                        x_validation=(
                            x_validation
                        ),
                        validation_dates=(
                            validation_feature_dates
                        ),
                        feature_index=(
                            feature_index
                        ),
                        random_state=(
                            42
                            + fold_index
                            * 1000
                            + feature_index
                        ),
                    )
                )

                probability = (
                    classifier
                    .predict_proba(
                        permuted_x
                    )[:, 1]
                )

                permuted_probabilities[
                    feature_name
                ].append(
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

            all_feature_dates.extend(
                validation_feature_dates
            )

        oof_probability = np.concatenate(
            baseline_probabilities
        )

        oof_y = np.concatenate(
            all_targets
        )

        oof_future_return = np.concatenate(
            all_future_returns
        )

        baseline_auc = float(
            roc_auc_score(
                oof_y,
                oof_probability,
            )
        )

        baseline_ranking = (
            self
            ._ranking_signal_metrics(
                probabilities=(
                    oof_probability
                ),
                future_returns=(
                    oof_future_return
                ),
                feature_dates=(
                    all_feature_dates
                ),
            )
        )

        feature_results = []

        for feature_name in (
            selected_feature_names
        ):
            probability = np.concatenate(
                permuted_probabilities[
                    feature_name
                ]
            )

            permuted_auc = float(
                roc_auc_score(
                    oof_y,
                    probability,
                )
            )

            ranking = (
                self
                ._ranking_signal_metrics(
                    probabilities=(
                        probability
                    ),
                    future_returns=(
                        oof_future_return
                    ),
                    feature_dates=(
                        all_feature_dates
                    ),
                )
            )

            baseline_ic = (
                baseline_ranking[
                    "spearman_ic_mean"
                ]
            )

            permuted_ic = (
                ranking[
                    "spearman_ic_mean"
                ]
            )

            feature_results.append(
                {
                    "feature":
                        feature_name,

                    "permuted_roc_auc":
                        permuted_auc,

                    "roc_auc_drop":
                        baseline_auc
                        - permuted_auc,

                    "permuted_spearman_ic_mean":
                        permuted_ic,

                    "spearman_ic_drop": (
                        baseline_ic
                        - permuted_ic
                        if (
                            baseline_ic
                            is not None
                            and permuted_ic
                            is not None
                        )
                        else None
                    ),

                    "permuted_top10_excess_mean_pct":
                        ranking[
                            "top10_excess_mean_pct"
                        ],

                    "top10_excess_drop_pct":
                        baseline_ranking[
                            "top10_excess_mean_pct"
                        ]
                        - ranking[
                            "top10_excess_mean_pct"
                        ],

                    "permuted_top20_excess_mean_pct":
                        ranking[
                            "top20_excess_mean_pct"
                        ],

                    "top20_excess_drop_pct":
                        baseline_ranking[
                            "top20_excess_mean_pct"
                        ]
                        - ranking[
                            "top20_excess_mean_pct"
                        ],

                    "permuted_long_short_10_mean_pct":
                        ranking[
                            "long_short_10_mean_pct"
                        ],

                    "long_short_10_drop_pct":
                        baseline_ranking[
                            "long_short_10_mean_pct"
                        ]
                        - ranking[
                            "long_short_10_mean_pct"
                        ],
                }
            )

        feature_results.sort(
            key=lambda row: (
                row[
                    "top10_excess_drop_pct"
                ],
                row[
                    "spearman_ic_drop"
                ]
                if row[
                    "spearman_ic_drop"
                ]
                is not None
                else -999.0,
                row[
                    "roc_auc_drop"
                ],
            ),
            reverse=True,
        )

        harmful_features = [
            row["feature"]
            for row
            in feature_results
            if (
                row[
                    "top10_excess_drop_pct"
                ] < 0.0
                and (
                    row[
                        "spearman_ic_drop"
                    ] is None
                    or row[
                        "spearman_ic_drop"
                    ] <= 0.0
                )
            )
        ]

        return {
            "status":
                "pass",

            "phase":
                "6.3-1",

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

            "feature_names":
                selected_feature_names,

            "fold_count":
                folds,

            "test_dataset_used":
                False,

            "locked_test_first_date":
                locked_test_first_date
                .isoformat(),

            "permutation_rule": (
                "각 validation 날짜 내부에서 "
                "feature 1개만 shuffle 후 "
                "동일 OOF 모델로 재예측"
            ),

            "baseline": {
                "roc_auc":
                    baseline_auc,

                **baseline_ranking,
            },

            "feature_results":
                feature_results,

            "harmful_feature_candidates":
                harmful_features,
        }

    def run_oof_objective_experiment(
        self,
        *,
        feature_version: str,
        experiment: str,
        horizon: int = 5,
        folds: int = 4,
        initial_train_ratio: float = 0.55,
    ) -> dict:
        experiments = {
            "binary_logistic_26": {
                "model_mode":
                    "classifier",
                "ranking_objective":
                    "rank:pairwise",
                "ranking_label_mode":
                    "binary",
            },

            "rank_pairwise_binary_26": {
                "model_mode":
                    "ranker",
                "ranking_objective":
                    "rank:pairwise",
                "ranking_label_mode":
                    "binary",
            },

            "rank_pairwise_quintile_26": {
                "model_mode":
                    "ranker",
                "ranking_objective":
                    "rank:pairwise",
                "ranking_label_mode":
                    "quintile",
            },

            "rank_ndcg_binary_26": {
                "model_mode":
                    "ranker",
                "ranking_objective":
                    "rank:ndcg",
                "ranking_label_mode":
                    "binary",
            },

            "rank_ndcg_quintile_26": {
                "model_mode":
                    "ranker",
                "ranking_objective":
                    "rank:ndcg",
                "ranking_label_mode":
                    "quintile",
            },
        }

        if experiment not in experiments:
            raise ValueError(
                "지원하지 않는 Objective 실험: "
                f"{experiment}"
            )

        config = experiments[
            experiment
        ]

        result = (
            HistoricalMlOofService(
                self.db
            )
            .run(
                feature_version=(
                    feature_version
                ),
                horizon=horizon,
                folds=folds,
                initial_train_ratio=(
                    initial_train_ratio
                ),
                excluded_features={
                    "rsi_14",
                },
                comparison_only=True,
                classifier_params=(
                    dict(
                        HistoricalMlOofService
                        .FINAL_PARAMS
                    )
                ),
                model_mode=(
                    config[
                        "model_mode"
                    ]
                ),
                ranking_objective=(
                    config[
                        "ranking_objective"
                    ]
                ),
                ranking_label_mode=(
                    config[
                        "ranking_label_mode"
                    ]
                ),
            )
        )

        return {
            "status":
                "pass",

            "phase":
                "6.3-4",

            "experiment":
                experiment,

            "removed_features": [
                "rsi_14",
            ],

            "comparison_rule":
                (
                    "rsi_14를 제외한 26개 Feature, "
                    "기존 FINAL_PARAMS, 동일 OOF 구간을 "
                    "고정하고 학습 objective와 "
                    "ranking relevance label만 비교"
                ),

            **result,
        }

    def run_oof_model_tuning_experiment(
        self,
        *,
        feature_version: str,
        experiment: str,
        horizon: int = 5,
        folds: int = 4,
        initial_train_ratio: float = 0.55,
    ) -> dict:
        candidate_params = {
            candidate[
                "name"
            ]: {
                key:
                    value
                for (
                    key,
                    value,
                )
                in candidate.items()
                if key != "name"
            }
            for candidate
            in (
                HistoricalMlWalkForwardTuningService
                ._candidate_params()
            )
        }

        experiments = {
            "current_final_26":
                dict(
                    HistoricalMlOofService
                    .FINAL_PARAMS
                ),

            "shallow_regularized_26":
                candidate_params[
                    "shallow_regularized"
                ],

            "strong_regularization_26":
                candidate_params[
                    "strong_regularization"
                ],

            "very_shallow_26":
                candidate_params[
                    "very_shallow"
                ],

            "very_shallow_strong_26":
                candidate_params[
                    "very_shallow_strong"
                ],

            "shallow_slow_26":
                candidate_params[
                    "shallow_slow"
                ],
        }

        if experiment not in experiments:
            raise ValueError(
                "지원하지 않는 Model tuning 실험: "
                f"{experiment}"
            )

        classifier_params = dict(
            experiments[
                experiment
            ]
        )

        result = (
            HistoricalMlOofService(
                self.db
            )
            .run(
                feature_version=(
                    feature_version
                ),
                horizon=horizon,
                folds=folds,
                initial_train_ratio=(
                    initial_train_ratio
                ),
                excluded_features={
                    "rsi_14",
                },
                comparison_only=True,
                classifier_params=(
                    classifier_params
                ),
            )
        )

        return {
            "status":
                "pass",

            "phase":
                "6.3-3",

            "experiment":
                experiment,

            "removed_features": [
                "rsi_14",
            ],

            "classifier_params":
                classifier_params,

            "comparison_rule":
                (
                    "rsi_14를 제외한 26개 Feature와 "
                    "동일 OOF 구간을 고정하고 "
                    "XGBoost 파라미터만 변경하여 재학습"
                ),

            **result,
        }

    def run_oof_feature_subset_experiment(
        self,
        *,
        feature_version: str,
        experiment: str,
        horizon: int = 5,
        folds: int = 4,
        initial_train_ratio: float = 0.55,
    ) -> dict:
        experiments = {
            "full_27": set(),

            "no_adx_14": {
                "adx_14",
            },

            "no_rsi_14": {
                "rsi_14",
            },

            "no_adx_rsi": {
                "adx_14",
                "rsi_14",
            },

            "no_flagged_4": {
                "stochastic_d_3",
                "adx_14",
                "macd_hist_ratio",
                "rsi_14",
            },
        }

        if experiment not in experiments:
            raise ValueError(
                "지원하지 않는 Feature subset 실험: "
                f"{experiment}"
            )

        removed_features = (
            experiments[
                experiment
            ]
        )

        result = (
            HistoricalMlOofService(
                self.db
            )
            .run(
                feature_version=(
                    feature_version
                ),
                horizon=horizon,
                folds=folds,
                initial_train_ratio=(
                    initial_train_ratio
                ),
                excluded_features=(
                    removed_features
                ),
                comparison_only=True,
            )
        )

        return {
            "status":
                "pass",

            "phase":
                "6.3-2",

            "experiment":
                experiment,

            "removed_features":
                sorted(
                    removed_features
                ),

            "comparison_rule":
                (
                    "동일 OOF 기간, 동일 XGBoost "
                    "파라미터에서 지정 Feature를 제거한 뒤 "
                    "모델을 처음부터 재학습"
                ),

            **result,
        }

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