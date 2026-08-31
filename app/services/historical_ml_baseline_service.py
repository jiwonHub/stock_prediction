from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    log_loss,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)
from sqlalchemy.orm import Session

from app.services.historical_ml_training_service import (
    HistoricalMlTrainingService,
)


class HistoricalMlBaselineService:
    MODEL_NAME = "xgboost-historical-baseline"
    MODEL_VERSION = "phase6-baseline-v1"

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

    @staticmethod
    def _make_models():
        try:
            from xgboost import (
                XGBClassifier,
                XGBRegressor,
            )

        except ImportError as e:
            raise RuntimeError(
                "Phase 6 XGBoost baseline을 위해 "
                "xgboost 패키지가 필요합니다."
            ) from e

        regressor = XGBRegressor(
            n_estimators=500,
            max_depth=4,
            learning_rate=0.03,
            min_child_weight=5,
            subsample=0.85,
            colsample_bytree=0.80,
            reg_alpha=0.05,
            reg_lambda=1.50,
            objective="reg:squarederror",
            tree_method="hist",
            random_state=(
                HistoricalMlBaselineService
                .RANDOM_STATE
            ),
            n_jobs=4,
        )

        classifier = XGBClassifier(
            n_estimators=500,
            max_depth=4,
            learning_rate=0.03,
            min_child_weight=5,
            subsample=0.85,
            colsample_bytree=0.80,
            reg_alpha=0.05,
            reg_lambda=1.50,
            objective="binary:logistic",
            eval_metric="logloss",
            tree_method="hist",
            random_state=(
                HistoricalMlBaselineService
                .RANDOM_STATE
            ),
            n_jobs=4,
        )

        return (
            regressor,
            classifier,
        )

    @staticmethod
    def _regression_metrics(
        *,
        y_true: np.ndarray,
        y_pred: np.ndarray,
    ) -> dict:
        mae = float(
            mean_absolute_error(
                y_true,
                y_pred,
            )
        )

        rmse = float(
            np.sqrt(
                mean_squared_error(
                    y_true,
                    y_pred,
                )
            )
        )

        r2 = float(
            r2_score(
                y_true,
                y_pred,
            )
        )

        direction_accuracy = float(
            accuracy_score(
                y_true > 0.0,
                y_pred > 0.0,
            )
            * 100.0
        )

        correlation = 0.0

        if (
            len(
                y_true
            )
            > 1
            and np.std(
                y_true
            )
            > 0.0
            and np.std(
                y_pred
            )
            > 0.0
        ):
            correlation = float(
                np.corrcoef(
                    y_true,
                    y_pred,
                )[0, 1]
            )

        return {
            "mae":
                mae,

            "rmse":
                rmse,

            "r2":
                r2,

            "direction_accuracy_pct":
                direction_accuracy,

            "prediction_target_correlation":
                correlation,

            "prediction_mean":
                float(
                    np.mean(
                        y_pred
                    )
                ),

            "target_mean":
                float(
                    np.mean(
                        y_true
                    )
                ),
        }

    @staticmethod
    def _classification_metrics(
        *,
        y_true_return: np.ndarray,
        probability: np.ndarray,
    ) -> dict:
        y_true = (
            y_true_return
            > 0.0
        ).astype(
            np.int32
        )

        y_pred = (
            probability
            >= 0.50
        ).astype(
            np.int32
        )

        accuracy = float(
            accuracy_score(
                y_true,
                y_pred,
            )
            * 100.0
        )

        balanced_accuracy = float(
            balanced_accuracy_score(
                y_true,
                y_pred,
            )
            * 100.0
        )

        auc = None

        if len(
            np.unique(
                y_true
            )
        ) >= 2:
            auc = float(
                roc_auc_score(
                    y_true,
                    probability,
                )
            )

        clipped_probability = np.clip(
            probability,
            1e-7,
            1.0 - 1e-7,
        )

        loss = float(
            log_loss(
                y_true,
                clipped_probability,
                labels=[
                    0,
                    1,
                ],
            )
        )

        positive_rate = float(
            np.mean(
                y_true
            )
            * 100.0
        )

        majority_baseline = float(
            max(
                np.mean(
                    y_true
                ),
                1.0
                - np.mean(
                    y_true
                ),
            )
            * 100.0
        )

        return {
            "accuracy_pct":
                accuracy,

            "balanced_accuracy_pct":
                balanced_accuracy,

            "roc_auc":
                auc,

            "log_loss":
                loss,

            "positive_rate_pct":
                positive_rate,

            "majority_baseline_accuracy_pct":
                majority_baseline,

            "accuracy_edge_vs_majority_pct":
                accuracy
                - majority_baseline,

            "probability_mean":
                float(
                    np.mean(
                        probability
                    )
                ),
        }

    @staticmethod
    def _feature_importance(
        *,
        model,
        feature_names: list[str],
        limit: int = 20,
    ) -> list[dict]:
        raw = getattr(
            model,
            "feature_importances_",
            None,
        )

        if raw is None:
            return []

        pairs = list(
            zip(
                feature_names,
                raw,
                strict=True,
            )
        )

        pairs.sort(
            key=lambda item: float(
                item[1]
            ),
            reverse=True,
        )

        return [
            {
                "feature":
                    feature_name,

                "importance":
                    float(
                        importance
                    ),
            }
            for (
                feature_name,
                importance,
            )
            in pairs[
                :limit
            ]
        ]

    def train_baseline(
        self,
        *,
        feature_version: str,
        horizon: int,
        train_ratio: float = 0.70,
        valid_ratio: float = 0.15,
        save_model: bool = True,
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

        x_train = np.asarray(
            split.x_train,
            dtype=np.float32,
        )

        y_train = np.asarray(
            split.y_train,
            dtype=np.float32,
        )

        x_valid = np.asarray(
            split.x_valid,
            dtype=np.float32,
        )

        y_valid = np.asarray(
            split.y_valid,
            dtype=np.float32,
        )

        x_test = np.asarray(
            split.x_test,
            dtype=np.float32,
        )

        y_test = np.asarray(
            split.y_test,
            dtype=np.float32,
        )

        y_train_class = (
            y_train
            > 0.0
        ).astype(
            np.int32
        )

        if len(
            np.unique(
                y_train_class
            )
        ) < 2:
            raise ValueError(
                "Train Dataset에 상승/비상승 "
                "두 클래스가 모두 필요합니다."
            )

        (
            regressor,
            classifier,
        ) = self._make_models()

        regressor.fit(
            x_train,
            y_train,
        )

        classifier.fit(
            x_train,
            y_train_class,
        )

        valid_return_prediction = (
            regressor.predict(
                x_valid
            )
        )

        test_return_prediction = (
            regressor.predict(
                x_test
            )
        )

        valid_probability = (
            classifier
            .predict_proba(
                x_valid
            )[:, 1]
        )

        test_probability = (
            classifier
            .predict_proba(
                x_test
            )[:, 1]
        )

        validation_regression = (
            self._regression_metrics(
                y_true=(
                    y_valid
                ),
                y_pred=(
                    valid_return_prediction
                ),
            )
        )

        test_regression = (
            self._regression_metrics(
                y_true=(
                    y_test
                ),
                y_pred=(
                    test_return_prediction
                ),
            )
        )

        validation_classification = (
            self._classification_metrics(
                y_true_return=(
                    y_valid
                ),
                probability=(
                    valid_probability
                ),
            )
        )

        test_classification = (
            self._classification_metrics(
                y_true_return=(
                    y_test
                ),
                probability=(
                    test_probability
                ),
            )
        )

        model_path = None

        if save_model:
            model_dir = Path(
                ".cache/models/phase6"
            )

            model_dir.mkdir(
                parents=True,
                exist_ok=True,
            )

            path = (
                model_dir
                / (
                    "historical_"
                    f"h{horizon}_"
                    f"{self.MODEL_VERSION}.joblib"
                )
            )

            joblib.dump(
                {
                    "model_name":
                        self.MODEL_NAME,

                    "model_version":
                        self.MODEL_VERSION,

                    "feature_version":
                        feature_version,

                    "feature_names":
                        split.feature_names,

                    "horizon":
                        horizon,

                    "regressor":
                        regressor,

                    "classifier":
                        classifier,

                    "train_first_date":
                        min(
                            split.train_dates
                        ),

                    "train_last_date":
                        max(
                            split.train_dates
                        ),

                    "validation_first_date":
                        min(
                            split.valid_dates
                        ),

                    "validation_last_date":
                        max(
                            split.valid_dates
                        ),

                    "test_first_date":
                        min(
                            split.test_dates
                        ),

                    "test_last_date":
                        max(
                            split.test_dates
                        ),
                },
                path,
            )

            model_path = str(
                path
            )

        return {
            "status":
                "pass",

            "model_name":
                self.MODEL_NAME,

            "model_version":
                self.MODEL_VERSION,

            "feature_version":
                feature_version,

            "target_horizon":
                horizon,

            "feature_count":
                len(
                    split.feature_names
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

            "test_rows":
                int(
                    x_test.shape[
                        0
                    ]
                ),

            "train_period": {
                "first":
                    min(
                        split.train_dates
                    ).isoformat(),

                "last":
                    max(
                        split.train_dates
                    ).isoformat(),
            },

            "validation_period": {
                "first":
                    min(
                        split.valid_dates
                    ).isoformat(),

                "last":
                    max(
                        split.valid_dates
                    ).isoformat(),
            },

            "test_period": {
                "first":
                    min(
                        split.test_dates
                    ).isoformat(),

                "last":
                    max(
                        split.test_dates
                    ).isoformat(),
            },

            "validation": {
                "regression":
                    validation_regression,

                "classification":
                    validation_classification,
            },

            "test": {
                "regression":
                    test_regression,

                "classification":
                    test_classification,
            },

            "regression_top_features":
                self._feature_importance(
                    model=(
                        regressor
                    ),
                    feature_names=(
                        split.feature_names
                    ),
                ),

            "classification_top_features":
                self._feature_importance(
                    model=(
                        classifier
                    ),
                    feature_names=(
                        split.feature_names
                    ),
                ),

            "model_path":
                model_path,
        }