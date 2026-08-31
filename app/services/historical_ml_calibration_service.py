from __future__ import annotations

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    brier_score_loss,
    log_loss,
    roc_auc_score,
)
from sqlalchemy.orm import Session

from app.services.historical_ml_training_service import (
    HistoricalMlTrainingService,
)


class HistoricalMlCalibrationService:
    BEST_PARAMS = {
        "n_estimators": 600,
        "max_depth": 3,
        "learning_rate": 0.03,
        "min_child_weight": 20,
        "subsample": 0.85,
        "colsample_bytree": 0.75,
        "reg_alpha": 0.25,
        "reg_lambda": 3.0,
    }

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
    def _make_classifier():
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
                HistoricalMlCalibrationService
                .RANDOM_STATE
            ),
            n_jobs=4,
            **(
                HistoricalMlCalibrationService
                .BEST_PARAMS
            ),
        )

    @staticmethod
    def _expected_calibration_error(
        *,
        y_true: np.ndarray,
        probability: np.ndarray,
        bins: int = 10,
    ) -> float:
        edges = np.linspace(
            0.0,
            1.0,
            bins + 1,
        )

        total = len(
            y_true
        )

        if total == 0:
            return 0.0

        ece = 0.0

        for index in range(
            bins
        ):
            lower = edges[
                index
            ]

            upper = edges[
                index + 1
            ]

            if index == (
                bins - 1
            ):
                mask = (
                    (probability >= lower)
                    & (probability <= upper)
                )
            else:
                mask = (
                    (probability >= lower)
                    & (probability < upper)
                )

            count = int(
                np.sum(
                    mask
                )
            )

            if count == 0:
                continue

            confidence = float(
                np.mean(
                    probability[
                        mask
                    ]
                )
            )

            accuracy = float(
                np.mean(
                    y_true[
                        mask
                    ]
                )
            )

            ece += (
                count
                / total
                * abs(
                    confidence
                    - accuracy
                )
            )

        return float(
            ece
        )

    @staticmethod
    def _metrics(
        *,
        y_true: np.ndarray,
        probability: np.ndarray,
    ) -> dict:
        probability = np.clip(
            np.asarray(
                probability,
                dtype=np.float64,
            ),
            1e-7,
            1.0 - 1e-7,
        )

        y_true = np.asarray(
            y_true,
            dtype=np.int32,
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

        return {
            "roc_auc":
                auc,

            "log_loss":
                float(
                    log_loss(
                        y_true,
                        probability,
                        labels=[
                            0,
                            1,
                        ],
                    )
                ),

            "brier_score":
                float(
                    brier_score_loss(
                        y_true,
                        probability,
                    )
                ),

            "ece_10":
                (
                    HistoricalMlCalibrationService
                    ._expected_calibration_error(
                        y_true=y_true,
                        probability=(
                            probability
                        ),
                        bins=10,
                    )
                ),

            "probability_mean":
                float(
                    np.mean(
                        probability
                    )
                ),

            "actual_positive_rate":
                float(
                    np.mean(
                        y_true
                    )
                ),
        }

    @staticmethod
    def _logit(
        probability: np.ndarray,
    ) -> np.ndarray:
        probability = np.clip(
            probability,
            1e-6,
            1.0 - 1e-6,
        )

        return np.log(
            probability
            / (
                1.0
                - probability
            )
        )

    def inspect_calibration(
        self,
        *,
        feature_version: str,
        horizon: int = 5,
        calibration_ratio: float = 0.60,
    ) -> dict:
        if not (
            0.50
            <= calibration_ratio
            <= 0.80
        ):
            raise ValueError(
                "calibration_ratio는 "
                "0.50~0.80 범위여야 합니다."
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

        x_train = np.asarray(
            split.x_train[
                :,
                selected_indexes
            ],
            dtype=np.float32,
        )

        y_train = (
            np.asarray(
                split.y_train
            )
            > 0.0
        ).astype(
            np.int32
        )

        x_valid = np.asarray(
            split.x_valid[
                :,
                selected_indexes
            ],
            dtype=np.float32,
        )

        y_valid = (
            np.asarray(
                split.y_valid
            )
            > 0.0
        ).astype(
            np.int32
        )

        valid_dates = list(
            split.valid_dates
        )

        unique_dates = sorted(
            set(
                valid_dates
            )
        )

        cut = int(
            len(
                unique_dates
            )
            * calibration_ratio
        )

        if (
            cut <= 0
            or cut >= len(
                unique_dates
            )
        ):
            raise ValueError(
                "Calibration 날짜 분리에 실패했습니다."
            )

        calibration_dates = set(
            unique_dates[
                :cut
            ]
        )

        evaluation_dates = set(
            unique_dates[
                cut:
            ]
        )

        calibration_indexes = [
            index
            for (
                index,
                feature_date,
            )
            in enumerate(
                valid_dates
            )
            if feature_date
            in calibration_dates
        ]

        evaluation_indexes = [
            index
            for (
                index,
                feature_date,
            )
            in enumerate(
                valid_dates
            )
            if feature_date
            in evaluation_dates
        ]

        classifier = (
            self._make_classifier()
        )

        classifier.fit(
            x_train,
            y_train,
        )

        raw_probability = (
            classifier
            .predict_proba(
                x_valid
            )[:, 1]
        )

        calibration_probability = (
            raw_probability[
                calibration_indexes
            ]
        )

        calibration_y = (
            y_valid[
                calibration_indexes
            ]
        )

        evaluation_probability = (
            raw_probability[
                evaluation_indexes
            ]
        )

        evaluation_y = (
            y_valid[
                evaluation_indexes
            ]
        )

        platt = LogisticRegression(
            random_state=(
                self.RANDOM_STATE
            ),
        )

        platt.fit(
            self._logit(
                calibration_probability
            ).reshape(
                -1,
                1,
            ),
            calibration_y,
        )

        sigmoid_probability = (
            platt
            .predict_proba(
                self._logit(
                    evaluation_probability
                ).reshape(
                    -1,
                    1,
                )
            )[:, 1]
        )

        isotonic = (
            IsotonicRegression(
                out_of_bounds="clip",
            )
        )

        isotonic.fit(
            calibration_probability,
            calibration_y,
        )

        isotonic_probability = (
            isotonic.predict(
                evaluation_probability
            )
        )

        raw_metrics = (
            self._metrics(
                y_true=(
                    evaluation_y
                ),
                probability=(
                    evaluation_probability
                ),
            )
        )

        sigmoid_metrics = (
            self._metrics(
                y_true=(
                    evaluation_y
                ),
                probability=(
                    sigmoid_probability
                ),
            )
        )

        isotonic_metrics = (
            self._metrics(
                y_true=(
                    evaluation_y
                ),
                probability=(
                    isotonic_probability
                ),
            )
        )

        candidates = {
            "raw":
                raw_metrics,

            "sigmoid":
                sigmoid_metrics,

            "isotonic":
                isotonic_metrics,
        }

        ranked = sorted(
            candidates.items(),
            key=lambda item: (
                item[
                    1
                ][
                    "brier_score"
                ],
                item[
                    1
                ][
                    "log_loss"
                ],
                item[
                    1
                ][
                    "ece_10"
                ],
            ),
        )

        best_name = ranked[
            0
        ][
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
                    selected_indexes
                ),

            "model_candidate":
                "strong_regularization",

            "model_params":
                dict(
                    self.BEST_PARAMS
                ),

            "validation_unique_dates":
                len(
                    unique_dates
                ),

            "calibration_ratio":
                calibration_ratio,

            "calibration_rows":
                len(
                    calibration_indexes
                ),

            "evaluation_rows":
                len(
                    evaluation_indexes
                ),

            "calibration_period": {
                "first":
                    min(
                        calibration_dates
                    ).isoformat(),

                "last":
                    max(
                        calibration_dates
                    ).isoformat(),
            },

            "evaluation_period": {
                "first":
                    min(
                        evaluation_dates
                    ).isoformat(),

                "last":
                    max(
                        evaluation_dates
                    ).isoformat(),
            },

            "selection_rule":
                (
                    "Calibration evaluation 구간의 "
                    "Brier Score 우선, "
                    "동률 시 LogLoss, ECE 사용. "
                    "Test Dataset 미사용."
                ),

            "raw":
                raw_metrics,

            "sigmoid":
                sigmoid_metrics,

            "isotonic":
                isotonic_metrics,

            "best_calibration":
                best_name,
        }