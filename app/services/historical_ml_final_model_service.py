from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
from sqlalchemy.orm import Session

from app.services.historical_dataset_service import (
    HistoricalDatasetService,
)
from app.services.historical_ml_oof_service import (
    HistoricalMlOofService,
)
from app.services.historical_ml_walk_forward_service import (
    HistoricalMlWalkForwardService,
)


class HistoricalMlFinalModelService:
    MODEL_NAME = "xgboost-ranking"
    MODEL_VERSION = "phase6-ranking-v1"

    FEATURE_VERSION = (
        "v5-historical-disclosure-1"
    )

    HORIZON = 5

    FEATURE_STRATEGY = (
        "stock_internal_only"
    )

    CALIBRATION = "raw"

    INTENDED_USE = (
        "top100_cross_sectional_ranking_factor"
    )

    MODEL_PATH = Path(
        ".cache/models/phase6/"
        "historical_h5_phase6-ranking-v1.joblib"
    )

    LOCKED_TEST_METRICS = {
        "roc_auc": 0.523536,
        "balanced_accuracy_pct": 51.7874,
        "spearman_ic_mean": 0.046998,
        "spearman_ic_positive_rate_pct": 55.0,
        "top10_excess_mean_pct": 0.7394,
        "top10_excess_positive_rate_pct": 57.0,
        "top20_excess_mean_pct": 0.4442,
        "long_short_10_mean_pct": 0.4283,
        "non_overlapping_positive_offsets": 4,
        "non_overlapping_total_offsets": 5,
        "non_overlapping_min_excess_pct": -0.2566,
    }

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

    def train_final_model(
        self,
    ) -> dict:
        dataset = (
            self.dataset_service
            .build_dataset(
                feature_version=(
                    self.FEATURE_VERSION
                ),
                horizon=self.HORIZON,
                feature_strategy=(
                    self.FEATURE_STRATEGY
                ),
            )
        )

        selected_indexes = (
            HistoricalMlWalkForwardService
            ._selected_indexes(
                feature_names=(
                    dataset.feature_names
                ),
                feature_strategy=(
                    self.FEATURE_STRATEGY
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

        x = np.asarray(
            dataset.x[
                :,
                selected_indexes
            ],
            dtype=np.float32,
        )

        y_return = np.asarray(
            dataset.y,
            dtype=np.float32,
        )

        y = (
            y_return > 0.0
        ).astype(
            np.int32
        )

        classifier = (
            HistoricalMlOofService
            ._make_classifier()
        )

        classifier.fit(
            x,
            y,
        )

        artifact = {
            "model_name":
                self.MODEL_NAME,

            "model_version":
                self.MODEL_VERSION,

            "created_at":
                datetime.now(
                    timezone.utc
                ).isoformat(),

            "feature_version":
                self.FEATURE_VERSION,

            "horizon":
                self.HORIZON,

            "feature_strategy":
                self.FEATURE_STRATEGY,

            "feature_names":
                selected_feature_names,

            "feature_count":
                len(
                    selected_feature_names
                ),

            "calibration":
                self.CALIBRATION,

            "intended_use":
                self.INTENDED_USE,

            "classifier":
                classifier,

            "model_params":
                dict(
                    HistoricalMlOofService
                    .FINAL_PARAMS
                ),

            "training_rows":
                int(
                    x.shape[
                        0
                    ]
                ),

            "training_first_date":
                min(
                    dataset.feature_dates
                ).isoformat(),

            "training_last_date":
                max(
                    dataset.feature_dates
                ).isoformat(),

            "locked_test_metrics":
                dict(
                    self.LOCKED_TEST_METRICS
                ),

            "production_policy": {
                "use_for_return_regression":
                    False,

                "use_as_hard_direction_classifier":
                    False,

                "use_as_cross_sectional_ranking_factor":
                    True,

                "diagnostic_threshold":
                    0.54,
            },
        }

        self.MODEL_PATH.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        joblib.dump(
            artifact,
            self.MODEL_PATH,
        )

        return {
            "status":
                "pass",

            "model_path":
                str(
                    self.MODEL_PATH
                ),

            "model_name":
                self.MODEL_NAME,

            "model_version":
                self.MODEL_VERSION,

            "feature_version":
                self.FEATURE_VERSION,

            "feature_strategy":
                self.FEATURE_STRATEGY,

            "feature_count":
                len(
                    selected_feature_names
                ),

            "feature_names":
                selected_feature_names,

            "training_rows":
                int(
                    x.shape[
                        0
                    ]
                ),

            "training_first_date":
                min(
                    dataset.feature_dates
                ).isoformat(),

            "training_last_date":
                max(
                    dataset.feature_dates
                ).isoformat(),

            "intended_use":
                self.INTENDED_USE,

            "calibration":
                self.CALIBRATION,

            "model_params":
                dict(
                    HistoricalMlOofService
                    .FINAL_PARAMS
                ),

            "locked_test_metrics":
                dict(
                    self.LOCKED_TEST_METRICS
                ),
        }

    @classmethod
    def load_artifact(
        cls,
    ) -> dict:
        if not cls.MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Final ML 모델이 없습니다: "
                f"{cls.MODEL_PATH}"
            )

        artifact = joblib.load(
            cls.MODEL_PATH
        )

        if (
            artifact.get(
                "model_version"
            )
            != cls.MODEL_VERSION
        ):
            raise RuntimeError(
                "Final ML 모델 버전이 "
                "일치하지 않습니다."
            )

        return artifact