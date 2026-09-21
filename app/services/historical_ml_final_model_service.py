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

    REFIT_VERSION = (
        "phase7-full-refit-v1"
    )

    PRODUCTION_ML_WEIGHT = 0.05

    LOCKED_TEST_PERIOD = {
        "first_date": "2026-03-27",
        "last_date": "2026-09-07",
        "rows": 10989,
        "stocks": 99,
        "unique_dates": 111,
    }

    LOCKED_TEST_METRICS = {
        "roc_auc": 0.5207027187097952,
        "log_loss": 0.6990452951752307,
        "brier_score": 0.2527430470513363,
        "ece_10": 0.05387913951319229,
        "spearman_ic_mean": 0.049837966113434655,
        "spearman_ic_positive_rate_pct": 62.16216216216216,
        "top10_excess_mean_pct": 0.6749339314148266,
        "top10_excess_positive_rate_pct": 57.65765765765766,
        "top20_excess_mean_pct": 0.3474987019282773,
        "top20_excess_positive_rate_pct": 49.549549549549546,
        "long_short_10_mean_pct": 0.6991332580402561,
        "long_short_10_positive_rate_pct": 52.25225225225225,
        "top10_cost20_net_excess_pct": 0.5313862393740749,
        "top10_cost20_positive_offsets": 5,
        "top10_cost20_total_offsets": 5,
        "buffer_cost20_net_excess_pct": 0.6910665616668437,
        "buffer_cost20_positive_offsets": 5,
        "buffer_cost20_total_offsets": 5,
        "buffer_cumulative_excess_pct": 15.89906476144653,
        "buffer_max_drawdown_pct": -32.24993819283844,
        "buffer_turnover_pct": 57.66798418972333,
    }

    FINAL_SELECTION = {
        "decision":
            "keep_production_baseline_27",

        "feature_count":
            27,

        "removed_features":
            [],

        "required_feature":
            "rsi_14",

        "objective":
            "binary:logistic",

        "ml_weight":
            0.05,

        "locked_test_consumed":
            True,

        "locked_test_consumed_through":
            "2026-09-07",
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

        if len(
            selected_feature_names
        ) != 27:
            raise RuntimeError(
                "Phase 7 Production Feature 개수가 "
                "27개가 아닙니다: "
                f"{len(selected_feature_names)}"
            )

        if "rsi_14" not in selected_feature_names:
            raise RuntimeError(
                "Phase 7 Production Feature에 "
                "rsi_14가 없습니다."
            )

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

            "refit_version":
                self.REFIT_VERSION,

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

            "locked_test_period":
                dict(
                    self.LOCKED_TEST_PERIOD
                ),

            "locked_test_metrics":
                dict(
                    self.LOCKED_TEST_METRICS
                ),

            "final_selection":
                dict(
                    self.FINAL_SELECTION
                ),

            "next_holdout_must_start_after":
                max(
                    dataset.feature_dates
                ).isoformat(),

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

            "refit_version":
                self.REFIT_VERSION,

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

            "locked_test_period":
                dict(
                    self.LOCKED_TEST_PERIOD
                ),

            "locked_test_metrics":
                dict(
                    self.LOCKED_TEST_METRICS
                ),

            "final_selection":
                dict(
                    self.FINAL_SELECTION
                ),

            "next_holdout_must_start_after":
                max(
                    dataset.feature_dates
                ).isoformat(),

            "training_policy":
                "full_dataset_after_locked_test_consumed",
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