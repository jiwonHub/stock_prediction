from __future__ import annotations

import random

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
)
from sklearn.preprocessing import StandardScaler
from sqlalchemy.orm import Session
from torch import nn
from torch.utils.data import (
    DataLoader,
    TensorDataset,
)

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


class StockMlp(nn.Module):
    def __init__(
        self,
        input_size: int,
    ):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(
                input_size,
                64,
            ),
            nn.ReLU(),
            nn.Dropout(
                0.20
            ),

            nn.Linear(
                64,
                32,
            ),
            nn.ReLU(),
            nn.Dropout(
                0.15
            ),

            nn.Linear(
                32,
                1,
            ),
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        return self.network(
            x
        ).squeeze(
            -1
        )


class HistoricalDlMlpService:
    RANDOM_STATE = 42

    EPOCHS = 60
    BATCH_SIZE = 512
    LEARNING_RATE = 0.001
    WEIGHT_DECAY = 0.0001

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
    def _device() -> torch.device:
        if (
            hasattr(
                torch.backends,
                "mps",
            )
            and torch.backends.mps.is_available()
        ):
            return torch.device(
                "mps"
            )

        if torch.cuda.is_available():
            return torch.device(
                "cuda"
            )

        return torch.device(
            "cpu"
        )

    @classmethod
    def _seed_everything(
        cls,
    ) -> None:
        random.seed(
            cls.RANDOM_STATE
        )

        np.random.seed(
            cls.RANDOM_STATE
        )

        torch.manual_seed(
            cls.RANDOM_STATE
        )

    @classmethod
    def _train_model(
        cls,
        *,
        x_train: np.ndarray,
        y_train: np.ndarray,
        input_size: int,
        device: torch.device,
    ) -> StockMlp:
        cls._seed_everything()

        model = StockMlp(
            input_size=input_size
        ).to(
            device
        )

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=cls.LEARNING_RATE,
            weight_decay=(
                cls.WEIGHT_DECAY
            ),
        )

        criterion = (
            nn.BCEWithLogitsLoss()
        )

        x_tensor = torch.tensor(
            x_train,
            dtype=torch.float32,
        )

        y_tensor = torch.tensor(
            y_train,
            dtype=torch.float32,
        )

        dataset = TensorDataset(
            x_tensor,
            y_tensor,
        )

        generator = (
            torch.Generator()
        )

        generator.manual_seed(
            cls.RANDOM_STATE
        )

        loader = DataLoader(
            dataset,
            batch_size=(
                cls.BATCH_SIZE
            ),
            shuffle=True,
            num_workers=0,
            generator=generator,
        )

        model.train()

        for _ in range(
            cls.EPOCHS
        ):
            for (
                batch_x,
                batch_y,
            ) in loader:
                batch_x = batch_x.to(
                    device
                )

                batch_y = batch_y.to(
                    device
                )

                optimizer.zero_grad(
                    set_to_none=True
                )

                logits = model(
                    batch_x
                )

                loss = criterion(
                    logits,
                    batch_y,
                )

                loss.backward()

                optimizer.step()

        return model

    @staticmethod
    def _predict_probability(
        *,
        model: StockMlp,
        x: np.ndarray,
        device: torch.device,
    ) -> np.ndarray:
        model.eval()

        probabilities = []

        with torch.no_grad():
            for start in range(
                0,
                len(x),
                2048,
            ):
                batch = torch.tensor(
                    x[
                        start:
                        start + 2048
                    ],
                    dtype=torch.float32,
                    device=device,
                )

                logits = model(
                    batch
                )

                probability = (
                    torch.sigmoid(
                        logits
                    )
                    .detach()
                    .cpu()
                    .numpy()
                )

                probabilities.append(
                    probability
                )

        return np.concatenate(
            probabilities
        ).astype(
            np.float64
        )

    def run_walk_forward(
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

        device = self._device()

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

            x_validation = x_all[
                validation_indexes
            ]

            y_train = (
                y_all[
                    train_indexes
                ]
                > 0.0
            ).astype(
                np.float32
            )

            y_validation = (
                y_all[
                    validation_indexes
                ]
                > 0.0
            ).astype(
                np.int32
            )

            scaler = StandardScaler()

            x_train_scaled = (
                scaler.fit_transform(
                    x_train
                )
                .astype(
                    np.float32
                )
            )

            x_validation_scaled = (
                scaler.transform(
                    x_validation
                )
                .astype(
                    np.float32
                )
            )

            model = self._train_model(
                x_train=(
                    x_train_scaled
                ),
                y_train=(
                    y_train
                ),
                input_size=len(
                    selected_indexes
                ),
                device=device,
            )

            probability = (
                self._predict_probability(
                    model=model,
                    x=(
                        x_validation_scaled
                    ),
                    device=device,
                )
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
                probability >= 0.50
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
                        (
                            accuracy
                            - majority_accuracy
                        ),

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

        auc_values = np.asarray(
            [
                row[
                    "roc_auc"
                ]
                for row
                in fold_results
            ],
            dtype=np.float64,
        )

        balanced_values = np.asarray(
            [
                row[
                    "balanced_accuracy_pct"
                ]
                for row
                in fold_results
            ],
            dtype=np.float64,
        )

        logloss_values = np.asarray(
            [
                row[
                    "log_loss"
                ]
                for row
                in fold_results
            ],
            dtype=np.float64,
        )

        brier_values = np.asarray(
            [
                row[
                    "brier_score"
                ]
                for row
                in fold_results
            ],
            dtype=np.float64,
        )

        return {
            "status":
                "pass",

            "model":
                "pytorch-mlp-baseline",

            "target_horizon":
                horizon,

            "feature_version":
                feature_version,

            "feature_strategy":
                "stock_internal_only",

            "feature_count":
                len(
                    selected_feature_names
                ),

            "architecture":
                "27 -> 64 -> 32 -> 1",

            "epochs":
                self.EPOCHS,

            "batch_size":
                self.BATCH_SIZE,

            "learning_rate":
                self.LEARNING_RATE,

            "device":
                str(
                    device
                ),

            "test_dataset_used":
                False,

            "locked_test_first_date":
                locked_test_first_date
                .isoformat(),

            "folds":
                fold_results,

            "summary": {
                "roc_auc_mean":
                    float(
                        np.mean(
                            auc_values
                        )
                    ),

                "roc_auc_std":
                    float(
                        np.std(
                            auc_values
                        )
                    ),

                "roc_auc_min":
                    float(
                        np.min(
                            auc_values
                        )
                    ),

                "roc_auc_max":
                    float(
                        np.max(
                            auc_values
                        )
                    ),

                "folds_auc_above_0_50":
                    int(
                        np.sum(
                            auc_values > 0.50
                        )
                    ),

                "folds_auc_above_0_52":
                    int(
                        np.sum(
                            auc_values > 0.52
                        )
                    ),

                "balanced_accuracy_mean_pct":
                    float(
                        np.mean(
                            balanced_values
                        )
                    ),

                "log_loss_mean":
                    float(
                        np.mean(
                            logloss_values
                        )
                    ),

                "brier_score_mean":
                    float(
                        np.mean(
                            brier_values
                        )
                    ),
            },
        }