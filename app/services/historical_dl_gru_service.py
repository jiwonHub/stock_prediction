from __future__ import annotations

import random

import numpy as np
import torch
from sklearn.metrics import (
    balanced_accuracy_score,
    log_loss,
    roc_auc_score,
)
from sqlalchemy.orm import Session
from torch import nn
from torch.utils.data import (
    DataLoader,
    TensorDataset,
)

from app.services.historical_dl_sequence_service import (
    HistoricalDlSequenceService,
)
from app.services.historical_ml_training_service import (
    HistoricalMlTrainingService,
)
from app.services.historical_ml_oof_service import (
    HistoricalMlOofService,
)


class _StockGruClassifier(nn.Module):
    def __init__(
        self,
        *,
        input_size: int,
        hidden_size: int = 64,
    ):
        super().__init__()

        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=2,
            batch_first=True,
            dropout=0.20,
        )

        self.head = nn.Sequential(
            nn.LayerNorm(
                hidden_size
            ),
            nn.Linear(
                hidden_size,
                32,
            ),
            nn.GELU(),
            nn.Dropout(
                0.20
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
        output, _ = self.gru(
            x
        )

        last_hidden = output[
            :,
            -1,
            :
        ]

        return self.head(
            last_hidden
        ).squeeze(
            -1
        )


class HistoricalDlGruService:
    RANDOM_STATE = 42

    def __init__(
        self,
        db: Session,
    ):
        self.db = db

        self.sequence_service = (
            HistoricalDlSequenceService(
                db
            )
        )

        self.training_service = (
            HistoricalMlTrainingService(
                db
            )
        )

    @staticmethod
    def _device() -> torch.device:
        if (
            torch.backends.mps
            .is_available()
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
    def _set_seed(
        cls,
        seed_offset: int = 0,
    ) -> None:
        seed = (
            cls.RANDOM_STATE
            + seed_offset
        )

        random.seed(
            seed
        )

        np.random.seed(
            seed
        )

        torch.manual_seed(
            seed
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
    def _normalize(
        *,
        x_train: np.ndarray,
        x_validation: np.ndarray,
    ) -> tuple[
        np.ndarray,
        np.ndarray,
    ]:
        mean = np.mean(
            x_train,
            axis=(
                0,
                1,
            ),
            keepdims=True,
        )

        std = np.std(
            x_train,
            axis=(
                0,
                1,
            ),
            keepdims=True,
        )

        std = np.where(
            std < 1e-6,
            1.0,
            std,
        )

        normalized_train = (
            x_train
            - mean
        ) / std

        normalized_validation = (
            x_validation
            - mean
        ) / std

        return (
            normalized_train.astype(
                np.float32
            ),
            normalized_validation.astype(
                np.float32
            ),
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

    def run_walk_forward(
        self,
        *,
        feature_version: str,
        horizon: int = 5,
        sequence_length: int = 20,
        folds: int = 4,
        initial_train_ratio: float = 0.55,
        epochs: int = 15,
        batch_size: int = 256,
    ) -> dict:
        if folds < 3:
            raise ValueError(
                "folds는 3 이상이어야 합니다."
            )

        dataset = (
            self.sequence_service
            .build_sequence_dataset(
                feature_version=(
                    feature_version
                ),
                horizon=horizon,
                sequence_length=(
                    sequence_length
                ),
                include_locked_test=False,
            )
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

        locked_test_first_date = min(
            split.test_dates
        )

        unique_dates = sorted(
            set(
                dataset.feature_dates
            )
        )

        initial_train_count = int(
            len(
                unique_dates
            )
            * initial_train_ratio
        )

        remaining_count = (
            len(
                unique_dates
            )
            - initial_train_count
        )

        fold_size = (
            remaining_count
            // folds
        )

        if fold_size <= horizon:
            raise ValueError(
                "Fold 기간이 너무 짧습니다."
            )

        device = self._device()

        fold_results = []

        all_probabilities = []
        all_future_returns = []
        all_stock_codes = []
        all_feature_dates = []

        for fold_index in range(
            folds
        ):
            self._set_seed(
                fold_index
            )

            validation_start_index = (
                initial_train_count
                + (
                    fold_index
                    * fold_size
                )
            )

            if fold_index == (
                folds - 1
            ):
                validation_end_index = (
                    len(
                        unique_dates
                    )
                )
            else:
                validation_end_index = (
                    validation_start_index
                    + fold_size
                )

            train_end_index = (
                validation_start_index
                - horizon
            )

            train_dates = set(
                unique_dates[
                    :train_end_index
                ]
            )

            purge_dates = (
                unique_dates[
                    train_end_index:
                    validation_start_index
                ]
            )

            validation_dates = set(
                unique_dates[
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

            x_train = np.asarray(
                dataset.x[
                    train_indexes
                ],
                dtype=np.float32,
            )

            y_train = np.asarray(
                dataset.y[
                    train_indexes
                ],
                dtype=np.float32,
            )

            x_validation = np.asarray(
                dataset.x[
                    validation_indexes
                ],
                dtype=np.float32,
            )

            y_validation = np.asarray(
                dataset.y[
                    validation_indexes
                ],
                dtype=np.int32,
            )

            (
                x_train,
                x_validation,
            ) = self._normalize(
                x_train=x_train,
                x_validation=(
                    x_validation
                ),
            )

            train_dataset = (
                TensorDataset(
                    torch.from_numpy(
                        x_train
                    ),
                    torch.from_numpy(
                        y_train
                    ),
                )
            )

            loader_generator = (
                torch.Generator()
            )

            loader_generator.manual_seed(
                self.RANDOM_STATE
                + fold_index
            )

            train_loader = DataLoader(
                train_dataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=0,
                generator=(
                    loader_generator
                ),
            )

            model = (
                _StockGruClassifier(
                    input_size=(
                        len(
                            dataset.feature_names
                        )
                    ),
                    hidden_size=64,
                )
                .to(
                    device
                )
            )

            criterion = (
                nn.BCEWithLogitsLoss()
            )

            optimizer = (
                torch.optim.AdamW(
                    model.parameters(),
                    lr=1e-3,
                    weight_decay=1e-4,
                )
            )

            for _ in range(
                epochs
            ):
                model.train()

                for (
                    batch_x,
                    batch_y,
                ) in train_loader:
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

                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(),
                        max_norm=1.0,
                    )

                    optimizer.step()

            model.eval()

            probabilities = []

            validation_tensor = (
                torch.from_numpy(
                    x_validation
                )
            )

            validation_loader = (
                DataLoader(
                    TensorDataset(
                        validation_tensor
                    ),
                    batch_size=(
                        batch_size
                    ),
                    shuffle=False,
                    num_workers=0,
                )
            )

            with torch.no_grad():
                for (
                    batch_x,
                ) in validation_loader:
                    batch_x = batch_x.to(
                        device
                    )

                    logits = model(
                        batch_x
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

            probability = np.concatenate(
                probabilities
            ).astype(
                np.float64
            )

            all_probabilities.append(
                probability
            )

            all_future_returns.append(
                np.asarray(
                    dataset.future_returns[
                        validation_indexes
                    ],
                    dtype=np.float32,
                )
            )

            all_stock_codes.extend(
                [
                    dataset.stock_codes[
                        index
                    ]
                    for index
                    in validation_indexes
                ]
            )

            all_feature_dates.extend(
                [
                    dataset.feature_dates[
                        index
                    ]
                    for index
                    in validation_indexes
                ]
            )

            prediction = (
                probability
                >= 0.50
            ).astype(
                np.int32
            )

            auc = float(
                roc_auc_score(
                    y_validation,
                    probability,
                )
            )

            balanced = float(
                balanced_accuracy_score(
                    y_validation,
                    prediction,
                )
                * 100.0
            )

            loss_value = float(
                log_loss(
                    y_validation,
                    np.clip(
                        probability,
                        1e-7,
                        1.0 - 1e-7,
                    ),
                    labels=[
                        0,
                        1,
                    ],
                )
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

                    "purge_period": {
                        "dates":
                            len(
                                purge_dates
                            ),

                        "first":
                            purge_dates[
                                0
                            ].isoformat(),

                        "last":
                            purge_dates[
                                -1
                            ].isoformat(),
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
                        auc,

                    "balanced_accuracy_pct":
                        balanced,

                    "log_loss":
                        loss_value,

                    "probability_mean":
                        float(
                            np.mean(
                                probability
                            )
                        ),

                    "actual_positive_rate":
                        float(
                            np.mean(
                                y_validation
                            )
                        ),
                }
            )

            del model

            if device.type == "mps":
                torch.mps.empty_cache()

            elif device.type == "cuda":
                torch.cuda.empty_cache()

        oof_probability = np.concatenate(
            all_probabilities
        )

        oof_future_returns = np.concatenate(
            all_future_returns
        )

        ranking_metrics = (
            HistoricalMlOofService
            ._ranking_metrics(
                probabilities=(
                    oof_probability
                ),
                future_returns=(
                    oof_future_returns
                ),
                stock_codes=(
                    all_stock_codes
                ),
                feature_dates=(
                    all_feature_dates
                ),
                rebalance_step=horizon,
            )
        )

        auc_values = [
            row["roc_auc"]
            for row
            in fold_results
        ]

        balanced_values = [
            row[
                "balanced_accuracy_pct"
            ]
            for row
            in fold_results
        ]

        logloss_values = [
            row["log_loss"]
            for row
            in fold_results
        ]

        return {
            "status":
                "pass",

            "model":
                "gru",

            "feature_version":
                feature_version,

            "target_horizon":
                horizon,

            "feature_strategy":
                "stock_internal_only",

            "sequence_length":
                sequence_length,

            "feature_count":
                len(
                    dataset.feature_names
                ),

            "device":
                str(
                    device
                ),

            "epochs":
                epochs,

            "fold_count":
                folds,

            "test_dataset_used":
                False,

            "locked_test_first_date":
                locked_test_first_date
                .isoformat(),

            "folds":
                fold_results,

            "oof_ranking_metrics":
                ranking_metrics,

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

                "folds_auc_above_0_50":
                    sum(
                        value > 0.50
                        for value
                        in auc_values
                    ),

                "folds_auc_above_0_52":
                    sum(
                        value > 0.52
                        for value
                        in auc_values
                    ),

                "balanced_accuracy_mean_pct":
                    self._mean(
                        balanced_values
                    ),

                "log_loss_mean":
                    self._mean(
                        logloss_values
                    ),
            },
        }