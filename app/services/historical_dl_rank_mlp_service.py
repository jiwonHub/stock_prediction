from __future__ import annotations

import random

import numpy as np
import torch
from scipy.stats import rankdata
from sqlalchemy.orm import Session
from torch import nn
from torch.utils.data import (
    DataLoader,
    TensorDataset,
)

from app.services.historical_dataset_service import (
    HistoricalDatasetService,
)
from app.services.historical_ml_oof_service import (
    HistoricalMlOofService,
)
from app.services.historical_ml_training_service import (
    HistoricalMlTrainingService,
)
from app.services.historical_ml_walk_forward_service import (
    HistoricalMlWalkForwardService,
)


class _RankMlp(nn.Module):
    def __init__(
        self,
        *,
        input_size: int,
    ):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(
                input_size,
                64,
            ),
            nn.LayerNorm(
                64
            ),
            nn.GELU(),
            nn.Dropout(
                0.20
            ),
            nn.Linear(
                64,
                32,
            ),
            nn.GELU(),
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


class HistoricalDlRankMlpService:
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
    def _build_rank_targets(
        *,
        future_returns: np.ndarray,
        feature_dates: list,
    ) -> np.ndarray:
        targets = np.zeros(
            len(
                future_returns
            ),
            dtype=np.float32,
        )

        grouped: dict = {}

        for (
            index,
            feature_date,
        ) in enumerate(
            feature_dates
        ):
            grouped.setdefault(
                feature_date,
                [],
            ).append(
                index
            )

        for indexes in grouped.values():
            index_array = np.asarray(
                indexes,
                dtype=np.int64,
            )

            values = future_returns[
                index_array
            ]

            if len(
                values
            ) <= 1:
                targets[
                    index_array
                ] = 0.0

                continue

            ranks = rankdata(
                values,
                method="average",
            )

            percentile = (
                ranks - 1.0
            ) / (
                len(
                    ranks
                )
                - 1.0
            )

            normalized_rank = (
                percentile
                * 2.0
                - 1.0
            )

            targets[
                index_array
            ] = normalized_rank.astype(
                np.float32
            )

        return targets

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
            axis=0,
            keepdims=True,
        )

        std = np.std(
            x_train,
            axis=0,
            keepdims=True,
        )

        std = np.where(
            std < 1e-6,
            1.0,
            std,
        )

        return (
            (
                (
                    x_train
                    - mean
                )
                / std
            ).astype(
                np.float32
            ),
            (
                (
                    x_validation
                    - mean
                )
                / std
            ).astype(
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
        folds: int = 4,
        initial_train_ratio: float = 0.55,
        epochs: int = 20,
        batch_size: int = 512,
    ) -> dict:
        dataset = (
            self.dataset_service
            .build_dataset(
                feature_version=(
                    feature_version
                ),
                horizon=horizon,
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

        walk_forward_last_date = max(
            split.valid_dates
        )

        locked_test_first_date = min(
            split.test_dates
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

        eligible_indexes = [
            index
            for (
                index,
                feature_date,
            )
            in enumerate(
                dataset.feature_dates
            )
            if feature_date
            <= walk_forward_last_date
        ]

        x = np.asarray(
            dataset.x[
                eligible_indexes
            ][
                :,
                selected_indexes
            ],
            dtype=np.float32,
        )

        future_returns = np.asarray(
            dataset.y[
                eligible_indexes
            ],
            dtype=np.float32,
        )

        feature_dates = [
            dataset.feature_dates[
                index
            ]
            for index
            in eligible_indexes
        ]

        stock_codes = [
            dataset.stock_codes[
                index
            ]
            for index
            in eligible_indexes
        ]

        rank_targets = (
            self._build_rank_targets(
                future_returns=(
                    future_returns
                ),
                feature_dates=(
                    feature_dates
                ),
            )
        )

        unique_dates = sorted(
            set(
                feature_dates
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

        device = self._device()

        fold_results = []

        all_scores = []
        all_future_returns = []
        all_stock_codes = []
        all_feature_dates = []

        for fold_index in range(
            folds
        ):
            self._set_seed(
                fold_index
            )

            validation_start = (
                initial_train_count
                + (
                    fold_index
                    * fold_size
                )
            )

            if fold_index == (
                folds - 1
            ):
                validation_end = len(
                    unique_dates
                )
            else:
                validation_end = (
                    validation_start
                    + fold_size
                )

            train_end = (
                validation_start
                - horizon
            )

            train_dates = set(
                unique_dates[
                    :train_end
                ]
            )

            validation_dates = set(
                unique_dates[
                    validation_start:
                    validation_end
                ]
            )

            train_indexes = (
                self._indexes_for_dates(
                    feature_dates=(
                        feature_dates
                    ),
                    allowed_dates=(
                        train_dates
                    ),
                )
            )

            validation_indexes = (
                self._indexes_for_dates(
                    feature_dates=(
                        feature_dates
                    ),
                    allowed_dates=(
                        validation_dates
                    ),
                )
            )

            x_train = x[
                train_indexes
            ]

            y_train = rank_targets[
                train_indexes
            ]

            x_validation = x[
                validation_indexes
            ]

            (
                x_train,
                x_validation,
            ) = self._normalize(
                x_train=x_train,
                x_validation=(
                    x_validation
                ),
            )

            train_dataset = TensorDataset(
                torch.from_numpy(
                    x_train
                ),
                torch.from_numpy(
                    y_train
                ),
            )

            generator = torch.Generator()

            generator.manual_seed(
                self.RANDOM_STATE
                + fold_index
            )

            train_loader = DataLoader(
                train_dataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=0,
                generator=generator,
            )

            model = (
                _RankMlp(
                    input_size=len(
                        selected_indexes
                    ),
                )
                .to(
                    device
                )
            )

            criterion = (
                nn.SmoothL1Loss(
                    beta=0.20
                )
            )

            optimizer = (
                torch.optim.AdamW(
                    model.parameters(),
                    lr=5e-4,
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

                    score = model(
                        batch_x
                    )

                    loss = criterion(
                        score,
                        batch_y,
                    )

                    loss.backward()

                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(),
                        max_norm=1.0,
                    )

                    optimizer.step()

            model.eval()

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
                    batch_size=batch_size,
                    shuffle=False,
                    num_workers=0,
                )
            )

            score_batches = []

            with torch.no_grad():
                for (
                    batch_x,
                ) in validation_loader:
                    batch_x = batch_x.to(
                        device
                    )

                    score = model(
                        batch_x
                    )

                    score_batches.append(
                        score
                        .detach()
                        .cpu()
                        .numpy()
                    )

            scores = np.concatenate(
                score_batches
            ).astype(
                np.float64
            )

            validation_returns = (
                future_returns[
                    validation_indexes
                ]
            )

            validation_stock_codes = [
                stock_codes[
                    index
                ]
                for index
                in validation_indexes
            ]

            validation_feature_dates = [
                feature_dates[
                    index
                ]
                for index
                in validation_indexes
            ]

            ranking = (
                HistoricalMlOofService
                ._ranking_metrics(
                    probabilities=scores,
                    future_returns=(
                        validation_returns
                    ),
                    stock_codes=(
                        validation_stock_codes
                    ),
                    feature_dates=(
                        validation_feature_dates
                    ),
                    rebalance_step=horizon,
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

                    "validation_first":
                        min(
                            validation_dates
                        ).isoformat(),

                    "validation_last":
                        max(
                            validation_dates
                        ).isoformat(),

                    "ic_mean":
                        ranking[
                            "spearman_ic_mean"
                        ],

                    "ic_positive_rate_pct":
                        ranking[
                            "spearman_ic_positive_rate_pct"
                        ],

                    "top10_excess_mean_pct":
                        ranking[
                            "top10_excess_mean_pct"
                        ],

                    "top10_excess_positive_rate_pct":
                        ranking[
                            "top10_excess_positive_rate_pct"
                        ],

                    "long_short_10_mean_pct":
                        ranking[
                            "long_short_10_mean_pct"
                        ],
                }
            )

            all_scores.append(
                scores
            )

            all_future_returns.append(
                validation_returns
            )

            all_stock_codes.extend(
                validation_stock_codes
            )

            all_feature_dates.extend(
                validation_feature_dates
            )

            del model

            if device.type == "mps":
                torch.mps.empty_cache()

            elif device.type == "cuda":
                torch.cuda.empty_cache()

        oof_scores = np.concatenate(
            all_scores
        )

        oof_returns = np.concatenate(
            all_future_returns
        )

        oof_ranking = (
            HistoricalMlOofService
            ._ranking_metrics(
                probabilities=(
                    oof_scores
                ),
                future_returns=(
                    oof_returns
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

        fold_ic_values = [
            row["ic_mean"]
            for row
            in fold_results
        ]

        fold_top10_values = [
            row[
                "top10_excess_mean_pct"
            ]
            for row
            in fold_results
        ]

        return {
            "status":
                "pass",

            "model":
                "mlp_cross_sectional_rank",

            "target":
                "5d_cross_sectional_return_percentile",

            "target_range":
                [
                    -1.0,
                    1.0,
                ],

            "feature_version":
                feature_version,

            "feature_strategy":
                "stock_internal_only",

            "feature_count":
                len(
                    selected_indexes
                ),

            "target_horizon":
                horizon,

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

            "oof_ranking_metrics":
                oof_ranking,

            "summary": {
                "fold_ic_mean":
                    self._mean(
                        fold_ic_values
                    ),

                "fold_ic_std":
                    self._std(
                        fold_ic_values
                    ),

                "fold_ic_positive":
                    sum(
                        value > 0.0
                        for value
                        in fold_ic_values
                    ),

                "fold_top10_excess_mean_pct":
                    self._mean(
                        fold_top10_values
                    ),

                "fold_top10_positive":
                    sum(
                        value > 0.0
                        for value
                        in fold_top10_values
                    ),
            },
        }