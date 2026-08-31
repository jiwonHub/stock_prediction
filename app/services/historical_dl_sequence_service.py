from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sqlalchemy.orm import Session

from app.services.historical_dataset_service import (
    HistoricalDatasetService,
)
from app.services.historical_ml_training_service import (
    HistoricalMlTrainingService,
)
from app.services.historical_ml_walk_forward_service import (
    HistoricalMlWalkForwardService,
)


@dataclass
class HistoricalDlSequenceDataset:
    x: np.ndarray
    y: np.ndarray
    future_returns: np.ndarray
    stock_codes: list[str]
    feature_dates: list
    feature_names: list[str]
    sequence_length: int


class HistoricalDlSequenceService:
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

    def build_sequence_dataset(
        self,
        *,
        feature_version: str,
        horizon: int = 5,
        sequence_length: int = 20,
        include_locked_test: bool = False,
    ) -> HistoricalDlSequenceDataset:
        if sequence_length < 5:
            raise ValueError(
                "sequence_length는 "
                "5 이상이어야 합니다."
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

        split = (
            self.training_service
            .build_temporal_split(
                feature_version=(
                    feature_version
                ),
                horizon=horizon,
            )
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

        if include_locked_test:
            max_allowed_date = max(
                dataset.feature_dates
            )
        else:
            max_allowed_date = max(
                split.valid_dates
            )

        stock_row_indexes: dict[
            str,
            list[int],
        ] = {}

        for (
            row_index,
            stock_code,
        ) in enumerate(
            dataset.stock_codes
        ):
            feature_date = (
                dataset.feature_dates[
                    row_index
                ]
            )

            if (
                feature_date
                > max_allowed_date
            ):
                continue

            stock_row_indexes.setdefault(
                stock_code,
                [],
            ).append(
                row_index
            )

        sequence_x = []
        sequence_y = []
        sequence_future_returns = []
        sequence_stock_codes = []
        sequence_feature_dates = []

        for (
            stock_code,
            row_indexes,
        ) in stock_row_indexes.items():
            row_indexes.sort(
                key=lambda index: (
                    dataset.feature_dates[
                        index
                    ]
                )
            )

            if len(
                row_indexes
            ) < sequence_length:
                continue

            for end_position in range(
                sequence_length - 1,
                len(
                    row_indexes
                ),
            ):
                start_position = (
                    end_position
                    - sequence_length
                    + 1
                )

                window_indexes = (
                    row_indexes[
                        start_position:
                        end_position + 1
                    ]
                )

                target_index = (
                    row_indexes[
                        end_position
                    ]
                )

                window = np.asarray(
                    dataset.x[
                        window_indexes
                    ][
                        :,
                        selected_indexes
                    ],
                    dtype=np.float32,
                )

                if window.shape != (
                    sequence_length,
                    len(
                        selected_indexes
                    ),
                ):
                    continue

                if not np.all(
                    np.isfinite(
                        window
                    )
                ):
                    continue

                future_return = float(
                    dataset.y[
                        target_index
                    ]
                )

                if not np.isfinite(
                    future_return
                ):
                    continue

                sequence_x.append(
                    window
                )

                sequence_future_returns.append(
                    future_return
                )

                sequence_y.append(
                    1
                    if future_return > 0.0
                    else 0
                )

                sequence_stock_codes.append(
                    stock_code
                )

                sequence_feature_dates.append(
                    dataset.feature_dates[
                        target_index
                    ]
                )

        if not sequence_x:
            raise ValueError(
                "생성된 Sequence Dataset이 "
                "없습니다."
            )

        x = np.asarray(
            sequence_x,
            dtype=np.float32,
        )

        y = np.asarray(
            sequence_y,
            dtype=np.int64,
        )

        future_returns = np.asarray(
            sequence_future_returns,
            dtype=np.float32,
        )

        order = sorted(
            range(
                len(
                    sequence_feature_dates
                )
            ),
            key=lambda index: (
                sequence_feature_dates[
                    index
                ],
                sequence_stock_codes[
                    index
                ],
            ),
        )

        order_array = np.asarray(
            order,
            dtype=np.int64,
        )

        x = x[
            order_array
        ]

        y = y[
            order_array
        ]

        future_returns = (
            future_returns[
                order_array
            ]
        )

        stock_codes = [
            sequence_stock_codes[
                index
            ]
            for index
            in order
        ]

        feature_dates = [
            sequence_feature_dates[
                index
            ]
            for index
            in order
        ]

        return HistoricalDlSequenceDataset(
            x=x,
            y=y,
            future_returns=(
                future_returns
            ),
            stock_codes=(
                stock_codes
            ),
            feature_dates=(
                feature_dates
            ),
            feature_names=(
                selected_feature_names
            ),
            sequence_length=(
                sequence_length
            ),
        )

    def inspect(
        self,
        *,
        feature_version: str,
        horizon: int = 5,
        sequence_length: int = 20,
    ) -> dict:
        dataset = (
            self.build_sequence_dataset(
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

        unique_dates = sorted(
            set(
                dataset.feature_dates
            )
        )

        unique_stocks = set(
            dataset.stock_codes
        )

        return {
            "status":
                "pass",

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

            "x_shape":
                list(
                    dataset.x.shape
                ),

            "y_shape":
                list(
                    dataset.y.shape
                ),

            "samples":
                int(
                    len(
                        dataset.y
                    )
                ),

            "stocks":
                len(
                    unique_stocks
                ),

            "unique_dates":
                len(
                    unique_dates
                ),

            "first_date":
                min(
                    unique_dates
                ).isoformat(),

            "last_date":
                max(
                    unique_dates
                ).isoformat(),

            "positive_rate_pct":
                float(
                    np.mean(
                        dataset.y
                    )
                    * 100.0
                ),

            "locked_test_used":
                False,

            "feature_names":
                dataset.feature_names,
        }