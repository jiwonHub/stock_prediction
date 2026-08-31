from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import numpy as np
from sqlalchemy.orm import Session

from app.services.historical_dataset_service import (
    HistoricalDatasetService,
)
from app.services.historical_feature_service import (
    HistoricalFeatureService,
)


@dataclass(frozen=True)
class HistoricalMlSplitBundle:
    feature_version: str
    horizon: int
    feature_names: list[str]

    x_train: np.ndarray
    y_train: np.ndarray
    train_stock_codes: list[str]
    train_dates: list[date]

    x_valid: np.ndarray
    y_valid: np.ndarray
    valid_stock_codes: list[str]
    valid_dates: list[date]

    x_test: np.ndarray
    y_test: np.ndarray
    test_stock_codes: list[str]
    test_dates: list[date]

    source_rows: int
    total_unique_dates: int

    purged_train_dates: int
    purged_valid_dates: int


class HistoricalMlTrainingService:
    DEFAULT_FEATURE_VERSION = (
        HistoricalFeatureService
        .FEATURE_VERSION
    )

    SUPPORTED_HORIZONS = (
        1,
        5,
        20,
    )

    DEFAULT_TRAIN_RATIO = 0.70
    DEFAULT_VALID_RATIO = 0.15

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

    @staticmethod
    def _slice_rows(
        *,
        dates: list[date],
        allowed_dates: set[date],
        x: np.ndarray,
        y: np.ndarray,
        stock_codes: list[str],
    ) -> tuple[
        np.ndarray,
        np.ndarray,
        list[str],
        list[date],
    ]:
        indexes = [
            index
            for (
                index,
                feature_date,
            )
            in enumerate(
                dates
            )
            if feature_date
            in allowed_dates
        ]

        if not indexes:
            raise ValueError(
                "시간 분리 결과가 비어 있습니다."
            )

        index_array = np.asarray(
            indexes,
            dtype=np.int64,
        )

        return (
            x[
                index_array
            ],
            y[
                index_array
            ],
            [
                stock_codes[
                    index
                ]
                for index
                in indexes
            ],
            [
                dates[
                    index
                ]
                for index
                in indexes
            ],
        )

    def build_temporal_split(
        self,
        *,
        feature_version: str,
        horizon: int,
        train_ratio: float = (
            DEFAULT_TRAIN_RATIO
        ),
        valid_ratio: float = (
            DEFAULT_VALID_RATIO
        ),
    ) -> HistoricalMlSplitBundle:
        if horizon not in (
            self.SUPPORTED_HORIZONS
        ):
            raise ValueError(
                "horizon은 1, 5, 20만 "
                "지원합니다."
            )

        if not (
            0.50
            <= train_ratio
            <= 0.85
        ):
            raise ValueError(
                "train_ratio는 "
                "0.50~0.85 범위여야 합니다."
            )

        if not (
            0.05
            <= valid_ratio
            <= 0.30
        ):
            raise ValueError(
                "valid_ratio는 "
                "0.05~0.30 범위여야 합니다."
            )

        if (
            train_ratio
            + valid_ratio
            >= 0.95
        ):
            raise ValueError(
                "Test 구간을 위해 "
                "train_ratio + valid_ratio는 "
                "0.95 미만이어야 합니다."
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

        unique_dates = sorted(
            set(
                dataset.feature_dates
            )
        )

        if len(
            unique_dates
        ) < 120:
            raise ValueError(
                "시간 분리를 위한 거래일이 "
                "120일 미만입니다."
            )

        train_end = int(
            len(
                unique_dates
            )
            * train_ratio
        )

        valid_end = int(
            len(
                unique_dates
            )
            * (
                train_ratio
                + valid_ratio
            )
        )

        raw_train_dates = (
            unique_dates[
                :train_end
            ]
        )

        raw_valid_dates = (
            unique_dates[
                train_end:
                valid_end
            ]
        )

        test_dates = (
            unique_dates[
                valid_end:
            ]
        )

        if (
            len(
                raw_train_dates
            )
            <= horizon
            or len(
                raw_valid_dates
            )
            <= horizon
            or not test_dates
        ):
            raise ValueError(
                "Purged temporal split을 "
                "만들기 위한 거래일이 "
                "부족합니다."
            )

        train_dates = (
            raw_train_dates[
                :-horizon
            ]
        )

        valid_dates = (
            raw_valid_dates[
                :-horizon
            ]
        )

        (
            x_train,
            y_train,
            train_stock_codes,
            train_feature_dates,
        ) = self._slice_rows(
            dates=(
                dataset.feature_dates
            ),
            allowed_dates=set(
                train_dates
            ),
            x=dataset.x,
            y=dataset.y,
            stock_codes=(
                dataset.stock_codes
            ),
        )

        (
            x_valid,
            y_valid,
            valid_stock_codes,
            valid_feature_dates,
        ) = self._slice_rows(
            dates=(
                dataset.feature_dates
            ),
            allowed_dates=set(
                valid_dates
            ),
            x=dataset.x,
            y=dataset.y,
            stock_codes=(
                dataset.stock_codes
            ),
        )

        (
            x_test,
            y_test,
            test_stock_codes,
            test_feature_dates,
        ) = self._slice_rows(
            dates=(
                dataset.feature_dates
            ),
            allowed_dates=set(
                test_dates
            ),
            x=dataset.x,
            y=dataset.y,
            stock_codes=(
                dataset.stock_codes
            ),
        )

        if not (
            max(
                train_feature_dates
            )
            < min(
                valid_feature_dates
            )
            < max(
                valid_feature_dates
            )
            < min(
                test_feature_dates
            )
        ):
            raise ValueError(
                "Train/Validation/Test "
                "날짜 순서가 올바르지 않습니다."
            )

        return HistoricalMlSplitBundle(
            feature_version=(
                feature_version
            ),
            horizon=horizon,
            feature_names=(
                dataset.feature_names
            ),

            x_train=x_train,
            y_train=y_train,
            train_stock_codes=(
                train_stock_codes
            ),
            train_dates=(
                train_feature_dates
            ),

            x_valid=x_valid,
            y_valid=y_valid,
            valid_stock_codes=(
                valid_stock_codes
            ),
            valid_dates=(
                valid_feature_dates
            ),

            x_test=x_test,
            y_test=y_test,
            test_stock_codes=(
                test_stock_codes
            ),
            test_dates=(
                test_feature_dates
            ),

            source_rows=int(
                dataset.x.shape[
                    0
                ]
            ),

            total_unique_dates=len(
                unique_dates
            ),

            purged_train_dates=(
                horizon
            ),

            purged_valid_dates=(
                horizon
            ),
        )

    @staticmethod
    def _split_summary(
        *,
        x: np.ndarray,
        y: np.ndarray,
        stock_codes: list[str],
        dates: list[date],
    ) -> dict:
        positive_count = int(
            np.sum(
                y > 0.0
            )
        )

        negative_count = int(
            np.sum(
                y < 0.0
            )
        )

        zero_count = int(
            np.sum(
                y == 0.0
            )
        )

        return {
            "rows":
                int(
                    x.shape[
                        0
                    ]
                ),

            "features":
                int(
                    x.shape[
                        1
                    ]
                ),

            "stocks":
                len(
                    set(
                        stock_codes
                    )
                ),

            "unique_dates":
                len(
                    set(
                        dates
                    )
                ),

            "first_date":
                min(
                    dates
                ).isoformat(),

            "last_date":
                max(
                    dates
                ).isoformat(),

            "positive":
                positive_count,

            "negative":
                negative_count,

            "zero":
                zero_count,

            "positive_rate_pct":
                round(
                    positive_count
                    / len(
                        y
                    )
                    * 100.0,
                    4,
                ),

            "target_mean":
                float(
                    np.mean(
                        y
                    )
                ),

            "target_std":
                float(
                    np.std(
                        y
                    )
                ),
        }

    def inspect_temporal_split(
        self,
        *,
        feature_version: str,
        horizon: int,
        train_ratio: float = (
            DEFAULT_TRAIN_RATIO
        ),
        valid_ratio: float = (
            DEFAULT_VALID_RATIO
        ),
    ) -> dict:
        split = (
            self.build_temporal_split(
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

        usable_rows = int(
            split.x_train.shape[
                0
            ]
            + split.x_valid.shape[
                0
            ]
            + split.x_test.shape[
                0
            ]
        )

        return {
            "status":
                "pass",

            "feature_version":
                split.feature_version,

            "target_horizon":
                split.horizon,

            "feature_count":
                len(
                    split.feature_names
                ),

            "source_rows":
                split.source_rows,

            "usable_rows":
                usable_rows,

            "purged_rows":
                split.source_rows
                - usable_rows,

            "total_unique_dates_before_purge":
                split.total_unique_dates,

            "train_ratio":
                train_ratio,

            "valid_ratio":
                valid_ratio,

            "test_ratio":
                round(
                    1.0
                    - train_ratio
                    - valid_ratio,
                    4,
                ),

            "purge_rule":
                (
                    "Train/Validation 끝에서 "
                    f"각각 {horizon}개 거래일 제거"
                ),

            "purged_train_dates":
                split.purged_train_dates,

            "purged_valid_dates":
                split.purged_valid_dates,

            "train":
                self._split_summary(
                    x=split.x_train,
                    y=split.y_train,
                    stock_codes=(
                        split
                        .train_stock_codes
                    ),
                    dates=(
                        split
                        .train_dates
                    ),
                ),

            "validation":
                self._split_summary(
                    x=split.x_valid,
                    y=split.y_valid,
                    stock_codes=(
                        split
                        .valid_stock_codes
                    ),
                    dates=(
                        split
                        .valid_dates
                    ),
                ),

            "test":
                self._split_summary(
                    x=split.x_test,
                    y=split.y_test,
                    stock_codes=(
                        split
                        .test_stock_codes
                    ),
                    dates=(
                        split
                        .test_dates
                    ),
                ),
        }