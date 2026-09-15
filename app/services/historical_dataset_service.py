from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date

import numpy as np
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.models.future import (
    StockFeatureSnapshot,
)
from app.services.historical_feature_service import (
    HistoricalFeatureService,
)


@dataclass(frozen=True)
class HistoricalDatasetBundle:
    stock_codes: list[str]
    feature_dates: list[date]
    feature_names: list[str]

    x: np.ndarray
    y: np.ndarray

    horizon: int
    feature_version: str


class HistoricalDatasetService:
    TARGET_COLUMNS = {
        1:
            StockFeatureSnapshot
            .target_return_1d,

        5:
            StockFeatureSnapshot
            .target_return_5d,

        20:
            StockFeatureSnapshot
            .target_return_20d,
    }

    def __init__(
        self,
        db: Session,
    ):
        self.db = db

    def get_stock_date_ranges(
        self,
        *,
        feature_version: str,
        horizon: int,
    ) -> list[tuple[str, date, date]]:
        target_column = (
            self.TARGET_COLUMNS
            .get(
                horizon
            )
        )

        if target_column is None:
            raise ValueError(
                "지원하지 않는 Target horizon입니다: "
                f"{horizon}"
            )

        rows = self.db.execute(
            select(
                StockFeatureSnapshot.stock_code,
                func.min(
                    StockFeatureSnapshot.feature_date
                ),
                func.max(
                    StockFeatureSnapshot.feature_date
                ),
            )
            .where(
                StockFeatureSnapshot.feature_version
                == feature_version,
                target_column.is_not(None),
            )
            .group_by(
                StockFeatureSnapshot.stock_code
            )
            .order_by(
                StockFeatureSnapshot.stock_code.asc()
            )
        ).all()

        return [
            (
                str(stock_code),
                first_date,
                last_date,
            )
            for (
                stock_code,
                first_date,
                last_date,
            )
            in rows
            if (
                first_date is not None
                and last_date is not None
            )
        ]

    def get_stock_date_ranges(
        self,
        *,
        feature_version: str,
        horizon: int,
    ) -> list[tuple[str, date, date]]:
        target_column = (
            self.TARGET_COLUMNS
            .get(
                horizon
            )
        )

        if target_column is None:
            raise ValueError(
                "지원하지 않는 Target horizon입니다: "
                f"{horizon}"
            )

        rows = self.db.execute(
            select(
                StockFeatureSnapshot.stock_code,
                func.min(
                    StockFeatureSnapshot.feature_date
                ),
                func.max(
                    StockFeatureSnapshot.feature_date
                ),
            )
            .where(
                StockFeatureSnapshot.feature_version
                == feature_version,
                target_column.is_not(None),
            )
            .group_by(
                StockFeatureSnapshot.stock_code
            )
            .order_by(
                StockFeatureSnapshot.stock_code.asc()
            )
        ).all()

        return [
            (
                str(stock_code),
                first_date,
                last_date,
            )
            for (
                stock_code,
                first_date,
                last_date,
            ) in rows
            if (
                first_date is not None
                and last_date is not None
            )
        ]

    def build_dataset(
        self,
        *,
        feature_version: str,
        horizon: int,
        feature_strategy: str = "full",
    ) -> HistoricalDatasetBundle:

        target_column = (
            self.TARGET_COLUMNS
            .get(
                horizon
            )
        )

        if target_column is None:
            raise ValueError(
                "지원하지 않는 Target horizon입니다: "
                f"{horizon}"
            )

        feature_names = (
            HistoricalFeatureService
            .feature_names_for_strategy(
                feature_strategy=(
                    feature_strategy
                )
            )
        )

        snapshots = list(
            self.db.scalars(
                select(
                    StockFeatureSnapshot
                )
                .where(
                    StockFeatureSnapshot
                    .feature_version
                    == feature_version,

                    target_column
                    .is_not(
                        None
                    ),
                )
                .order_by(
                    StockFeatureSnapshot
                    .feature_date
                    .asc(),

                    StockFeatureSnapshot
                    .stock_code
                    .asc(),
                )
            ).all()
        )

        if not snapshots:
            raise ValueError(
                "학습 가능한 Historical Snapshot이 "
                "없습니다."
            )

        matrix_rows: list[
            list[float]
        ] = []

        targets: list[
            float
        ] = []

        stock_codes: list[
            str
        ] = []

        feature_dates: list[
            date
        ] = []

        for snapshot in snapshots:
            features = (
                snapshot.features
                or {}
            )

            row: list[
                float
            ] = []

            for feature_name in (
                feature_names
            ):
                if (
                    feature_name
                    not in features
                ):
                    raise ValueError(
                        "Historical Feature 누락: "
                        f"{snapshot.stock_code} "
                        f"{snapshot.feature_date} "
                        f"{feature_name}"
                    )

                value = (
                    features[
                        feature_name
                    ]
                )

                if value is None:
                    raise ValueError(
                        "Historical Feature NULL: "
                        f"{snapshot.stock_code} "
                        f"{snapshot.feature_date} "
                        f"{feature_name}"
                    )

                try:
                    number = float(
                        value
                    )

                except (
                    TypeError,
                    ValueError,
                ) as e:
                    raise ValueError(
                        "Historical Feature 비숫자: "
                        f"{snapshot.stock_code} "
                        f"{snapshot.feature_date} "
                        f"{feature_name}="
                        f"{value}"
                    ) from e

                if not math.isfinite(
                    number
                ):
                    raise ValueError(
                        "Historical Feature "
                        "NaN/Infinity: "
                        f"{snapshot.stock_code} "
                        f"{snapshot.feature_date} "
                        f"{feature_name}"
                    )

                row.append(
                    number
                )

            target_value = getattr(
                snapshot,
                target_column.key,
            )

            if target_value is None:
                continue

            target = float(
                target_value
            )

            if not math.isfinite(
                target
            ):
                raise ValueError(
                    "Target NaN/Infinity: "
                    f"{snapshot.stock_code} "
                    f"{snapshot.feature_date}"
                )

            matrix_rows.append(
                row
            )

            targets.append(
                target
            )

            stock_codes.append(
                snapshot.stock_code
            )

            feature_dates.append(
                snapshot.feature_date
            )

        x = np.asarray(
            matrix_rows,
            dtype=np.float64,
        )

        y = np.asarray(
            targets,
            dtype=np.float64,
        )

        if (
            x.ndim != 2
            or x.shape[
                1
            ] != len(
                feature_names
            )
        ):
            raise ValueError(
                "Historical Feature Matrix "
                "shape가 올바르지 않습니다."
            )

        if (
            x.shape[
                0
            ]
            != y.shape[
                0
            ]
        ):
            raise ValueError(
                "Feature와 Target 행 수가 "
                "일치하지 않습니다."
            )

        if not np.all(
            np.isfinite(
                x
            )
        ):
            raise ValueError(
                "Feature Matrix에 "
                "NaN/Infinity가 존재합니다."
            )

        if not np.all(
            np.isfinite(
                y
            )
        ):
            raise ValueError(
                "Target에 "
                "NaN/Infinity가 존재합니다."
            )

        return HistoricalDatasetBundle(
            stock_codes=stock_codes,
            feature_dates=feature_dates,
            feature_names=(
                feature_names
            ),
            x=x,
            y=y,
            horizon=horizon,
            feature_version=(
                feature_version
            ),
        )

    def inspect_dataset(
        self,
        *,
        feature_version: str,
        horizon: int,
        preview_rows: int = 3,
    ) -> dict:
        bundle = (
            self.build_dataset(
                feature_version=(
                    feature_version
                ),
                horizon=horizon,
            )
        )

        x = bundle.x
        y = bundle.y

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

        unique_dates = sorted(
            set(
                bundle
                .feature_dates
            )
        )

        unique_stocks = sorted(
            set(
                bundle
                .stock_codes
            )
        )

        preview = []

        preview_count = min(
            preview_rows,
            len(
                bundle.stock_codes
            ),
        )

        for index in range(
            preview_count
        ):
            preview.append(
                {
                    "stock_code":
                        bundle
                        .stock_codes[
                            index
                        ],

                    "feature_date":
                        bundle
                        .feature_dates[
                            index
                        ]
                        .isoformat(),

                    "target_return":
                        float(
                            y[
                                index
                            ]
                        ),

                    "first_5_features":
                        {
                            name:
                                float(
                                    x[
                                        index,
                                        column_index,
                                    ]
                                )

                            for (
                                column_index,
                                name,
                            )
                            in enumerate(
                                bundle
                                .feature_names[
                                    :5
                                ]
                            )
                        },
                }
            )

        return {
            "status":
                "pass",

            "feature_version":
                feature_version,

            "target_horizon":
                horizon,

            "rows":
                int(
                    x.shape[
                        0
                    ]
                ),

            "columns":
                int(
                    x.shape[
                        1
                    ]
                ),

            "shape": [
                int(
                    x.shape[
                        0
                    ]
                ),
                int(
                    x.shape[
                        1
                    ]
                ),
            ],

            "unique_stocks":
                len(
                    unique_stocks
                ),

            "unique_dates":
                len(
                    unique_dates
                ),

            "first_date":
                unique_dates[
                    0
                ].isoformat(),

            "last_date":
                unique_dates[
                    -1
                ].isoformat(),

            "invalid_feature_cells":
                int(
                    np.sum(
                        ~np.isfinite(
                            x
                        )
                    )
                ),

            "invalid_target_cells":
                int(
                    np.sum(
                        ~np.isfinite(
                            y
                        )
                    )
                ),

            "target": {
                "positive":
                    positive_count,

                "negative":
                    negative_count,

                "zero":
                    zero_count,

                "positive_rate_pct":
                    round(
                        positive_count
                        / len(y)
                        * 100.0,
                        4,
                    ),

                "mean":
                    float(
                        np.mean(
                            y
                        )
                    ),

                "median":
                    float(
                        np.median(
                            y
                        )
                    ),

                "std":
                    float(
                        np.std(
                            y
                        )
                    ),

                "min":
                    float(
                        np.min(
                            y
                        )
                    ),

                "p01":
                    float(
                        np.quantile(
                            y,
                            0.01,
                        )
                    ),

                "p99":
                    float(
                        np.quantile(
                            y,
                            0.99,
                        )
                    ),

                "max":
                    float(
                        np.max(
                            y
                        )
                    ),
            },

            "feature_names":
                bundle
                .feature_names,

            "preview":
                preview,
        }