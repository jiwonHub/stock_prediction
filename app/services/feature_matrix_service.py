from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.models.future import (
    StockFeatureSnapshot,
)
from app.services.feature_quality_service import (
    FeatureQualityService,
)


@dataclass(frozen=True)
class FeatureMatrixBundle:
    stock_codes: list[str]
    feature_names: list[str]
    feature_date: object

    x_raw: np.ndarray
    x: np.ndarray

    missing_mask: np.ndarray
    imputation_values: dict[str, float]


class FeatureMatrixService:
    def __init__(
        self,
        db: Session,
    ):
        self.db = db

    def build_latest(
        self,
        *,
        feature_version: str,
        limit: int = 100,
    ) -> FeatureMatrixBundle:
        feature_names = list(
            FeatureQualityService
            .EXPECTED_FEATURES
        )

        latest_date = (
            self.db.scalar(
                select(
                    func.max(
                        StockFeatureSnapshot
                        .feature_date
                    )
                )
                .where(
                    StockFeatureSnapshot
                    .feature_version
                    == feature_version
                )
            )
        )

        if latest_date is None:
            raise ValueError(
                "Feature Snapshot이 없습니다: "
                f"{feature_version}"
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
                    StockFeatureSnapshot
                    .feature_date
                    == latest_date,
                )
                .order_by(
                    StockFeatureSnapshot
                    .stock_code
                    .asc()
                )
                .limit(
                    limit
                )
            ).all()
        )

        if not snapshots:
            raise ValueError(
                "Feature Matrix를 만들 "
                "Snapshot이 없습니다."
            )

        matrix_rows: list[
            list[float]
        ] = []

        stock_codes: list[str] = []

        for snapshot in snapshots:
            features = (
                snapshot.features
                or {}
            )

            row: list[float] = []

            for feature_name in (
                feature_names
            ):
                value = features.get(
                    feature_name
                )

                if value is None:
                    row.append(
                        np.nan
                    )
                    continue

                try:
                    number = float(
                        value
                    )

                except (
                    TypeError,
                    ValueError,
                ) as e:
                    raise ValueError(
                        "비숫자 Feature 발견: "
                        f"{snapshot.stock_code} "
                        f"{feature_name}="
                        f"{value}"
                    ) from e

                if not np.isfinite(
                    number
                ):
                    raise ValueError(
                        "NaN/Infinity Feature 발견: "
                        f"{snapshot.stock_code} "
                        f"{feature_name}"
                    )

                row.append(
                    number
                )

            matrix_rows.append(
                row
            )

            stock_codes.append(
                snapshot.stock_code
            )

        x_raw = np.asarray(
            matrix_rows,
            dtype=np.float64,
        )

        missing_mask = np.isnan(
            x_raw
        )

        x = x_raw.copy()

        imputation_values: dict[
            str,
            float,
        ] = {}

        for column_index, feature_name in enumerate(
            feature_names
        ):
            column = x[
                :,
                column_index
            ]

            valid_values = column[
                np.isfinite(
                    column
                )
            ]

            if len(
                valid_values
            ) == 0:
                raise ValueError(
                    "전체 종목이 결측인 Feature: "
                    f"{feature_name}"
                )

            median = float(
                np.median(
                    valid_values
                )
            )

            imputation_values[
                feature_name
            ] = median

            missing_indexes = (
                np.isnan(
                    column
                )
            )

            if np.any(
                missing_indexes
            ):
                x[
                    missing_indexes,
                    column_index,
                ] = median

        if not np.all(
            np.isfinite(
                x
            )
        ):
            raise ValueError(
                "Feature Matrix 보정 후에도 "
                "NaN/Infinity가 존재합니다."
            )

        return FeatureMatrixBundle(
            stock_codes=stock_codes,
            feature_names=(
                feature_names
            ),
            feature_date=(
                latest_date
            ),
            x_raw=x_raw,
            x=x,
            missing_mask=(
                missing_mask
            ),
            imputation_values=(
                imputation_values
            ),
        )

    def inspect_latest(
        self,
        *,
        feature_version: str,
        limit: int = 100,
        preview_rows: int = 3,
    ) -> dict:
        bundle = (
            self.build_latest(
                feature_version=(
                    feature_version
                ),
                limit=limit,
            )
        )

        raw_missing = int(
            np.isnan(
                bundle.x_raw
            ).sum()
        )

        remaining_invalid = int(
            (
                ~np.isfinite(
                    bundle.x
                )
            ).sum()
        )

        features_with_imputation = []

        for column_index, name in enumerate(
            bundle.feature_names
        ):
            missing_count = int(
                bundle.missing_mask[
                    :,
                    column_index
                ].sum()
            )

            if missing_count > 0:
                features_with_imputation.append(
                    {
                        "feature":
                            name,

                        "missing_count":
                            missing_count,

                        "median":
                            bundle
                            .imputation_values[
                                name
                            ],
                    }
                )

        preview = []

        preview_count = min(
            preview_rows,
            len(
                bundle.stock_codes
            ),
        )

        for row_index in range(
            preview_count
        ):
            preview.append(
                {
                    "stock_code":
                        bundle
                        .stock_codes[
                            row_index
                        ],

                    "first_10_features":
                        {
                            name:
                                float(
                                    bundle.x[
                                        row_index,
                                        column_index,
                                    ]
                                )

                            for (
                                column_index,
                                name,
                            ) in enumerate(
                                bundle
                                .feature_names[
                                    :10
                                ]
                            )
                        },
                }
            )

        return {
            "status":
                "pass"
                if remaining_invalid == 0
                else "fail",

            "feature_version":
                feature_version,

            "feature_date":
                bundle
                .feature_date
                .isoformat(),

            "rows":
                int(
                    bundle.x.shape[
                        0
                    ]
                ),

            "columns":
                int(
                    bundle.x.shape[
                        1
                    ]
                ),

            "shape": [
                int(
                    bundle.x.shape[
                        0
                    ]
                ),
                int(
                    bundle.x.shape[
                        1
                    ]
                ),
            ],

            "raw_missing_cells":
                raw_missing,

            "remaining_invalid_cells":
                remaining_invalid,

            "feature_count":
                len(
                    bundle.feature_names
                ),

            "features_with_imputation":
                features_with_imputation,

            "imputed_feature_count":
                len(
                    features_with_imputation
                ),

            "stock_codes":
                bundle.stock_codes,

            "feature_names":
                bundle.feature_names,

            "preview":
                preview,
        }