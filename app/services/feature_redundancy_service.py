from __future__ import annotations

import numpy as np
from sqlalchemy.orm import Session

from app.services.historical_dataset_service import (
    HistoricalDatasetService,
)
from app.utils.advanced_technical_features import (
    ADVANCED_TECHNICAL_FEATURE_NAMES,
)


class FeatureRedundancyService:
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

    def inspect(
        self,
        *,
        feature_version: str,
        horizon: int = 5,
        correlation_threshold: float = 0.95,
    ) -> dict:
        bundle = (
            self.dataset_service
            .build_dataset(
                feature_version=(
                    feature_version
                ),
                horizon=horizon,
            )
        )

        x = bundle.x

        feature_names = (
            bundle.feature_names
        )

        if x.shape[0] < 2:
            raise ValueError(
                "상관관계 분석에 필요한 "
                "학습 행이 부족합니다."
            )

        if x.shape[1] != len(
            feature_names
        ):
            raise ValueError(
                "Feature 이름과 Matrix column 수가 "
                "일치하지 않습니다."
            )

        feature_index = {
            name: index
            for index, name
            in enumerate(
                feature_names
            )
        }

        technical_names = [
            name
            for name
            in ADVANCED_TECHNICAL_FEATURE_NAMES
            if name in feature_index
        ]

        zero_variance_features = []
        feature_stats = []

        for name in feature_names:
            index = (
                feature_index[
                    name
                ]
            )

            column = x[
                :,
                index
            ]

            std = float(
                np.std(
                    column
                )
            )

            minimum = float(
                np.min(
                    column
                )
            )

            maximum = float(
                np.max(
                    column
                )
            )

            unique_count = int(
                np.unique(
                    column
                ).size
            )

            if (
                std <= 1e-12
                or unique_count <= 1
            ):
                zero_variance_features.append(
                    name
                )

            feature_stats.append(
                {
                    "feature":
                        name,

                    "mean":
                        float(
                            np.mean(
                                column
                            )
                        ),

                    "std":
                        std,

                    "min":
                        minimum,

                    "max":
                        maximum,

                    "unique_count":
                        unique_count,
                }
            )

        correlation_matrix = (
            np.corrcoef(
                x,
                rowvar=False,
            )
        )

        high_correlation_pairs = []

        technical_high_correlation_pairs = []

        exact_duplicate_pairs = []

        for left_index in range(
            len(
                feature_names
            )
        ):
            for right_index in range(
                left_index + 1,
                len(
                    feature_names
                ),
            ):
                left_name = (
                    feature_names[
                        left_index
                    ]
                )

                right_name = (
                    feature_names[
                        right_index
                    ]
                )

                correlation = float(
                    correlation_matrix[
                        left_index,
                        right_index,
                    ]
                )

                if not np.isfinite(
                    correlation
                ):
                    continue

                absolute = abs(
                    correlation
                )

                if absolute >= 0.999999:
                    exact_duplicate_pairs.append(
                        {
                            "left":
                                left_name,

                            "right":
                                right_name,

                            "correlation":
                                correlation,
                        }
                    )

                if absolute >= (
                    correlation_threshold
                ):
                    pair = {
                        "left":
                            left_name,

                        "right":
                            right_name,

                        "correlation":
                            correlation,

                        "absolute_correlation":
                            absolute,
                    }

                    high_correlation_pairs.append(
                        pair
                    )

                    if (
                        left_name
                        in technical_names
                        or right_name
                        in technical_names
                    ):
                        technical_high_correlation_pairs.append(
                            pair
                        )

        high_correlation_pairs.sort(
            key=lambda row:
                row[
                    "absolute_correlation"
                ],
            reverse=True,
        )

        technical_high_correlation_pairs.sort(
            key=lambda row:
                row[
                    "absolute_correlation"
                ],
            reverse=True,
        )

        technical_stats = [
            row
            for row
            in feature_stats
            if row[
                "feature"
            ] in technical_names
        ]

        warnings = []

        if zero_variance_features:
            warnings.append(
                "분산이 사실상 0인 Feature가 있습니다."
            )

        if exact_duplicate_pairs:
            warnings.append(
                "거의 완전히 동일한 Feature 쌍이 있습니다."
            )

        if technical_high_correlation_pairs:
            warnings.append(
                "고급 기술지표와 강한 상관관계를 가진 "
                "Feature 쌍이 있습니다."
            )

        return {
            "status":
                "warning"
                if warnings
                else "pass",

            "feature_version":
                feature_version,

            "target_horizon":
                horizon,

            "dataset_rows":
                int(
                    x.shape[
                        0
                    ]
                ),

            "feature_count":
                int(
                    x.shape[
                        1
                    ]
                ),

            "advanced_technical_feature_count":
                len(
                    technical_names
                ),

            "correlation_threshold":
                correlation_threshold,

            "zero_variance_features":
                zero_variance_features,

            "zero_variance_count":
                len(
                    zero_variance_features
                ),

            "exact_duplicate_pairs":
                exact_duplicate_pairs,

            "exact_duplicate_pair_count":
                len(
                    exact_duplicate_pairs
                ),

            "high_correlation_pair_count":
                len(
                    high_correlation_pairs
                ),

            "technical_high_correlation_pair_count":
                len(
                    technical_high_correlation_pairs
                ),

            "technical_high_correlation_pairs":
                technical_high_correlation_pairs[
                    :50
                ],

            "top_high_correlation_pairs":
                high_correlation_pairs[
                    :50
                ],

            "technical_feature_stats":
                technical_stats,

            "warnings":
                warnings,
        }