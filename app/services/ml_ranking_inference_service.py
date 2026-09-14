from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import xgboost as xgb
from scipy.stats import rankdata
from sqlalchemy.orm import Session

from app.repositories.stock_repository import (
    StockRepository,
)
from app.services.historical_ml_final_model_service import (
    HistoricalMlFinalModelService,
)
from app.services.market_data_service import (
    MarketDataService,
)
from app.utils.advanced_technical_features import (
    ADVANCED_TECHNICAL_FEATURE_NAMES,
    build_advanced_technical_features,
)
from app.utils.ml_features import (
    FEATURE_NAMES,
    build_latest_feature_dict,
)


class MlRankingInferenceService:
    HISTORY_CALENDAR_DAYS = 1200

    EXPECTED_FEATURE_NAMES = (
        *FEATURE_NAMES,
        "return_60d",
        *ADVANCED_TECHNICAL_FEATURE_NAMES,
    )

    def __init__(
        self,
        db: Session,
    ):
        self.db = db

        self.stock_repository = (
            StockRepository(
                db
            )
        )

        self.market_service = (
            MarketDataService(
                db
            )
        )

        self.artifact = (
            HistoricalMlFinalModelService
            .load_artifact()
        )

        self.classifier = (
            self.artifact[
                "classifier"
            ]
        )

        self.feature_names = list(
            self.artifact[
                "feature_names"
            ]
        )

        self._validate_artifact()
        
    def _validate_artifact(
        self,
    ) -> None:
        if (
            self.artifact.get(
                "model_name"
            )
            != HistoricalMlFinalModelService
            .MODEL_NAME
        ):
            raise RuntimeError(
                "ML Ranking 모델 이름이 "
                "일치하지 않습니다: "
                f"{self.artifact.get('model_name')}"
            )

        if (
            self.artifact.get(
                "model_version"
            )
            != HistoricalMlFinalModelService
            .MODEL_VERSION
        ):
            raise RuntimeError(
                "ML Ranking 모델 버전이 "
                "일치하지 않습니다: "
                f"{self.artifact.get('model_version')}"
            )

        if (
            self.artifact.get(
                "feature_version"
            )
            != HistoricalMlFinalModelService
            .FEATURE_VERSION
        ):
            raise RuntimeError(
                "ML Ranking Feature 버전이 "
                "일치하지 않습니다: "
                f"{self.artifact.get('feature_version')}"
            )

        if (
            self.artifact.get(
                "horizon"
            )
            != HistoricalMlFinalModelService
            .HORIZON
        ):
            raise RuntimeError(
                "ML Ranking Horizon이 "
                "일치하지 않습니다: "
                f"{self.artifact.get('horizon')}"
            )

        if (
            self.artifact.get(
                "feature_strategy"
            )
            != HistoricalMlFinalModelService
            .FEATURE_STRATEGY
        ):
            raise RuntimeError(
                "ML Ranking Feature Strategy가 "
                "일치하지 않습니다: "
                f"{self.artifact.get('feature_strategy')}"
            )

        if (
            self.artifact.get(
                "calibration"
            )
            != HistoricalMlFinalModelService
            .CALIBRATION
        ):
            raise RuntimeError(
                "ML Ranking Calibration이 "
                "일치하지 않습니다: "
                f"{self.artifact.get('calibration')}"
            )

        if (
            self.artifact.get(
                "intended_use"
            )
            != HistoricalMlFinalModelService
            .INTENDED_USE
        ):
            raise RuntimeError(
                "ML Ranking Intended Use가 "
                "일치하지 않습니다: "
                f"{self.artifact.get('intended_use')}"
            )

        expected_feature_names = list(
            self.EXPECTED_FEATURE_NAMES
        )

        if (
            self.artifact.get(
                "feature_count"
            )
            != len(
                expected_feature_names
            )
        ):
            raise RuntimeError(
                "ML Ranking Artifact Feature 개수가 "
                "일치하지 않습니다: "
                f"{self.artifact.get('feature_count')}"
            )

        if (
            self.feature_names
            != expected_feature_names
        ):
            raise RuntimeError(
                "ML Ranking Feature 이름 또는 "
                "순서가 학습 기준과 일치하지 않습니다.\n"
                f"expected={expected_feature_names}\n"
                f"actual={self.feature_names}"
            )

    def _build_feature_vector(
        self,
        stock_code: str,
    ) -> tuple[
        np.ndarray,
        date,
        dict[str, float],
    ]:
        rows = (
            self.stock_repository
            .get_daily_prices(
                stock_code=stock_code,
                start_date=(
                    date.today()
                    - timedelta(
                        days=self.HISTORY_CALENDAR_DAYS
                    )
                ),
            )
        )

        return (
            self._build_feature_vector_from_rows(
                stock_code,
                rows,
            )
        )


    def _build_feature_vector_from_rows(
        self,
        stock_code: str,
        rows: list,
    ) -> tuple[
        np.ndarray,
        date,
        dict[str, float],
    ]:
        rows = sorted(
            rows,
            key=lambda row: row.trade_date,
        )

        if len(rows) < 61:
            raise ValueError(
                "일봉 데이터가 부족합니다: "
                f"{stock_code} "
                f"{len(rows)}건"
            )

        technical = build_latest_feature_dict(
            rows
        )

        advanced = (
            build_advanced_technical_features(
                rows
            )
        )

        features = {
            **technical,
            **advanced,
        }

        missing = [
            feature_name
            for feature_name
            in self.feature_names
            if (
                feature_name not in features
                or features[
                    feature_name
                ] is None
            )
        ]

        if missing:
            raise ValueError(
                "ML Feature 누락: "
                f"{stock_code} "
                f"{missing}"
            )

        values = np.asarray(
            [
                float(
                    features[
                        feature_name
                    ]
                )
                for feature_name
                in self.feature_names
            ],
            dtype=np.float32,
        )

        if not np.all(
            np.isfinite(
                values
            )
        ):
            raise ValueError(
                "ML Feature에 "
                "NaN/Inf가 있습니다: "
                f"{stock_code}"
            )

        return (
            values.reshape(
                1,
                -1,
            ),
            rows[-1].trade_date,
            {
                feature_name:
                    float(
                        features[
                            feature_name
                        ]
                    )
                for feature_name
                in self.feature_names
            },
        )

    def _predict_feature_contributions(
        self,
        x: np.ndarray,
    ) -> tuple[
        dict[str, float],
        float,
    ]:
        booster = (
            self.classifier
            .get_booster()
        )

        matrix = xgb.DMatrix(
            x
        )

        values = booster.predict(
            matrix,
            pred_contribs=True,
        )

        if (
            values.ndim != 2
            or values.shape[0] != 1
            or values.shape[1]
            != len(self.feature_names) + 1
        ):
            raise RuntimeError(
                "XGBoost Feature Contribution "
                "형식이 예상과 다릅니다: "
                f"shape={values.shape}"
            )

        row = values[0]

        contributions = {
            feature_name:
                float(
                    row[index]
                )
            for index, feature_name
            in enumerate(
                self.feature_names
            )
        }

        bias = float(
            row[-1]
        )

        return (
            contributions,
            bias,
        )

    def predict_stock(
        self,
        stock_code: str,
    ) -> dict:
        (
            x,
            feature_date,
            features,
        ) = self._build_feature_vector(
            stock_code
        )

        probability = float(
            self.classifier
            .predict_proba(
                x
            )[
                0,
                1,
            ]
        )

        probability = max(
            0.0,
            min(
                1.0,
                probability,
            ),
        )

        (
            feature_contributions,
            contribution_bias,
        ) = (
            self._predict_feature_contributions(
                x
            )
        )

        stock = (
            self.stock_repository
            .get_stock(
                stock_code
            )
        )

        return {
            "stock_code":
                stock_code,

            "stock_name":
                (
                    stock.name
                    if stock is not None
                    else stock_code
                ),

            "feature_date":
                feature_date,

            "raw_probability":
                probability,

            "raw_probability_pct":
                probability
                * 100.0,

            "features":
                features,

            "feature_contributions":
                feature_contributions,

            "contribution_bias":
                contribution_bias,
        }

    def score_current_universe(
        self,
        *,
        limit: int = 100,
        include_explanations: bool = True,
    ) -> dict:
        from collections import defaultdict

        from sqlalchemy import select

        from app.models.stock import Stock
        from app.models.stock_price import StockPrice

        stock_codes = (
            self.market_service
            .get_current_universe_stock_codes(
                limit=limit,
            )
        )

        if not stock_codes:
            raise ValueError(
                "현재 TOP Universe가 없습니다."
            )

        start_date = (
            date.today()
            - timedelta(
                days=self.HISTORY_CALENDAR_DAYS
            )
        )

        # TOP100 일봉을 종목별 100번 조회하지 않고
        # 한 번에 가져온다.
        price_rows = (
            self.db
            .scalars(
                select(
                    StockPrice
                )
                .where(
                    StockPrice.stock_code.in_(
                        stock_codes
                    ),
                    StockPrice.trade_date
                    >= start_date,
                )
                .order_by(
                    StockPrice.stock_code.asc(),
                    StockPrice.trade_date.asc(),
                )
            )
            .all()
        )

        rows_by_stock = defaultdict(
            list
        )

        for row in price_rows:
            rows_by_stock[
                row.stock_code
            ].append(
                row
            )

        # 종목명도 100번 get_stock 하지 않고
        # 한 번에 가져온다.
        stock_rows = (
            self.db
            .scalars(
                select(
                    Stock
                )
                .where(
                    Stock.code.in_(
                        stock_codes
                    )
                )
            )
            .all()
        )

        stock_name_map = {
            stock.code: stock.name
            for stock in stock_rows
        }

        prepared = []
        skipped = []

        for stock_code in stock_codes:
            try:
                (
                    x,
                    feature_date,
                    features,
                ) = (
                    self
                    ._build_feature_vector_from_rows(
                        stock_code,
                        rows_by_stock.get(
                            stock_code,
                            [],
                        ),
                    )
                )

                prepared.append(
                    {
                        "stock_code":
                            stock_code,

                        "stock_name":
                            stock_name_map.get(
                                stock_code,
                                stock_code,
                            ),

                        "x":
                            x,

                        "feature_date":
                            feature_date,

                        "features":
                            features,
                    }
                )

            except Exception as e:
                skipped.append(
                    {
                        "stock_code":
                            stock_code,

                        "reason":
                            str(e),
                    }
                )

        if not prepared:
            raise ValueError(
                "ML Ranking 추론에 성공한 "
                "종목이 없습니다."
            )

        # XGBoost도 종목별 100회가 아니라
        # 전체 matrix 한 번에 추론한다.
        x_matrix = np.vstack(
            [
                row["x"]
                for row in prepared
            ]
        )

        probabilities = np.asarray(
            self.classifier
            .predict_proba(
                x_matrix
            )[:, 1],
            dtype=np.float64,
        )

        probabilities = np.clip(
            probabilities,
            0.0,
            1.0,
        )

        scored = []

        for (
            prepared_row,
            probability,
        ) in zip(
            prepared,
            probabilities,
        ):
            if include_explanations:
                (
                    feature_contributions,
                    contribution_bias,
                ) = (
                    self
                    ._predict_feature_contributions(
                        prepared_row[
                            "x"
                        ]
                    )
                )

            else:
                feature_contributions = {}
                contribution_bias = 0.0

            probability = float(
                probability
            )

            scored.append(
                {
                    "stock_code":
                        prepared_row[
                            "stock_code"
                        ],

                    "stock_name":
                        prepared_row[
                            "stock_name"
                        ],

                    "feature_date":
                        prepared_row[
                            "feature_date"
                        ],

                    "raw_probability":
                        probability,

                    "raw_probability_pct":
                        probability * 100.0,

                    "features":
                        prepared_row[
                            "features"
                        ],

                    "feature_contributions":
                        feature_contributions,

                    "contribution_bias":
                        contribution_bias,
                }
            )

        if len(
            probabilities
        ) == 1:
            percentiles = np.asarray(
                [50.0],
                dtype=np.float64,
            )

        else:
            ranks = rankdata(
                probabilities,
                method="average",
            )

            percentiles = (
                (
                    ranks
                    - 1.0
                )
                / (
                    len(
                        ranks
                    )
                    - 1.0
                )
                * 100.0
            )

        for (
            row,
            percentile,
        ) in zip(
            scored,
            percentiles,
        ):
            row[
                "ml_score"
            ] = round(
                float(
                    percentile
                ),
                4,
            )

        scored.sort(
            key=lambda row: (
                row[
                    "ml_score"
                ],
                row[
                    "raw_probability"
                ],
            ),
            reverse=True,
        )

        for (
            index,
            row,
        ) in enumerate(
            scored,
            start=1,
        ):
            row[
                "ml_rank"
            ] = index

        feature_dates = [
            row[
                "feature_date"
            ]
            for row in scored
        ]

        return {
            "status":
                "pass",

            "model_name":
                self.artifact[
                    "model_name"
                ],

            "model_version":
                self.artifact[
                    "model_version"
                ],

            "horizon_days":
                self.artifact[
                    "horizon"
                ],

            "feature_strategy":
                self.artifact[
                    "feature_strategy"
                ],

            "feature_count":
                len(
                    self.feature_names
                ),

            "requested_universe":
                len(
                    stock_codes
                ),

            "scored_count":
                len(
                    scored
                ),

            "skipped_count":
                len(
                    skipped
                ),

            "feature_date_min":
                min(
                    feature_dates
                ).isoformat(),

            "feature_date_max":
                max(
                    feature_dates
                ).isoformat(),

            "raw_probability_min_pct":
                float(
                    np.min(
                        probabilities
                    )
                    * 100.0
                ),

            "raw_probability_mean_pct":
                float(
                    np.mean(
                        probabilities
                    )
                    * 100.0
                ),

            "raw_probability_max_pct":
                float(
                    np.max(
                        probabilities
                    )
                    * 100.0
                ),

            "scores":
                scored,

            "skipped":
                skipped,
        }

