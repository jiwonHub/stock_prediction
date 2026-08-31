from __future__ import annotations

from datetime import date, timedelta

from sqlalchemy.orm import Session

from app.repositories.stock_repository import (
    StockRepository,
)
from app.utils.advanced_technical_features import (
    ADVANCED_TECHNICAL_FEATURE_NAMES,
    build_advanced_technical_features,
)


class TechnicalAnalysisService:
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

    def inspect_stock(
        self,
        *,
        stock_code: str,
    ) -> dict:
        rows = (
            self.stock_repository
            .get_daily_prices(
                stock_code=(
                    stock_code
                ),
                start_date=(
                    date.today()
                    - timedelta(
                        days=365
                    )
                ),
            )
        )

        if len(
            rows
        ) < 61:
            raise ValueError(
                "일봉 데이터가 부족합니다: "
                f"{stock_code} "
                f"{len(rows)}건"
            )

        rows = sorted(
            rows,
            key=lambda row:
                row.trade_date,
        )

        features = (
            build_advanced_technical_features(
                rows
            )
        )

        return {
            "status":
                "pass",

            "stock_code":
                stock_code,

            "feature_date":
                rows[
                    -1
                ]
                .trade_date
                .isoformat(),

            "price_rows":
                len(
                    rows
                ),

            "feature_count":
                len(
                    features
                ),

            "expected_feature_count":
                len(
                    ADVANCED_TECHNICAL_FEATURE_NAMES
                ),

            "features":
                features,
        }