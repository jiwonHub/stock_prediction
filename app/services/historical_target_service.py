from __future__ import annotations

from sqlalchemy.orm import Session

from app.repositories.feature_repository import (
    FeatureRepository,
)
from app.repositories.stock_repository import (
    StockRepository,
)
from app.services.market_data_service import (
    MarketDataService,
)


class HistoricalTargetService:
    HORIZONS = (
        1,
        5,
        20,
    )

    def __init__(
        self,
        db: Session,
    ):
        self.db = db

        self.feature_repository = (
            FeatureRepository(
                db
            )
        )

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

    def get_current_universe_stock_codes(
        self,
        *,
        limit: int = 100,
    ) -> list[str]:
        return (
            self.market_service
            .get_current_universe_stock_codes(
                limit=limit,
            )
        )

    def build_stock_targets(
        self,
        *,
        stock_code: str,
        feature_version: str,
    ) -> dict:
        snapshots = (
            self.feature_repository
            .get_stock_snapshots(
                stock_code=(
                    stock_code
                ),
                feature_version=(
                    feature_version
                ),
            )
        )

        if not snapshots:
            raise ValueError(
                "Historical Feature Snapshot이 없습니다: "
                f"{stock_code}"
            )

        first_feature_date = (
            snapshots[
                0
            ].feature_date
        )

        price_rows = (
            self.stock_repository
            .get_daily_prices(
                stock_code=(
                    stock_code
                ),
                start_date=(
                    first_feature_date
                ),
            )
        )

        if not price_rows:
            raise ValueError(
                "종목 일봉이 없습니다: "
                f"{stock_code}"
            )

        price_index_by_date = {
            row.trade_date:
                index
            for index, row
            in enumerate(
                price_rows
            )
        }

        updates: list[
            dict
        ] = []

        filled = {
            horizon: 0
            for horizon
            in self.HORIZONS
        }

        pending = {
            horizon: 0
            for horizon
            in self.HORIZONS
        }

        missing_base_price = 0

        for snapshot in snapshots:
            price_index = (
                price_index_by_date
                .get(
                    snapshot
                    .feature_date
                )
            )

            target_values = {
                1: None,
                5: None,
                20: None,
            }

            if price_index is None:
                missing_base_price += 1

                for horizon in (
                    self.HORIZONS
                ):
                    pending[
                        horizon
                    ] += 1

            else:
                base_close = float(
                    price_rows[
                        price_index
                    ].close
                )

                if base_close <= 0.0:
                    missing_base_price += 1

                    for horizon in (
                        self.HORIZONS
                    ):
                        pending[
                            horizon
                        ] += 1

                else:
                    for horizon in (
                        self.HORIZONS
                    ):
                        future_index = (
                            price_index
                            + horizon
                        )

                        if (
                            future_index
                            >= len(
                                price_rows
                            )
                        ):
                            pending[
                                horizon
                            ] += 1

                            continue

                        future_close = float(
                            price_rows[
                                future_index
                            ].close
                        )

                        if future_close <= 0.0:
                            pending[
                                horizon
                            ] += 1

                            continue

                        target_values[
                            horizon
                        ] = (
                            future_close
                            / base_close
                            - 1.0
                        )

                        filled[
                            horizon
                        ] += 1

            updates.append(
                {
                    "id":
                        snapshot.id,

                    "target_return_1d":
                        target_values[
                            1
                        ],

                    "target_return_5d":
                        target_values[
                            5
                        ],

                    "target_return_20d":
                        target_values[
                            20
                        ],
                }
            )

        updated = (
            self.feature_repository
            .update_targets(
                updates
            )
        )

        return {
            "stock_code":
                stock_code,

            "snapshots":
                len(
                    snapshots
                ),

            "updated":
                updated,

            "filled_1d":
                filled[
                    1
                ],

            "filled_5d":
                filled[
                    5
                ],

            "filled_20d":
                filled[
                    20
                ],

            "pending_1d":
                pending[
                    1
                ],

            "pending_5d":
                pending[
                    5
                ],

            "pending_20d":
                pending[
                    20
                ],

            "missing_base_price":
                missing_base_price,
        }