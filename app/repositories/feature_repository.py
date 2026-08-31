from __future__ import annotations
from sqlalchemy import select, update
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.orm import Session

from app.models.future import (
    StockFeatureSnapshot,
)


class FeatureRepository:
    def __init__(
        self,
        db: Session,
    ):
        self.db = db

    def upsert_snapshots(
        self,
        rows: list[dict],
    ) -> int:
        if not rows:
            return 0

        stmt = insert(
            StockFeatureSnapshot
        ).values(
            rows
        )

        stmt = (
            stmt
            .on_conflict_do_update(
                index_elements=[
                    StockFeatureSnapshot.stock_code,
                    StockFeatureSnapshot.feature_date,
                    StockFeatureSnapshot.feature_version,
                ],
                set_={
                    "features":
                        stmt.excluded.features,
                },
            )
        )

        self.db.execute(
            stmt
        )

        self.db.commit()

        return len(
            rows
        )
    
    def get_stock_snapshots(
        self,
        *,
        stock_code: str,
        feature_version: str,
    ) -> list[StockFeatureSnapshot]:
        stmt = (
            select(
                StockFeatureSnapshot
            )
            .where(
                StockFeatureSnapshot.stock_code
                == stock_code,
                StockFeatureSnapshot.feature_version
                == feature_version,
            )
            .order_by(
                StockFeatureSnapshot
                .feature_date
                .asc()
            )
        )

        return list(
            self.db.scalars(
                stmt
            ).all()
        )

    def update_targets(
        self,
        rows: list[dict],
        *,
        batch_size: int = 1000,
    ) -> int:
        if not rows:
            return 0

        updated = 0

        for start in range(
            0,
            len(rows),
            batch_size,
        ):
            batch = rows[
                start:
                start + batch_size
            ]

            self.db.execute(
                update(
                    StockFeatureSnapshot
                ).execution_options(
                    synchronize_session=False
                ),
                batch,
            )

            self.db.commit()

            updated += len(
                batch
            )

        return updated