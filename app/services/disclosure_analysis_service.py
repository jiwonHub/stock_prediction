from __future__ import annotations

from collections import Counter
from datetime import datetime

from sqlalchemy import select
from sqlalchemy.dialects.postgresql import (
    insert as pg_insert,
)
from sqlalchemy.orm import Session

from app.models.future import (
    Disclosure,
    DisclosureAnalysis,
)
from app.utils.disclosure_event_taxonomy import (
    classify_disclosure_event,
)


class DisclosureAnalysisService:
    MODEL_NAME = "taxonomy-rule"
    MODEL_VERSION = "v1"

    def __init__(
        self,
        db: Session,
    ):
        self.db = db

    def analyze_stock(
        self,
        *,
        stock_code: str,
    ) -> dict:
        disclosures = list(
            self.db.scalars(
                select(
                    Disclosure
                )
                .where(
                    Disclosure.stock_code
                    == stock_code
                )
                .order_by(
                    Disclosure.receipt_date.asc(),
                    Disclosure.id.asc(),
                )
            )
            .all()
        )

        if not disclosures:
            return {
                "stock_code": stock_code,
                "processed": 0,
                "created": 0,
                "updated": 0,
                "other": 0,
                "event_counts": {},
            }

        disclosure_ids = [
            disclosure.id
            for disclosure
            in disclosures
        ]

        existing_ids = set(
            self.db.scalars(
                select(
                    DisclosureAnalysis.disclosure_id
                )
                .where(
                    DisclosureAnalysis.disclosure_id.in_(
                        disclosure_ids
                    ),
                    DisclosureAnalysis.model_name
                    == self.MODEL_NAME,
                    DisclosureAnalysis.model_version
                    == self.MODEL_VERSION,
                )
            )
            .all()
        )

        payloads: list[dict] = []

        event_counts = Counter()

        other_count = 0

        analyzed_at = (
            datetime.utcnow()
        )

        for disclosure in disclosures:
            result = (
                classify_disclosure_event(
                    disclosure.report_name
                )
            )

            event_type = str(
                result[
                    "event_type"
                ]
            )

            direction_score = float(
                result[
                    "direction_prior"
                ]
            )

            importance_score = float(
                result[
                    "importance_prior"
                ]
            )

            impact_score = (
                direction_score
                * importance_score
            )

            event_counts[
                event_type
            ] += 1

            if event_type == "OTHER":
                other_count += 1

            matched_keyword = (
                result[
                    "matched_keyword"
                ]
            )

            is_correction = bool(
                result[
                    "is_correction"
                ]
            )

            normalized_name = str(
                result[
                    "normalized_name"
                ]
            )

            payloads.append(
                {
                    "disclosure_id":
                        disclosure.id,

                    "stock_code":
                        stock_code,

                    "model_name":
                        self.MODEL_NAME,

                    "model_version":
                        self.MODEL_VERSION,

                    "event_type":
                        event_type,

                    "direction_score":
                        direction_score,

                    "importance_score":
                        importance_score,

                    "impact_score":
                        impact_score,

                    "matched_keyword":
                        matched_keyword,

                    "is_correction":
                        is_correction,

                    "normalized_name":
                        normalized_name,

                    "rationale":
                        (
                            "taxonomy keyword="
                            f"{matched_keyword}"
                            if matched_keyword
                            else
                            "taxonomy fallback=OTHER"
                        ),

                    "analysis_json": {
                        "report_name":
                            disclosure.report_name,

                        "receipt_no":
                            disclosure.receipt_no,

                        "receipt_date":
                            (
                                disclosure
                                .receipt_date
                                .isoformat()
                                if disclosure.receipt_date
                                else None
                            ),

                        "direction_prior":
                            direction_score,

                        "importance_prior":
                            importance_score,
                    },

                    "analyzed_at":
                        analyzed_at,
                }
            )

        chunk_size = 1000

        for start in range(
            0,
            len(payloads),
            chunk_size,
        ):
            chunk = payloads[
                start:
                start + chunk_size
            ]

            stmt = (
                pg_insert(
                    DisclosureAnalysis
                )
                .values(
                    chunk
                )
            )

            stmt = (
                stmt.on_conflict_do_update(
                    constraint=(
                        "uq_disclosure_analysis"
                    ),
                    set_={
                        "stock_code":
                            stmt.excluded.stock_code,

                        "event_type":
                            stmt.excluded.event_type,

                        "direction_score":
                            stmt.excluded.direction_score,

                        "importance_score":
                            stmt.excluded.importance_score,

                        "impact_score":
                            stmt.excluded.impact_score,

                        "matched_keyword":
                            stmt.excluded.matched_keyword,

                        "is_correction":
                            stmt.excluded.is_correction,

                        "normalized_name":
                            stmt.excluded.normalized_name,

                        "rationale":
                            stmt.excluded.rationale,

                        "analysis_json":
                            stmt.excluded.analysis_json,

                        "analyzed_at":
                            stmt.excluded.analyzed_at,
                    },
                )
            )

            self.db.execute(
                stmt
            )

        self.db.commit()

        created = sum(
            1
            for disclosure
            in disclosures
            if disclosure.id
            not in existing_ids
        )

        updated = (
            len(disclosures)
            - created
        )

        return {
            "stock_code":
                stock_code,

            "processed":
                len(disclosures),

            "created":
                created,

            "updated":
                updated,

            "other":
                other_count,

            "event_counts":
                dict(
                    event_counts
                ),
        }