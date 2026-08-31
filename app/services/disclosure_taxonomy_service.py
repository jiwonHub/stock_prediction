from __future__ import annotations

from collections import Counter
from collections import defaultdict

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models.future import Disclosure
from app.utils.disclosure_event_taxonomy import (
    DISCLOSURE_EVENT_TYPES,
    classify_disclosure_event,
)


class DisclosureTaxonomyService:
    def __init__(
        self,
        db: Session,
    ):
        self.db = db

    def inspect(
        self,
        *,
        stock_codes: list[str],
        sample_per_type: int = 5,
    ) -> dict:
        if not stock_codes:
            raise ValueError(
                "분석할 종목이 없습니다."
            )

        stmt = (
            select(
                Disclosure.stock_code,
                Disclosure.report_name,
                Disclosure.receipt_date,
            )
            .where(
                Disclosure.stock_code
                .in_(
                    stock_codes
                )
            )
            .order_by(
                Disclosure.receipt_date
                .asc()
            )
        )

        rows = (
            self.db.execute(
                stmt
            )
            .all()
        )

        if not rows:
            raise ValueError(
                "분석할 공시가 없습니다."
            )

        event_counts = Counter()
        event_stocks = defaultdict(
            set
        )

        examples = defaultdict(
            list
        )

        unclassified = Counter()

        correction_count = 0

        positive_count = 0
        neutral_count = 0
        negative_count = 0

        dates = []

        for (
            stock_code,
            report_name,
            receipt_date,
        ) in rows:
            result = (
                classify_disclosure_event(
                    report_name
                )
            )

            event_type = (
                result[
                    "event_type"
                ]
            )

            event_counts[
                event_type
            ] += 1

            if stock_code:
                event_stocks[
                    event_type
                ].add(
                    stock_code
                )

            if receipt_date:
                dates.append(
                    receipt_date
                )

            if result[
                "is_correction"
            ]:
                correction_count += 1

            direction = float(
                result[
                    "direction_prior"
                ]
            )

            if direction > 0.0:
                positive_count += 1

            elif direction < 0.0:
                negative_count += 1

            else:
                neutral_count += 1

            if (
                len(
                    examples[
                        event_type
                    ]
                )
                < sample_per_type
            ):
                example = {
                    "stock_code":
                        stock_code,

                    "report_name":
                        report_name,

                    "matched_keyword":
                        result[
                            "matched_keyword"
                        ],

                    "direction_prior":
                        direction,

                    "importance_prior":
                        result[
                            "importance_prior"
                        ],
                }

                if (
                    example
                    not in examples[
                        event_type
                    ]
                ):
                    examples[
                        event_type
                    ].append(
                        example
                    )

            if event_type == "OTHER":
                unclassified[
                    report_name
                ] += 1

        total = len(
            rows
        )

        other_count = (
            event_counts[
                "OTHER"
            ]
        )

        classified_count = (
            total
            - other_count
        )

        event_summary = []

        for event_type in (
            DISCLOSURE_EVENT_TYPES
        ):
            count = (
                event_counts[
                    event_type
                ]
            )

            event_summary.append(
                {
                    "event_type":
                        event_type,

                    "count":
                        count,

                    "rate_pct":
                        round(
                            count
                            / total
                            * 100.0,
                            4,
                        ),

                    "stock_count":
                        len(
                            event_stocks[
                                event_type
                            ]
                        ),

                    "examples":
                        examples[
                            event_type
                        ],
                }
            )

        top_unclassified = [
            {
                "report_name":
                    report_name,

                "count":
                    count,
            }
            for (
                report_name,
                count,
            ) in (
                unclassified
                .most_common(
                    30
                )
            )
        ]

        return {
            "status":
                "pass",

            "stock_count":
                len(
                    stock_codes
                ),

            "disclosure_count":
                total,

            "first_date":
                (
                    min(
                        dates
                    ).isoformat()
                    if dates
                    else None
                ),

            "last_date":
                (
                    max(
                        dates
                    ).isoformat()
                    if dates
                    else None
                ),

            "classified_count":
                classified_count,

            "other_count":
                other_count,

            "match_rate_pct":
                round(
                    classified_count
                    / total
                    * 100.0,
                    4,
                ),

            "correction_count":
                correction_count,

            "correction_rate_pct":
                round(
                    correction_count
                    / total
                    * 100.0,
                    4,
                ),

            "direction_prior": {
                "positive":
                    positive_count,

                "neutral":
                    neutral_count,

                "negative":
                    negative_count,
            },

            "event_types":
                event_summary,

            "top_unclassified":
                top_unclassified,
        }