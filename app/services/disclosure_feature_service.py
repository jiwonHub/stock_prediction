from __future__ import annotations

from datetime import date
from datetime import timedelta

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models.future import (
    Disclosure,
    DisclosureAnalysis,
)


DISCLOSURE_FEATURE_NAMES = (
    "disclosure_count_1d",
    "disclosure_count_7d",
    "disclosure_count_30d",
    "disclosure_count_90d",
    "disclosure_net_impact_7d",
    "disclosure_net_impact_30d",
    "disclosure_net_impact_90d",
    "disclosure_positive_impact_30d",
    "disclosure_negative_impact_30d",
    "disclosure_high_importance_count_30d",
    "disclosure_avg_importance_30d",
    "disclosure_correction_count_30d",
    "disclosure_event_diversity_30d",
    "disclosure_earnings_count_30d",
    "disclosure_contract_order_count_30d",
    "disclosure_financing_dilution_count_30d",
    "disclosure_shareholder_return_count_30d",
    "disclosure_mna_restructuring_count_30d",
    "disclosure_legal_risk_count_30d",
    "disclosure_credit_guarantee_count_30d",
    "disclosure_treasury_stock_count_30d",
    "disclosure_management_decision_count_30d",
    "disclosure_ir_guidance_count_30d",
    "disclosure_securities_issuance_count_30d",
)


class DisclosureFeatureService:
    MODEL_NAME = "taxonomy-rule"
    MODEL_VERSION = "v1"

    EVENT_FEATURE_MAP = {
        "EARNINGS":
            "disclosure_earnings_count_30d",

        "CONTRACT_ORDER":
            "disclosure_contract_order_count_30d",

        "FINANCING_DILUTION":
            "disclosure_financing_dilution_count_30d",

        "SHAREHOLDER_RETURN":
            "disclosure_shareholder_return_count_30d",

        "MNA_RESTRUCTURING":
            "disclosure_mna_restructuring_count_30d",

        "LEGAL_RISK":
            "disclosure_legal_risk_count_30d",

        "CREDIT_GUARANTEE":
            "disclosure_credit_guarantee_count_30d",

        "TREASURY_STOCK":
            "disclosure_treasury_stock_count_30d",

        "MANAGEMENT_DECISION":
            "disclosure_management_decision_count_30d",

        "IR_GUIDANCE":
            "disclosure_ir_guidance_count_30d",

        "SECURITIES_ISSUANCE":
            "disclosure_securities_issuance_count_30d",
    }

    def __init__(
        self,
        db: Session,
    ):
        self.db = db

        self._history_cache: dict[
            str,
            list[dict],
        ] = {}

    def _get_history_rows(
        self,
        *,
        stock_code: str,
    ) -> list[dict]:
        cached = (
            self._history_cache
            .get(
                stock_code
            )
        )

        if cached is not None:
            return cached

        rows = (
            self.db.execute(
                select(
                    Disclosure.receipt_date.label(
                        "receipt_date"
                    ),
                    DisclosureAnalysis.event_type.label(
                        "event_type"
                    ),
                    DisclosureAnalysis.direction_score.label(
                        "direction_score"
                    ),
                    DisclosureAnalysis.importance_score.label(
                        "importance_score"
                    ),
                    DisclosureAnalysis.impact_score.label(
                        "impact_score"
                    ),
                    DisclosureAnalysis.is_correction.label(
                        "is_correction"
                    ),
                )
                .join(
                    Disclosure,
                    Disclosure.id
                    == DisclosureAnalysis.disclosure_id,
                )
                .where(
                    DisclosureAnalysis.stock_code
                    == stock_code,
                    DisclosureAnalysis.model_name
                    == self.MODEL_NAME,
                    DisclosureAnalysis.model_version
                    == self.MODEL_VERSION,
                    Disclosure.receipt_date
                    .is_not(None),
                )
                .order_by(
                    Disclosure.receipt_date.asc()
                )
            )
            .mappings()
            .all()
        )

        result = [
            dict(
                row
            )
            for row
            in rows
        ]

        self._history_cache[
            stock_code
        ] = result

        return result

    def build_features(
        self,
        *,
        stock_code: str,
        as_of_date: date,
    ) -> dict[str, float]:
        start_date = (
            as_of_date
            - timedelta(
                days=90
            )
        )

        rows = [
            row
            for row
            in self._get_history_rows(
                stock_code=(
                    stock_code
                )
            )
            if (
                row[
                    "receipt_date"
                ]
                >= start_date
                and row[
                    "receipt_date"
                ]
                < as_of_date
            )
        ]

        features = {
            name: 0.0
            for name
            in DISCLOSURE_FEATURE_NAMES
        }

        windows = {
            1: [],
            7: [],
            30: [],
            90: [],
        }

        for row in rows:
            receipt_date = (
                row[
                    "receipt_date"
                ]
            )

            age_days = (
                as_of_date
                - receipt_date
            ).days

            for window_days in windows:
                if (
                    1
                    <= age_days
                    <= window_days
                ):
                    windows[
                        window_days
                    ].append(
                        row
                    )

        rows_1d = windows[1]
        rows_7d = windows[7]
        rows_30d = windows[30]
        rows_90d = windows[90]

        features[
            "disclosure_count_1d"
        ] = float(
            len(
                rows_1d
            )
        )

        features[
            "disclosure_count_7d"
        ] = float(
            len(
                rows_7d
            )
        )

        features[
            "disclosure_count_30d"
        ] = float(
            len(
                rows_30d
            )
        )

        features[
            "disclosure_count_90d"
        ] = float(
            len(
                rows_90d
            )
        )

        features[
            "disclosure_net_impact_7d"
        ] = float(
            sum(
                float(
                    row[
                        "impact_score"
                    ]
                    or 0.0
                )
                for row
                in rows_7d
            )
        )

        features[
            "disclosure_net_impact_30d"
        ] = float(
            sum(
                float(
                    row[
                        "impact_score"
                    ]
                    or 0.0
                )
                for row
                in rows_30d
            )
        )

        features[
            "disclosure_net_impact_90d"
        ] = float(
            sum(
                float(
                    row[
                        "impact_score"
                    ]
                    or 0.0
                )
                for row
                in rows_90d
            )
        )

        features[
            "disclosure_positive_impact_30d"
        ] = float(
            sum(
                max(
                    float(
                        row[
                            "impact_score"
                        ]
                        or 0.0
                    ),
                    0.0,
                )
                for row
                in rows_30d
            )
        )

        features[
            "disclosure_negative_impact_30d"
        ] = float(
            sum(
                min(
                    float(
                        row[
                            "impact_score"
                        ]
                        or 0.0
                    ),
                    0.0,
                )
                for row
                in rows_30d
            )
        )

        features[
            "disclosure_high_importance_count_30d"
        ] = float(
            sum(
                1
                for row
                in rows_30d
                if float(
                    row[
                        "importance_score"
                    ]
                    or 0.0
                )
                >= 0.80
            )
        )

        if rows_30d:
            features[
                "disclosure_avg_importance_30d"
            ] = float(
                sum(
                    float(
                        row[
                            "importance_score"
                        ]
                        or 0.0
                    )
                    for row
                    in rows_30d
                )
                / len(
                    rows_30d
                )
            )

        features[
            "disclosure_correction_count_30d"
        ] = float(
            sum(
                1
                for row
                in rows_30d
                if bool(
                    row[
                        "is_correction"
                    ]
                )
            )
        )

        features[
            "disclosure_event_diversity_30d"
        ] = float(
            len(
                {
                    str(
                        row[
                            "event_type"
                        ]
                    )
                    for row
                    in rows_30d
                    if (
                        row[
                            "event_type"
                        ]
                        and row[
                            "event_type"
                        ]
                        != "OTHER"
                    )
                }
            )
        )

        for row in rows_30d:
            event_type = str(
                row[
                    "event_type"
                ]
                or ""
            )

            feature_name = (
                self.EVENT_FEATURE_MAP.get(
                    event_type
                )
            )

            if feature_name:
                features[
                    feature_name
                ] += 1.0

        return features

    def inspect(
        self,
        *,
        stock_code: str,
        as_of_date: date,
    ) -> dict:
        features = (
            self.build_features(
                stock_code=(
                    stock_code
                ),
                as_of_date=(
                    as_of_date
                ),
            )
        )

        invalid = []

        for (
            feature_name,
            value,
        ) in features.items():
            if not isinstance(
                value,
                (int, float),
            ):
                invalid.append(
                    feature_name
                )

        return {
            "status":
                (
                    "pass"
                    if not invalid
                    else "fail"
                ),

            "stock_code":
                stock_code,

            "as_of_date":
                as_of_date.isoformat(),

            "cutoff_rule":
                (
                    "receipt_date < "
                    "as_of_date"
                ),

            "model_name":
                self.MODEL_NAME,

            "model_version":
                self.MODEL_VERSION,

            "feature_count":
                len(
                    features
                ),

            "expected_feature_count":
                len(
                    DISCLOSURE_FEATURE_NAMES
                ),

            "invalid_features":
                invalid,

            "features":
                features,
        }