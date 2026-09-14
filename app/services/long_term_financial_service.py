from __future__ import annotations

from statistics import (
    mean,
    pstdev,
)

from sqlalchemy.orm import Session

from app.models.financial_statement import (
    FinancialStatement,
)
from app.repositories.stock_repository import (
    StockRepository,
)
from app.services.financial_analysis_service import (
    FinancialAnalysisService,
)
from app.utils.financial_scoring import (
    safe_ratio,
)


class LongTermFinancialService:
    HISTORY_YEARS = 5
    REPORT_CODE = "11011"

    CAPEX_RULES = {
        "property_plant_equipment_acquisition": {
            "names": [
                "유형자산의 취득",
                "유형자산 취득",
                "유형자산의취득",
                "유형자산취득",
            ],
            "statements": [
                "CF",
            ],
        },
        "intangible_asset_acquisition": {
            "names": [
                "무형자산의 취득",
                "무형자산 취득",
                "무형자산의취득",
                "무형자산취득",
            ],
            "statements": [
                "CF",
            ],
        },
    }

    def __init__(
        self,
        db: Session,
    ):
        self.db = db

        self.repository = (
            StockRepository(
                db
            )
        )

        self.financial_service = (
            FinancialAnalysisService(
                db
            )
        )

    @staticmethod
    def _normalize(
        text: str | None,
    ) -> str:
        return "".join(
            str(
                text
                or ""
            )
            .lower()
            .split()
        )

    @staticmethod
    def _number(
        value,
    ) -> float | None:
        if value is None:
            return None

        return float(
            value
        )

    @staticmethod
    def _preferred_rows(
        rows: list[
            FinancialStatement
        ],
    ) -> list[
        FinancialStatement
    ]:
        if not rows:
            return []

        cfs_rows = [
            row
            for row
            in rows
            if row.fs_div
            == "CFS"
        ]

        if cfs_rows:
            return cfs_rows

        ofs_rows = [
            row
            for row
            in rows
            if row.fs_div
            == "OFS"
        ]

        return (
            ofs_rows
            or rows
        )

    def _pick_capex_row(
        self,
        rows: list[
            FinancialStatement
        ],
        rule_name: str,
    ) -> (
        FinancialStatement
        | None
    ):
        rule = (
            self.CAPEX_RULES[
                rule_name
            ]
        )

        names = [
            self._normalize(
                value
            )
            for value
            in rule[
                "names"
            ]
        ]

        statements = set(
            rule[
                "statements"
            ]
        )

        candidates = [
            row
            for row
            in rows
            if row.sj_div
            in statements
        ]

        exact = [
            row
            for row
            in candidates
            if self._normalize(
                row.account_nm
            )
            in names
        ]

        if exact:
            return sorted(
                exact,
                key=lambda row: (
                    (
                        0
                        if not row.account_detail
                        else 1
                    ),
                    row.id,
                ),
            )[0]

        fuzzy = []

        for row in candidates:
            account_name = (
                self._normalize(
                    row.account_nm
                )
            )

            if any(
                name
                and name
                in account_name
                for name
                in names
            ):
                fuzzy.append(
                    row
                )

        if not fuzzy:
            return None

        return sorted(
            fuzzy,
            key=lambda row: (
                (
                    0
                    if not row.account_detail
                    else 1
                ),
                row.id,
            ),
        )[0]

    def _extract_period(
        self,
        *,
        rows: list[
            FinancialStatement
        ],
        amount_field: str,
        year: int,
    ) -> dict:
        rows = (
            self._preferred_rows(
                rows
            )
        )

        picked = {
            name:
                self.financial_service
                ._pick_row(
                    rows,
                    name,
                )
            for name
            in (
                "revenue",
                "operating_income",
                "net_income",
                "total_assets",
                "total_liabilities",
                "total_equity",
                "operating_cash_flow",
            )
        }

        def amount(
            metric_name: str,
        ) -> float | None:
            row = (
                picked[
                    metric_name
                ]
            )

            if row is None:
                return None

            return self._number(
                getattr(
                    row,
                    amount_field,
                )
            )

        revenue = amount(
            "revenue"
        )

        operating_income = amount(
            "operating_income"
        )

        net_income = amount(
            "net_income"
        )

        total_assets = amount(
            "total_assets"
        )

        total_liabilities = amount(
            "total_liabilities"
        )

        total_equity = amount(
            "total_equity"
        )

        operating_cash_flow = amount(
            "operating_cash_flow"
        )

        capex_values: list[
            float
        ] = []

        for rule_name in (
            self.CAPEX_RULES
        ):
            row = (
                self._pick_capex_row(
                    rows,
                    rule_name,
                )
            )

            if row is None:
                continue

            value = self._number(
                getattr(
                    row,
                    amount_field,
                )
            )

            if value is not None:
                capex_values.append(
                    abs(
                        value
                    )
                )

        capital_expenditure = (
            sum(
                capex_values
            )
            if capex_values
            else None
        )

        free_cash_flow = None

        if (
            operating_cash_flow
            is not None
            and capital_expenditure
            is not None
        ):
            free_cash_flow = (
                operating_cash_flow
                - capital_expenditure
            )

        return {
            "year":
                str(
                    year
                ),

            "revenue":
                revenue,

            "operatingIncome":
                operating_income,

            "netIncome":
                net_income,

            "totalAssets":
                total_assets,

            "totalLiabilities":
                total_liabilities,

            "totalEquity":
                total_equity,

            "operatingCashFlow":
                operating_cash_flow,

            "capitalExpenditure":
                capital_expenditure,

            "freeCashFlow":
                free_cash_flow,

            "operatingMargin":
                safe_ratio(
                    operating_income,
                    revenue,
                ),

            "roe":
                safe_ratio(
                    net_income,
                    total_equity,
                ),

            "debtRatio":
                safe_ratio(
                    total_liabilities,
                    total_equity,
                ),

            "operatingCashFlowMargin":
                safe_ratio(
                    operating_cash_flow,
                    revenue,
                ),

            "freeCashFlowMargin":
                safe_ratio(
                    free_cash_flow,
                    revenue,
                ),
        }

    @staticmethod
    def _cagr(
        *,
        start_value: float | None,
        end_value: float | None,
        periods: int,
    ) -> float | None:
        if (
            start_value is None
            or end_value is None
            or start_value <= 0.0
            or end_value <= 0.0
            or periods <= 0
        ):
            return None

        return (
            (
                (
                    end_value
                    / start_value
                )
                ** (
                    1.0
                    / periods
                )
            )
            - 1.0
        ) * 100.0

    @staticmethod
    def _ratio(
        values: list[
            float | None
        ],
        predicate,
    ) -> float | None:
        available = [
            value
            for value
            in values
            if value is not None
        ]

        if not available:
            return None

        matched = sum(
            1
            for value
            in available
            if predicate(
                value
            )
        )

        return (
            matched
            / len(
                available
            )
            * 100.0
        )

    @staticmethod
    def _trend_state(
        value: float | None,
        *,
        improving_threshold: float,
        deteriorating_threshold: float,
    ) -> str:
        if value is None:
            return "UNAVAILABLE"

        if (
            value
            >= improving_threshold
        ):
            return "IMPROVING"

        if (
            value
            <= deteriorating_threshold
        ):
            return "DETERIORATING"

        return "STABLE"

    def build(
        self,
        *,
        stock_code: str,
    ) -> dict:
        rows = (
            self.repository
            .get_annual_financial_rows(
                stock_code=(
                    stock_code
                ),
                report_code=(
                    self.REPORT_CODE
                ),
            )
        )

        if not rows:
            return self._empty()

        rows_by_report_year: dict[
            int,
            list[
                FinancialStatement
            ],
        ] = {}

        for row in rows:
            try:
                report_year = int(
                    row.business_year
                )
            except ValueError:
                continue

            rows_by_report_year.setdefault(
                report_year,
                [],
            ).append(
                row
            )

        yearly: dict[
            int,
            dict,
        ] = {}

        for report_year in sorted(
            rows_by_report_year,
            reverse=True,
        ):
            report_rows = (
                rows_by_report_year[
                    report_year
                ]
            )

            for (
                amount_field,
                offset,
            ) in (
                (
                    "thstrm_amount",
                    0,
                ),
                (
                    "frmtrm_amount",
                    -1,
                ),
                (
                    "bfefrmtrm_amount",
                    -2,
                ),
            ):
                year = (
                    report_year
                    + offset
                )

                if year in yearly:
                    continue

                point = (
                    self._extract_period(
                        rows=(
                            report_rows
                        ),
                        amount_field=(
                            amount_field
                        ),
                        year=year,
                    )
                )

                core_values = (
                    point[
                        "revenue"
                    ],
                    point[
                        "operatingIncome"
                    ],
                    point[
                        "netIncome"
                    ],
                    point[
                        "operatingCashFlow"
                    ],
                )

                if not any(
                    value is not None
                    for value
                    in core_values
                ):
                    continue

                yearly[
                    year
                ] = point

        if not yearly:
            return self._empty()

        selected_years = sorted(
            yearly,
            reverse=True,
        )[
            :self.HISTORY_YEARS
        ]

        points = [
            yearly[
                year
            ]
            for year
            in sorted(
                selected_years
            )
        ]

        first = (
            points[
                0
            ]
        )

        latest = (
            points[
                -1
            ]
        )

        year_span = (
            int(
                latest[
                    "year"
                ]
            )
            - int(
                first[
                    "year"
                ]
            )
        )

        roe_values = [
            float(
                point[
                    "roe"
                ]
            )
            for point
            in points
            if point[
                "roe"
            ]
            is not None
        ]

        operating_margin_change = None

        if (
            first[
                "operatingMargin"
            ]
            is not None
            and latest[
                "operatingMargin"
            ]
            is not None
        ):
            operating_margin_change = (
                float(
                    latest[
                        "operatingMargin"
                    ]
                )
                - float(
                    first[
                        "operatingMargin"
                    ]
                )
            )

        debt_ratio_change = None

        if (
            first[
                "debtRatio"
            ]
            is not None
            and latest[
                "debtRatio"
            ]
            is not None
        ):
            debt_ratio_change = (
                float(
                    latest[
                        "debtRatio"
                    ]
                )
                - float(
                    first[
                        "debtRatio"
                    ]
                )
            )

        debt_improvement = (
            -debt_ratio_change
            if debt_ratio_change
            is not None
            else None
        )

        revenue_cagr = (
            self._cagr(
                start_value=(
                    first[
                        "revenue"
                    ]
                ),
                end_value=(
                    latest[
                        "revenue"
                    ]
                ),
                periods=year_span,
            )
        )

        operating_income_cagr = (
            self._cagr(
                start_value=(
                    first[
                        "operatingIncome"
                    ]
                ),
                end_value=(
                    latest[
                        "operatingIncome"
                    ]
                ),
                periods=year_span,
            )
        )

        net_income_cagr = (
            self._cagr(
                start_value=(
                    first[
                        "netIncome"
                    ]
                ),
                end_value=(
                    latest[
                        "netIncome"
                    ]
                ),
                periods=year_span,
            )
        )

        free_cash_flow_cagr = (
            self._cagr(
                start_value=(
                    first[
                        "freeCashFlow"
                    ]
                ),
                end_value=(
                    latest[
                        "freeCashFlow"
                    ]
                ),
                periods=year_span,
            )
        )

        return {
            "available":
                len(
                    points
                )
                >= 3,

            "startYear":
                first[
                    "year"
                ],

            "endYear":
                latest[
                    "year"
                ],

            "yearCount":
                len(
                    points
                ),

            "revenueCagr":
                self._round(
                    revenue_cagr
                ),

            "operatingIncomeCagr":
                self._round(
                    operating_income_cagr
                ),

            "netIncomeCagr":
                self._round(
                    net_income_cagr
                ),

            "freeCashFlowCagr":
                self._round(
                    free_cash_flow_cagr
                ),

            "averageRoe":
                self._round(
                    (
                        mean(
                            roe_values
                        )
                        if roe_values
                        else None
                    )
                ),

            "roeStdDev":
                self._round(
                    (
                        pstdev(
                            roe_values
                        )
                        if len(
                            roe_values
                        )
                        >= 2
                        else (
                            0.0
                            if roe_values
                            else None
                        )
                    )
                ),

            "roePositiveYearRatio":
                self._round(
                    self._ratio(
                        [
                            point[
                                "roe"
                            ]
                            for point
                            in points
                        ],
                        lambda value:
                            value
                            > 0.0,
                    ),
                    1,
                ),

            "operatingMarginChange":
                self._round(
                    operating_margin_change
                ),

            "operatingMarginTrend":
                self._trend_state(
                    operating_margin_change,
                    improving_threshold=(
                        2.0
                    ),
                    deteriorating_threshold=(
                        -2.0
                    ),
                ),

            "debtRatioChange":
                self._round(
                    debt_ratio_change
                ),

            "debtRatioTrend":
                self._trend_state(
                    debt_improvement,
                    improving_threshold=(
                        20.0
                    ),
                    deteriorating_threshold=(
                        -20.0
                    ),
                ),

            "positiveOperatingCashFlowRatio":
                self._round(
                    self._ratio(
                        [
                            point[
                                "operatingCashFlow"
                            ]
                            for point
                            in points
                        ],
                        lambda value:
                            value
                            > 0.0,
                    ),
                    1,
                ),

            "positiveFreeCashFlowRatio":
                self._round(
                    self._ratio(
                        [
                            point[
                                "freeCashFlow"
                            ]
                            for point
                            in points
                        ],
                        lambda value:
                            value
                            > 0.0,
                    ),
                    1,
                ),

            "latestFreeCashFlow":
                self._round(
                    latest[
                        "freeCashFlow"
                    ],
                    0,
                ),

            "latestFreeCashFlowMargin":
                self._round(
                    latest[
                        "freeCashFlowMargin"
                    ]
                ),

            "points":
                points,
        }

    @staticmethod
    def _round(
        value: float | None,
        digits: int = 2,
    ) -> float | None:
        if value is None:
            return None

        return round(
            float(
                value
            ),
            digits,
        )

    @staticmethod
    def _empty() -> dict:
        return {
            "available":
                False,

            "startYear":
                None,

            "endYear":
                None,

            "yearCount":
                0,

            "revenueCagr":
                None,

            "operatingIncomeCagr":
                None,

            "netIncomeCagr":
                None,

            "freeCashFlowCagr":
                None,

            "averageRoe":
                None,

            "roeStdDev":
                None,

            "roePositiveYearRatio":
                None,

            "operatingMarginChange":
                None,

            "operatingMarginTrend":
                "UNAVAILABLE",

            "debtRatioChange":
                None,

            "debtRatioTrend":
                "UNAVAILABLE",

            "positiveOperatingCashFlowRatio":
                None,

            "positiveFreeCashFlowRatio":
                None,

            "latestFreeCashFlow":
                None,

            "latestFreeCashFlowMargin":
                None,

            "points":
                [],
        }