from __future__ import annotations

from decimal import Decimal

from sqlalchemy.orm import Session

from app.models.financial_metric import FinancialMetric
from app.models.financial_statement import FinancialStatement
from app.repositories.stock_repository import StockRepository
from app.schemas.financial import FinancialAnalysisResponse
from app.utils.financial_scoring import (
    clamp_score,
    growth_rate,
    piecewise_score,
    safe_ratio,
    weighted_score,
)


class FinancialAnalysisService:
    ACCOUNT_RULES = {
        "revenue": {
            "ids": [
                "ifrs-full_Revenue",
                "ifrs-full_RevenueFromContractsWithCustomers",
                "dart_Revenue",
            ],
            "names": [
                "매출액",
                "수익(매출액)",
                "영업수익",
                "매출",
            ],
            "statements": ["IS", "CIS"],
        },
        "operating_income": {
            "ids": [
                "dart_OperatingIncomeLoss",
                "ifrs-full_ProfitLossFromOperatingActivities",
            ],
            "names": [
                "영업이익",
                "영업이익(손실)",
                "영업손익",
            ],
            "statements": ["IS", "CIS"],
        },
        "net_income": {
            "ids": [
                "ifrs-full_ProfitLoss",
                "ifrs-full_ProfitLossAttributableToOwnersOfParent",
            ],
            "names": [
                "당기순이익",
                "당기순이익(손실)",
                "연결당기순이익",
                "지배기업 소유주지분 순이익",
            ],
            "statements": ["IS", "CIS"],
        },
        "total_assets": {
            "ids": ["ifrs-full_Assets"],
            "names": ["자산총계", "자산 총계"],
            "statements": ["BS"],
        },
        "total_liabilities": {
            "ids": ["ifrs-full_Liabilities"],
            "names": ["부채총계", "부채 총계"],
            "statements": ["BS"],
        },
        "total_equity": {
            "ids": ["ifrs-full_Equity"],
            "names": ["자본총계", "자본 총계"],
            "statements": ["BS"],
        },
        "operating_cash_flow": {
            "ids": [
                "ifrs-full_CashFlowsFromUsedInOperatingActivities",
                "dart_CashFlowsFromUsedInOperatingActivities",
            ],
            "names": [
                "영업활동현금흐름",
                "영업활동으로 인한 현금흐름",
                "영업활동 현금흐름",
            ],
            "statements": ["CF"],
        },
    }

    def __init__(self, db: Session):
        self.db = db
        self.repository = StockRepository(db)

    @staticmethod
    def _normalize(text: str | None) -> str:
        return "".join(str(text or "").lower().split())

    def _pick_row(
        self,
        rows: list[FinancialStatement],
        metric_name: str,
    ) -> FinancialStatement | None:
        rule = self.ACCOUNT_RULES[metric_name]
        ids = [self._normalize(value) for value in rule["ids"]]
        names = [self._normalize(value) for value in rule["names"]]
        statements = set(rule["statements"])

        candidates = [row for row in rows if row.sj_div in statements]

        def rank(row: FinancialStatement) -> tuple[int, int, int]:
            account_id = self._normalize(row.account_id)
            account_name = self._normalize(row.account_nm)
            account_detail = self._normalize(getattr(row, "account_detail", ""))

            id_rank = ids.index(account_id) if account_id in ids else 999
            name_rank = names.index(account_name) if account_name in names else 999
            detail_penalty = 0 if not account_detail else 1
            return (min(id_rank, name_rank), detail_penalty, row.id)

        exact = [
            row
            for row in candidates
            if self._normalize(row.account_id) in ids
            or self._normalize(row.account_nm) in names
        ]
        if exact:
            return sorted(exact, key=rank)[0]

        fuzzy = []
        for row in candidates:
            account_name = self._normalize(row.account_nm)
            if any(name and name in account_name for name in names):
                fuzzy.append(row)

        return sorted(fuzzy, key=rank)[0] if fuzzy else None

    @staticmethod
    def _amount(row: FinancialStatement | None, field: str) -> Decimal | None:
        if row is None:
            return None
        return getattr(row, field)

    @staticmethod
    def _cash_flow_quality(
        operating_cash_flow: Decimal | None,
        net_income: Decimal | None,
    ) -> float | None:
        if operating_cash_flow is None or net_income is None:
            return None

        ocf = float(operating_cash_flow)
        ni = float(net_income)

        if ni > 0:
            return ocf / ni
        if ocf > 0:
            return 1.0
        if ocf < 0:
            return -1.0
        return 0.0

    @staticmethod
    def _build_note(values: dict[str, float | None], scores: dict[str, float]) -> str:
        strengths: list[str] = []
        cautions: list[str] = []

        if values.get("revenue_growth") is not None:
            if values["revenue_growth"] >= 10:
                strengths.append("매출 성장세가 양호함")
            elif values["revenue_growth"] < 0:
                cautions.append("매출이 전년 대비 감소함")

        if values.get("operating_margin") is not None:
            if values["operating_margin"] >= 15:
                strengths.append("영업이익률이 높은 편임")
            elif values["operating_margin"] < 5:
                cautions.append("영업이익률이 낮은 편임")

        if values.get("roe") is not None:
            if values["roe"] >= 15:
                strengths.append("ROE가 양호함")
            elif values["roe"] < 5:
                cautions.append("ROE가 낮음")

        if values.get("debt_ratio") is not None:
            if values["debt_ratio"] <= 100:
                strengths.append("부채비율이 안정적인 편임")
            elif values["debt_ratio"] >= 200:
                cautions.append("부채비율이 높은 편임")

        if values.get("operating_cash_flow") is not None:
            if values["operating_cash_flow"] > 0:
                strengths.append("영업현금흐름이 플러스임")
            else:
                cautions.append("영업현금흐름이 마이너스임")

        if not strengths:
            strengths.append("뚜렷한 강점 신호가 아직 부족함")
        if not cautions:
            cautions.append("핵심 재무지표에서 큰 경고 신호는 적음")

        return (
            f"강점: {', '.join(strengths[:3])}. "
            f"주의: {', '.join(cautions[:3])}. "
            f"재무점수 {scores['financial_score']:.1f}/100."
        )

    def analyze(
        self,
        *,
        stock_code: str,
        business_year: str,
        report_code: str = "11011",
    ) -> FinancialAnalysisResponse:
        stock = self.repository.get_stock(stock_code)
        if stock is None:
            raise ValueError(f"등록되지 않은 종목입니다: {stock_code}")

        rows = self.repository.get_financial_rows(
            stock_code=stock_code,
            business_year=business_year,
            report_code=report_code,
        )
        if not rows:
            raise ValueError(
                f"{stock_code} {business_year} {report_code} 재무제표가 없습니다. "
                "먼저 /v1/admin/sync/financials/{stock_code}를 실행하세요."
            )

        fs_div = rows[0].fs_div
        picked = {name: self._pick_row(rows, name) for name in self.ACCOUNT_RULES}

        revenue = self._amount(picked["revenue"], "thstrm_amount")
        operating_income = self._amount(picked["operating_income"], "thstrm_amount")
        net_income = self._amount(picked["net_income"], "thstrm_amount")
        total_assets = self._amount(picked["total_assets"], "thstrm_amount")
        total_liabilities = self._amount(picked["total_liabilities"], "thstrm_amount")
        total_equity = self._amount(picked["total_equity"], "thstrm_amount")
        operating_cash_flow = self._amount(picked["operating_cash_flow"], "thstrm_amount")

        previous_revenue = self._amount(picked["revenue"], "frmtrm_amount")
        previous_operating_income = self._amount(picked["operating_income"], "frmtrm_amount")
        previous_net_income = self._amount(picked["net_income"], "frmtrm_amount")

        values = {
            "revenue_growth": growth_rate(revenue, previous_revenue),
            "operating_income_growth": growth_rate(operating_income, previous_operating_income),
            "net_income_growth": growth_rate(net_income, previous_net_income),
            "operating_margin": safe_ratio(operating_income, revenue),
            "net_margin": safe_ratio(net_income, revenue),
            "roe": safe_ratio(net_income, total_equity),
            "roa": safe_ratio(net_income, total_assets),
            "debt_ratio": safe_ratio(total_liabilities, total_equity),
            "equity_ratio": safe_ratio(total_equity, total_assets),
            "operating_cash_flow_margin": safe_ratio(operating_cash_flow, revenue),
            "cash_flow_quality": self._cash_flow_quality(operating_cash_flow, net_income),
            "operating_cash_flow": float(operating_cash_flow) if operating_cash_flow is not None else None,
        }

        growth_score = weighted_score([
            (piecewise_score(values["revenue_growth"], [(-20, 0), (-10, 20), (0, 50), (10, 70), (20, 85), (40, 100)]), 0.40),
            (piecewise_score(values["operating_income_growth"], [(-30, 0), (-10, 20), (0, 50), (15, 75), (30, 90), (60, 100)]), 0.35),
            (piecewise_score(values["net_income_growth"], [(-30, 0), (-10, 20), (0, 50), (15, 75), (30, 90), (60, 100)]), 0.25),
        ])

        profitability_score = weighted_score([
            (piecewise_score(values["operating_margin"], [(0, 10), (5, 40), (10, 65), (20, 85), (30, 100)]), 0.35),
            (piecewise_score(values["net_margin"], [(0, 10), (3, 35), (7, 60), (15, 85), (25, 100)]), 0.25),
            (piecewise_score(values["roe"], [(0, 10), (5, 35), (10, 60), (15, 80), (25, 100)]), 0.40),
        ])

        debt_score = None
        if values["debt_ratio"] is not None:
            debt_score = piecewise_score(
                -values["debt_ratio"],
                [(-500, 0), (-300, 15), (-200, 35), (-150, 55), (-100, 75), (-50, 90), (-30, 100)],
            )

        stability_score = weighted_score([
            (debt_score, 0.65),
            (piecewise_score(values["equity_ratio"], [(10, 10), (20, 30), (30, 50), (40, 70), (60, 90), (80, 100)]), 0.35),
        ])

        quality_score = None
        if values["cash_flow_quality"] is not None:
            quality_score = piecewise_score(
                values["cash_flow_quality"],
                [(-1, 0), (0, 10), (0.5, 50), (1.0, 80), (1.5, 100)],
            )

        cash_flow_score = weighted_score([
            (piecewise_score(values["operating_cash_flow_margin"], [(-10, 0), (0, 35), (5, 60), (10, 80), (20, 100)]), 0.60),
            (quality_score, 0.40),
        ])

        core_metrics = [
            revenue,
            operating_income,
            net_income,
            total_assets,
            total_liabilities,
            total_equity,
            operating_cash_flow,
        ]
        completeness = sum(value is not None for value in core_metrics) / len(core_metrics) * 100.0

        growth_score_value = clamp_score(growth_score)
        profitability_score_value = clamp_score(profitability_score)
        stability_score_value = clamp_score(stability_score)
        cash_flow_score_value = clamp_score(cash_flow_score)

        raw_financial_score = weighted_score([
            (growth_score if growth_score is not None else None, 0.30),
            (profitability_score if profitability_score is not None else None, 0.30),
            (stability_score if stability_score is not None else None, 0.25),
            (cash_flow_score if cash_flow_score is not None else None, 0.15),
        ]) or 0.0

        completeness_factor = 0.60 + 0.40 * (completeness / 100.0)
        financial_score_value = clamp_score(raw_financial_score * completeness_factor)

        scores = {
            "growth_score": growth_score_value,
            "profitability_score": profitability_score_value,
            "stability_score": stability_score_value,
            "cash_flow_score": cash_flow_score_value,
            "financial_score": financial_score_value,
        }
        note = self._build_note(values, scores)

        metric = self.repository.upsert_financial_metric(
            {
                "stock_code": stock_code,
                "business_year": business_year,
                "report_code": report_code,
                "fs_div": fs_div,
                "revenue": revenue,
                "operating_income": operating_income,
                "net_income": net_income,
                "total_assets": total_assets,
                "total_liabilities": total_liabilities,
                "total_equity": total_equity,
                "operating_cash_flow": operating_cash_flow,
                "previous_revenue": previous_revenue,
                "previous_operating_income": previous_operating_income,
                "previous_net_income": previous_net_income,
                **{key: values[key] for key in [
                    "revenue_growth",
                    "operating_income_growth",
                    "net_income_growth",
                    "operating_margin",
                    "net_margin",
                    "roe",
                    "roa",
                    "debt_ratio",
                    "equity_ratio",
                    "operating_cash_flow_margin",
                    "cash_flow_quality",
                ]},
                **scores,
                "data_completeness": round(completeness, 2),
                "analysis_note": note,
            }
        )

        return self._to_response(stock.name, metric)

    def get_analysis(
        self,
        *,
        stock_code: str,
        business_year: str,
        report_code: str = "11011",
    ) -> FinancialAnalysisResponse:
        stock = self.repository.get_stock(stock_code)
        if stock is None:
            raise ValueError(f"등록되지 않은 종목입니다: {stock_code}")

        metric = self.repository.get_financial_metric(
            stock_code=stock_code,
            business_year=business_year,
            report_code=report_code,
        )
        if metric is None:
            raise ValueError(
                f"{stock_code} {business_year} 재무분석 결과가 없습니다. "
                "먼저 /v1/admin/analyze/financials/{stock_code}를 실행하세요."
            )

        return self._to_response(stock.name, metric)

    @staticmethod
    def _to_response(stock_name: str, metric: FinancialMetric) -> FinancialAnalysisResponse:
        def money(value):
            return float(value) if value is not None else None

        return FinancialAnalysisResponse(
            stockCode=metric.stock_code,
            stockName=stock_name,
            businessYear=metric.business_year,
            reportCode=metric.report_code,
            fsDiv=metric.fs_div,
            revenue=money(metric.revenue),
            operatingIncome=money(metric.operating_income),
            netIncome=money(metric.net_income),
            totalAssets=money(metric.total_assets),
            totalLiabilities=money(metric.total_liabilities),
            totalEquity=money(metric.total_equity),
            operatingCashFlow=money(metric.operating_cash_flow),
            revenueGrowth=metric.revenue_growth,
            operatingIncomeGrowth=metric.operating_income_growth,
            netIncomeGrowth=metric.net_income_growth,
            operatingMargin=metric.operating_margin,
            netMargin=metric.net_margin,
            roe=metric.roe,
            roa=metric.roa,
            debtRatio=metric.debt_ratio,
            equityRatio=metric.equity_ratio,
            operatingCashFlowMargin=metric.operating_cash_flow_margin,
            cashFlowQuality=metric.cash_flow_quality,
            growthScore=metric.growth_score,
            profitabilityScore=metric.profitability_score,
            stabilityScore=metric.stability_score,
            cashFlowScore=metric.cash_flow_score,
            financialScore=metric.financial_score,
            dataCompleteness=metric.data_completeness,
            analysisNote=metric.analysis_note or "",
        )
