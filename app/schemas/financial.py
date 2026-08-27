from pydantic import BaseModel


class FinancialAnalysisResponse(BaseModel):
    stockCode: str
    stockName: str
    businessYear: str
    reportCode: str
    fsDiv: str

    revenue: float | None
    operatingIncome: float | None
    netIncome: float | None
    totalAssets: float | None
    totalLiabilities: float | None
    totalEquity: float | None
    operatingCashFlow: float | None

    revenueGrowth: float | None
    operatingIncomeGrowth: float | None
    netIncomeGrowth: float | None
    operatingMargin: float | None
    netMargin: float | None
    roe: float | None
    roa: float | None
    debtRatio: float | None
    equityRatio: float | None
    operatingCashFlowMargin: float | None
    cashFlowQuality: float | None

    growthScore: float
    profitabilityScore: float
    stabilityScore: float
    cashFlowScore: float
    financialScore: float
    dataCompleteness: float
    analysisNote: str


class FinancialBatchResponse(BaseModel):
    success: bool = True
    analyzedCount: int
    skippedCount: int
    message: str
