from pydantic import BaseModel, ConfigDict

from app.schemas.financial import (
    FinancialAnalysisResponse,
)
from app.schemas.ranking import (
    RankingExplanationResponse,
)


class StockResponse(BaseModel):
    model_config = ConfigDict(
        populate_by_name=True,
    )

    code: str
    name: str
    market: str
    currentPrice: float
    change: float
    changeRate: float
    marketCap: float


class ChartPointResponse(BaseModel):
    time: str
    price: float

    open: float | None = None
    high: float | None = None
    low: float | None = None
    close: float | None = None
    volume: int | None = None


class ChartPageResponse(BaseModel):
    candles: list[ChartPointResponse]
    nextBefore: str | None
    hasMore: bool


class SyncResponse(BaseModel):
    success: bool = True
    count: int
    message: str

class TechnicalAnalysisResponse(BaseModel):
    status: str
    stock_code: str
    feature_date: str
    price_rows: int
    feature_count: int
    expected_feature_count: int
    features: dict[str, float]


class TechnicalInterpretationItemResponse(
    BaseModel
):
    state: str
    label: str
    description: str


class TechnicalInterpretationResponse(
    BaseModel
):
    headline: str

    trend: (
        TechnicalInterpretationItemResponse
    )

    momentum: (
        TechnicalInterpretationItemResponse
    )

    trendStrength: (
        TechnicalInterpretationItemResponse
    )

    volatility: (
        TechnicalInterpretationItemResponse
    )

    volume: (
        TechnicalInterpretationItemResponse
    )

    priceLocation: (
        TechnicalInterpretationItemResponse
    )

    reboundSignal: (
        TechnicalInterpretationItemResponse
    )


class InvestorFlowPeriodResponse(BaseModel):
    foreignNetBuyVolume: int
    institutionNetBuyVolume: int
    individualNetBuyVolume: int
    foreignInstitutionNetBuyVolume: int


class InvestorFlowAnalysisResponse(BaseModel):
    available: bool
    latestDate: str | None
    foreignHoldingRatio: float | None

    fiveDays: InvestorFlowPeriodResponse
    twentyDays: InvestorFlowPeriodResponse
    sixtyDays: InvestorFlowPeriodResponse


class InvestorFlowInterpretationItemResponse(
    BaseModel
):
    state: str
    label: str
    description: str


class InvestorFlowInterpretationResponse(
    BaseModel
):
    available: bool
    tone: str
    headline: str

    shortTerm: (
        InvestorFlowInterpretationItemResponse
    )

    participantAlignment: (
        InvestorFlowInterpretationItemResponse
    )

    trend: (
        InvestorFlowInterpretationItemResponse
    )

    foreignHolding: (
        InvestorFlowInterpretationItemResponse
    )


class ProgramTradingPeriodResponse(BaseModel):
    arbitrageNetBuyVolume: int
    nonArbitrageNetBuyVolume: int
    totalNetBuyVolume: int


class ShortSellingPeriodResponse(BaseModel):
    shortSellingVolume: int
    shortSellingAmount: int

    averageVolumeRate: float | None
    averageAmountRate: float | None

    latestVolumeRate: float | None
    latestAmountRate: float | None


class CreditSidePeriodResponse(BaseModel):
    newQuantity: int
    returnQuantity: int
    netNewQuantity: int

    latestBalanceQuantity: int
    balanceChangeQuantity: int

    latestBalanceRate: float | None
    averageTradingRate: float | None


class CreditTradingPeriodResponse(BaseModel):
    marginLoan: CreditSidePeriodResponse
    stockLoan: CreditSidePeriodResponse


class SecuritiesLendingPeriodResponse(BaseModel):
    executionQuantity: int
    repaymentQuantity: int
    netLendingQuantity: int

    latestBalanceQuantity: int
    balanceChangeQuantity: int
    latestBalanceAmount: int


class TradingSignalsPeriodResponse(BaseModel):
    program: ProgramTradingPeriodResponse
    shortSelling: ShortSellingPeriodResponse
    credit: CreditTradingPeriodResponse
    securitiesLending: SecuritiesLendingPeriodResponse


class StockTradingSignalsResponse(BaseModel):
    available: bool

    latestDates: dict[
        str,
        str | None,
    ]

    fiveDays: TradingSignalsPeriodResponse
    twentyDays: TradingSignalsPeriodResponse
    sixtyDays: TradingSignalsPeriodResponse


class StockRelativeStrengthPeriodResponse(
    BaseModel
):
    sessionCount: int
    startDate: str | None

    stockReturn: float | None
    sectorReturn: float | None
    marketReturn: float | None

    sectorExcessReturn: float | None
    marketExcessReturn: float | None


class StockRelativeStrengthResponse(
    BaseModel
):
    available: bool

    market: str | None
    sectorCode: str | None
    sectorName: str | None
    asOfDate: str | None

    twentyDays: (
        StockRelativeStrengthPeriodResponse
        | None
    )

    sixtyDays: (
        StockRelativeStrengthPeriodResponse
        | None
    )


class BenchmarkPeriodResponse(BaseModel):
    sessionCount: int
    baseDate: str | None

    stockReturn: float | None
    benchmarkReturn: float | None
    excessReturn: float | None


class BenchmarkPerformanceResponse(BaseModel):
    available: bool

    market: str | None
    benchmark: str | None
    benchmarkIndexCode: str | None

    asOfDate: str | None

    stockClose: float | None
    benchmarkClose: float | None

    oneDay: (
        BenchmarkPeriodResponse
        | None
    )

    fiveDays: (
        BenchmarkPeriodResponse
        | None
    )

    twentyDays: (
        BenchmarkPeriodResponse
        | None
    )


class BenchmarkInterpretationItemResponse(
    BaseModel
):
    state: str
    label: str
    description: str


class BenchmarkInterpretationResponse(
    BaseModel
):
    available: bool
    tone: str
    headline: str

    trend: (
        BenchmarkInterpretationItemResponse
    )

    oneDay: (
        BenchmarkInterpretationItemResponse
    )

    fiveDays: (
        BenchmarkInterpretationItemResponse
    )

    twentyDays: (
        BenchmarkInterpretationItemResponse
    )


class MarketRegimeGuidanceSignalResponse(
    BaseModel
):
    type: str
    label: str
    text: str


class MarketRegimeGuidanceResponse(
    BaseModel
):
    tone: str
    posture: str
    label: str
    headline: str

    signals: list[
        MarketRegimeGuidanceSignalResponse
    ]


class MarketRegimeAnalysisResponse(BaseModel):
    available: bool
    market: str | None
    featureDate: str | None

    state: str | None
    label: str | None
    description: str

    trend: str | None
    volatility: str | None
    risk: str | None

    guidance: (
        MarketRegimeGuidanceResponse
        | None
    ) = None


class StockNewsAnalysisResponse(BaseModel):
    id: str
    title: str
    summary: str
    source: str | None
    publishedAt: str | None
    sentimentScore: float
    importanceScore: float
    relatedStockCodes: list[str]
    url: str | None


class StockDisclosureAnalysisResponse(BaseModel):
    id: str
    stockCode: str
    stockName: str
    title: str
    reportName: str
    receivedAt: str | None
    importanceScore: float
    impactScore: float
    summary: str
    url: str | None


class StockAnalysisBriefPointResponse(BaseModel):
    type: str
    text: str


class StockFinancialPeerComparisonResponse(
    BaseModel
):
    available: bool
    sectorName: str | None
    peerCount: int
    rank: int | None
    topPercent: float | None
    financialScore: float | None
    peerAverageScore: float | None
    label: str
    description: str


class StockFinancialMetricPeerResponse(
    BaseModel
):
    key: str
    label: str
    unit: str
    higherIsBetter: bool
    available: bool

    value: float | None
    peerAverage: float | None
    peerMedian: float | None
    differenceFromMedian: float | None

    peerRank: int | None
    peerCount: int
    topPercent: float | None
    state: str


class StockFinancialMetricComparisonResponse(
    BaseModel
):
    available: bool
    sectorName: str | None
    businessYear: str | None
    reportCode: str | None
    fsDiv: str | None
    peerCount: int

    metrics: list[
        StockFinancialMetricPeerResponse
    ]


class StockValuationMultipleResponse(
    BaseModel
):
    value: float | None
    comparisonAvailable: bool

    sectorBenchmark: float | None
    marketBenchmark: float | None

    sectorRelativePercent: float | None
    marketRelativePercent: float | None


class StockValuationAnalysisResponse(
    BaseModel
):
    available: bool
    snapshotDate: str | None

    sectorCode: str | None
    sectorName: str | None

    price: float | None
    marketCap: float | None

    per: StockValuationMultipleResponse
    pbr: StockValuationMultipleResponse

    eps: float | None
    bps: float | None
    dividendYield: float | None

    source: str | None


class StockLongTermFinancialPointResponse(
    BaseModel
):
    year: str

    revenue: float | None
    operatingIncome: float | None
    netIncome: float | None

    totalAssets: float | None
    totalLiabilities: float | None
    totalEquity: float | None

    operatingCashFlow: float | None
    capitalExpenditure: float | None
    freeCashFlow: float | None

    operatingMargin: float | None
    roe: float | None
    debtRatio: float | None

    operatingCashFlowMargin: float | None
    freeCashFlowMargin: float | None


class StockLongTermFinancialResponse(
    BaseModel
):
    available: bool

    startYear: str | None
    endYear: str | None
    yearCount: int

    revenueCagr: float | None
    operatingIncomeCagr: float | None
    netIncomeCagr: float | None
    freeCashFlowCagr: float | None

    averageRoe: float | None
    roeStdDev: float | None
    roePositiveYearRatio: float | None

    operatingMarginChange: float | None
    operatingMarginTrend: str

    debtRatioChange: float | None
    debtRatioTrend: str

    positiveOperatingCashFlowRatio: (
        float
        | None
    )

    positiveFreeCashFlowRatio: (
        float
        | None
    )

    latestFreeCashFlow: float | None
    latestFreeCashFlowMargin: float | None

    points: list[
        StockLongTermFinancialPointResponse
    ]


class StockAnalysisBriefResponse(BaseModel):
    tone: str
    label: str
    headline: str
    description: str
    points: list[
        StockAnalysisBriefPointResponse
    ]
    financialPeer: (
        StockFinancialPeerComparisonResponse
    )


class StockAnalysisResponse(BaseModel):
    stockCode: str
    stockName: str

    ranking: RankingExplanationResponse

    technical: TechnicalAnalysisResponse

    technicalInterpretation: (
        TechnicalInterpretationResponse
    )

    financial: (
        FinancialAnalysisResponse
        | None
    )

    valuation: (
        StockValuationAnalysisResponse
    )

    financialPeerMetrics: (
        StockFinancialMetricComparisonResponse
    )

    longTermFinancial: (
        StockLongTermFinancialResponse
    )

    investorFlow: (
        InvestorFlowAnalysisResponse
    )

    investorFlowInterpretation: (
        InvestorFlowInterpretationResponse
    )

    tradingSignals: (
        StockTradingSignalsResponse
    )

    benchmarkPerformance: (
        BenchmarkPerformanceResponse
    )

    relativeStrength: (
        StockRelativeStrengthResponse
    )

    benchmarkInterpretation: (
        BenchmarkInterpretationResponse
    )
    marketRegime: (
        MarketRegimeAnalysisResponse
    )

    analysisBrief: (
        StockAnalysisBriefResponse
    )

    news: list[
        StockNewsAnalysisResponse
    ]
    disclosures: list[
        StockDisclosureAnalysisResponse
    ]