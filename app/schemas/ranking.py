from datetime import date

from pydantic import BaseModel, ConfigDict, Field


class RankingResponse(BaseModel):
    rank: int
    stockCode: str
    stockName: str

    sectorName: str | None = None
    sectorRank: int | None = None
    sectorPeerCount: int = 0

    currentPrice: float
    changeRate: float
    totalScore: float
    predictedReturn: float
    upsideProbability: float
    financialScore: float = 0.0
    growthScore: float = 0.0
    profitabilityScore: float = 0.0
    stabilityScore: float = 0.0
    cashFlowScore: float = 0.0

    qualityScore: float | None = None
    valueScore: float | None = None
    financialHealthScore: float | None = None
    relativeStrengthScore: float | None = None
    momentumScore: float | None = None
    flowScore: float | None = None
    riskScore: float | None = None
    mlScore: float = 0.0
    dataCoverage: float = 0.0


class RankingSnapshotResponse(BaseModel):
    asOfDate: date
    latestMarketDate: date | None = None

    rankingVersion: str
    horizonDays: int
    universe: str

    itemCount: int
    requestedCount: int

    integrityOk: bool
    isStale: bool

    items: list[RankingResponse]


class RankingFactorResponse(BaseModel):
    model_config = ConfigDict(
        populate_by_name=True,
    )

    feature: str
    label: str
    value: float

    valueText: str = Field(
        validation_alias="value_text",
    )

    contribution: float

    contributionShare: float = Field(
        validation_alias="contribution_share",
    )


class RankingExplanationResponse(BaseModel):
    model_config = ConfigDict(
        populate_by_name=True,
    )

    stockCode: str = Field(
        validation_alias="stock_code",
    )
    stockName: str = Field(
        validation_alias="stock_name",
    )

    rank: int | None = None

    totalScore: float | None = Field(
        default=None,
        validation_alias="total_score",
    )

    qualityScore: float | None = Field(
        default=None,
        validation_alias="quality_score",
    )

    growthScore: float | None = Field(
        default=None,
        validation_alias="growth_score",
    )

    valueScore: float | None = Field(
        default=None,
        validation_alias="value_score",
    )

    financialHealthScore: float | None = Field(
        default=None,
        validation_alias="financial_health_score",
    )

    relativeStrengthScore: float | None = Field(
        default=None,
        validation_alias="relative_strength_score",
    )

    flowScore: float | None = Field(
        default=None,
        validation_alias="flow_score",
    )

    dataCoverage: float | None = Field(
        default=None,
        validation_alias="data_coverage",
    )

    mlRank: int | None = Field(
        default=None,
        validation_alias="ml_rank",
    )

    mlScore: float | None = Field(
        default=None,
        validation_alias="ml_score",
    )

    summary: str

    positiveFactors: list[
        RankingFactorResponse
    ] = Field(
        validation_alias="positive_factors",
    )

    negativeFactors: list[
        RankingFactorResponse
    ] = Field(
        validation_alias="negative_factors",
    )

    method: str