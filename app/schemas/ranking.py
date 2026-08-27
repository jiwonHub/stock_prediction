from pydantic import BaseModel


class RankingResponse(BaseModel):
    rank: int
    stockCode: str
    stockName: str
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
