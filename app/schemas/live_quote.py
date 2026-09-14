from pydantic import BaseModel


class LiveQuoteResponse(BaseModel):
    stockCode: str
    currentPrice: float
    change: float
    changeRate: float
    timestamp: str | None
    currency: str
    source: str
    stale: bool