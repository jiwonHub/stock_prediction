from pydantic import BaseModel, ConfigDict


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


class SyncResponse(BaseModel):
    success: bool = True
    count: int
    message: str
