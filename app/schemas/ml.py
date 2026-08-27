from datetime import date, datetime

from pydantic import BaseModel


class StockPredictionResponse(BaseModel):
    stockCode: str
    stockName: str
    horizonDays: int
    predictedReturn: float
    upsideProbability: float
    mlScore: float
    modelType: str
    modelVersion: str
    trainingRows: int
    validationMae: float | None = None
    validationAccuracy: float | None = None
    featureDate: date
    trainedAt: datetime


class MlBatchResponse(BaseModel):
    trainedCount: int
    skippedCount: int
    message: str
