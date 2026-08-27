from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.core.exceptions import ConfigurationError, ExternalApiError
from app.schemas.ml import StockPredictionResponse
from app.services.ml_prediction_service import MlPredictionService


router = APIRouter(
    prefix="/stocks",
    tags=["ml-prediction"],
)


@router.get(
    "/{stock_code}/prediction",
    response_model=StockPredictionResponse,
)
async def get_stock_prediction(
    stock_code: str,
    horizon_days: int = Query(default=5, ge=1, le=20),
    auto_train: bool = Query(default=False),
    days: int = Query(default=1200, ge=180, le=3650),
    db: Session = Depends(get_db),
):
    try:
        return await MlPredictionService(db).get_prediction(
            stock_code,
            horizon_days=horizon_days,
            auto_train=auto_train,
            days=days,
        )
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    except ConfigurationError as e:
        raise HTTPException(status_code=500, detail=str(e)) from e
    except ExternalApiError as e:
        raise HTTPException(status_code=502, detail=str(e)) from e
