from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.core.exceptions import ConfigurationError, ExternalApiError
from app.schemas.ml import MlBatchResponse, StockPredictionResponse
from app.services.ml_prediction_service import MlPredictionService


router = APIRouter(
    prefix="/admin/ml",
    tags=["admin-ml"],
)

@router.post(
    "/predict/batch",
    response_model=MlBatchResponse,
)
async def predict_stock_models_batch(
    limit: int = Query(
        default=100,
        ge=1,
        le=100,
    ),
    horizon_days: int = Query(
        default=5,
        ge=1,
        le=20,
    ),
    db: Session = Depends(get_db),
):
    service = MlPredictionService(
        db
    )

    stock_codes = (
        service.repository
        .get_stock_codes_with_predictions(
            horizon_days=horizon_days,
            limit=limit,
        )
    )

    predicted = 0
    skipped = 0
    total = len(stock_codes)

    for (
        index,
        stock_code,
    ) in enumerate(
        stock_codes,
        start=1,
    ):
        print(
            f"[PREDICT] "
            f"{index}/{total} "
            f"{stock_code}",
            flush=True,
        )

        try:
            await service.predict_existing(
                stock_code,
                horizon_days=horizon_days,
            )

            predicted += 1

        except (
            ValueError,
            ConfigurationError,
            ExternalApiError,
            SQLAlchemyError,
        ) as e:
            db.rollback()

            skipped += 1

            print(
                f"[PREDICT] "
                f"{stock_code} "
                f"건너뜀: {e}",
                flush=True,
            )

    return MlBatchResponse(
        trainedCount=predicted,
        skippedCount=skipped,
        message=(
            f"기존 모델 예측 "
            f"{predicted}개 완료, "
            f"{skipped}개 건너뜀"
        ),
    )

@router.post(
    "/train/batch",
    response_model=MlBatchResponse,
)
async def train_stock_models_batch(
    limit: int = Query(default=10, ge=1, le=100),
    horizon_days: int = Query(default=5, ge=1, le=20),
    days: int = Query(default=1200, ge=180, le=3650),
    sync_prices: bool = Query(default=True),
    db: Session = Depends(get_db),
):
    service = MlPredictionService(db)
    stock_codes = service.repository.get_stock_codes_for_ml(limit=limit)

    trained = 0
    skipped = 0

    for stock_code in stock_codes:
        try:
            await service.train(
                stock_code,
                horizon_days=horizon_days,
                days=days,
                sync_prices=sync_prices,
            )
            trained += 1
        except (
            ValueError,
            ConfigurationError,
            ExternalApiError,
            SQLAlchemyError,
        ):
            db.rollback()
            skipped += 1

    return MlBatchResponse(
        trainedCount=trained,
        skippedCount=skipped,
        message=f"ML 모델 {trained}개 학습 완료, {skipped}개 건너뜀",
    )


@router.post(
    "/train/{stock_code}",
    response_model=StockPredictionResponse,
)
async def train_stock_model(
    stock_code: str,
    horizon_days: int = Query(default=5, ge=1, le=20),
    days: int = Query(default=1200, ge=180, le=3650),
    sync_prices: bool = Query(default=True),
    db: Session = Depends(get_db),
):
    try:
        return await MlPredictionService(db).train(
            stock_code,
            horizon_days=horizon_days,
            days=days,
            sync_prices=sync_prices,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except ConfigurationError as e:
        raise HTTPException(status_code=500, detail=str(e)) from e
    except ExternalApiError as e:
        raise HTTPException(status_code=502, detail=str(e)) from e
