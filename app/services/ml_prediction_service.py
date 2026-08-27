from __future__ import annotations

from datetime import date, datetime, timedelta
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.metrics import accuracy_score, mean_absolute_error
from sqlalchemy.orm import Session

from app.models.stock_prediction import StockPrediction
from app.repositories.stock_repository import StockRepository
from app.schemas.ml import StockPredictionResponse
from app.services.stock_service import StockService
from app.utils.ml_features import FEATURE_NAMES, build_dataset


MODEL_VERSION = "phase4-v1"
DEFAULT_HORIZON_DAYS = 5


class MlPredictionService:
    def __init__(self, db: Session):
        self.db = db
        self.repository = StockRepository(db)

    @staticmethod
    def _response(prediction: StockPrediction, stock_name: str) -> StockPredictionResponse:
        return StockPredictionResponse(
            stockCode=prediction.stock_code,
            stockName=stock_name,
            horizonDays=prediction.horizon_days,
            predictedReturn=float(prediction.predicted_return),
            upsideProbability=float(prediction.upside_probability),
            mlScore=float(prediction.ml_score),
            modelType=prediction.model_type,
            modelVersion=prediction.model_version,
            trainingRows=prediction.training_rows,
            validationMae=(
                float(prediction.validation_mae)
                if prediction.validation_mae is not None
                else None
            ),
            validationAccuracy=(
                float(prediction.validation_accuracy)
                if prediction.validation_accuracy is not None
                else None
            ),
            featureDate=prediction.feature_date,
            trainedAt=prediction.trained_at,
        )

    @staticmethod
    def _ml_score(predicted_return: float, upside_probability: float) -> float:
        probability_score = max(0.0, min(100.0, upside_probability))
        return_score = max(0.0, min(100.0, (predicted_return + 10.0) / 20.0 * 100.0))
        return round(probability_score * 0.70 + return_score * 0.30, 2)

    @staticmethod
    def _make_models():
        try:
            from xgboost import XGBClassifier, XGBRegressor

            regressor = XGBRegressor(
                n_estimators=300,
                max_depth=4,
                learning_rate=0.03,
                subsample=0.9,
                colsample_bytree=0.9,
                objective="reg:squarederror",
                random_state=42,
                n_jobs=2,
            )
            classifier = XGBClassifier(
                n_estimators=300,
                max_depth=4,
                learning_rate=0.03,
                subsample=0.9,
                colsample_bytree=0.9,
                objective="binary:logistic",
                eval_metric="logloss",
                random_state=42,
                n_jobs=2,
            )
            return regressor, classifier, "xgboost"
        except Exception:
            regressor = HistGradientBoostingRegressor(
                max_iter=250,
                learning_rate=0.05,
                max_depth=5,
                l2_regularization=1.0,
                random_state=42,
            )
            classifier = HistGradientBoostingClassifier(
                max_iter=250,
                learning_rate=0.05,
                max_depth=5,
                l2_regularization=1.0,
                random_state=42,
            )
            return regressor, classifier, "sklearn-hist-gradient-boosting"

    async def train(
        self,
        stock_code: str,
        *,
        horizon_days: int = DEFAULT_HORIZON_DAYS,
        days: int = 1200,
        sync_prices: bool = True,
    ) -> StockPredictionResponse:
        stock = self.repository.get_stock(stock_code)
        if stock is None:
            raise ValueError(f"등록되지 않은 종목입니다: {stock_code}")

        start_date = date.today() - timedelta(days=max(days, 180))
        rows = self.repository.get_daily_prices(stock_code=stock_code, start_date=start_date)

        if sync_prices and len(rows) < 260:
            await StockService(self.db).sync_daily_prices(stock_code, days=days)
            rows = self.repository.get_daily_prices(stock_code=stock_code, start_date=start_date)

        dataset = build_dataset(rows, horizon_days=horizon_days)
        sample_count = len(dataset.x)
        split_index = int(sample_count * 0.80)
        split_index = max(60, min(split_index, sample_count - 20))

        x_train = dataset.x[:split_index]
        x_valid = dataset.x[split_index:]
        y_return_train = dataset.y_return[:split_index]
        y_return_valid = dataset.y_return[split_index:]
        y_up_train = dataset.y_up[:split_index]
        y_up_valid = dataset.y_up[split_index:]

        regressor, classifier, model_type = self._make_models()
        regressor.fit(x_train, y_return_train)

        valid_return_prediction = regressor.predict(x_valid)
        validation_mae = float(mean_absolute_error(y_return_valid, valid_return_prediction))

        classifier_model = classifier
        if len(np.unique(y_up_train)) >= 2:
            classifier.fit(x_train, y_up_train)
            valid_up_prediction = classifier.predict(x_valid)
            validation_accuracy = float(accuracy_score(y_up_valid, valid_up_prediction)) * 100.0
            upside_probability = float(classifier.predict_proba(dataset.latest_x)[0][1]) * 100.0
        else:
            classifier_model = None
            baseline = float(np.mean(y_up_train)) if len(y_up_train) else 0.5
            validation_accuracy = float(
                accuracy_score(y_up_valid, np.full_like(y_up_valid, int(baseline >= 0.5)))
            ) * 100.0
            upside_probability = baseline * 100.0
            model_type = f"{model_type}+baseline-classifier"

        predicted_return = float(regressor.predict(dataset.latest_x)[0])
        predicted_return = max(-30.0, min(30.0, predicted_return))
        upside_probability = max(0.0, min(100.0, upside_probability))
        ml_score = self._ml_score(predicted_return, upside_probability)

        model_dir = Path(".cache/models")
        model_dir.mkdir(parents=True, exist_ok=True)
        model_path = model_dir / f"{stock_code}_h{horizon_days}_{MODEL_VERSION}.joblib"
        joblib.dump(
            {
                "regressor": regressor,
                "classifier": classifier_model,
                "feature_names": FEATURE_NAMES,
                "horizon_days": horizon_days,
                "model_version": MODEL_VERSION,
            },
            model_path,
        )

        prediction = self.repository.upsert_stock_prediction(
            {
                "stock_code": stock_code,
                "horizon_days": horizon_days,
                "model_version": MODEL_VERSION,
                "model_type": model_type,
                "model_path": str(model_path),
                "predicted_return": round(predicted_return, 4),
                "upside_probability": round(upside_probability, 2),
                "ml_score": ml_score,
                "training_rows": sample_count,
                "validation_mae": round(validation_mae, 4),
                "validation_accuracy": round(validation_accuracy, 2),
                "feature_date": dataset.latest_date,
                "trained_at": datetime.utcnow(),
            }
        )

        return self._response(prediction, stock.name)
    
    async def predict_existing(
        self,
        stock_code: str,
        *,
        horizon_days: int = DEFAULT_HORIZON_DAYS,
    ) -> StockPredictionResponse:
        stock = self.repository.get_stock(
            stock_code
        )

        if stock is None:
            raise ValueError(
                f"등록되지 않은 종목입니다: "
                f"{stock_code}"
            )

        existing = (
            self.repository
            .get_stock_prediction(
                stock_code=stock_code,
                horizon_days=horizon_days,
            )
        )

        if (
            existing is None
            or not existing.model_path
        ):
            raise ValueError(
                f"{stock_code} "
                "학습 완료 모델이 없습니다."
            )

        model_path = Path(
            existing.model_path
        )

        if not model_path.exists():
            raise ValueError(
                f"{stock_code} 모델 파일을 "
                f"찾을 수 없습니다: "
                f"{model_path}"
            )

        rows = (
            self.repository
            .get_daily_prices(
                stock_code=stock_code,
                start_date=(
                    date.today()
                    - timedelta(days=500)
                ),
            )
        )

        dataset = build_dataset(
            rows,
            horizon_days=horizon_days,
        )

        bundle = joblib.load(
            model_path
        )

        regressor = bundle.get(
            "regressor"
        )

        classifier = bundle.get(
            "classifier"
        )

        if regressor is None:
            raise ValueError(
                f"{stock_code} 모델 파일에 "
                "regressor가 없습니다."
            )

        predicted_return = float(
            regressor.predict(
                dataset.latest_x
            )[0]
        )

        predicted_return = max(
            -30.0,
            min(
                30.0,
                predicted_return,
            ),
        )

        if (
            classifier is not None
            and hasattr(
                classifier,
                "predict_proba",
            )
        ):
            upside_probability = (
                float(
                    classifier
                    .predict_proba(
                        dataset.latest_x
                    )[0][1]
                )
                * 100.0
            )
        else:
            upside_probability = float(
                existing
                .upside_probability
                or 50.0
            )

        upside_probability = max(
            0.0,
            min(
                100.0,
                upside_probability,
            ),
        )

        ml_score = self._ml_score(
            predicted_return,
            upside_probability,
        )

        current_price = float(
            stock.current_price
            or 0.0
        )

        predicted_price = (
            current_price
            * (
                1.0
                + predicted_return
                / 100.0
            )
            if current_price > 0.0
            else None
        )

        prediction = (
            self.repository
            .upsert_stock_prediction(
                {
                    "stock_code": (
                        stock_code
                    ),
                    "horizon_days": (
                        horizon_days
                    ),
                    "model_version": (
                        existing
                        .model_version
                    ),
                    "model_type": (
                        existing
                        .model_type
                    ),
                    "model_path": (
                        existing
                        .model_path
                    ),
                    "prediction_date": (
                        date.today()
                    ),
                    "target_date": (
                        date.today()
                        + timedelta(
                            days=horizon_days
                        )
                    ),
                    "predicted_return": (
                        round(
                            predicted_return,
                            4,
                        )
                    ),
                    "predicted_price": (
                        predicted_price
                    ),
                    "upside_probability": (
                        round(
                            upside_probability,
                            2,
                        )
                    ),
                    "ml_score": (
                        ml_score
                    ),
                    "training_rows": (
                        existing
                        .training_rows
                    ),
                    "validation_mae": (
                        existing
                        .validation_mae
                    ),
                    "validation_accuracy": (
                        existing
                        .validation_accuracy
                    ),
                    "feature_date": (
                        dataset.latest_date
                    ),
                    "trained_at": (
                        existing
                        .trained_at
                    ),
                }
            )
        )

        return self._response(
            prediction,
            stock.name,
        )

    async def get_prediction(
        self,
        stock_code: str,
        *,
        horizon_days: int = DEFAULT_HORIZON_DAYS,
        auto_train: bool = False,
        days: int = 1200,
    ) -> StockPredictionResponse:
        stock = self.repository.get_stock(stock_code)
        if stock is None:
            raise ValueError(f"등록되지 않은 종목입니다: {stock_code}")

        prediction = self.repository.get_stock_prediction(
            stock_code=stock_code,
            horizon_days=horizon_days,
        )

        if prediction is None:
            if not auto_train:
                raise ValueError(
                    f"{stock_code} ML 예측이 없습니다. 먼저 /v1/admin/ml/train/{stock_code}를 실행하세요."
                )
            return await self.train(
                stock_code,
                horizon_days=horizon_days,
                days=days,
                sync_prices=True,
            )

        if not auto_train:
            return await self.predict_existing(
                stock_code,
                horizon_days=horizon_days,
            )

        latest_prices = (
            self.repository
            .get_daily_prices(
                stock_code=stock_code,
                start_date=(
                    date.today()
                    - timedelta(days=14)
                ),
            )
        )

        if (
            latest_prices
            and latest_prices[-1].trade_date
            > prediction.feature_date
        ):
            return await self.train(
                stock_code,
                horizon_days=horizon_days,
                days=days,
                sync_prices=True,
            )

        return self._response(
            prediction,
            stock.name,
        )
