# Stock AI Phase 4 Server

Phase 4 adds machine-learning price prediction on top of the Phase 3 financial analysis server.

## Prediction target
- Horizon: default 5 trading days
- Regression: expected return (%)
- Classification: probability of a positive return
- Features: 1/5/20-day return, SMA ratios, 20-day volatility, RSI14, volume ratio, intraday range, close position
- Validation: chronological 80/20 split (no random shuffle)

## Model
- If `xgboost` is installed, XGBoost regressor/classifier is used.
- Otherwise scikit-learn HistGradientBoosting is used automatically.

## New endpoints
- `POST /v1/admin/ml/train/{stock_code}`
- `POST /v1/admin/ml/train/batch`
- `GET /v1/stocks/{stock_code}/prediction`

## Ranking
Phase 4 total score:
- financial score 45%
- ML score 55%

If only one side exists, the available score is used.

## Migration
```bash
docker exec -i stock-ai-postgres psql -U stock -d stock_ai < sql/phase4_migration.sql
```

## Install
```bash
python -m pip install -r requirements.txt
```

Optional XGBoost:
```bash
python -m pip install -r requirements-xgboost.txt
```

## Run
```bash
python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```
