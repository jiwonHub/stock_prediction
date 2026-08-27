from __future__ import annotations

from dataclasses import dataclass

import numpy as np


FEATURE_NAMES = [
    "return_1d",
    "return_5d",
    "return_20d",
    "sma5_ratio",
    "sma20_ratio",
    "sma60_ratio",
    "volatility_20d",
    "rsi_14",
    "volume_ratio_20d",
    "intraday_range",
    "close_position",
]


@dataclass(frozen=True)
class MlDataset:
    x: np.ndarray
    y_return: np.ndarray
    y_up: np.ndarray
    latest_x: np.ndarray
    latest_date: object


def _safe_ratio(numerator: float, denominator: float) -> float:
    if denominator == 0.0:
        return 0.0
    return numerator / denominator


def _pct_return(current: float, previous: float) -> float:
    if previous == 0.0:
        return 0.0
    return current / previous - 1.0


def _rsi(closes: np.ndarray, end_index: int, period: int = 14) -> float:
    start = end_index - period
    if start < 0:
        return 50.0

    window = closes[start : end_index + 1]
    changes = np.diff(window)
    gains = np.clip(changes, 0.0, None)
    losses = np.clip(-changes, 0.0, None)
    avg_gain = float(np.mean(gains)) if len(gains) else 0.0
    avg_loss = float(np.mean(losses)) if len(losses) else 0.0

    if avg_loss == 0.0:
        return 100.0 if avg_gain > 0.0 else 50.0

    rs = avg_gain / avg_loss
    return 100.0 - 100.0 / (1.0 + rs)


def _feature_vector(rows: list, index: int) -> list[float]:
    closes = np.asarray([float(row.close) for row in rows], dtype=float)
    volumes = np.asarray([float(row.volume or 0) for row in rows], dtype=float)

    close = float(closes[index])
    ret_1 = _pct_return(close, float(closes[index - 1]))
    ret_5 = _pct_return(close, float(closes[index - 5]))
    ret_20 = _pct_return(close, float(closes[index - 20]))

    sma_5 = float(np.mean(closes[index - 4 : index + 1]))
    sma_20 = float(np.mean(closes[index - 19 : index + 1]))
    sma_60 = float(np.mean(closes[index - 59 : index + 1]))

    daily_returns = []
    for j in range(index - 19, index + 1):
        daily_returns.append(_pct_return(float(closes[j]), float(closes[j - 1])))
    volatility_20 = float(np.std(np.asarray(daily_returns, dtype=float)))

    volume_mean = float(np.mean(volumes[index - 19 : index + 1]))
    volume_ratio = _safe_ratio(float(volumes[index]), volume_mean) - 1.0 if volume_mean > 0 else 0.0

    high = float(rows[index].high)
    low = float(rows[index].low)
    day_range = high - low
    intraday_range = _safe_ratio(day_range, close)
    close_position = _safe_ratio(close - low, day_range) if day_range > 0 else 0.5

    return [
        ret_1,
        ret_5,
        ret_20,
        _safe_ratio(close, sma_5) - 1.0,
        _safe_ratio(close, sma_20) - 1.0,
        _safe_ratio(close, sma_60) - 1.0,
        volatility_20,
        _rsi(closes, index) / 100.0,
        volume_ratio,
        intraday_range,
        close_position,
    ]


def build_dataset(rows: list, *, horizon_days: int) -> MlDataset:
    if len(rows) < 120:
        raise ValueError("ML 학습에는 최소 120거래일 이상의 일봉 데이터가 필요합니다.")

    start_index = 60
    last_train_index = len(rows) - horizon_days - 1

    if last_train_index <= start_index:
        raise ValueError("학습 가능한 주가 데이터가 부족합니다.")

    features: list[list[float]] = []
    targets: list[float] = []
    labels: list[int] = []

    for index in range(start_index, last_train_index + 1):
        current_close = float(rows[index].close)
        future_close = float(rows[index + horizon_days].close)
        if current_close <= 0.0:
            continue

        future_return = (future_close / current_close - 1.0) * 100.0
        features.append(_feature_vector(rows, index))
        targets.append(future_return)
        labels.append(1 if future_return > 0.0 else 0)

    if len(features) < 80:
        raise ValueError("학습 샘플이 80개 미만이라 모델을 만들 수 없습니다.")

    latest_index = len(rows) - 1
    latest_x = np.asarray([_feature_vector(rows, latest_index)], dtype=float)

    return MlDataset(
        x=np.asarray(features, dtype=float),
        y_return=np.asarray(targets, dtype=float),
        y_up=np.asarray(labels, dtype=int),
        latest_x=latest_x,
        latest_date=rows[latest_index].trade_date,
    )
