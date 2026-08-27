from __future__ import annotations

from decimal import Decimal


def safe_ratio(
    numerator: Decimal | float | None,
    denominator: Decimal | float | None,
    *,
    multiplier: float = 100.0,
) -> float | None:
    if numerator is None or denominator is None:
        return None

    denominator_value = float(denominator)
    if denominator_value == 0.0:
        return None

    return float(numerator) / denominator_value * multiplier


def growth_rate(
    current: Decimal | float | None,
    previous: Decimal | float | None,
) -> float | None:
    if current is None or previous is None:
        return None

    previous_value = float(previous)
    if previous_value == 0.0:
        return None

    return (float(current) - previous_value) / abs(previous_value) * 100.0


def piecewise_score(value: float | None, points: list[tuple[float, float]]) -> float | None:
    if value is None:
        return None

    if value <= points[0][0]:
        return float(points[0][1])

    if value >= points[-1][0]:
        return float(points[-1][1])

    for index in range(1, len(points)):
        left_x, left_score = points[index - 1]
        right_x, right_score = points[index]

        if value <= right_x:
            distance = right_x - left_x
            if distance == 0:
                return float(right_score)

            ratio = (value - left_x) / distance
            return float(left_score + (right_score - left_score) * ratio)

    return float(points[-1][1])


def weighted_score(items: list[tuple[float | None, float]]) -> float | None:
    available = [(value, weight) for value, weight in items if value is not None]
    if not available:
        return None

    total_weight = sum(weight for _, weight in available)
    if total_weight <= 0:
        return None

    return sum(float(value) * weight for value, weight in available) / total_weight


def clamp_score(value: float | None) -> float:
    if value is None:
        return 0.0
    return round(max(0.0, min(100.0, value)), 2)
