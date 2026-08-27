from decimal import Decimal, InvalidOperation


def to_float(value: object, default: float = 0.0) -> float:
    if value is None:
        return default

    text = str(value).replace(",", "").strip()

    if text in {"", "-", "None", "null"}:
        return default

    try:
        return float(text)
    except (TypeError, ValueError):
        return default


def to_int(value: object, default: int = 0) -> int:
    if value is None:
        return default

    text = str(value).replace(",", "").strip()

    if text in {"", "-", "None", "null"}:
        return default

    try:
        return int(float(text))
    except (TypeError, ValueError):
        return default


def to_decimal_or_none(value: object) -> Decimal | None:
    if value is None:
        return None

    text = str(value).replace(",", "").strip()

    if text in {"", "-", "None", "null"}:
        return None

    negative = text.startswith("(") and text.endswith(")")

    if negative:
        text = text[1:-1]

    try:
        result = Decimal(text)
    except InvalidOperation:
        return None

    return -result if negative else result
