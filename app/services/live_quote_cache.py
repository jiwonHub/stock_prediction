import time
from dataclasses import dataclass, replace
from threading import RLock


@dataclass(
    frozen=True,
    slots=True,
)
class LiveQuote:
    stock_code: str
    current_price: float
    change: float
    change_rate: float
    timestamp: str | None
    currency: str
    source: str
    fetched_at: float
    stale: bool = False


class LiveQuoteCache:
    def __init__(self):
        self._quotes: dict[
            str,
            LiveQuote,
        ] = {}

        self._lock = RLock()

    def get_many(
        self,
        stock_codes: list[str],
        *,
        max_age_seconds: float | None,
    ) -> dict[str, LiveQuote]:
        now = time.monotonic()

        with self._lock:
            result: dict[
                str,
                LiveQuote,
            ] = {}

            for stock_code in stock_codes:
                quote = self._quotes.get(
                    stock_code
                )

                if quote is None:
                    continue

                if max_age_seconds is not None:
                    age = (
                        now
                        - quote.fetched_at
                    )

                    if age > max_age_seconds:
                        continue

                result[
                    stock_code
                ] = quote

            return result

    def get_stale_many(
        self,
        stock_codes: list[str],
    ) -> dict[str, LiveQuote]:
        with self._lock:
            return {
                stock_code: replace(
                    quote,
                    stale=True,
                )
                for stock_code in stock_codes
                if (
                    quote := self._quotes.get(
                        stock_code
                    )
                ) is not None
            }

    def set_many(
        self,
        quotes: list[LiveQuote],
    ) -> None:
        if not quotes:
            return

        with self._lock:
            for quote in quotes:
                self._quotes[
                    quote.stock_code
                ] = quote

    def size(self) -> int:
        with self._lock:
            return len(
                self._quotes
            )


live_quote_cache = LiveQuoteCache()