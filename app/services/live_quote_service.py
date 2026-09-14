import asyncio
import time
from datetime import date, datetime
from zoneinfo import ZoneInfo

from sqlalchemy.orm import Session

from app.clients.toss_client import toss_client
from app.core.config import settings
from app.core.exceptions import ExternalApiError
from app.repositories.stock_repository import (
    StockRepository,
)
from app.schemas.live_quote import (
    LiveQuoteResponse,
)
from app.services.live_quote_cache import (
    LiveQuote,
    live_quote_cache,
)
from app.utils.numbers import to_float


_refresh_lock = asyncio.Lock()


class LiveQuoteService:
    def __init__(
        self,
        db: Session,
    ):
        self.db = db

        self.repository = StockRepository(
            db
        )

    @staticmethod
    def _normalize_symbols(
        stock_codes: list[str],
    ) -> list[str]:
        symbols = list(
            dict.fromkeys(
                str(stock_code).strip()
                for stock_code in stock_codes
                if str(stock_code).strip()
            )
        )

        if len(symbols) > 200:
            raise ValueError(
                "실시간 현재가는 한 요청에 "
                "최대 200종목까지 조회할 수 있습니다."
            )

        return symbols

    @staticmethod
    def _quote_trade_date(
        output: dict,
    ) -> date:
        timestamp = str(
            output.get(
                "timestamp",
                "",
            )
        ).strip()

        if timestamp:
            try:
                return datetime.fromisoformat(
                    timestamp
                ).date()

            except ValueError:
                pass

        return datetime.now(
            ZoneInfo(
                "Asia/Seoul"
            )
        ).date()

    @staticmethod
    def _to_response(
        quote: LiveQuote,
    ) -> LiveQuoteResponse:
        return LiveQuoteResponse(
            stockCode=quote.stock_code,
            currentPrice=quote.current_price,
            change=quote.change,
            changeRate=quote.change_rate,
            timestamp=quote.timestamp,
            currency=quote.currency,
            source=quote.source,
            stale=quote.stale,
        )

    async def _refresh(
        self,
        stock_codes: list[str],
    ) -> None:
        if not stock_codes:
            return

        outputs = await toss_client.get_prices(
            stock_codes
        )

        output_by_code = {
            str(
                output.get(
                    "symbol",
                    "",
                )
            ).strip(): output
            for output in outputs
        }

        codes_by_trade_date: dict[
            date,
            list[str],
        ] = {}

        current_prices: dict[
            str,
            float,
        ] = {}

        for stock_code in stock_codes:
            output = output_by_code.get(
                stock_code
            )

            if output is None:
                continue

            current_price = to_float(
                output.get(
                    "lastPrice"
                )
            )

            if current_price <= 0.0:
                continue

            trade_date = (
                self._quote_trade_date(
                    output
                )
            )

            current_prices[
                stock_code
            ] = current_price

            codes_by_trade_date.setdefault(
                trade_date,
                [],
            ).append(
                stock_code
            )

        previous_closes: dict[
            str,
            float,
        ] = {}

        for (
            trade_date,
            trade_date_codes,
        ) in codes_by_trade_date.items():
            previous_closes.update(
                self.repository
                .get_latest_closes_before(
                    stock_codes=(
                        trade_date_codes
                    ),
                    trade_date=trade_date,
                )
            )

        fetched_at = time.monotonic()

        quotes: list[
            LiveQuote
        ] = []

        for (
            stock_code,
            current_price,
        ) in current_prices.items():
            previous_close = (
                previous_closes.get(
                    stock_code
                )
            )

            if (
                previous_close is None
                or previous_close <= 0.0
            ):
                continue

            output = output_by_code[
                stock_code
            ]

            change = (
                current_price
                - previous_close
            )

            change_rate = (
                change
                / previous_close
                * 100.0
            )

            quotes.append(
                LiveQuote(
                    stock_code=(
                        stock_code
                    ),
                    current_price=(
                        current_price
                    ),
                    change=change,
                    change_rate=(
                        change_rate
                    ),
                    timestamp=(
                        str(
                            output.get(
                                "timestamp",
                                "",
                            )
                        ).strip()
                        or None
                    ),
                    currency=(
                        str(
                            output.get(
                                "currency",
                                "KRW",
                            )
                        ).strip()
                        or "KRW"
                    ),
                    source="toss",
                    fetched_at=(
                        fetched_at
                    ),
                )
            )

        live_quote_cache.set_many(
            quotes
        )

        print(
            "[LIVE][TOSS-BATCH] "
            f"requested={len(stock_codes)} "
            f"cached={len(quotes)} "
            f"missing={len(stock_codes) - len(quotes)} "
            f"cacheSize={live_quote_cache.size()}",
            flush=True,
        )

    async def get_quotes(
        self,
        stock_codes: list[str],
        *,
        force: bool = False,
    ) -> list[LiveQuoteResponse]:
        symbols = self._normalize_symbols(
            stock_codes
        )

        if not symbols:
            return []

        max_age_seconds = float(
            settings.live_quote_cache_seconds
        )

        fresh = (
            {}
            if force
            else live_quote_cache.get_many(
                symbols,
                max_age_seconds=(
                    max_age_seconds
                ),
            )
        )

        missing = [
            stock_code
            for stock_code in symbols
            if stock_code not in fresh
        ]

        if missing:
            try:
                async with _refresh_lock:
                    if not force:
                        fresh = (
                            live_quote_cache
                            .get_many(
                                symbols,
                                max_age_seconds=(
                                    max_age_seconds
                                ),
                            )
                        )

                        missing = [
                            stock_code
                            for stock_code in symbols
                            if stock_code
                            not in fresh
                        ]

                    if missing:
                        await self._refresh(
                            missing
                        )

            except ExternalApiError:
                stale = (
                    live_quote_cache
                    .get_stale_many(
                        symbols
                    )
                )

                if len(stale) == len(
                    symbols
                ):
                    return [
                        self._to_response(
                            stale[
                                stock_code
                            ]
                        )
                        for stock_code
                        in symbols
                    ]

                raise

        quotes = (
            live_quote_cache.get_many(
                symbols,
                max_age_seconds=None,
            )
        )

        unresolved = [
            stock_code
            for stock_code in symbols
            if stock_code not in quotes
        ]

        if unresolved:
            raise ExternalApiError(
                "실시간 현재가를 "
                "가져오지 못한 종목: "
                + ", ".join(
                    unresolved[:10]
                )
            )

        return [
            self._to_response(
                quotes[
                    stock_code
                ]
            )
            for stock_code in symbols
        ]