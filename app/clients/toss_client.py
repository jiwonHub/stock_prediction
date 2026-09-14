import asyncio
import json
import random
import time
from pathlib import Path
from typing import Any

import httpx

from app.core.config import settings
from app.core.exceptions import ConfigurationError, ExternalApiError


class TossClient:
    TOKEN_PATH = "/oauth2/token"
    PRICES_PATH = "/api/v1/prices"
    CANDLES_PATH = "/api/v1/candles"
    INVESTOR_TRADING_PATH = (
        "/api/v1/stocks/{symbol}/investor-trading"
    )

    PROGRAM_TRADES_PATH = (
        "/api/v1/stocks/{symbol}/program-trades"
    )

    SHORT_SELLING_PATH = (
        "/api/v1/stocks/{symbol}/short-selling"
    )

    CREDIT_TRADES_PATH = (
        "/api/v1/stocks/{symbol}/credit-trades"
    )

    SECURITIES_LENDING_PATH = (
        "/api/v1/stocks/{symbol}/securities-lending"
    )

    MARKET_INDICATOR_PRICES_PATH = (
        "/api/v1/market-indicators/prices"
    )

    MARKET_INDICATOR_CANDLES_PATH = (
        "/api/v1/market-indicators/{symbol}/candles"
    )

    MARKET_INDICATOR_INVESTOR_TRADING_PATH = (
        "/api/v1/market-indicators/{symbol}/investor-trading"
    )

    EXCHANGE_RATE_PATH = "/api/v1/exchange-rate"

    MARKET_INDICATOR_SYMBOLS = {
        "KOSPI",
        "KOSDAQ",
        "KR_BOND_2Y",
        "KR_BOND_3Y",
        "KR_BOND_5Y",
        "KR_BOND_10Y",
        "KR_BOND_20Y",
        "KR_BOND_30Y",
    }

    def __init__(self):
        self._access_token: str | None = None
        self._expires_at: float = 0.0

        self._token_lock = asyncio.Lock()
        self._market_data_lock = asyncio.Lock()
        self._trading_trend_lock = asyncio.Lock()
        self._market_indicator_lock = asyncio.Lock()
        self._market_indicator_chart_lock = asyncio.Lock()
        self._market_info_lock = asyncio.Lock()

        self._last_market_data_request_at: float = 0.0
        self._last_trading_trend_request_at: float = 0.0
        self._last_market_indicator_request_at: float = 0.0
        self._last_market_indicator_chart_request_at: float = 0.0
        self._last_market_info_request_at: float = 0.0
        self._load_token_cache()

    def _require_keys(self) -> None:
        if (
            not settings.toss_client_id
            or not settings.toss_client_secret
        ):
            raise ConfigurationError(
                "TOSS_CLIENT_ID 또는 "
                "TOSS_CLIENT_SECRET이 비어 있습니다."
            )

    def _token_cache_path(self) -> Path:
        return Path(
            settings.toss_token_cache_path
        )

    def _load_token_cache(self) -> None:
        path = self._token_cache_path()

        if not path.exists():
            return

        try:
            payload = json.loads(
                path.read_text(
                    encoding="utf-8",
                )
            )

            access_token = str(
                payload.get(
                    "access_token",
                    "",
                )
            )

            expires_at = float(
                payload.get(
                    "expires_at",
                    0,
                )
            )

            margin = max(
                60,
                settings.toss_token_refresh_margin_seconds,
            )

            if (
                access_token
                and expires_at
                > time.time() + margin
            ):
                self._access_token = access_token
                self._expires_at = expires_at

        except Exception:
            self._access_token = None
            self._expires_at = 0.0

    def _save_token_cache(
        self,
        access_token: str,
        expires_at: float,
    ) -> None:
        path = self._token_cache_path()

        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        path.write_text(
            json.dumps(
                {
                    "access_token": access_token,
                    "expires_at": expires_at,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

    def _clear_token(self) -> None:
        self._access_token = None
        self._expires_at = 0.0

        path = self._token_cache_path()

        try:
            if path.exists():
                path.unlink()
        except OSError:
            pass

    @staticmethod
    def _error_message(
        response: httpx.Response,
    ) -> str:
        request_id = (
            response.headers.get(
                "X-Request-Id"
            )
            or response.headers.get(
                "x-request-id"
            )
            or ""
        )

        try:
            payload = response.json()
        except ValueError:
            payload = {}

        error = payload.get("error")

        if isinstance(error, dict):
            code = str(
                error.get(
                    "code",
                    "",
                )
            )

            message = str(
                error.get(
                    "message",
                    "",
                )
            )

        else:
            code = str(
                error or ""
            )

            message = str(
                payload.get(
                    "error_description",
                    "",
                )
            )

        parts = [
            f"HTTP {response.status_code}",
        ]

        if code:
            parts.append(code)

        if message:
            parts.append(message)

        if request_id:
            parts.append(
                f"requestId={request_id}"
            )

        return " / ".join(parts)

    @staticmethod
    async def _retry_wait(
        response: httpx.Response | None,
        attempt: int,
    ) -> None:
        retry_after: float | None = None

        if response is not None:
            retry_text = response.headers.get(
                "Retry-After"
            )

            if retry_text:
                try:
                    retry_after = float(
                        retry_text
                    )
                except ValueError:
                    retry_after = None

        if retry_after is None:
            retry_after = (
                0.5 * (2**attempt)
                + random.uniform(
                    0.0,
                    0.25,
                )
            )

        await asyncio.sleep(
            max(
                0.1,
                retry_after,
            )
        )

    async def _issue_token(self) -> str:
        self._require_keys()

        for attempt in range(4):
            response: httpx.Response | None = None

            try:
                async with httpx.AsyncClient(
                    base_url=settings.toss_base_url,
                    timeout=(
                        settings
                        .toss_request_timeout_seconds
                    ),
                ) as client:
                    response = await client.post(
                        self.TOKEN_PATH,
                        headers={
                            "Content-Type":
                                "application/x-www-form-urlencoded",
                            "Accept":
                                "application/json",
                        },
                        data={
                            "grant_type":
                                "client_credentials",
                            "client_id":
                                settings.toss_client_id,
                            "client_secret":
                                settings.toss_client_secret,
                        },
                    )

            except httpx.RequestError as e:
                if attempt < 3:
                    await self._retry_wait(
                        None,
                        attempt,
                    )
                    continue

                raise ExternalApiError(
                    "Toss 토큰 서버 연결 실패"
                ) from e

            if response.status_code == 429:
                if attempt < 3:
                    await self._retry_wait(
                        response,
                        attempt,
                    )
                    continue

            if response.status_code >= 500:
                if attempt < 3:
                    await self._retry_wait(
                        response,
                        attempt,
                    )
                    continue

            if response.status_code != 200:
                raise ExternalApiError(
                    "Toss 토큰 발급 실패: "
                    + self._error_message(
                        response
                    )
                )

            try:
                payload = response.json()
            except ValueError as e:
                raise ExternalApiError(
                    "Toss 토큰 응답 JSON 파싱 실패"
                ) from e

            access_token = str(
                payload.get(
                    "access_token",
                    "",
                )
            )

            if not access_token:
                raise ExternalApiError(
                    "Toss 토큰 응답에 "
                    "access_token이 없습니다."
                )

            try:
                expires_in = int(
                    payload.get(
                        "expires_in",
                        3600,
                    )
                )
            except (
                TypeError,
                ValueError,
            ):
                expires_in = 3600

            margin = max(
                60,
                settings.toss_token_refresh_margin_seconds,
            )

            self._access_token = access_token

            self._expires_at = (
                time.time()
                + max(
                    60,
                    expires_in - margin,
                )
            )

            self._save_token_cache(
                self._access_token,
                self._expires_at,
            )

            return self._access_token

        raise ExternalApiError(
            "Toss 토큰 발급 재시도 횟수를 "
            "초과했습니다."
        )

    async def get_access_token(
        self,
        *,
        force: bool = False,
    ) -> str:
        margin = max(
            60,
            settings.toss_token_refresh_margin_seconds,
        )

        if (
            not force
            and self._access_token
            and self._expires_at
            > time.time() + margin
        ):
            return self._access_token

        async with self._token_lock:
            if (
                not force
                and self._access_token
                and self._expires_at
                > time.time() + margin
            ):
                return self._access_token

            if force:
                self._clear_token()

            return await self._issue_token()

    async def _respect_market_data_rate_limit(
        self,
    ) -> None:
        async with self._market_data_lock:
            now = time.monotonic()

            elapsed = (
                now
                - self._last_market_data_request_at
            )

            wait_seconds = (
                settings
                .toss_market_data_min_interval_seconds
                - elapsed
            )

            if wait_seconds > 0:
                await asyncio.sleep(
                    wait_seconds
                )

            self._last_market_data_request_at = (
                time.monotonic()
            )

    async def _respect_trading_trend_rate_limit(
        self,
    ) -> None:
        async with self._trading_trend_lock:
            now = time.monotonic()

            elapsed = (
                now
                - self._last_trading_trend_request_at
            )

            wait_seconds = (
                settings
                .toss_trading_trend_min_interval_seconds
                - elapsed
            )

            if wait_seconds > 0:
                await asyncio.sleep(
                    wait_seconds
                )

            self._last_trading_trend_request_at = (
                time.monotonic()
            )

    async def _respect_market_indicator_rate_limit(
        self,
    ) -> None:
        async with self._market_indicator_lock:
            now = time.monotonic()

            elapsed = (
                now
                - self._last_market_indicator_request_at
            )

            wait_seconds = (
                settings
                .toss_market_indicator_min_interval_seconds
                - elapsed
            )

            if wait_seconds > 0:
                await asyncio.sleep(
                    wait_seconds
                )

            self._last_market_indicator_request_at = (
                time.monotonic()
            )

    async def _respect_market_indicator_chart_rate_limit(
        self,
    ) -> None:
        async with self._market_indicator_chart_lock:
            now = time.monotonic()

            elapsed = (
                now
                - self._last_market_indicator_chart_request_at
            )

            wait_seconds = (
                settings
                .toss_market_indicator_chart_min_interval_seconds
                - elapsed
            )

            if wait_seconds > 0:
                await asyncio.sleep(
                    wait_seconds
                )

            self._last_market_indicator_chart_request_at = (
                time.monotonic()
            )

    async def _respect_market_info_rate_limit(
        self,
    ) -> None:
        async with self._market_info_lock:
            now = time.monotonic()

            elapsed = (
                now
                - self._last_market_info_request_at
            )

            wait_seconds = (
                settings
                .toss_market_info_min_interval_seconds
                - elapsed
            )

            if wait_seconds > 0:
                await asyncio.sleep(
                    wait_seconds
                )

            self._last_market_info_request_at = (
                time.monotonic()
            )

    async def _get_market_data(
        self,
        *,
        path: str,
        params: dict[str, str],
    ) -> dict[str, Any]:
        self._require_keys()

        token_refreshed = False

        for attempt in range(4):
            await self._respect_market_data_rate_limit()

            token = await self.get_access_token()

            response: httpx.Response | None = None

            try:
                async with httpx.AsyncClient(
                    base_url=settings.toss_base_url,
                    timeout=(
                        settings
                        .toss_request_timeout_seconds
                    ),
                ) as client:
                    response = await client.get(
                        path,
                        headers={
                            "Authorization":
                                f"Bearer {token}",
                            "Accept":
                                "application/json",
                        },
                        params=params,
                    )

            except httpx.RequestError as e:
                if attempt < 3:
                    await self._retry_wait(
                        None,
                        attempt,
                    )
                    continue

                raise ExternalApiError(
                    "Toss Market Data 연결 실패"
                ) from e

            if (
                response.status_code == 401
                and not token_refreshed
            ):
                token_refreshed = True

                await self.get_access_token(
                    force=True,
                )

                continue

            if response.status_code == 429:
                if attempt < 3:
                    await self._retry_wait(
                        response,
                        attempt,
                    )
                    continue

            if response.status_code >= 500:
                if attempt < 3:
                    await self._retry_wait(
                        response,
                        attempt,
                    )
                    continue

            if response.status_code != 200:
                raise ExternalApiError(
                    "Toss Market Data 오류: "
                    + self._error_message(
                        response
                    )
                )

            try:
                payload = response.json()
            except ValueError as e:
                raise ExternalApiError(
                    "Toss Market Data "
                    "응답 JSON 파싱 실패"
                ) from e

            return dict(payload)

        raise ExternalApiError(
            "Toss Market Data 호출 "
            "재시도 횟수를 초과했습니다."
        )

    async def _get_trading_trend(
        self,
        *,
        path: str,
        params: dict[str, str],
    ) -> dict[str, Any]:
        self._require_keys()

        token_refreshed = False

        for attempt in range(4):
            await self._respect_trading_trend_rate_limit()

            token = await self.get_access_token()

            response: httpx.Response | None = None

            try:
                async with httpx.AsyncClient(
                    base_url=settings.toss_base_url,
                    timeout=(
                        settings
                        .toss_request_timeout_seconds
                    ),
                ) as client:
                    response = await client.get(
                        path,
                        headers={
                            "Authorization":
                                f"Bearer {token}",
                            "Accept":
                                "application/json",
                        },
                        params=params,
                    )

            except httpx.RequestError as e:
                if attempt < 3:
                    await self._retry_wait(
                        None,
                        attempt,
                    )
                    continue

                raise ExternalApiError(
                    "Toss Trading Trend 연결 실패"
                ) from e

            if (
                response.status_code == 401
                and not token_refreshed
            ):
                token_refreshed = True

                await self.get_access_token(
                    force=True,
                )

                continue

            if response.status_code == 429:
                if attempt < 3:
                    await self._retry_wait(
                        response,
                        attempt,
                    )
                    continue

            if response.status_code >= 500:
                if attempt < 3:
                    await self._retry_wait(
                        response,
                        attempt,
                    )
                    continue

            if response.status_code != 200:
                raise ExternalApiError(
                    "Toss Trading Trend 오류: "
                    + self._error_message(
                        response
                    )
                )

            try:
                payload = response.json()
            except ValueError as e:
                raise ExternalApiError(
                    "Toss Trading Trend "
                    "응답 JSON 파싱 실패"
                ) from e

            return dict(
                payload
            )

        raise ExternalApiError(
            "Toss Trading Trend 호출 "
            "재시도 횟수를 초과했습니다."
        )

    async def get_prices(
        self,
        stock_codes: list[str],
    ) -> list[dict[str, Any]]:
        symbols = [
            str(code).strip()
            for code in stock_codes
            if str(code).strip()
        ]

        symbols = list(
            dict.fromkeys(
                symbols
            )
        )

        if not symbols:
            return []

        if len(symbols) > 200:
            raise ValueError(
                "Toss 현재가는 한 요청에 "
                "최대 200종목까지 조회할 수 있습니다."
            )

        payload = await self._get_market_data(
            path=self.PRICES_PATH,
            params={
                "symbols": ",".join(
                    symbols
                ),
            },
        )

        result = payload.get("result")

        if not isinstance(
            result,
            list,
        ):
            raise ExternalApiError(
                "Toss 현재가 응답 형식이 "
                "올바르지 않습니다."
            )

        return [
            dict(row)
            for row in result
            if isinstance(
                row,
                dict,
            )
        ]

    async def get_candles(
        self,
        stock_code: str,
        *,
        interval: str,
        count: int = 200,
        before: str | None = None,
        adjusted: bool = True,
    ) -> dict[str, Any]:
        code = stock_code.strip()

        if not code:
            raise ValueError(
                "stock_code가 비어 있습니다."
            )

        normalized_interval = (
            interval.strip().lower()
        )

        if normalized_interval not in {
            "1m",
            "1d",
        }:
            raise ValueError(
                "Toss 캔들 interval은 "
                "1m 또는 1d만 지원합니다."
            )

        if count < 1 or count > 200:
            raise ValueError(
                "Toss 캔들 count는 "
                "1~200 범위여야 합니다."
            )

        params = {
            "symbol": code,
            "interval": normalized_interval,
            "count": str(count),
            "adjusted": (
                "true"
                if adjusted
                else "false"
            ),
        }

        if before:
            params["before"] = (
                before.strip()
            )

        payload = await self._get_market_data(
            path=self.CANDLES_PATH,
            params=params,
        )

        result = payload.get(
            "result"
        )

        if not isinstance(
            result,
            dict,
        ):
            raise ExternalApiError(
                "Toss 캔들 응답 형식이 "
                "올바르지 않습니다."
            )

        raw_candles = result.get(
            "candles"
        )

        if not isinstance(
            raw_candles,
            list,
        ):
            raise ExternalApiError(
                "Toss 캔들 목록이 "
                "올바르지 않습니다."
            )

        next_before = str(
            result.get(
                "nextBefore",
                "",
            )
            or ""
        ).strip()

        return {
            "candles": [
                dict(row)
                for row in raw_candles
                if isinstance(
                    row,
                    dict,
                )
            ],
            "nextBefore": (
                next_before
                or None
            ),
        }

    async def get_daily_candles(
        self,
        stock_code: str,
        *,
        before: str | None = None,
        count: int = 200,
        adjusted: bool = True,
    ) -> dict[str, Any]:
        return await self.get_candles(
            stock_code,
            interval="1d",
            count=count,
            before=before,
            adjusted=adjusted,
        )

    async def get_daily_candle_history(
        self,
        stock_code: str,
        *,
        adjusted: bool = True,
        max_pages: int = 500,
    ) -> list[dict[str, Any]]:
        if max_pages < 1:
            raise ValueError(
                "max_pages는 1 이상이어야 합니다."
            )

        code = stock_code.strip()

        if not code:
            raise ValueError(
                "stock_code가 비어 있습니다."
            )

        before: str | None = None

        seen_cursors: set[str] = set()

        candles_by_timestamp: dict[
            str,
            dict[str, Any],
        ] = {}

        for page in range(
            1,
            max_pages + 1,
        ):
            payload = (
                await self.get_daily_candles(
                    code,
                    before=before,
                    count=200,
                    adjusted=adjusted,
                )
            )

            candles = list(
                payload.get(
                    "candles",
                    [],
                )
            )

            for candle in candles:
                timestamp = str(
                    candle.get(
                        "timestamp",
                        "",
                    )
                ).strip()

                if timestamp:
                    candles_by_timestamp[
                        timestamp
                    ] = candle

            next_before = str(
                payload.get(
                    "nextBefore",
                    "",
                )
                or ""
            ).strip()

            print(
                "[TOSS CANDLE] "
                f"{code} "
                f"page={page} "
                f"rows={len(candles)} "
                f"total={len(candles_by_timestamp)} "
                f"nextBefore={next_before or '-'}",
                flush=True,
            )

            if not candles:
                break

            if not next_before:
                break

            if next_before in seen_cursors:
                break

            seen_cursors.add(
                next_before
            )

            before = next_before

        return [
            candles_by_timestamp[key]
            for key in sorted(
                candles_by_timestamp
            )
        ]
    
    async def get_investor_trading(
        self,
        stock_code: str,
        *,
        count: int = 60,
        until: str | None = None,
    ) -> dict[str, Any]:
        code = stock_code.strip()

        if not code:
            raise ValueError(
                "stock_code가 비어 있습니다."
            )

        if count < 1 or count > 100:
            raise ValueError(
                "Toss 투자자별 매매동향 count는 "
                "1~100 범위여야 합니다."
            )

        params = {
            "count": str(
                count
            ),
        }

        if until:
            params["until"] = (
                until.strip()
            )

        payload = await self._get_trading_trend(
            path=(
                self.INVESTOR_TRADING_PATH.format(
                    symbol=code,
                )
            ),
            params=params,
        )

        result = payload.get(
            "result"
        )

        if not isinstance(
            result,
            dict,
        ):
            raise ExternalApiError(
                "Toss 투자자별 매매동향 "
                "응답 형식이 올바르지 않습니다."
            )

        records = result.get(
            "records"
        )

        if not isinstance(
            records,
            list,
        ):
            raise ExternalApiError(
                "Toss 투자자별 매매동향 records "
                "형식이 올바르지 않습니다."
            )

        next_until = str(
            result.get(
                "nextUntil",
                "",
            )
            or ""
        ).strip()

        return {
            "records": [
                dict(row)
                for row in records
                if isinstance(
                    row,
                    dict,
                )
            ],
            "nextUntil": (
                next_until
                or None
            ),
        }
    
    async def _get_stock_trading_records(
        self,
        stock_code: str,
        *,
        path: str,
        count: int = 60,
        until: str | None = None,
    ) -> dict[str, Any]:
        code = stock_code.strip()

        if not code:
            raise ValueError(
                "stock_code가 비어 있습니다."
            )

        if count < 1 or count > 100:
            raise ValueError(
                "Toss 거래동향 count는 "
                "1~100 범위여야 합니다."
            )

        params = {
            "count": str(count),
        }

        if until:
            params["until"] = (
                until.strip()
            )

        payload = (
            await self._get_trading_trend(
                path=path.format(
                    symbol=code,
                ),
                params=params,
            )
        )

        result = payload.get(
            "result"
        )

        if not isinstance(
            result,
            dict,
        ):
            raise ExternalApiError(
                "Toss 거래동향 응답 형식이 "
                "올바르지 않습니다."
            )

        records = result.get(
            "records"
        )

        if not isinstance(
            records,
            list,
        ):
            raise ExternalApiError(
                "Toss 거래동향 records 형식이 "
                "올바르지 않습니다."
            )

        next_until = str(
            result.get(
                "nextUntil",
                "",
            )
            or ""
        ).strip()

        return {
            "records": [
                dict(row)
                for row in records
                if isinstance(
                    row,
                    dict,
                )
            ],
            "nextUntil": (
                next_until
                or None
            ),
        }

    async def get_program_trades(
        self,
        stock_code: str,
        *,
        count: int = 60,
        until: str | None = None,
    ) -> dict[str, Any]:
        return await self._get_stock_trading_records(
            stock_code,
            path=self.PROGRAM_TRADES_PATH,
            count=count,
            until=until,
        )

    async def get_short_selling(
        self,
        stock_code: str,
        *,
        count: int = 60,
        until: str | None = None,
    ) -> dict[str, Any]:
        return await self._get_stock_trading_records(
            stock_code,
            path=self.SHORT_SELLING_PATH,
            count=count,
            until=until,
        )

    async def get_credit_trades(
        self,
        stock_code: str,
        *,
        count: int = 60,
        until: str | None = None,
    ) -> dict[str, Any]:
        return await self._get_stock_trading_records(
            stock_code,
            path=self.CREDIT_TRADES_PATH,
            count=count,
            until=until,
        )

    async def get_securities_lending(
        self,
        stock_code: str,
        *,
        count: int = 60,
        until: str | None = None,
    ) -> dict[str, Any]:
        return await self._get_stock_trading_records(
            stock_code,
            path=self.SECURITIES_LENDING_PATH,
            count=count,
            until=until,
        )
    
    async def get_market_indicator_prices(
        self,
        symbols: list[str],
    ) -> list[dict[str, Any]]:
        normalized = list(
            dict.fromkeys(
                str(symbol).strip().upper()
                for symbol in symbols
                if str(symbol).strip()
            )
        )

        if not normalized:
            return []

        invalid = [
            symbol
            for symbol in normalized
            if symbol
            not in self.MARKET_INDICATOR_SYMBOLS
        ]

        if invalid:
            raise ValueError(
                "지원하지 않는 Toss 시장지표 심볼: "
                + ",".join(
                    invalid
                )
            )

        await self._respect_market_indicator_rate_limit()

        payload = await self._get_market_data(
            path=self.MARKET_INDICATOR_PRICES_PATH,
            params={
                "symbols": ",".join(
                    normalized
                ),
            },
        )

        result = payload.get(
            "result"
        )

        if not isinstance(
            result,
            list,
        ):
            raise ExternalApiError(
                "Toss 시장지표 현재가 응답 형식이 "
                "올바르지 않습니다."
            )

        return [
            dict(row)
            for row in result
            if isinstance(
                row,
                dict,
            )
        ]

    async def get_market_indicator_candles(
        self,
        symbol: str,
        *,
        interval: str = "1d",
        count: int = 200,
        before: str | None = None,
    ) -> dict[str, Any]:
        code = (
            symbol.strip().upper()
        )

        if code not in self.MARKET_INDICATOR_SYMBOLS:
            raise ValueError(
                "지원하지 않는 Toss 시장지표 심볼입니다."
            )

        normalized_interval = (
            interval.strip().lower()
        )

        if normalized_interval not in {
            "1m",
            "1d",
        }:
            raise ValueError(
                "Toss 시장지표 캔들 interval은 "
                "1m 또는 1d만 지원합니다."
            )

        if (
            code.startswith(
                "KR_BOND_"
            )
            and normalized_interval != "1d"
        ):
            raise ValueError(
                "Toss 국채 시장지표는 "
                "1d 캔들만 지원합니다."
            )

        if count < 1 or count > 200:
            raise ValueError(
                "Toss 시장지표 캔들 count는 "
                "1~200 범위여야 합니다."
            )

        params = {
            "interval": normalized_interval,
            "count": str(
                count
            ),
        }

        if before:
            params["before"] = (
                before.strip()
            )

        await (
            self
            ._respect_market_indicator_chart_rate_limit()
        )

        payload = await self._get_market_data(
            path=(
                self.MARKET_INDICATOR_CANDLES_PATH
                .format(
                    symbol=code,
                )
            ),
            params=params,
        )

        result = payload.get(
            "result"
        )

        if not isinstance(
            result,
            dict,
        ):
            raise ExternalApiError(
                "Toss 시장지표 캔들 응답 형식이 "
                "올바르지 않습니다."
            )

        candles = result.get(
            "candles"
        )

        if not isinstance(
            candles,
            list,
        ):
            raise ExternalApiError(
                "Toss 시장지표 캔들 목록이 "
                "올바르지 않습니다."
            )

        next_before = str(
            result.get(
                "nextBefore",
                "",
            )
            or ""
        ).strip()

        return {
            "candles": [
                dict(row)
                for row in candles
                if isinstance(
                    row,
                    dict,
                )
            ],
            "nextBefore": (
                next_before
                or None
            ),
        }

    async def get_market_indicator_investor_trading(
        self,
        symbol: str,
        *,
        interval: str = "1d",
        count: int = 60,
        until: str | None = None,
    ) -> dict[str, Any]:
        code = (
            symbol.strip().upper()
        )

        if code not in {
            "KOSPI",
            "KOSDAQ",
        }:
            raise ValueError(
                "Toss 시장 투자자 매매대금은 "
                "KOSPI/KOSDAQ만 지원합니다."
            )

        normalized_interval = (
            interval.strip().lower()
        )

        if normalized_interval not in {
            "1d",
            "1w",
            "1mo",
            "1y",
        }:
            raise ValueError(
                "Toss 시장 투자자 interval은 "
                "1d/1w/1mo/1y만 지원합니다."
            )

        if count < 1 or count > 100:
            raise ValueError(
                "Toss 시장 투자자 count는 "
                "1~100 범위여야 합니다."
            )

        params = {
            "interval": normalized_interval,
            "count": str(
                count
            ),
        }

        if until:
            params["until"] = (
                until.strip()
            )

        await self._respect_market_indicator_rate_limit()

        payload = await self._get_market_data(
            path=(
                self.MARKET_INDICATOR_INVESTOR_TRADING_PATH
                .format(
                    symbol=code,
                )
            ),
            params=params,
        )

        result = payload.get(
            "result"
        )

        if not isinstance(
            result,
            dict,
        ):
            raise ExternalApiError(
                "Toss 시장 투자자 매매대금 응답 형식이 "
                "올바르지 않습니다."
            )

        records = result.get(
            "records"
        )

        if not isinstance(
            records,
            list,
        ):
            raise ExternalApiError(
                "Toss 시장 투자자 records 형식이 "
                "올바르지 않습니다."
            )

        next_until = str(
            result.get(
                "nextUntil",
                "",
            )
            or ""
        ).strip()

        return {
            "records": [
                dict(row)
                for row in records
                if isinstance(
                    row,
                    dict,
                )
            ],
            "nextUntil": (
                next_until
                or None
            ),
        }

    async def get_exchange_rate(
        self,
        *,
        base_currency: str = "USD",
        quote_currency: str = "KRW",
        date_time: str | None = None,
    ) -> dict[str, Any]:
        base = (
            base_currency.strip().upper()
        )

        quote = (
            quote_currency.strip().upper()
        )

        if (
            base not in {
                "KRW",
                "USD",
            }
            or quote not in {
                "KRW",
                "USD",
            }
            or base == quote
        ):
            raise ValueError(
                "Toss 환율은 KRW/USD 상호 환산만 "
                "지원합니다."
            )

        params = {
            "baseCurrency": base,
            "quoteCurrency": quote,
        }

        if date_time:
            params["dateTime"] = (
                date_time.strip()
            )

        await self._respect_market_info_rate_limit()

        payload = await self._get_market_data(
            path=self.EXCHANGE_RATE_PATH,
            params=params,
        )

        result = payload.get(
            "result"
        )

        if not isinstance(
            result,
            dict,
        ):
            raise ExternalApiError(
                "Toss 환율 응답 형식이 "
                "올바르지 않습니다."
            )

        return dict(
            result
        )

    async def get_current_price(
        self,
        stock_code: str,
    ) -> dict[str, Any]:
        code = stock_code.strip()

        if not code:
            raise ValueError(
                "stock_code가 비어 있습니다."
            )

        rows = await self.get_prices(
            [
                code,
            ]
        )

        for row in rows:
            if str(
                row.get(
                    "symbol",
                    "",
                )
            ) == code:
                return row

        if rows:
            return rows[0]

        raise ExternalApiError(
            f"Toss 현재가 데이터 없음: {code}"
        )


toss_client = TossClient()