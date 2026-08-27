import asyncio
import json
import time
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

import httpx

from app.clients.dart_client import DartClient
from app.core.config import settings
from app.core.exceptions import ConfigurationError, ExternalApiError


class KisClient:
    TOKEN_PATH = "/oauth2/tokenP"

    CURRENT_PRICE_PATH = (
        "/uapi/domestic-stock/v1/quotations/inquire-price"
    )
    CURRENT_PRICE_TR_ID = "FHKST01010100"

    DAILY_PRICE_PATH = (
        "/uapi/domestic-stock/v1/quotations/"
        "inquire-daily-itemchartprice"
    )
    DAILY_PRICE_TR_ID = "FHKST03010100"

    INTRADAY_PATH = (
        "/uapi/domestic-stock/v1/quotations/"
        "inquire-time-itemchartprice"
    )
    INTRADAY_TR_ID = "FHKST03010200"

    def __init__(self):
        self._access_token: str | None = None
        self._expires_at: float = 0.0
        self._last_request_at: float = 0.0
        self._rate_lock = asyncio.Lock()

        self._load_token_cache()

    def _require_keys(self) -> None:
        if not settings.kis_app_key or not settings.kis_app_secret:
            raise ConfigurationError(
                "KIS_APP_KEY 또는 KIS_APP_SECRET이 비어 있습니다. "
                ".env에 설정하세요."
            )

    def _token_cache_path(self) -> Path:
        return Path(settings.kis_token_cache_path)

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

            token = str(
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

            if token and expires_at > time.time() + 60:
                self._access_token = token
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

    async def _issue_token(self) -> str:
        self._require_keys()

        async with httpx.AsyncClient(
            base_url=settings.kis_base_url,
            timeout=30.0,
        ) as client:
            response = await client.post(
                self.TOKEN_PATH,
                headers={
                    "Content-Type": "application/json",
                },
                json={
                    "grant_type": "client_credentials",
                    "appkey": settings.kis_app_key,
                    "appsecret": settings.kis_app_secret,
                },
            )

        response.raise_for_status()

        payload = response.json()
        token = payload.get("access_token")

        if not token:
            raise ExternalApiError(
                f"KIS 토큰 발급 실패: {payload}"
            )

        expires_in = int(
            payload.get(
                "expires_in",
                86400,
            )
        )

        self._access_token = str(token)
        self._expires_at = time.time() + max(
            60,
            expires_in - 120,
        )

        self._save_token_cache(
            self._access_token,
            self._expires_at,
        )

        return self._access_token

    async def get_access_token(
        self,
        *,
        force: bool = False,
    ) -> str:
        if (
            not force
            and self._access_token
            and self._expires_at > time.time() + 60
        ):
            return self._access_token

        return await self._issue_token()

    async def _respect_rate_limit(self) -> None:
        async with self._rate_lock:
            now = time.monotonic()

            elapsed = now - self._last_request_at
            wait_seconds = (
                settings.kis_min_interval_seconds - elapsed
            )

            if wait_seconds > 0:
                await asyncio.sleep(wait_seconds)

            self._last_request_at = time.monotonic()

    async def _get(
        self,
        *,
        path: str,
        tr_id: str,
        params: dict[str, str],
    ) -> dict[str, Any]:
        self._require_keys()

        for attempt in range(4):
            await self._respect_rate_limit()

            token = await self.get_access_token()

            async with httpx.AsyncClient(
                base_url=settings.kis_base_url,
                timeout=30.0,
            ) as client:
                response = await client.get(
                    path,
                    headers={
                        "Content-Type": "application/json; charset=utf-8",
                        "authorization": f"Bearer {token}",
                        "appkey": settings.kis_app_key,
                        "appsecret": settings.kis_app_secret,
                        "tr_id": tr_id,
                        "custtype": "P",
                    },
                    params=params,
                )

            if response.status_code >= 500:
                if attempt < 3:
                    await asyncio.sleep(
                        0.5 * (2**attempt)
                    )
                    continue

                raise ExternalApiError(
                    "KIS 서버 오류: "
                    f"HTTP {response.status_code}"
                )

            try:
                response.raise_for_status()
            except httpx.HTTPStatusError as e:
                raise ExternalApiError(
                    "KIS HTTP 오류: "
                    f"{response.status_code}"
                ) from e

            try:
                payload = response.json()
            except ValueError as e:
                raise ExternalApiError(
                    "KIS 응답 JSON 파싱 실패"
                ) from e

            if str(payload.get("rt_cd", "")) == "0":
                return payload

            message_code = str(
                payload.get(
                    "msg_cd",
                    "",
                )
            )

            if message_code == "EGW00123" and attempt == 0:
                await self.get_access_token(
                    force=True,
                )
                continue

            if message_code == "EGW00201" and attempt < 3:
                await asyncio.sleep(
                    0.5 * (2**attempt)
                )
                continue

            raise ExternalApiError(
                "KIS API 오류: "
                f"{payload.get('msg1')} ({message_code})"
            )

        raise ExternalApiError(
            "KIS API 호출 재시도 횟수를 초과했습니다."
        )

    async def get_current_price(
        self,
        stock_code: str,
    ) -> dict[str, Any]:
        payload = await self._get(
            path=self.CURRENT_PRICE_PATH,
            tr_id=self.CURRENT_PRICE_TR_ID,
            params={
                "FID_COND_MRKT_DIV_CODE": settings.kis_market_div_code,
                "FID_INPUT_ISCD": stock_code,
            },
        )

        output = payload.get("output") or {}

        if not output:
            raise ExternalApiError(
                f"KIS 현재가 데이터 없음: {stock_code}"
            )

        return dict(output)

    async def get_daily_prices(
        self,
        stock_code: str,
        *,
        start_date: date,
        end_date: date,
    ) -> list[dict[str, Any]]:
        all_rows: list[dict[str, Any]] = []

        current_end = end_date

        while current_end >= start_date:
            current_start = max(
                start_date,
                current_end - timedelta(days=119),
            )

            payload = await self._get(
                path=self.DAILY_PRICE_PATH,
                tr_id=self.DAILY_PRICE_TR_ID,
                params={
                    "FID_COND_MRKT_DIV_CODE": settings.kis_market_div_code,
                    "FID_INPUT_ISCD": stock_code,
                    "FID_INPUT_DATE_1": current_start.strftime("%Y%m%d"),
                    "FID_INPUT_DATE_2": current_end.strftime("%Y%m%d"),
                    "FID_PERIOD_DIV_CODE": "D",
                    "FID_ORG_ADJ_PRC": "0",
                },
            )

            rows = list(
                payload.get("output2") or []
            )

            all_rows.extend(rows)

            if current_start <= start_date:
                break

            current_end = (
                current_start
                - timedelta(days=1)
            )

        unique_rows: dict[str, dict[str, Any]] = {}

        for row in all_rows:
            date_key = str(
                row.get(
                    "stck_bsop_date",
                    "",
                )
            )

            if date_key:
                unique_rows[date_key] = dict(row)

        return [
            unique_rows[key]
            for key in sorted(unique_rows)
        ]

    async def get_intraday_prices(
        self,
        stock_code: str,
        *,
        end_hour: str = "153000",
        include_previous: bool = True,
    ) -> list[dict[str, Any]]:
        current_hour = end_hour
        all_rows: list[dict[str, Any]] = []

        for _ in range(10):
            payload = await self._get(
                path=self.INTRADAY_PATH,
                tr_id=self.INTRADAY_TR_ID,
                params={
                    "FID_COND_MRKT_DIV_CODE": settings.kis_market_div_code,
                    "FID_INPUT_ISCD": stock_code,
                    "FID_INPUT_HOUR_1": current_hour,
                    "FID_PW_DATA_INCU_YN": "Y" if include_previous else "N",
                    "FID_ETC_CLS_CODE": "",
                },
            )

            rows = list(
                payload.get(
                    "output2",
                )
                or []
            )

            if not rows:
                break

            all_rows.extend(rows)

            times = [
                str(row.get("stck_cntg_hour", ""))
                for row in rows
                if row.get("stck_cntg_hour")
            ]

            if not times:
                break

            earliest = min(times)

            if earliest <= "090000" or len(rows) < 30:
                break

            current_hour = earliest

        seen: set[str] = set()
        result: list[dict[str, Any]] = []

        for row in reversed(all_rows):
            time_key = str(
                row.get(
                    "stck_cntg_hour",
                    "",
                )
            )

            if not time_key or time_key in seen:
                continue

            seen.add(time_key)
            result.append(dict(row))

        return result


kis_client = KisClient()
dart_client = DartClient()
