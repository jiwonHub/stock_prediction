from __future__ import annotations

from datetime import date
from typing import Any
from urllib.parse import quote

import httpx

from app.core.config import settings
from app.core.exceptions import (
    ConfigurationError,
    ExternalApiError,
)


class EcosClient:
    BASE_RATE_STAT_CODE = (
        "722Y001"
    )
    BASE_RATE_ITEM_CODE = (
        "0101000"
    )

    USD_KRW_STAT_CODE = (
        "731Y001"
    )
    USD_KRW_ITEM_CODE = (
        "0000001"
    )

    KTB_3Y_STAT_CODE = (
        "817Y002"
    )
    KTB_3Y_ITEM_CODE = (
        "010200000"
    )

    KTB_10Y_STAT_CODE = (
        "817Y002"
    )
    KTB_10Y_ITEM_CODE = (
        "010210000"
    )

    def _require_key(
        self,
    ) -> None:
        if not settings.ecos_api_key:
            raise ConfigurationError(
                "ECOS_API_KEY가 "
                "비어 있습니다. "
                "한국은행 ECOS "
                "Open API 인증키를 "
                ".env에 설정하세요."
            )

    async def statistic_search(
        self,
        *,
        stat_code: str,
        cycle: str,
        start: str,
        end: str,
        item_code1: str,
        item_code2: str | None = None,
        item_code3: str | None = None,
        item_code4: str | None = None,
        page_size: int = 1000,
    ) -> list[dict[str, Any]]:
        self._require_key()

        page_size = max(
            1,
            min(
                1000,
                page_size,
            ),
        )

        start_count = 1

        result: list[
            dict[str, Any]
        ] = []

        while True:
            end_count = (
                start_count
                + page_size
                - 1
            )

            segments = [
                "StatisticSearch",
                settings.ecos_api_key,
                "json",
                "kr",
                str(start_count),
                str(end_count),
                stat_code,
                cycle,
                start,
                end,
                item_code1,
            ]

            for value in (
                item_code2,
                item_code3,
                item_code4,
            ):
                if value is None:
                    break

                segments.append(
                    value
                )

            path = "/".join(
                quote(
                    segment,
                    safe="",
                )
                for segment
                in segments
            )

            async with (
                httpx.AsyncClient(
                    base_url=(
                        settings
                        .ecos_base_url
                        .rstrip("/")
                    ),
                    timeout=30.0,
                )
            ) as client:
                response = (
                    await client.get(
                        f"/{path}"
                    )
                )

            try:
                response.raise_for_status()
            except (
                httpx.HTTPStatusError
            ) as e:
                raise ExternalApiError(
                    "ECOS HTTP 오류: "
                    f"{response.status_code}"
                ) from e

            try:
                payload = (
                    response.json()
                )
            except ValueError as e:
                raise ExternalApiError(
                    "ECOS 응답 JSON "
                    "파싱 실패"
                ) from e

            error = payload.get(
                "RESULT"
            )

            if error:
                code = str(
                    error.get(
                        "CODE",
                        "",
                    )
                )

                message = str(
                    error.get(
                        "MESSAGE",
                        "",
                    )
                )

                if code in {
                    "INFO-200",
                    "200",
                }:
                    return result

                raise ExternalApiError(
                    "ECOS API 오류: "
                    f"{message} "
                    f"({code})"
                )

            body = (
                payload.get(
                    "StatisticSearch"
                )
                or {}
            )

            rows = list(
                body.get(
                    "row"
                )
                or []
            )

            result.extend(
                dict(row)
                for row
                in rows
            )

            total_count = int(
                body.get(
                    "list_total_count",
                    len(result),
                )
                or len(result)
            )

            if (
                not rows
                or len(result)
                >= total_count
                or len(rows)
                < page_size
            ):
                break

            start_count = (
                end_count + 1
            )

        return result

    async def get_base_rate(
        self,
        *,
        start_date: date,
        end_date: date,
    ) -> list[dict[str, Any]]:
        return await (
            self.statistic_search(
                stat_code=(
                    self
                    .BASE_RATE_STAT_CODE
                ),
                cycle="D",
                start=(
                    start_date
                    .strftime(
                        "%Y%m%d"
                    )
                ),
                end=(
                    end_date
                    .strftime(
                        "%Y%m%d"
                    )
                ),
                item_code1=(
                    self
                    .BASE_RATE_ITEM_CODE
                ),
            )
        )

    async def get_usd_krw(
        self,
        *,
        start_date: date,
        end_date: date,
    ) -> list[dict[str, Any]]:
        return await (
            self.statistic_search(
                stat_code=(
                    self
                    .USD_KRW_STAT_CODE
                ),
                cycle="D",
                start=(
                    start_date
                    .strftime(
                        "%Y%m%d"
                    )
                ),
                end=(
                    end_date
                    .strftime(
                        "%Y%m%d"
                    )
                ),
                item_code1=(
                    self
                    .USD_KRW_ITEM_CODE
                ),
            )
        )

    async def get_ktb_3y(
        self,
        *,
        start_date: date,
        end_date: date,
    ) -> list[dict[str, Any]]:
        return await (
            self.statistic_search(
                stat_code=(
                    self
                    .KTB_3Y_STAT_CODE
                ),
                cycle="D",
                start=(
                    start_date
                    .strftime(
                        "%Y%m%d"
                    )
                ),
                end=(
                    end_date
                    .strftime(
                        "%Y%m%d"
                    )
                ),
                item_code1=(
                    self
                    .KTB_3Y_ITEM_CODE
                ),
            )
        )

    async def get_ktb_10y(
        self,
        *,
        start_date: date,
        end_date: date,
    ) -> list[dict[str, Any]]:
        return await (
            self.statistic_search(
                stat_code=(
                    self
                    .KTB_10Y_STAT_CODE
                ),
                cycle="D",
                start=(
                    start_date
                    .strftime(
                        "%Y%m%d"
                    )
                ),
                end=(
                    end_date
                    .strftime(
                        "%Y%m%d"
                    )
                ),
                item_code1=(
                    self
                    .KTB_10Y_ITEM_CODE
                ),
            )
        )


ecos_client = EcosClient()