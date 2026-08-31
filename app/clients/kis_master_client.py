from __future__ import annotations
import asyncio
import time
from io import BytesIO
from zipfile import ZipFile

import httpx

from app.core.exceptions import ExternalApiError


class KisMasterClient:
    CACHE_SECONDS = 21600.0
    KOSPI_URL = (
        "https://new.real.download.dws.co.kr/"
        "common/master/kospi_code.mst.zip"
    )

    KOSDAQ_URL = (
        "https://new.real.download.dws.co.kr/"
        "common/master/kosdaq_code.mst.zip"
    )

    def __init__(self) -> None:
        self._universe_cache: list[dict] | None = None
        self._cache_expires_at = 0.0

    async def get_market_cap_universe(
        self,
        *,
        limit: int = 100,
    ) -> list[dict]:
        now = time.monotonic()

        if (
            self._universe_cache is None
            or now >= self._cache_expires_at
        ):
            kospi, kosdaq = await asyncio.gather(
                self._download_market(
                    market="KOSPI",
                ),
                self._download_market(
                    market="KOSDAQ",
                ),
            )

            rows = [
                *kospi,
                *kosdaq,
            ]

            rows = [
                row
                for row in rows
                if row["market_cap_100m"] > 0
                and self._is_common_stock(row)
            ]

            rows.sort(
                key=lambda row: row["market_cap_100m"],
                reverse=True,
            )

            self._universe_cache = rows

            self._cache_expires_at = (
                time.monotonic()
                + self.CACHE_SECONDS
            )

        return [
            dict(row)
            for row
            in self._universe_cache[:limit]
        ]

    async def _download_market(
        self,
        *,
        market: str,
    ) -> list[dict]:
        if market == "KOSPI":
            url = self.KOSPI_URL
            tail_size = 227

        elif market == "KOSDAQ":
            url = self.KOSDAQ_URL
            tail_size = 221

        else:
            raise ValueError(
                f"지원하지 않는 시장: {market}"
            )

        async with httpx.AsyncClient(
            timeout=httpx.Timeout(
                10.0,
                connect=5.0,
            ),
            follow_redirects=True,
        ) as client:
            response = await client.get(url)

        try:
            response.raise_for_status()
        except httpx.HTTPStatusError as e:
            raise ExternalApiError(
                f"KIS {market} 마스터 다운로드 실패: "
                f"HTTP {response.status_code}"
            ) from e

        try:
            with ZipFile(
                BytesIO(response.content)
            ) as zf:
                names = [
                    name
                    for name in zf.namelist()
                    if not name.endswith("/")
                ]

                if not names:
                    raise ValueError(
                        "압축파일 내부 파일 없음"
                    )

                content = zf.read(
                    names[0]
                )

        except Exception as e:
            raise ExternalApiError(
                f"KIS {market} 마스터 압축 해제 실패"
            ) from e

        rows: list[dict] = []

        for raw_line in content.splitlines():
            row = self._parse_line(
                raw_line,
                market=market,
                tail_size=tail_size,
            )

            if row is not None:
                rows.append(row)

        return rows

    def _parse_line(
        self,
        raw_line: bytes,
        *,
        market: str,
        tail_size: int,
    ) -> dict | None:
        line = raw_line.rstrip(
            b"\r\n"
        )

        if len(line) <= (
            tail_size + 21
        ):
            return None

        prefix = line[
            :-tail_size
        ]

        tail = line[
            -tail_size:
        ]

        code = (
            prefix[0:9]
            .decode(
                "cp949",
                errors="ignore",
            )
            .strip()
        )

        if len(code) > 6:
            code = code[-6:]

        if not code:
            return None

        name = (
            prefix[21:]
            .decode(
                "cp949",
                errors="ignore",
            )
            .strip()
        )

        if market == "KOSPI":
            sector_large_code = (
                self._text(
                    tail,
                    3,
                    7,
                )
            )

            sector_middle_code = (
                self._text(
                    tail,
                    7,
                    11,
                )
            )

            sector_small_code = (
                self._text(
                    tail,
                    11,
                    15,
                )
            )

            etp_code = self._text(
                tail,
                22,
                23,
            )

            spac_yn = self._text(
                tail,
                29,
                30,
            )

            preferred_code = (
                self._text(
                    tail,
                    158,
                    159,
                )
            )

            market_cap = (
                self._int(
                    tail,
                    212,
                    221,
                )
            )

        else:
            sector_large_code = (
                self._text(
                    tail,
                    3,
                    7,
                )
            )

            sector_middle_code = (
                self._text(
                    tail,
                    7,
                    11,
                )
            )

            sector_small_code = (
                self._text(
                    tail,
                    11,
                    15,
                )
            )

            etp_code = self._text(
                tail,
                18,
                19,
            )

            spac_yn = self._text(
                tail,
                24,
                25,
            )

            preferred_code = (
                self._text(
                    tail,
                    153,
                    154,
                )
            )

            market_cap = (
                self._int(
                    tail,
                    206,
                    215,
                )
            )

        return {
            "stock_code":
                code,

            "stock_name":
                name,

            "market":
                market,

            "market_cap_100m":
                market_cap,

            "sector_large_code":
                sector_large_code,

            "sector_middle_code":
                sector_middle_code,

            "sector_small_code":
                sector_small_code,

            "preferred_code":
                preferred_code,

            "etp_code":
                etp_code,

            "spac_yn":
                spac_yn,
        }

    @staticmethod
    def _is_common_stock(
        row: dict,
    ) -> bool:
        preferred = str(
            row.get(
                "preferred_code",
                "",
            )
        ).strip()

        if preferred not in {
            "",
            "0",
        }:
            return False

        spac = str(
            row.get(
                "spac_yn",
                "",
            )
        ).strip().upper()

        if spac == "Y":
            return False

        name = str(
            row.get(
                "stock_name",
                "",
            )
        )

        if "스팩" in name:
            return False

        return True

    @staticmethod
    def _text(
        data: bytes,
        start: int,
        end: int,
    ) -> str:
        return (
            data[start:end]
            .decode(
                "ascii",
                errors="ignore",
            )
            .strip()
        )

    @staticmethod
    def _int(
        data: bytes,
        start: int,
        end: int,
    ) -> int:
        text = (
            data[start:end]
            .decode(
                "ascii",
                errors="ignore",
            )
            .replace(",", "")
            .strip()
        )

        if not text:
            return 0

        try:
            return int(text)
        except ValueError:
            return 0


kis_master_client = KisMasterClient()