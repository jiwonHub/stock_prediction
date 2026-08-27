import io
import zipfile
from xml.etree import ElementTree

import httpx

from app.core.config import settings
from app.core.exceptions import ConfigurationError, ExternalApiError


class DartClient:
    BASE_URL = "https://opendart.fss.or.kr/api"

    def _require_key(self) -> None:
        if not settings.dart_api_key:
            raise ConfigurationError(
                "DART_API_KEY가 비어 있습니다. .env에 설정하세요."
            )

    async def fetch_corp_codes(self) -> list[dict]:
        self._require_key()

        async with httpx.AsyncClient(
            timeout=60.0,
            follow_redirects=True,
        ) as client:
            response = await client.get(
                f"{self.BASE_URL}/corpCode.xml",
                params={
                    "crtfc_key": settings.dart_api_key,
                },
            )

        response.raise_for_status()

        try:
            archive = zipfile.ZipFile(
                io.BytesIO(response.content)
            )
        except zipfile.BadZipFile as e:
            raise ExternalApiError(
                "OpenDART corpCode 응답이 ZIP 파일이 아닙니다. "
                "API Key와 응답 내용을 확인하세요."
            ) from e

        xml_names = [
            name
            for name in archive.namelist()
            if name.lower().endswith(".xml")
        ]

        if not xml_names:
            raise ExternalApiError(
                "OpenDART corpCode ZIP 내부 XML을 찾지 못했습니다."
            )

        xml_bytes = archive.read(xml_names[0])
        root = ElementTree.fromstring(xml_bytes)

        result: list[dict] = []

        for item in root.findall("list"):
            stock_code = (item.findtext("stock_code") or "").strip()

            if not stock_code:
                continue

            result.append(
                {
                    "corp_code": (item.findtext("corp_code") or "").strip(),
                    "corp_name": (item.findtext("corp_name") or "").strip(),
                    "corp_eng_name": (
                        item.findtext("corp_eng_name") or ""
                    ).strip(),
                    "stock_code": stock_code,
                    "modify_date": (
                        item.findtext("modify_date") or ""
                    ).strip(),
                }
            )

        return result

    async def fetch_financial_statements(
        self,
        *,
        corp_code: str,
        business_year: str,
        report_code: str,
        fs_div: str = "CFS",
    ) -> list[dict]:
        self._require_key()

        params = {
            "crtfc_key": settings.dart_api_key,
            "corp_code": corp_code,
            "bsns_year": business_year,
            "reprt_code": report_code,
            "fs_div": fs_div,
        }

        async with httpx.AsyncClient(
            timeout=45.0,
        ) as client:
            response = await client.get(
                f"{self.BASE_URL}/fnlttSinglAcntAll.json",
                params=params,
            )

        response.raise_for_status()

        payload = response.json()
        status = str(payload.get("status", ""))

        if status == "000":
            return list(payload.get("list") or [])

        if status == "013":
            return []

        raise ExternalApiError(
            f"OpenDART 재무제표 조회 실패: "
            f"{payload.get('message')} ({status})"
        )
    
    async def fetch_disclosures(
            self,
            *,
            corp_code: str | None = None,
            begin_date=None,
            end_date=None,
            page_count: int = 100,
        ) -> list[dict]:
            self._require_key()

            params = {
                "crtfc_key": settings.dart_api_key,
                "page_no": 1,
                "page_count": max(
                    1,
                    min(100, page_count),
                ),
            }

            if corp_code:
                params["corp_code"] = corp_code

            if begin_date is not None:
                params["bgn_de"] = (
                    begin_date.strftime("%Y%m%d")
                )

            if end_date is not None:
                params["end_de"] = (
                    end_date.strftime("%Y%m%d")
                )

            async with httpx.AsyncClient(
                timeout=30.0,
            ) as client:
                response = await client.get(
                    f"{self.BASE_URL}/list.json",
                    params=params,
                )

            response.raise_for_status()

            payload = response.json()
            status = str(
                payload.get("status", "")
            )

            if status == "000":
                return list(
                    payload.get("list") or []
                )

            if status == "013":
                return []

            raise ExternalApiError(
                "OpenDART 공시 조회 실패: "
                f"{payload.get('message')} "
                f"({status})"
            )
