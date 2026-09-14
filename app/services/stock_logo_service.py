import asyncio
import json
import re
from dataclasses import dataclass
from html.parser import HTMLParser
from time import monotonic
from urllib.parse import urljoin, urlparse

import httpx
from sqlalchemy.orm import Session

from app.clients.dart_client import DartClient
from app.repositories.stock_repository import StockRepository


@dataclass(frozen=True)
class LogoAsset:
    body: bytes
    content_type: str
    source: str
    asset_url: str


@dataclass(frozen=True)
class _AssetCandidate:
    url: str
    score: int
    source: str


@dataclass(frozen=True)
class _PageCandidate:
    url: str
    score: int


class _OfficialPageParser(HTMLParser):
    def __init__(
        self,
        *,
        page_url: str,
        company_names: tuple[str, ...],
        page_bonus: int = 0,
    ):
        super().__init__(
            convert_charrefs=True,
        )

        self.page_url = page_url

        self.company_names = tuple(
            name.strip().lower()
            for name in company_names
            if name and name.strip()
        )

        self.page_bonus = page_bonus

        self.assets: list[
            _AssetCandidate
        ] = []

        self.pages: list[
            _PageCandidate
        ] = []

        self.manifests: list[
            str
        ] = []

        self._anchor_href: (
            str | None
        ) = None

        self._anchor_text: list[
            str
        ] = []

        self._json_ld_active = False

        self._json_ld: list[
            str
        ] = []

    @staticmethod
    def _attrs(
        attrs: list[
            tuple[
                str,
                str | None,
            ]
        ],
    ) -> dict[
        str,
        str,
    ]:
        return {
            key.lower(): value or ""
            for key, value in attrs
        }

    @staticmethod
    def _size_score(
        sizes: str,
    ) -> int:
        best = 0

        for match in re.finditer(
            r"(\d{1,4})\s*[xX]\s*(\d{1,4})",
            sizes,
        ):
            best = max(
                best,
                min(
                    int(
                        match.group(1)
                    ),
                    int(
                        match.group(2)
                    ),
                ),
            )

        if best >= 512:
            return 100

        if best >= 256:
            return 80

        if best >= 180:
            return 70

        if best >= 128:
            return 60

        if best >= 96:
            return 45

        if best >= 64:
            return 30

        if best >= 32:
            return 10

        return 0

    def _company_bonus(
        self,
        text: str,
    ) -> int:
        compact_text = re.sub(
            r"[^0-9a-z가-힣]",
            "",
            text.lower(),
        )

        for name in (
            self.company_names
        ):
            compact_name = re.sub(
                r"[^0-9a-z가-힣]",
                "",
                name,
            )

            if (
                len(
                    compact_name
                )
                >= 3
                and compact_name
                in compact_text
            ):
                return 120

        return 0

    def _add_asset(
        self,
        raw_url: str,
        score: int,
        source: str,
    ) -> None:
        raw_url = raw_url.strip()

        if (
            not raw_url
            or raw_url.startswith(
                (
                    "data:",
                    "javascript:",
                )
            )
        ):
            return

        resolved = urljoin(
            self.page_url,
            raw_url,
        )

        if (
            urlparse(
                resolved
            ).scheme
            not in {
                "http",
                "https",
            }
        ):
            return

        self.assets.append(
            _AssetCandidate(
                url=resolved,
                score=(
                    score
                    + self.page_bonus
                ),
                source=source,
            )
        )

    def _add_page(
        self,
        raw_url: str,
        text: str,
    ) -> None:
        raw_url = raw_url.strip()

        if not raw_url:
            return

        resolved = (
            urljoin(
                self.page_url,
                raw_url,
            )
            .split(
                "#",
                1,
            )[0]
        )

        current_host = (
            urlparse(
                self.page_url
            ).hostname
            or ""
        ).lower()

        target_host = (
            urlparse(
                resolved
            ).hostname
            or ""
        ).lower()

        if (
            not current_host
            or not target_host
        ):
            return

        if not (
            target_host
            == current_host
            or target_host.endswith(
                f".{current_host}"
            )
            or current_host.endswith(
                f".{target_host}"
            )
        ):
            return

        hint = (
            f"{raw_url} {text}"
            .lower()
        )

        score = 0

        if any(
            marker in hint
            for marker in (
                "brand",
                "identity",
                "logo",
                "브랜드",
                "로고",
                "ci 소개",
                "ci소개",
            )
        ):
            score += 500

        if re.search(
            r"(^|[/_.\-\s])ci([/_.\-\s]|$)",
            hint,
        ):
            score += 500

        if any(
            marker in hint
            for marker in (
                "about-us",
                "about_us",
                "/about/",
                "company",
                "intro",
                "회사소개",
                "회사개요",
                "기업소개",
            )
        ):
            score += 250

        if score > 0:
            self.pages.append(
                _PageCandidate(
                    url=resolved,
                    score=score,
                )
            )

    def handle_starttag(
        self,
        tag: str,
        attrs: list[
            tuple[
                str,
                str | None,
            ]
        ],
    ) -> None:
        tag = tag.lower()

        values = self._attrs(
            attrs
        )

        if tag == "link":
            rel = (
                values.get(
                    "rel",
                    "",
                )
                .lower()
            )

            href = values.get(
                "href",
                "",
            )

            size_score = (
                self._size_score(
                    values.get(
                        "sizes",
                        "",
                    )
                )
            )

            if (
                "manifest"
                in rel
                and href
            ):
                self.manifests.append(
                    urljoin(
                        self.page_url,
                        href,
                    )
                )

            if (
                "apple-touch-icon"
                in rel
            ):
                self._add_asset(
                    href,
                    (
                        1000
                        + size_score
                    ),
                    "apple-touch-icon",
                )

            elif "icon" in rel:
                self._add_asset(
                    href,
                    (
                        650
                        + size_score
                    ),
                    "html-icon",
                )

        elif tag == "meta":
            key = (
                values.get(
                    "property"
                )
                or values.get(
                    "name"
                )
                or values.get(
                    "itemprop"
                )
                or ""
            ).lower()

            content = (
                values.get(
                    "content",
                    "",
                )
            )

            if "logo" in key:
                self._add_asset(
                    content,
                    920,
                    "meta-logo",
                )

            elif key in {
                "og:image",
                "twitter:image",
            }:
                self._add_asset(
                    content,
                    250,
                    "social-image",
                )

        elif tag == "img":
            src = (
                values.get(
                    "src"
                )
                or values.get(
                    "data-src"
                )
                or values.get(
                    "data-original"
                )
                or ""
            )

            hint = " ".join(
                (
                    src,
                    values.get(
                        "alt",
                        "",
                    ),
                    values.get(
                        "class",
                        "",
                    ),
                    values.get(
                        "id",
                        "",
                    ),
                    values.get(
                        "title",
                        "",
                    ),
                    values.get(
                        "itemprop",
                        "",
                    ),
                )
            ).lower()

            score = (
                120
                + self._company_bonus(
                    hint
                )
            )

            if "logo" in hint:
                score += 800

            if re.search(
                r"(^|[/_.\-\s])ci([/_.\-\s]|$)",
                hint,
            ):
                score += 650

            if any(
                marker in hint
                for marker in (
                    "brand",
                    "identity",
                    "symbol",
                    "emblem",
                    "wordmark",
                    "bi_",
                    "bi-",
                )
            ):
                score += 600

            if any(
                word in hint
                for word in (
                    "banner",
                    "visual",
                    "hero",
                    "background",
                    "main_visual",
                    "slide",
                    "promotion",
                    "event",
                )
            ):
                score -= 500

            self._add_asset(
                src,
                score,
                "html-image",
            )

        elif tag == "a":
            self._anchor_href = (
                values.get(
                    "href",
                    "",
                )
            )

            self._anchor_text = []

        elif (
            tag == "script"
            and "application/ld+json"
            in values.get(
                "type",
                "",
            ).lower()
        ):
            self._json_ld_active = True
            self._json_ld = []

    def handle_data(
        self,
        data: str,
    ) -> None:
        if (
            self._anchor_href
            is not None
        ):
            self._anchor_text.append(
                data
            )

        if self._json_ld_active:
            self._json_ld.append(
                data
            )

    def handle_endtag(
        self,
        tag: str,
    ) -> None:
        tag = tag.lower()

        if (
            tag == "a"
            and self._anchor_href
            is not None
        ):
            self._add_page(
                self._anchor_href,
                " ".join(
                    self._anchor_text
                ),
            )

            self._anchor_href = None
            self._anchor_text = []

        elif (
            tag == "script"
            and self._json_ld_active
        ):
            raw = (
                "".join(
                    self._json_ld
                )
                .strip()
            )

            self._json_ld_active = False
            self._json_ld = []

            if raw:
                self._parse_json_ld(
                    raw
                )

    def _parse_json_ld(
        self,
        raw: str,
    ) -> None:
        try:
            payload = json.loads(
                raw
            )

        except json.JSONDecodeError:
            return

        def visit(
            value,
        ) -> None:
            if isinstance(
                value,
                dict,
            ):
                for (
                    key,
                    item,
                ) in value.items():
                    if (
                        str(
                            key
                        ).lower()
                        == "logo"
                    ):
                        if isinstance(
                            item,
                            str,
                        ):
                            self._add_asset(
                                item,
                                950,
                                "json-ld-logo",
                            )

                        elif isinstance(
                            item,
                            dict,
                        ):
                            for url_key in (
                                "url",
                                "contentUrl",
                                "content_url",
                            ):
                                raw_url = (
                                    item.get(
                                        url_key
                                    )
                                )

                                if isinstance(
                                    raw_url,
                                    str,
                                ):
                                    self._add_asset(
                                        raw_url,
                                        950,
                                        "json-ld-logo",
                                    )

                    visit(
                        item
                    )

            elif isinstance(
                value,
                list,
            ):
                for item in value:
                    visit(
                        item
                    )

        visit(
            payload
        )


class StockLogoService:
    CACHE_TTL_SECONDS = (
        60
        * 60
    )

    MAX_IMAGE_BYTES = (
        3
        * 1024
        * 1024
    )

    MAX_DISCOVERY_PAGES = 5

    _resolve_semaphore = (
        asyncio.Semaphore(
            4
        )
    )

    _logo_cache: dict[
        str,
        tuple[
            float,
            LogoAsset | None,
        ],
    ] = {}

    _company_cache: dict[
        str,
        tuple[
            float,
            dict,
        ],
    ] = {}

    def __init__(
        self,
        db: Session,
    ):
        self.repository = (
            StockRepository(
                db
            )
        )

        self.dart_client = (
            DartClient()
        )

    @staticmethod
    def _normalize_homepage(
        value: str,
    ) -> str | None:
        homepage = (
            value.strip()
        )

        if not homepage:
            return None

        if not homepage.startswith(
            (
                "http://",
                "https://",
            )
        ):
            homepage = (
                f"https://{homepage}"
            )

        if not urlparse(
            homepage
        ).hostname:
            return None

        return homepage

    async def _get_company_info(
        self,
        *,
        stock_code: str,
        corp_code: str,
    ) -> dict:
        cached = (
            self._company_cache
            .get(
                stock_code
            )
        )

        if cached is not None:
            (
                cached_at,
                payload,
            ) = cached

            if (
                monotonic()
                - cached_at
                < self.CACHE_TTL_SECONDS
            ):
                return payload

        payload = (
            await self.dart_client
            .fetch_company_info(
                corp_code=(
                    corp_code
                ),
            )
        )

        self._company_cache[
            stock_code
        ] = (
            monotonic(),
            payload,
        )

        return payload

    async def _fetch_html(
        self,
        *,
        client: httpx.AsyncClient,
        url: str,
    ) -> tuple[
        str,
        str,
    ] | None:
        try:
            response = (
                await client.get(
                    url,
                    headers={
                        "User-Agent": (
                            "Mozilla/5.0 "
                            "(Macintosh; Intel Mac OS X 10_15_7) "
                            "AppleWebKit/537.36 "
                            "(KHTML, like Gecko) "
                            "Chrome/152.0.0.0 "
                            "Safari/537.36"
                        ),
                        "Accept": (
                            "text/html,"
                            "application/xhtml+xml,"
                            "application/xml;q=0.9,"
                            "*/*;q=0.8"
                        ),
                        "Accept-Language": (
                            "ko-KR,ko;q=0.9,"
                            "en-US;q=0.8,"
                            "en;q=0.7"
                        ),
                    },
                )
            )

            response.raise_for_status()

        except httpx.HTTPError:
            return None

        content_type = (
            response.headers.get(
                "content-type",
                "",
            )
            .split(
                ";",
                1,
            )[0]
            .strip()
            .lower()
        )

        if content_type not in {
            "text/html",
            "application/xhtml+xml",
        }:
            return None

        return (
            response.text,
            str(
                response.url
            ),
        )

    async def _manifest_assets(
        self,
        *,
        client: httpx.AsyncClient,
        manifest_url: str,
    ) -> list[
        _AssetCandidate
    ]:
        try:
            response = (
                await client.get(
                    manifest_url
                )
            )

            response.raise_for_status()

            payload = (
                response.json()
            )

        except (
            httpx.HTTPError,
            json.JSONDecodeError,
            ValueError,
        ):
            return []

        result: list[
            _AssetCandidate
        ] = []

        for icon in list(
            payload.get(
                "icons"
            )
            or []
        ):
            if not isinstance(
                icon,
                dict,
            ):
                continue

            src = str(
                icon.get(
                    "src"
                )
                or ""
            ).strip()

            if not src:
                continue

            size_score = (
                _OfficialPageParser
                ._size_score(
                    str(
                        icon.get(
                            "sizes"
                        )
                        or ""
                    )
                )
            )

            purpose_bonus = (
                40
                if "maskable"
                in str(
                    icon.get(
                        "purpose"
                    )
                    or ""
                ).lower()
                else 0
            )

            result.append(
                _AssetCandidate(
                    url=urljoin(
                        manifest_url,
                        src,
                    ),
                    score=(
                        1050
                        + size_score
                        + purpose_bonus
                    ),
                    source=(
                        "web-manifest"
                    ),
                )
            )

        return result

    @staticmethod
    def _dedupe_assets(
        assets: list[
            _AssetCandidate
        ],
    ) -> list[
        _AssetCandidate
    ]:
        best: dict[
            str,
            _AssetCandidate,
        ] = {}

        for asset in assets:
            current = (
                best.get(
                    asset.url
                )
            )

            if (
                current is None
                or asset.score
                > current.score
            ):
                best[
                    asset.url
                ] = asset

        return sorted(
            best.values(),
            key=lambda item: (
                item.score
            ),
            reverse=True,
        )

    @staticmethod
    def _dedupe_pages(
        pages: list[
            _PageCandidate
        ],
    ) -> list[
        _PageCandidate
    ]:
        best: dict[
            str,
            _PageCandidate,
        ] = {}

        for page in pages:
            normalized = (
                page.url.split(
                    "#",
                    1,
                )[0]
            )

            current = (
                best.get(
                    normalized
                )
            )

            if (
                current is None
                or page.score
                > current.score
            ):
                best[
                    normalized
                ] = (
                    _PageCandidate(
                        url=normalized,
                        score=page.score,
                    )
                )

        return sorted(
            best.values(),
            key=lambda item: (
                item.score
            ),
            reverse=True,
        )

    async def _download_candidate(
        self,
        *,
        client: httpx.AsyncClient,
        candidate: _AssetCandidate,
    ) -> LogoAsset | None:
        try:
            response = (
                await client.get(
                    candidate.url,
                    headers={
                        "User-Agent": (
                            "Mozilla/5.0 "
                            "(Macintosh; Intel Mac OS X 10_15_7) "
                            "AppleWebKit/537.36 "
                            "(KHTML, like Gecko) "
                            "Chrome/152.0.0.0 "
                            "Safari/537.36"
                        ),
                        "Accept": (
                            "image/webp,"
                            "image/png,"
                            "image/jpeg,"
                            "image/*;q=0.8,"
                            "*/*;q=0.5"
                        ),
                    },
                )
            )

            response.raise_for_status()

        except httpx.HTTPError:
            return None

        content_type = (
            response.headers.get(
                "content-type",
                "",
            )
            .split(
                ";",
                1,
            )[0]
            .strip()
            .lower()
        )

        if content_type not in {
            "image/png",
            "image/jpeg",
            "image/webp",
        }:
            return None

        body = (
            response.content
        )

        if (
            not body
            or len(
                body
            )
            > self.MAX_IMAGE_BYTES
        ):
            return None

        return LogoAsset(
            body=body,
            content_type=content_type,
            source=(
                candidate.source
            ),
            asset_url=str(
                response.url
            ),
        )

    async def _discover_logo(
        self,
        *,
        stock_code: str,
    ) -> LogoAsset | None:
        stock = (
            self.repository
            .get_stock(
                stock_code
            )
        )

        if stock is None:
            raise ValueError(
                "종목을 찾을 수 없습니다: "
                f"{stock_code}"
            )

        if not stock.corp_code:
            return None

        company = (
            await self._get_company_info(
                stock_code=(
                    stock_code
                ),
                corp_code=(
                    stock.corp_code
                ),
            )
        )

        homepage = (
            self._normalize_homepage(
                str(
                    company.get(
                        "hm_url"
                    )
                    or ""
                )
            )
        )

        if homepage is None:
            return None

        company_names = tuple(
            value
            for value in (
                stock.name,
                stock.english_name or "",
                str(
                    company.get(
                        "corp_name"
                    )
                    or ""
                ),
                str(
                    company.get(
                        "corp_name_eng"
                    )
                    or ""
                ),
            )
            if value
        )

        async with httpx.AsyncClient(
            timeout=8.0,
            follow_redirects=True,
        ) as client:
            homepage_result = (
                await self._fetch_html(
                    client=client,
                    url=homepage,
                )
            )

            if (
                homepage_result
                is None
            ):
                return None

            (
                homepage_html,
                final_homepage,
            ) = homepage_result

            parser = (
                _OfficialPageParser(
                    page_url=(
                        final_homepage
                    ),
                    company_names=(
                        company_names
                    ),
                )
            )

            parser.feed(
                homepage_html
            )

            assets = list(
                parser.assets
            )

            for manifest_url in (
                parser.manifests[:2]
            ):
                assets.extend(
                    await self._manifest_assets(
                        client=client,
                        manifest_url=(
                            manifest_url
                        ),
                    )
                )

            discovery_pages = (
                self._dedupe_pages(
                    parser.pages
                )[
                    : self.MAX_DISCOVERY_PAGES
                ]
            )

            if discovery_pages:
                page_results = (
                    await asyncio.gather(
                        *(
                            self._fetch_html(
                                client=client,
                                url=page.url,
                            )
                            for page
                            in discovery_pages
                        )
                    )
                )

                for (
                    page,
                    page_result,
                ) in zip(
                    discovery_pages,
                    page_results,
                ):
                    if (
                        page_result
                        is None
                    ):
                        continue

                    (
                        page_html,
                        final_page_url,
                    ) = page_result

                    page_parser = (
                        _OfficialPageParser(
                            page_url=(
                                final_page_url
                            ),
                            company_names=(
                                company_names
                            ),
                            page_bonus=min(
                                300,
                                (
                                    page.score
                                    // 2
                                ),
                            ),
                        )
                    )

                    page_parser.feed(
                        page_html
                    )

                    assets.extend(
                        page_parser.assets
                    )

                    for manifest_url in (
                        page_parser
                        .manifests[:1]
                    ):
                        assets.extend(
                            await self._manifest_assets(
                                client=client,
                                manifest_url=(
                                    manifest_url
                                ),
                            )
                        )

            for candidate in (
                self._dedupe_assets(
                    assets
                )[:20]
            ):
                result = (
                    await self._download_candidate(
                        client=client,
                        candidate=(
                            candidate
                        ),
                    )
                )

                if result is None:
                    continue

                print(
                    "[LOGO][OFFICIAL] "
                    f"{stock_code} "
                    f"source={result.source} "
                    f"asset={result.asset_url}",
                    flush=True,
                )

                return result

        return None

    async def get_logo(
        self,
        *,
        stock_code: str,
    ) -> LogoAsset | None:
        code = (
            stock_code.strip()
        )

        if not code:
            raise ValueError(
                "종목코드가 비어 있습니다."
            )

        cached = (
            self._logo_cache
            .get(
                code
            )
        )

        if cached is not None:
            (
                cached_at,
                asset,
            ) = cached

            if (
                monotonic()
                - cached_at
                < self.CACHE_TTL_SECONDS
            ):
                return asset

        async with (
            self._resolve_semaphore
        ):
            cached = (
                self._logo_cache
                .get(
                    code
                )
            )

            if cached is not None:
                (
                    cached_at,
                    asset,
                ) = cached

                if (
                    monotonic()
                    - cached_at
                    < self.CACHE_TTL_SECONDS
                ):
                    return asset

            asset = (
                await self._discover_logo(
                    stock_code=(
                        code
                    )
                )
            )

            self._logo_cache[
                code
            ] = (
                monotonic(),
                asset,
            )

            return asset