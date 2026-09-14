import json
from datetime import date, datetime, timedelta
from pathlib import Path

from sqlalchemy.orm import Session

from app.clients.kis_client import dart_client, kis_client
from app.clients.toss_client import toss_client
from app.core.config import settings
from app.core.exceptions import ConfigurationError, ExternalApiError
from app.repositories.stock_repository import StockRepository
from app.schemas.ranking import RankingResponse
from app.schemas.stock import ChartPointResponse, ChartPageResponse, StockResponse
from app.utils.numbers import to_decimal_or_none, to_float, to_int


class StockService:
    def __init__(
        self,
        db: Session,
    ):
        self.db = db
        self.repository = StockRepository(
            db
        )

    @staticmethod
    def _normalize_market_name(
        value: object,
    ) -> str | None:
        text = str(
            value or ""
        ).strip()

        if not text:
            return None

        upper = (
            text
            .upper()
            .replace(" ", "")
        )

        if (
            "KOSDAQ" in upper
            or upper.startswith("KSQ")
            or "코스닥" in text
        ):
            return "KOSDAQ"

        if (
            "KOSPI" in upper
            or "코스피" in text
            or "유가증권" in text
            or text == "거래소"
        ):
            return "KOSPI"

        if (
            "KONEX" in upper
            or "코넥스" in text
        ):
            return "KONEX"

        return text

    @staticmethod
    def _apply_kis_sign(
        value: float,
        sign_code: str,
    ) -> float:
        if sign_code in {"1", "2"}:
            return abs(value)

        if sign_code == "3":
            return 0.0

        if sign_code in {"4", "5"}:
            return -abs(value)

        return value

    @staticmethod
    def _stock_to_response(stock) -> StockResponse:
        return StockResponse(
            code=stock.code,
            name=stock.name,
            market=stock.market or "KRX",
            currentPrice=float(
                stock.current_price or 0.0
            ),
            change=float(
                stock.change or 0.0
            ),
            changeRate=float(
                stock.change_rate or 0.0
            ),
            marketCap=float(
                stock.market_cap or 0.0
            ),
        )

    async def search_stocks(
        self,
        query: str,
    ) -> list[StockResponse]:
        stocks = self.repository.search(
            query
        )

        return [
            self._stock_to_response(stock)
            for stock in stocks
        ]

    async def get_stock(
        self,
        stock_code: str,
        *,
        refresh: bool = False,
    ) -> StockResponse:
        stock = self.repository.get_stock(
            stock_code
        )

        if stock is None:
            raise ValueError(
                f"등록되지 않은 종목입니다: {stock_code}. "
                "먼저 /v1/admin/sync/stocks를 실행하세요."
            )

        should_refresh = refresh

        if stock.last_price_at is not None:
            age = (
                datetime.utcnow()
                - stock.last_price_at
            ).total_seconds()

            if age < settings.price_cache_seconds:
                should_refresh = False

        if should_refresh:
            await self.sync_current_price(
                stock_code
            )
            stock = self.repository.get_stock(
                stock_code
            )

        return self._stock_to_response(
            stock
        )

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

        return date.today()

    async def _sync_current_price_from_toss(
        self,
        stock_code: str,
    ) -> StockResponse:
        output = await toss_client.get_current_price(
            stock_code
        )

        current_price = to_float(
            output.get(
                "lastPrice"
            )
        )

        if current_price <= 0.0:
            raise ExternalApiError(
                "Toss 현재가가 올바르지 않습니다: "
                f"{stock_code}"
            )

        trade_date = self._quote_trade_date(
            output
        )

        previous_close = (
            self.repository
            .get_latest_close_before(
                stock_code=stock_code,
                trade_date=trade_date,
            )
        )

        if (
            previous_close is None
            or previous_close <= 0.0
        ):
            raise ExternalApiError(
                "직전 거래일 종가가 없습니다: "
                f"{stock_code}"
            )

        change = (
            current_price
            - previous_close
        )

        change_rate = (
            change
            / previous_close
            * 100.0
        )

        stock = (
            self.repository
            .update_realtime_price(
                stock_code=stock_code,
                current_price=current_price,
                change=change,
                change_rate=change_rate,
            )
        )

        print(
            "[PRICE][TOSS] "
            f"{stock_code} "
            f"{current_price:.0f} "
            f"{change_rate:+.2f}%",
            flush=True,
        )

        return self._stock_to_response(
            stock
        )

    async def _sync_current_price_from_kis(
        self,
        stock_code: str,
        *,
        market_override: str | None = None,
    ) -> StockResponse:
        output = await kis_client.get_current_price(
            stock_code
        )

        market_cap_raw = to_float(
            output.get("hts_avls"),
            default=0.0,
        )

        market_cap = (
            market_cap_raw * 100_000_000.0
            if market_cap_raw > 0
            else None
        )

        sign_code = str(
            output.get(
                "prdy_vrss_sign",
                "",
            )
        )

        change = self._apply_kis_sign(
            to_float(
                output.get("prdy_vrss")
            ),
            sign_code,
        )

        change_rate = self._apply_kis_sign(
            to_float(
                output.get("prdy_ctrt")
            ),
            sign_code,
        )

        stock = self.repository.update_quote(
            stock_code=stock_code,
            current_price=to_float(
                output.get("stck_prpr")
            ),
            change=change,
            change_rate=change_rate,
            market_cap=market_cap,
            per=to_float(
                output.get("per"),
                default=0.0,
            )
            or None,
            pbr=to_float(
                output.get("pbr"),
                default=0.0,
            )
            or None,
            eps=to_float(
                output.get("eps"),
                default=0.0,
            )
            or None,
            bps=to_float(
                output.get("bps"),
                default=0.0,
            )
            or None,
            market=(
                market_override
                or self._normalize_market_name(
                    output.get(
                        "rprs_mrkt_kor_name"
                    )
                )
            ),
            sector_name=(
                str(
                    output.get(
                        "bstp_kor_isnm",
                        "",
                    )
                ).strip()
                or None
            ),
        )

        print(
            "[PRICE][KIS-FALLBACK] "
            f"{stock_code}",
            flush=True,
        )

        return self._stock_to_response(
            stock
        )

    async def sync_current_price(
        self,
        stock_code: str,
        *,
        market_override: str | None = None,
    ) -> StockResponse:
        try:
            return await self._sync_current_price_from_toss(
                stock_code
            )

        except (
            ConfigurationError,
            ExternalApiError,
        ) as toss_error:
            self.db.rollback()

            print(
                "[PRICE][TOSS-FAIL] "
                f"{stock_code}: {toss_error}",
                flush=True,
            )

            return await self._sync_current_price_from_kis(
                stock_code,
                market_override=market_override,
            )
    
    async def sync_current_prices(
        self,
        stock_codes: list[str],
    ) -> tuple[int, int]:
        symbols = list(
            dict.fromkeys(
                str(stock_code).strip()
                for stock_code in stock_codes
                if str(stock_code).strip()
            )
        )

        if not symbols:
            return 0, 0

        success_codes: set[str] = set()
        fallback_codes: list[str] = []

        try:
            outputs: list[dict] = []

            batch_size = 200

            for start in range(
                0,
                len(symbols),
                batch_size,
            ):
                batch_symbols = symbols[
                    start:
                    start + batch_size
                ]

                batch_outputs = await (
                    toss_client.get_prices(
                        batch_symbols
                    )
                )

                outputs.extend(
                    batch_outputs
                )

            output_by_code = {
                str(
                    output.get(
                        "symbol",
                        "",
                    )
                ): output
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

            for stock_code in symbols:
                output = output_by_code.get(
                    stock_code
                )

                if output is None:
                    fallback_codes.append(
                        stock_code
                    )
                    continue

                current_price = to_float(
                    output.get(
                        "lastPrice"
                    )
                )

                if current_price <= 0.0:
                    fallback_codes.append(
                        stock_code
                    )
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

            update_rows: list[dict] = []

            for stock_code, current_price in (
                current_prices.items()
            ):
                previous_close = (
                    previous_closes.get(
                        stock_code
                    )
                )

                if (
                    previous_close is None
                    or previous_close <= 0.0
                ):
                    fallback_codes.append(
                        stock_code
                    )
                    continue

                change = (
                    current_price
                    - previous_close
                )

                change_rate = (
                    change
                    / previous_close
                    * 100.0
                )

                update_rows.append(
                    {
                        "stock_code": (
                            stock_code
                        ),
                        "current_price": (
                            current_price
                        ),
                        "change": change,
                        "change_rate": (
                            change_rate
                        ),
                    }
                )

            toss_updated_codes = (
                self.repository
                .update_realtime_prices(
                    update_rows
                )
            )

            success_codes.update(
                toss_updated_codes
            )

            requested_toss_codes = {
                str(row["stock_code"])
                for row in update_rows
            }

            fallback_codes.extend(
                sorted(
                    requested_toss_codes
                    - toss_updated_codes
                )
            )

            print(
                "[PRICE][TOSS-BATCH] "
                f"requested={len(symbols)} "
                f"updated={len(toss_updated_codes)} "
                f"fallback={len(set(fallback_codes))}",
                flush=True,
            )

        except (
            ConfigurationError,
            ExternalApiError,
        ) as toss_error:
            self.db.rollback()

            fallback_codes = list(
                symbols
            )

            print(
                "[PRICE][TOSS-BATCH-FAIL] "
                f"{toss_error}",
                flush=True,
            )

        fallback_codes = list(
            dict.fromkeys(
                stock_code
                for stock_code in fallback_codes
                if stock_code
                not in success_codes
            )
        )

        skipped = 0
        fallback_total = len(
            fallback_codes
        )

        for index, stock_code in enumerate(
            fallback_codes,
            start=1,
        ):
            print(
                "[PRICE][KIS-FALLBACK] "
                f"{index}/{fallback_total} "
                f"{stock_code}",
                flush=True,
            )

            try:
                await self._sync_current_price_from_kis(
                    stock_code
                )

                success_codes.add(
                    stock_code
                )

            except (
                ValueError,
                ConfigurationError,
                ExternalApiError,
            ) as e:
                self.db.rollback()
                skipped += 1

                print(
                    "[PRICE][SKIP] "
                    f"{stock_code}: {e}",
                    flush=True,
                )

        return len(success_codes), skipped


    async def sync_ranked_current_prices(
        self,
        *,
        limit: int = 100,
    ) -> tuple[int, int]:
        rankings = await self.get_rankings(
            limit=limit,
        )

        stock_codes = [
            item.stockCode
            for item in rankings
        ]

        return await self.sync_current_prices(
            stock_codes
        )

    async def sync_stocks_from_dart(
        self,
    ) -> int:
        corp_rows = await dart_client.fetch_corp_codes()

        rows = []

        for item in corp_rows:
            modify_date_text = item.get(
                "modify_date",
                "",
            )

            modified_at = None

            if modify_date_text:
                try:
                    modified_at = datetime.strptime(
                        modify_date_text,
                        "%Y%m%d",
                    )
                except ValueError:
                    modified_at = None

            rows.append(
                {
                    "code": item["stock_code"],
                    "corp_code": item["corp_code"],
                    "name": item["corp_name"],
                    "english_name": (
                        item.get("corp_eng_name")
                        or None
                    ),
                    "market": "KRX",
                    "modified_at": modified_at,
                }
            )

        return self.repository.upsert_stocks(
            rows
        )

    @staticmethod
    def _parse_toss_daily_candle(
        row: dict,
    ) -> dict | None:
        timestamp = str(
            row.get(
                "timestamp",
                "",
            )
        ).strip()

        if not timestamp:
            return None

        try:
            trade_date = (
                datetime.fromisoformat(
                    timestamp
                ).date()
            )
        except ValueError:
            return None

        close = to_float(
            row.get(
                "closePrice"
            )
        )

        if close <= 0.0:
            return None

        return {
            "trade_date": trade_date,
            "open": to_float(
                row.get(
                    "openPrice"
                )
            ),
            "high": to_float(
                row.get(
                    "highPrice"
                )
            ),
            "low": to_float(
                row.get(
                    "lowPrice"
                )
            ),
            "close": close,
            "volume": to_int(
                row.get(
                    "volume"
                )
            ),
        }

    async def backfill_daily_prices_from_toss(
        self,
        stock_code: str,
        *,
        max_pages: int = 500,
    ) -> int:
        if self.repository.get_stock(
            stock_code
        ) is None:
            raise ValueError(
                f"등록되지 않은 종목입니다: {stock_code}"
            )

        candles = (
            await toss_client
            .get_daily_candle_history(
                stock_code,
                adjusted=True,
                max_pages=max_pages,
            )
        )

        parsed_rows: list[dict] = []

        for candle in candles:
            parsed = (
                self._parse_toss_daily_candle(
                    candle
                )
            )

            if parsed is not None:
                parsed_rows.append(
                    parsed
                )

        if not parsed_rows:
            raise ExternalApiError(
                "Toss 일봉 데이터를 "
                "저장할 수 없습니다: "
                f"{stock_code}"
            )

        inserted = (
            self.repository
            .insert_missing_daily_prices(
                stock_code=stock_code,
                rows=parsed_rows,
            )
        )

        print(
            "[PRICE][TOSS-BACKFILL] "
            f"{stock_code} "
            f"received={len(parsed_rows)} "
            f"inserted={inserted} "
            f"oldest={parsed_rows[0]['trade_date']} "
            f"latest={parsed_rows[-1]['trade_date']}",
            flush=True,
        )

        return inserted

    async def _load_or_create_toss_backfill_manifest(
        self,
    ) -> list[dict]:
        manifest_path = Path(
            ".cache/toss_backfill_top100.json"
        )

        if manifest_path.exists():
            try:
                payload = json.loads(
                    manifest_path.read_text(
                        encoding="utf-8"
                    )
                )

                stocks = payload.get(
                    "stocks"
                )

                if (
                    isinstance(stocks, list)
                    and stocks
                ):
                    print(
                        "[PRICE][TOSS-BACKFILL-MANIFEST] "
                        f"loaded={len(stocks)}",
                        flush=True,
                    )

                    return stocks

            except (
                OSError,
                json.JSONDecodeError,
            ):
                pass

        rankings = await self.get_rankings(
            limit=100,
        )

        stocks = [
            {
                "rank": ranking.rank,
                "stockCode": ranking.stockCode,
                "stockName": ranking.stockName,
            }
            for ranking in rankings
        ]

        if not stocks:
            raise ExternalApiError(
                "Toss Backfill 대상 TOP100을 "
                "생성할 수 없습니다."
            )

        manifest_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        temp_path = manifest_path.with_suffix(
            ".tmp"
        )

        temp_path.write_text(
            json.dumps(
                {
                    "createdAt": (
                        datetime.utcnow()
                        .isoformat()
                    ),
                    "stocks": stocks,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

        temp_path.replace(
            manifest_path
        )

        print(
            "[PRICE][TOSS-BACKFILL-MANIFEST] "
            f"created={len(stocks)} "
            f"path={manifest_path}",
            flush=True,
        )

        return stocks

    async def backfill_ranked_daily_prices_from_toss(
        self,
        *,
        offset: int = 0,
        limit: int = 10,
        max_pages: int = 500,
    ) -> dict:
        if offset < 0:
            raise ValueError(
                "offset은 0 이상이어야 합니다."
            )

        if limit < 1 or limit > 100:
            raise ValueError(
                "limit은 1~100 범위여야 합니다."
            )

        manifest = (
            await self
            ._load_or_create_toss_backfill_manifest()
        )

        selected = manifest[
            offset:
            offset + limit
        ]

        results: list[dict] = []

        success = 0
        failed = 0
        inserted_total = 0

        total = len(
            selected
        )

        for index, ranking in enumerate(
            selected,
            start=1,
        ):
            rank = int(
                ranking["rank"]
            )

            stock_code = str(
                ranking["stockCode"]
            )

            stock_name = str(
                ranking["stockName"]
            )

            print(
                "[PRICE][TOSS-BACKFILL-BATCH] "
                f"{index}/{total} "
                f"rank={rank} "
                f"code={stock_code} "
                f"name={stock_name}",
                flush=True,
            )

            try:
                inserted = (
                    await self
                    .backfill_daily_prices_from_toss(
                        stock_code,
                        max_pages=max_pages,
                    )
                )

                success += 1
                inserted_total += inserted

                results.append(
                    {
                        "rank": rank,
                        "stockCode": stock_code,
                        "stockName": stock_name,
                        "status": "success",
                        "inserted": inserted,
                    }
                )

            except Exception as e:
                self.db.rollback()

                failed += 1

                results.append(
                    {
                        "rank": rank,
                        "stockCode": stock_code,
                        "stockName": stock_name,
                        "status": "failed",
                        "inserted": 0,
                        "error": (
                            f"{type(e).__name__}: "
                            f"{e}"
                        ),
                    }
                )

                print(
                    "[PRICE][TOSS-BACKFILL-FAIL] "
                    f"{stock_code} "
                    f"{type(e).__name__}: {e}",
                    flush=True,
                )

        print(
            "[PRICE][TOSS-BACKFILL-SUMMARY] "
            f"offset={offset} "
            f"requested={total} "
            f"success={success} "
            f"failed={failed} "
            f"inserted={inserted_total}",
            flush=True,
        )

        return {
            "offset": offset,
            "requested": total,
            "success": success,
            "failed": failed,
            "inserted": inserted_total,
            "nextOffset": (
                offset + total
            ),
            "hasMore": (
                offset + total
                < len(manifest)
            ),
            "results": results,
        }

    async def sync_daily_prices(
        self,
        stock_code: str,
        *,
        days: int = 365,
    ) -> int:
        if self.repository.get_stock(stock_code) is None:
            raise ValueError(
                f"등록되지 않은 종목입니다: {stock_code}"
            )

        end_date = date.today()
        start_date = (
            end_date
            - timedelta(
                days=max(
                    7,
                    days,
                )
            )
        )

        rows = await kis_client.get_daily_prices(
            stock_code,
            start_date=start_date,
            end_date=end_date,
        )

        parsed_rows = []

        for row in rows:
            date_text = str(
                row.get(
                    "stck_bsop_date",
                    "",
                )
            )

            if not date_text:
                continue

            try:
                trade_date = datetime.strptime(
                    date_text,
                    "%Y%m%d",
                ).date()
            except ValueError:
                continue

            parsed_rows.append(
                {
                    "trade_date": trade_date,
                    "open": to_float(
                        row.get("stck_oprc")
                    ),
                    "high": to_float(
                        row.get("stck_hgpr")
                    ),
                    "low": to_float(
                        row.get("stck_lwpr")
                    ),
                    "close": to_float(
                        row.get("stck_clpr")
                    ),
                    "volume": to_int(
                        row.get("acml_vol")
                    ),
                }
            )

        return self.repository.upsert_daily_prices(
            stock_code=stock_code,
            rows=parsed_rows,
        )
    
    def get_chart_page(
        self,
        stock_code: str,
        *,
        interval: str = "1d",
        before: date | None = None,
        count: int = 200,
    ) -> ChartPageResponse:
        if count < 1 or count > 200:
            raise ValueError(
                "count는 1~200 범위여야 합니다."
            )

        if interval not in {
            "1d",
            "1w",
            "1mo",
            "1y",
        }:
            raise ValueError(
                f"지원하지 않는 interval입니다: "
                f"{interval}"
            )

        if self.repository.get_stock(
            stock_code
        ) is None:
            raise ValueError(
                f"등록되지 않은 종목입니다: "
                f"{stock_code}"
            )

        if interval == "1d":
            rows, has_more = (
                self.repository
                .get_daily_price_page(
                    stock_code=stock_code,
                    before=before,
                    limit=count,
                )
            )

            candles = [
                ChartPointResponse(
                    time=datetime.combine(
                        row.trade_date,
                        datetime.min.time(),
                    ).isoformat(),
                    price=float(
                        row.close
                    ),
                    open=float(
                        row.open
                    ),
                    high=float(
                        row.high
                    ),
                    low=float(
                        row.low
                    ),
                    close=float(
                        row.close
                    ),
                    volume=int(
                        row.volume
                        or 0
                    ),
                )
                for row in rows
            ]

            oldest_date = (
                rows[0].trade_date
                if rows
                else None
            )

        else:
            rows, has_more = (
                self.repository
                .get_aggregated_price_page(
                    stock_code=stock_code,
                    interval=interval,
                    before=before,
                    limit=count,
                )
            )

            candles = [
                ChartPointResponse(
                    time=datetime.combine(
                        row["bucket_start"],
                        datetime.min.time(),
                    ).isoformat(),
                    price=float(
                        row["close"]
                    ),
                    open=float(
                        row["open"]
                    ),
                    high=float(
                        row["high"]
                    ),
                    low=float(
                        row["low"]
                    ),
                    close=float(
                        row["close"]
                    ),
                    volume=int(
                        row["volume"]
                        or 0
                    ),
                )
                for row in rows
            ]

            oldest_date = (
                rows[0]["bucket_start"]
                if rows
                else None
            )

        next_before = None

        if (
            has_more
            and oldest_date is not None
        ):
            next_before = (
                oldest_date.isoformat()
            )

        return ChartPageResponse(
            candles=candles,
            nextBefore=next_before,
            hasMore=has_more,
        )
        if count < 1 or count > 200:
            raise ValueError(
                "count는 1~200 범위여야 합니다."
            )

        if self.repository.get_stock(
            stock_code
        ) is None:
            raise ValueError(
                f"등록되지 않은 종목입니다: "
                f"{stock_code}"
            )

        rows, has_more = (
            self.repository
            .get_daily_price_page(
                stock_code=stock_code,
                before=before,
                limit=count,
            )
        )

        candles = [
            ChartPointResponse(
                time=datetime.combine(
                    row.trade_date,
                    datetime.min.time(),
                ).isoformat(),
                price=float(
                    row.close
                ),
                open=float(
                    row.open
                ),
                high=float(
                    row.high
                ),
                low=float(
                    row.low
                ),
                close=float(
                    row.close
                ),
                volume=int(
                    row.volume
                    or 0
                ),
            )
            for row in rows
        ]

        next_before = None

        if (
            has_more
            and rows
        ):
            next_before = (
                rows[0]
                .trade_date
                .isoformat()
            )

        return ChartPageResponse(
            candles=candles,
            nextBefore=next_before,
            hasMore=has_more,
        )

    async def get_chart(
        self,
        stock_code: str,
        *,
        period: str,
        refresh: bool = False,
    ) -> list[ChartPointResponse]:
        if self.repository.get_stock(stock_code) is None:
            raise ValueError(
                f"등록되지 않은 종목입니다: {stock_code}"
            )

        normalized = period.upper()

        if normalized == "1D":
            rows = await kis_client.get_intraday_prices(
                stock_code
            )

            today = date.today()

            result = []

            for row in rows:
                time_text = str(
                    row.get(
                        "stck_cntg_hour",
                        "",
                    )
                )

                if len(time_text) != 6:
                    continue

                try:
                    dt = datetime.combine(
                        today,
                        datetime.strptime(
                            time_text,
                            "%H%M%S",
                        ).time(),
                    )
                except ValueError:
                    continue

                result.append(
                    ChartPointResponse(
                        time=dt.isoformat(),
                        price=to_float(
                            row.get("stck_prpr")
                        ),
                    )
                )

            return result

        days_map = {
            "1W": 14,
            "1M": 45,
            "3M": 120,
            "1Y": 400,
        }

        days = days_map.get(
            normalized,
            45,
        )

        if refresh:
            await self.sync_daily_prices(
                stock_code,
                days=days,
            )

        start_date = (
            date.today()
            - timedelta(
                days=days,
            )
        )

        rows = self.repository.get_daily_prices(
            stock_code=stock_code,
            start_date=start_date,
        )

        return [
            ChartPointResponse(
                time=datetime.combine(
                    row.trade_date,
                    datetime.min.time(),
                ).isoformat(),
                price=float(row.close),
                open=float(row.open),
                high=float(row.high),
                low=float(row.low),
                close=float(row.close),
                volume=int(row.volume or 0),
            )
            for row in rows
        ]

    async def sync_financials(
        self,
        stock_code: str,
        *,
        business_year: str,
        report_code: str,
    ) -> tuple[int, str]:
        stock = self.repository.get_stock(
            stock_code
        )

        if stock is None:
            raise ValueError(
                f"등록되지 않은 종목입니다: {stock_code}"
            )

        if not stock.corp_code:
            raise ValueError(
                f"DART corp_code가 없는 종목입니다: {stock_code}"
            )

        fs_div = "CFS"

        rows = await dart_client.fetch_financial_statements(
            corp_code=stock.corp_code,
            business_year=business_year,
            report_code=report_code,
            fs_div=fs_div,
        )

        if not rows:
            fs_div = "OFS"

            rows = await dart_client.fetch_financial_statements(
                corp_code=stock.corp_code,
                business_year=business_year,
                report_code=report_code,
                fs_div=fs_div,
            )

        parsed_rows = []

        for row in rows:
            parsed_rows.append(
                {
                    "stock_code": stock_code,
                    "corp_code": stock.corp_code,
                    "business_year": business_year,
                    "report_code": report_code,
                    "fs_div": str(
                        row.get(
                            "fs_div",
                            fs_div,
                        )
                    ),
                    "fs_nm": row.get("fs_nm"),
                    "sj_div": str(
                        row.get(
                            "sj_div",
                            "",
                        )
                    ),
                    "sj_nm": row.get("sj_nm"),
                    "account_id": str(
                        row.get(
                            "account_id",
                            "",
                        )
                    ),
                    "account_nm": str(
                        row.get(
                            "account_nm",
                            "",
                        )
                    ),
                    "account_detail": str(
                        row.get(
                            "account_detail",
                            "",
                        )
                    ),
                    "currency": row.get("currency"),
                    "thstrm_amount": to_decimal_or_none(
                        row.get("thstrm_amount")
                    ),
                    "frmtrm_amount": to_decimal_or_none(
                        row.get("frmtrm_amount")
                    ),
                    "bfefrmtrm_amount": to_decimal_or_none(
                        row.get("bfefrmtrm_amount")
                    ),
                    "thstrm_nm": row.get("thstrm_nm"),
                    "frmtrm_nm": row.get("frmtrm_nm"),
                    "bfefrmtrm_nm": row.get("bfefrmtrm_nm"),
                    "raw_json": json.dumps(
                        row,
                        ensure_ascii=False,
                    ),
                }
            )

        count = self.repository.replace_financials(
            stock_code=stock_code,
            business_year=business_year,
            report_code=report_code,
            fs_div=fs_div,
            rows=parsed_rows,
        )

        return count, fs_div

    async def get_rankings(
        self,
        *,
        limit: int,
        force_recompute: bool = False,
    ) -> list[RankingResponse]:
        if not force_recompute:
            from app.services.market_context_service import (
                MarketContextService,
            )

            (
                snapshot,
                snapshot_items,
            ) = (
                self.repository
                .get_latest_ranking_snapshot_items(
                    ranking_version=(
                        MarketContextService
                        .RANKING_VERSION
                    ),
                    horizon_days=(
                        MarketContextService
                        .PRIMARY_HORIZON_DAYS
                    ),
                    universe="KRX",
                    limit=limit,
                )
            )

            required_component_keys = {
                "quality",
                "growth",
                "value",
                "financial_health",
                "relative_strength",
                "flow",
                "ml",
                "data_coverage",
            }

            snapshot_has_required_components = (
                snapshot is not None
                and len(snapshot_items)
                >= MarketContextService.MIN_SNAPSHOT_RANKS
                and all(
                    (
                        required_component_keys
                        .issubset(
                            (
                                item
                                .score_components_json
                                or {}
                            ).keys()
                        )
                        and (
                            item
                            .score_components_json
                            or {}
                        ).get(
                            "quality"
                        ) is not None
                        and (
                            item
                            .score_components_json
                            or {}
                        ).get(
                            "value"
                        ) is not None
                    )
                    for item
                    in snapshot_items
                )
            )
            snapshot_quality_count = sum(
                1
                for item in snapshot_items
                if (
                    item.score_components_json
                    or {}
                ).get(
                    "quality"
                ) is not None
            )

            snapshot_value_count = sum(
                1
                for item in snapshot_items
                if (
                    item.score_components_json
                    or {}
                ).get(
                    "value"
                ) is not None
            )

            snapshot_samples = [
                {
                    "stock": item.stock_code,
                    "quality": (
                        item.score_components_json
                        or {}
                    ).get(
                        "quality"
                    ),
                    "value": (
                        item.score_components_json
                        or {}
                    ).get(
                        "value"
                    ),
                    "total": float(
                        item.total_score
                    ),
                }
                for item in snapshot_items[:5]
            ]

            print(
                "[RANKING][SNAPSHOT-CHECK] "
                f"version={MarketContextService.RANKING_VERSION} "
                f"date={snapshot.as_of_date if snapshot is not None else None} "
                f"items={len(snapshot_items)} "
                f"keysValid={snapshot_has_required_components} "
                f"quality={snapshot_quality_count} "
                f"value={snapshot_value_count} "
                f"samples={snapshot_samples}",
                flush=True,
            )

            if snapshot_has_required_components:
                stock_codes = [
                    item.stock_code
                    for item in snapshot_items
                ]

                contexts = (
                    self.repository
                    .get_ranking_context_by_stock_codes(
                        stock_codes
                    )
                )

                context_map = {
                    stock.code: (
                        stock,
                        metric,
                    )
                    for (
                        stock,
                        metric,
                    )
                    in contexts
                }

                cached_result: list[
                    RankingResponse
                ] = []

                for item in snapshot_items:
                    context = (
                        context_map.get(
                            item.stock_code
                        )
                    )

                    if context is None:
                        continue

                    (
                        stock,
                        metric,
                    ) = context

                    components = (
                        item.score_components_json
                        or {}
                    )

                    cached_result.append(
                        RankingResponse(
                            rank=int(
                                item.rank
                            ),
                            stockCode=(
                                stock.code
                            ),
                            stockName=(
                                stock.name
                            ),

                            sectorName=(
                                components.get(
                                    "sector_name"
                                )
                                or stock.sector_name
                            ),

                            sectorRank=(
                                int(
                                    components[
                                        "sector_rank"
                                    ]
                                )
                                if components.get(
                                    "sector_rank"
                                ) is not None
                                else None
                            ),

                            sectorPeerCount=int(
                                components.get(
                                    "sector_peer_count"
                                )
                                or 0
                            ),

                            currentPrice=float(
                                stock.current_price
                                or 0.0
                            ),
                            changeRate=float(
                                stock.change_rate
                                or 0.0
                            ),
                            totalScore=round(
                                float(
                                    item.total_score
                                ),
                                2,
                            ),
                            predictedReturn=float(
                                item.predicted_return
                                or 0.0
                            ),
                            upsideProbability=float(
                                item.upside_probability
                                or 0.0
                            ),
                            financialScore=float(
                                metric.financial_score
                                if metric is not None
                                else item.financial_score
                                or 0.0
                            ),
                            growthScore=float(
                                components.get(
                                    "growth"
                                )
                                or 0.0
                            ),
                            profitabilityScore=float(
                                metric.profitability_score
                                if metric is not None
                                else 0.0
                            ),
                            stabilityScore=float(
                                metric.stability_score
                                if metric is not None
                                else 0.0
                            ),
                            cashFlowScore=float(
                                metric.cash_flow_score
                                if metric is not None
                                else 0.0
                            ),
                            qualityScore=(
                                float(
                                    components[
                                        "quality"
                                    ]
                                )
                                if components.get(
                                    "quality"
                                ) is not None
                                else None
                            ),
                            valueScore=(
                                float(
                                    components[
                                        "value"
                                    ]
                                )
                                if components.get(
                                    "value"
                                ) is not None
                                else None
                            ),
                            financialHealthScore=(
                                float(
                                    components[
                                        "financial_health"
                                    ]
                                )
                                if components.get(
                                    "financial_health"
                                ) is not None
                                else None
                            ),
                            relativeStrengthScore=(
                                float(
                                    components[
                                        "relative_strength"
                                    ]
                                )
                                if components.get(
                                    "relative_strength"
                                ) is not None
                                else None
                            ),
                            momentumScore=(
                                float(
                                    components[
                                        "momentum"
                                    ]
                                )
                                if components.get(
                                    "momentum"
                                ) is not None
                                else None
                            ),
                            flowScore=(
                                float(
                                    components[
                                        "flow"
                                    ]
                                )
                                if components.get(
                                    "flow"
                                ) is not None
                                else None
                            ),
                            riskScore=(
                                float(
                                    components[
                                        "risk"
                                    ]
                                )
                                if components.get(
                                    "risk"
                                ) is not None
                                else None
                            ),
                            mlScore=float(
                                components.get(
                                    "ml"
                                )
                                or 0.0
                            ),
                            dataCoverage=float(
                                components.get(
                                    "data_coverage"
                                )
                                or 0.0
                            ),
                        )
                    )

                if len(
                    cached_result
                ) >= MarketContextService.MIN_SNAPSHOT_RANKS:
                    print(
                        "[RANKING][API][SNAPSHOT] "
                        f"date={snapshot.as_of_date} "
                        f"count={len(cached_result)} "
                        f"quality={sum(1 for row in cached_result if row.qualityScore is not None)} "
                        f"value={sum(1 for row in cached_result if row.valueScore is not None)} "
                        f"samples={[{'stock': row.stockCode, 'quality': row.qualityScore, 'value': row.valueScore, 'total': row.totalScore} for row in cached_result[:5]]}",
                        flush=True,
                    )

                    return (
                        cached_result[
                            :limit
                        ]
                    )

            print(
                "[RANKING][SNAPSHOT-MISS] "
                "일반 조회에서는 랭킹을 재계산하지 않습니다. "
                "DailyPipeline Snapshot을 기다립니다.",
                flush=True,
            )

            return []

        print(
            "[RANKING][RECOMPUTE] "
            f"force={force_recompute}",
            flush=True,
        )

        from app.services.composite_ranking_service import (
            CompositeRankingService,
        )

        composite_result = (
            CompositeRankingService(
                self.db
            )
            .score_current_universe(
                limit=100,
            )
        )

        composite_rows = (
            composite_result[
                "scores"
            ]
        )

        stock_codes = [
            row[
                "stock_code"
            ]
            for row
            in composite_rows
        ]

        contexts = (
            self.repository
            .get_ranking_context_by_stock_codes(
                stock_codes
            )
        )

        context_map = {
            stock.code: (
                stock,
                metric,
            )
            for (
                stock,
                metric,
            )
            in contexts
        }

        result: list[
            RankingResponse
        ] = []

        for composite_row in composite_rows:
            stock_code = (
                composite_row[
                    "stock_code"
                ]
            )

            context = (
                context_map.get(
                    stock_code
                )
            )

            if context is None:
                continue

            (
                stock,
                metric,
            ) = context

            financial_score = float(
                composite_row[
                    "financial_score"
                ]
            )

            result.append(
                RankingResponse(
                    rank=int(
                        composite_row[
                            "rank"
                        ]
                    ),
                    stockCode=(
                        stock.code
                    ),
                    stockName=(
                        stock.name
                    ),

                    sectorName=(
                        composite_row.get(
                            "sector_name"
                        )
                        or stock.sector_name
                    ),

                    sectorRank=(
                        int(
                            composite_row[
                                "sector_rank"
                            ]
                        )
                        if composite_row.get(
                            "sector_rank"
                        ) is not None
                        else None
                    ),

                    sectorPeerCount=int(
                        composite_row.get(
                            "sector_peer_count"
                        )
                        or 0
                    ),

                    currentPrice=float(
                        stock.current_price
                        or 0.0
                    ),
                    changeRate=float(
                        stock.change_rate
                        or 0.0
                    ),
                    totalScore=round(
                        float(
                            composite_row[
                                "total_score"
                            ]
                        ),
                        2,
                    ),
                    predictedReturn=0.0,
                    upsideProbability=0.0,
                    financialScore=(
                        financial_score
                    ),
                    growthScore=float(
                        composite_row.get(
                            "growth_score"
                        )
                        or 0.0
                    ),
                    profitabilityScore=(
                        float(
                            metric
                            .profitability_score
                        )
                        if metric
                        else 0.0
                    ),
                    stabilityScore=(
                        float(
                            metric
                            .stability_score
                        )
                        if metric
                        else 0.0
                    ),
                    cashFlowScore=(
                        float(
                            metric
                            .cash_flow_score
                        )
                        if metric
                        else 0.0
                    ),
                    qualityScore=(
                        float(
                            composite_row[
                                "quality_score"
                            ]
                        )
                        if composite_row.get(
                            "quality_score"
                        ) is not None
                        else None
                    ),
                    valueScore=(
                        float(
                            composite_row[
                                "value_score"
                            ]
                        )
                        if composite_row.get(
                            "value_score"
                        ) is not None
                        else None
                    ),
                    financialHealthScore=(
                        float(
                            composite_row[
                                "financial_health_score"
                            ]
                        )
                        if composite_row.get(
                            "financial_health_score"
                        ) is not None
                        else None
                    ),
                    relativeStrengthScore=(
                        float(
                            composite_row[
                                "relative_strength_score"
                            ]
                        )
                        if composite_row.get(
                            "relative_strength_score"
                        ) is not None
                        else None
                    ),
                    momentumScore=(
                        float(
                            composite_row[
                                "momentum_score"
                            ]
                        )
                        if composite_row.get(
                            "momentum_score"
                        ) is not None
                        else None
                    ),
                    flowScore=(
                        float(
                            composite_row[
                                "flow_score"
                            ]
                        )
                        if composite_row.get(
                            "flow_score"
                        ) is not None
                        else None
                    ),
                    riskScore=(
                        float(
                            composite_row[
                                "risk_score"
                            ]
                        )
                        if composite_row.get(
                            "risk_score"
                        ) is not None
                        else None
                    ),
                    mlScore=float(
                        composite_row.get(
                            "ml_score"
                        )
                        or 0.0
                    ),
                    dataCoverage=float(
                        composite_row.get(
                            "data_coverage"
                        )
                        or 0.0
                    ),
                )
            )

            if (
                len(
                    result
                )
                >= limit
            ):
                break

        return result