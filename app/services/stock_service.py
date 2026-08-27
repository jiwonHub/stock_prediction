import json
from datetime import date, datetime, timedelta

from sqlalchemy.orm import Session

from app.clients.kis_client import dart_client, kis_client
from app.core.config import settings
from app.core.exceptions import ConfigurationError, ExternalApiError
from app.repositories.stock_repository import StockRepository
from app.schemas.ranking import RankingResponse
from app.schemas.stock import ChartPointResponse, StockResponse
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

    async def sync_current_price(
        self,
        stock_code: str,
    ) -> StockResponse:
        output = await kis_client.get_current_price(
            stock_code
        )

        market_cap_raw = to_float(
            output.get("hts_avls"),
            default=0.0,
        )

        # KIS hts_avls는 억원 단위이므로 원 단위로 저장.
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
                str(
                    output.get(
                        "rprs_mrkt_kor_name",
                        "",
                    )
                ).strip()
                or None
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

        return self._stock_to_response(
            stock
        )
    
    async def sync_current_prices(
        self,
        stock_codes: list[str],
    ) -> tuple[int, int]:
        success = 0
        skipped = 0
        total = len(stock_codes)

        for index, stock_code in enumerate(stock_codes, start=1):
            print(
                f"[PRICE] {index}/{total} {stock_code} 현재가 동기화",
                flush=True,
            )

            try:
                await self.sync_current_price(stock_code)
                success += 1
            except (
                ValueError,
                ConfigurationError,
                ExternalApiError,
            ) as e:
                self.db.rollback()
                skipped += 1

                print(
                    f"[PRICE] {index}/{total} "
                    f"{stock_code} 건너뜀: {e}",
                    flush=True,
                )

        return success, skipped


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
    ) -> list[RankingResponse]:
        candidates = self.repository.get_phase4_ranking_candidates(
            horizon_days=5,
            scan_limit=max(200, limit * 5),
        )

        scored = []

        for stock, metric, prediction in candidates:
            financial_score = float(metric.financial_score) if metric else 0.0
            ml_score = float(prediction.ml_score) if prediction else 0.0

            if metric is not None and prediction is not None:
                total_score = financial_score * 0.45 + ml_score * 0.55
            elif prediction is not None:
                total_score = ml_score
            else:
                total_score = financial_score

            scored.append(
                (
                    total_score,
                    stock,
                    metric,
                    prediction,
                )
            )

        scored.sort(
            key=lambda item: (
                item[0],
                float(item[1].market_cap or 0.0),
            ),
            reverse=True,
        )

        result: list[RankingResponse] = []
        seen_codes: set[str] = set()

        for total_score, stock, metric, prediction in scored:
            if stock.code in seen_codes:
                continue

            seen_codes.add(stock.code)
            rank = len(result) + 1
            financial_score = float(metric.financial_score) if metric else 0.0

            result.append(
                RankingResponse(
                    rank=rank,
                    stockCode=stock.code,
                    stockName=stock.name,
                    currentPrice=float(stock.current_price or 0.0),
                    changeRate=float(stock.change_rate or 0.0),
                    totalScore=round(float(total_score), 2),
                    predictedReturn=(
                        float(prediction.predicted_return) if prediction else 0.0
                    ),
                    upsideProbability=(
                        float(prediction.upside_probability) if prediction else 0.0
                    ),
                    financialScore=financial_score,
                    growthScore=float(metric.growth_score) if metric else 0.0,
                    profitabilityScore=(
                        float(metric.profitability_score) if metric else 0.0
                    ),
                    stabilityScore=float(metric.stability_score) if metric else 0.0,
                    cashFlowScore=float(metric.cash_flow_score) if metric else 0.0,
                )
            )

            if len(result) >= limit:
                break

        return result

