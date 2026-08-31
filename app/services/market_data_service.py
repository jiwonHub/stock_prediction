from collections import defaultdict
from datetime import date, datetime, timedelta
from statistics import median

from sqlalchemy.orm import Session
from app.clients.kis_master_client import (
    kis_master_client,
)
from app.clients.ecos_client import ecos_client
from app.clients.kis_client import kis_client
from app.core.exceptions import (
    ConfigurationError,
    ExternalApiError,
)
from app.repositories.market_data_repository import (
    MarketDataRepository,
)
from app.repositories.stock_repository import (
    StockRepository,
)
from app.services.stock_service import StockService
from app.utils.numbers import (
    to_decimal_or_none,
    to_float,
    to_int,
)


class MarketDataService:
    def __init__(
        self,
        db: Session,
    ):
        self.db = db

        self.repository = (
            MarketDataRepository(db)
        )

        self.stock_repository = (
            StockRepository(db)
        )

    async def sync_market_context(
        self,
        *,
        days: int = 365,
    ) -> tuple[int, int]:
        index_count = (
            await self.sync_market_indices(
                days=days,
            )
        )

        macro_count = (
            await self.sync_macro_indicators(
                days=days,
            )
        )

        return (
            index_count,
            macro_count,
        )

    async def sync_market_indices(
        self,
        *,
        days: int = 365,
    ) -> int:
        end_date = date.today()

        start_date = (
            end_date
            - timedelta(
                days=max(
                    30,
                    days,
                )
            )
        )

        index_definitions = [
            (
                "0001",
                "KOSPI",
                "KOSPI",
            ),
            (
                "1001",
                "KOSDAQ",
                "KOSDAQ",
            ),
        ]

        total_count = 0

        for (
            index_code,
            index_name,
            market,
        ) in index_definitions:
            raw_rows = (
                await kis_client
                .get_index_daily_prices(
                    index_code,
                    start_date=start_date,
                    end_date=end_date,
                )
            )

            parsed_rows: list[
                dict
            ] = []

            for raw in raw_rows:
                trade_date = (
                    self._parse_yyyymmdd(
                        raw.get(
                            "stck_bsop_date"
                        )
                    )
                )

                if trade_date is None:
                    continue

                close = to_float(
                    raw.get(
                        "bstp_nmix_prpr"
                    )
                )

                if close <= 0:
                    continue

                parsed_rows.append(
                    {
                        "index_code":
                            index_code,
                        "index_name":
                            index_name,
                        "market":
                            market,
                        "trade_date":
                            trade_date,
                        "open":
                            self._optional_float(
                                raw.get(
                                    "bstp_nmix_oprc"
                                )
                            ),
                        "high":
                            self._optional_float(
                                raw.get(
                                    "bstp_nmix_hgpr"
                                )
                            ),
                        "low":
                            self._optional_float(
                                raw.get(
                                    "bstp_nmix_lwpr"
                                )
                            ),
                        "close":
                            close,
                        "change":
                            None,
                        "change_rate":
                            None,
                        "volume":
                            to_int(
                                raw.get(
                                    "acml_vol"
                                )
                            ),
                        "trading_value":
                            to_decimal_or_none(
                                raw.get(
                                    "acml_tr_pbmn"
                                )
                            ),
                        "source":
                            "KIS",
                        "raw_json":
                            dict(raw),
                    }
                )

            parsed_rows.sort(
                key=lambda row: (
                    row["trade_date"]
                )
            )

            self._apply_index_changes(
                parsed_rows
            )

            total_count += (
                self.repository
                .upsert_market_index_prices(
                    parsed_rows
                )
            )

        return total_count

    async def sync_macro_indicators(
        self,
        *,
        days: int = 365,
    ) -> int:
        end_date = date.today()

        start_date = (
            end_date
            - timedelta(
                days=max(
                    60,
                    days,
                )
            )
        )

        datasets = [
            (
                "BOK_BASE_RATE",
                "한국은행 기준금리",
                "interest_rate",
                await ecos_client
                .get_base_rate(
                    start_date=start_date,
                    end_date=end_date,
                ),
            ),
            (
                "USD_KRW",
                "원/달러 환율",
                "fx",
                await ecos_client
                .get_usd_krw(
                    start_date=start_date,
                    end_date=end_date,
                ),
            ),
            (
                "KTB_3Y",
                "국고채 3년",
                "bond_yield",
                await ecos_client
                .get_ktb_3y(
                    start_date=start_date,
                    end_date=end_date,
                ),
            ),
            (
                "KTB_10Y",
                "국고채 10년",
                "bond_yield",
                await ecos_client
                .get_ktb_10y(
                    start_date=start_date,
                    end_date=end_date,
                ),
            ),
        ]

        parsed_rows: list[
            dict
        ] = []

        for (
            indicator_code,
            indicator_name,
            indicator_type,
            raw_rows,
        ) in datasets:
            for raw in raw_rows:
                observed_date = (
                    self._parse_ecos_date(
                        raw.get(
                            "TIME"
                        )
                    )
                )

                value = (
                    self._optional_float(
                        raw.get(
                            "DATA_VALUE"
                        )
                    )
                )

                if (
                    observed_date is None
                    or value is None
                ):
                    continue

                parsed_rows.append(
                    {
                        "indicator_code":
                            indicator_code,
                        "indicator_name":
                            indicator_name,
                        "indicator_type":
                            indicator_type,
                        "frequency":
                            "D",
                        "observed_date":
                            observed_date,
                        "value":
                            value,
                        "unit":
                            (
                                str(
                                    raw.get(
                                        "UNIT_NAME",
                                        "",
                                    )
                                ).strip()
                                or None
                            ),
                        "source":
                            "ECOS",
                        "metadata_json":
                            dict(raw),
                    }
                )

        return (
            self.repository
            .upsert_macro_indicators(
                parsed_rows
            )
        )

    async def sync_stock_context(
        self,
        stock_code: str,
    ) -> tuple[int, int]:
        stock = (
            self.stock_repository
            .get_stock(
                stock_code
            )
        )

        if stock is None:
            raise ValueError(
                "등록되지 않은 종목입니다: "
                f"{stock_code}"
            )

        investor_count = (
            await self
            .sync_stock_investor_flow(
                stock_code
            )
        )

        valuation_count = (
            await self
            .sync_stock_valuation(
                stock_code
            )
        )

        return (
            investor_count,
            valuation_count,
        )

    async def sync_ranked_stock_context(
            self,
            *,
            limit: int = 100,
        ) -> tuple[
            int,
            int,
            int,
            int,
            int,
        ]:
            # DB에 이미 현재가가 있는 종목이 아니라
            # KIS 실시간 시가총액 상위에서
            # Universe를 직접 만든다.
            raw_universe = (
                await kis_master_client
                .get_market_cap_universe(
                    limit=max(
                        limit * 2,
                        200,
                    )
                )
            )

            universe: list[
                tuple[str, str]
            ] = []

            for raw in raw_universe:
                stock_code = str(
                    raw.get(
                        "stock_code",
                        "",
                    )
                ).strip()

                market = str(
                    raw.get(
                        "market",
                        "",
                    )
                ).strip()

                if not stock_code:
                    continue

                stock = (
                    self.stock_repository
                    .get_stock(
                        stock_code
                    )
                )

                if (
                    stock is None
                    or not stock.is_active
                ):
                    continue

                universe.append(
                    (
                        stock_code,
                        market,
                    )
                )

                if len(universe) >= limit:
                    break

            stock_codes = [
                stock_code
                for stock_code, _
                in universe
            ]

            deleted = (
                self.repository
                .delete_valuation_snapshots_not_in(
                    snapshot_date=date.today(),
                    stock_codes=stock_codes,
                )
            )

            print(
                "[MARKET] "
                f"현재 Universe {len(universe)}종목, "
                f"기존 당일 snapshot "
                f"{deleted}건 정리",
                flush=True,
            )

            success_stocks = 0
            investor_count = 0
            valuation_count = 0
            investor_failures = 0
            valuation_failures = 0

            total = len(universe)

            for (
                index,
                (
                    stock_code,
                    market,
                ),
            ) in enumerate(
                universe,
                start=1,
            ):
                stock = (
                    self.stock_repository
                    .get_stock(
                        stock_code
                    )
                )

                stock_name = (
                    stock.name
                    if stock is not None
                    else stock_code
                )

                print(
                    "[MARKET] "
                    f"{index}/{total} "
                    f"{stock_code} "
                    f"{stock_name}",
                    flush=True,
                )

                investor_ok = False
                valuation_ok = False

                try:
                    count = (
                        await self
                        .sync_stock_investor_flow(
                            stock_code
                        )
                    )

                    investor_count += (
                        count
                    )

                    investor_ok = True

                    print(
                        "[MARKET] "
                        f"{stock_code} "
                        "수급 "
                        f"{count}건",
                        flush=True,
                    )

                except (
                    ValueError,
                    ConfigurationError,
                    ExternalApiError,
                ) as e:
                    self.db.rollback()

                    investor_failures += 1

                    print(
                        "[MARKET] "
                        f"{stock_code} "
                        "수급 실패: "
                        f"{e}",
                        flush=True,
                    )

                try:
                    count = (
                        await self
                        .sync_stock_valuation(
                            stock_code,
                            market_override=market,
                        )
                    )

                    valuation_count += (
                        count
                    )

                    valuation_ok = True

                    print(
                        "[MARKET] "
                        f"{stock_code} "
                        "밸류에이션 "
                        f"{count}건",
                        flush=True,
                    )

                except (
                    ValueError,
                    ConfigurationError,
                    ExternalApiError,
                ) as e:
                    self.db.rollback()

                    valuation_failures += 1

                    print(
                        "[MARKET] "
                        f"{stock_code} "
                        "밸류에이션 실패: "
                        f"{e}",
                        flush=True,
                    )

                if (
                    investor_ok
                    and valuation_ok
                ):
                    success_stocks += 1

            return (
                success_stocks,
                investor_count,
                valuation_count,
                investor_failures,
                valuation_failures,
            )

    async def sync_stock_investor_flow(
        self,
        stock_code: str,
    ) -> int:
        raw_rows = (
            await kis_client
            .get_investor_flows(
                stock_code
            )
        )

        parsed_rows: list[
            dict
        ] = []

        for raw in raw_rows:
            trade_date = (
                self._parse_yyyymmdd(
                    raw.get(
                        "stck_bsop_date"
                    )
                )
            )

            if trade_date is None:
                continue

            parsed_rows.append(
                {
                    "stock_code":
                        stock_code,
                    "trade_date":
                        trade_date,

                    "foreign_net_buy_amount":
                        to_decimal_or_none(
                            raw.get(
                                "frgn_ntby_tr_pbmn"
                            )
                        ),

                    "institution_net_buy_amount":
                        to_decimal_or_none(
                            raw.get(
                                "orgn_ntby_tr_pbmn"
                            )
                        ),

                    "individual_net_buy_amount":
                        to_decimal_or_none(
                            raw.get(
                                "prsn_ntby_tr_pbmn"
                            )
                        ),

                    "foreign_net_buy_volume":
                        to_int(
                            raw.get(
                                "frgn_ntby_qty"
                            )
                        ),

                    "institution_net_buy_volume":
                        to_int(
                            raw.get(
                                "orgn_ntby_qty"
                            )
                        ),

                    "individual_net_buy_volume":
                        to_int(
                            raw.get(
                                "prsn_ntby_qty"
                            )
                        ),

                    # 현재 투자자 API에서는
                    # 보유비율을 직접 쓰지 않는다.
                    "foreign_holding_ratio":
                        None,

                    "source":
                        "KIS",

                    "raw_json":
                        dict(raw),
                }
            )

        return (
            self.repository
            .upsert_stock_investor_flows(
                parsed_rows
            )
        )

    async def sync_stock_valuation(
        self,
        stock_code: str,
        *,
        market_override: str | None = None,
    ) -> int:
        # 최신 PER/PBR/EPS/BPS와
        # 현재가를 먼저 갱신
        stock_service = (
            StockService(
                self.db
            )
        )

        await (
            stock_service
            .sync_current_price(
                stock_code,
                market_override=(
                    market_override
                ),
            )
        )

        stock = (
            self.stock_repository
            .get_stock(
                stock_code
            )
        )

        if stock is None:
            raise ValueError(
                "종목 정보를 찾을 수 없습니다: "
                f"{stock_code}"
            )

        row = {
            "stock_code":
                stock_code,
            "snapshot_date":
                date.today(),

            "price":
                (
                    float(
                        stock.current_price
                    )
                    if stock.current_price
                    is not None
                    else None
                ),

            "market_cap":
                (
                    float(
                        stock.market_cap
                    )
                    if stock.market_cap
                    is not None
                    else None
                ),

            "per":
                (
                    float(stock.per)
                    if stock.per
                    is not None
                    else None
                ),

            "pbr":
                (
                    float(stock.pbr)
                    if stock.pbr
                    is not None
                    else None
                ),

            "eps":
                (
                    float(stock.eps)
                    if stock.eps
                    is not None
                    else None
                ),

            "bps":
                (
                    float(stock.bps)
                    if stock.bps
                    is not None
                    else None
                ),

            # 향후 배당 데이터 수집 단계에서
            # 별도로 채운다.
            "dividend_yield":
                None,

            "sector_code":
                stock.sector_code,

            "sector_name":
                stock.sector_name,

            # 업종/시장 평균 valuation은
            # Feature Engine 단계에서 계산
            "sector_per":
                None,

            "sector_pbr":
                None,

            "market_per":
                None,

            "market_pbr":
                None,

            "source":
                "KIS",

            "raw_json":
                {
                    "stock_code":
                        stock.code,
                    "sector_name":
                        stock.sector_name,
                },
        }

        return (
            self.repository
            .upsert_stock_valuation_snapshots(
                [row]
            )
        )
    
    def refresh_valuation_benchmarks(
            self,
            *,
            snapshot_date: date | None = None,
        ) -> tuple[
            int,
            int,
            int,
        ]:
            target_date = (
                snapshot_date
                or date.today()
            )

            rows = (
                self.repository
                .get_valuation_rows_for_date(
                    snapshot_date=target_date
                )
            )

            if not rows:
                return (
                    0,
                    0,
                    0,
                )

            sector_per_values: dict[
                str,
                list[float],
            ] = defaultdict(list)

            sector_pbr_values: dict[
                str,
                list[float],
            ] = defaultdict(list)

            market_per_values: dict[
                str,
                list[float],
            ] = defaultdict(list)

            market_pbr_values: dict[
                str,
                list[float],
            ] = defaultdict(list)

            for (
                valuation,
                market,
            ) in rows:
                sector_key = (
                    valuation.sector_code
                    or valuation.sector_name
                )

                market_key = (
                    str(
                        market or "KRX"
                    )
                    .strip()
                    or "KRX"
                )

                per = (
                    float(
                        valuation.per
                    )
                    if valuation.per
                    is not None
                    else None
                )

                pbr = (
                    float(
                        valuation.pbr
                    )
                    if valuation.pbr
                    is not None
                    else None
                )

                # 적자기업의 음수 PER,
                # 0 또는 비정상 값은
                # benchmark 계산에서 제외.
                if (
                    per is not None
                    and per > 0.0
                ):
                    market_per_values[
                        market_key
                    ].append(per)

                    if sector_key:
                        sector_per_values[
                            sector_key
                        ].append(per)

                if (
                    pbr is not None
                    and pbr > 0.0
                ):
                    market_pbr_values[
                        market_key
                    ].append(pbr)

                    if sector_key:
                        sector_pbr_values[
                            sector_key
                        ].append(pbr)

            sector_per_median = {
                key: float(
                    median(values)
                )
                for key, values
                in sector_per_values.items()
                if values
            }

            sector_pbr_median = {
                key: float(
                    median(values)
                )
                for key, values
                in sector_pbr_values.items()
                if values
            }

            market_per_median = {
                key: float(
                    median(values)
                )
                for key, values
                in market_per_values.items()
                if values
            }

            market_pbr_median = {
                key: float(
                    median(values)
                )
                for key, values
                in market_pbr_values.items()
                if values
            }

            update_rows: list[
                dict
            ] = []

            for (
                valuation,
                market,
            ) in rows:
                sector_key = (
                    valuation.sector_code
                    or valuation.sector_name
                )

                market_key = (
                    str(
                        market or "KRX"
                    )
                    .strip()
                    or "KRX"
                )

                update_rows.append(
                    {
                        "stock_code":
                            valuation.stock_code,

                        "snapshot_date":
                            target_date,

                        "sector_per":
                            (
                                sector_per_median
                                .get(
                                    sector_key
                                )
                                if sector_key
                                else None
                            ),

                        "sector_pbr":
                            (
                                sector_pbr_median
                                .get(
                                    sector_key
                                )
                                if sector_key
                                else None
                            ),

                        "market_per":
                            market_per_median
                            .get(
                                market_key
                            ),

                        "market_pbr":
                            market_pbr_median
                            .get(
                                market_key
                            ),
                    }
                )

            updated = (
                self.repository
                .update_valuation_benchmarks(
                    update_rows
                )
            )

            return (
                updated,
                len(
                    sector_per_median
                ),
                len(
                    market_per_median
                ),
            )

    async def sync_sector_index(
        self,
        *,
        sector_code: str,
        sector_name: str,
        market: str = "KRX",
        days: int = 365,
    ) -> int:
        end_date = date.today()

        start_date = (
            end_date
            - timedelta(
                days=max(
                    1,
                    days,
                )
            )
        )

        raw_rows = (
            await kis_client
            .get_index_daily_prices(
                sector_code,
                start_date=start_date,
                end_date=end_date,
            )
        )

        parsed_rows: list[
            dict
        ] = []

        for raw in raw_rows:
            trade_date = (
                self._parse_yyyymmdd(
                    raw.get(
                        "stck_bsop_date"
                    )
                )
            )

            if trade_date is None:
                continue

            close = to_float(
                raw.get(
                    "bstp_nmix_prpr"
                )
            )

            if close <= 0:
                continue

            parsed_rows.append(
                {
                    "sector_code":
                        sector_code,

                    "sector_name":
                        sector_name,

                    "market":
                        market,

                    "trade_date":
                        trade_date,

                    "open":
                        self._optional_float(
                            raw.get(
                                "bstp_nmix_oprc"
                            )
                        ),

                    "high":
                        self._optional_float(
                            raw.get(
                                "bstp_nmix_hgpr"
                            )
                        ),

                    "low":
                        self._optional_float(
                            raw.get(
                                "bstp_nmix_lwpr"
                            )
                        ),

                    "close":
                        close,

                    "change_rate":
                        None,

                    "volume":
                        to_int(
                            raw.get(
                                "acml_vol"
                            )
                        ),

                    "trading_value":
                        to_decimal_or_none(
                            raw.get(
                                "acml_tr_pbmn"
                            )
                        ),

                    "source":
                        "KIS",

                    "raw_json":
                        dict(raw),
                }
            )

        parsed_rows.sort(
            key=lambda row: (
                row["trade_date"]
            )
        )

        self._apply_sector_changes(
            parsed_rows
        )

        return (
            self.repository
            .upsert_sector_index_prices(
                parsed_rows
            )
        )

    @staticmethod
    def _apply_index_changes(
        rows: list[dict],
    ) -> None:
        for index, row in enumerate(
            rows
        ):
            if index == 0:
                continue

            previous_close = (
                rows[index - 1][
                    "close"
                ]
            )

            if previous_close <= 0:
                continue

            change = (
                row["close"]
                - previous_close
            )

            row["change"] = (
                change
            )

            row["change_rate"] = (
                change
                / previous_close
                * 100.0
            )

    @staticmethod
    def _apply_sector_changes(
        rows: list[dict],
    ) -> None:
        for index, row in enumerate(
            rows
        ):
            if index == 0:
                continue

            previous_close = (
                rows[index - 1][
                    "close"
                ]
            )

            if previous_close <= 0:
                continue

            row["change_rate"] = (
                (
                    row["close"]
                    - previous_close
                )
                / previous_close
                * 100.0
            )

    @staticmethod
    def _parse_yyyymmdd(
        value: object,
    ) -> date | None:
        text = str(
            value or ""
        ).strip()

        if len(text) != 8:
            return None

        try:
            return (
                datetime.strptime(
                    text,
                    "%Y%m%d",
                ).date()
            )
        except ValueError:
            return None

    @staticmethod
    def _parse_ecos_date(
        value: object,
    ) -> date | None:
        text = str(
            value or ""
        ).strip()

        if len(text) == 8:
            try:
                return (
                    datetime.strptime(
                        text,
                        "%Y%m%d",
                    ).date()
                )
            except ValueError:
                return None

        if len(text) == 6:
            try:
                return (
                    datetime.strptime(
                        text,
                        "%Y%m",
                    ).date()
                )
            except ValueError:
                return None

        return None

    @staticmethod
    def _optional_float(
        value: object,
    ) -> float | None:
        if value is None:
            return None

        text = (
            str(value)
            .replace(",", "")
            .strip()
        )

        if text in {
            "",
            "-",
            "None",
            "null",
        }:
            return None

        try:
            return float(text)
        except (
            TypeError,
            ValueError,
        ):
            return None
    
    def get_current_universe_stock_codes(
        self,
        *,
        limit: int = 100,
    ) -> list[str]:
        latest_snapshot_date = (
            self.repository
            .get_latest_valuation_snapshot_date(
                on_or_before=date.today(),
            )
        )

        if latest_snapshot_date is None:
            return []

        rows = (
            self.repository
            .get_valuation_rows_for_date(
                snapshot_date=(
                    latest_snapshot_date
                ),
            )
        )

        stock_codes = [
            snapshot.stock_code
            for snapshot, _market
            in rows
        ]

        return stock_codes[
            :limit
        ]
    
    @staticmethod
    def _close_on_or_before(
        rows: list,
        target_date: date,
    ) -> float | None:
        for row in reversed(rows):
            if row.trade_date > target_date:
                continue

            close = float(row.close)

            if close > 0:
                return close

        return None

    @classmethod
    def _return_between_dates(
        cls,
        rows: list,
        *,
        start_date: date,
        end_date: date,
    ) -> float | None:
        start_close = cls._close_on_or_before(
            rows,
            start_date,
        )

        end_close = cls._close_on_or_before(
            rows,
            end_date,
        )

        if (
            start_close is None
            or end_close is None
            or start_close <= 0
        ):
            return None

        return round(
            (
                end_close
                / start_close
                - 1.0
            )
            * 100.0,
            4,
        )

    def calculate_stock_relative_strength(
        self,
        stock_code: str,
    ) -> dict:
        stock = (
            self.stock_repository
            .get_stock(stock_code)
        )

        if stock is None:
            raise ValueError(
                "등록되지 않은 종목입니다: "
                f"{stock_code}"
            )

        market = str(
            stock.market or ""
        ).strip()

        sector_code = str(
            stock.sector_code or ""
        ).strip()

        if market not in {
            "KOSPI",
            "KOSDAQ",
        }:
            raise ValueError(
                "시장 구분이 올바르지 않습니다: "
                f"{stock_code} {market}"
            )

        if not sector_code:
            raise ValueError(
                "업종 코드가 없습니다: "
                f"{stock_code}"
            )

        market_index_code = (
            "0001"
            if market == "KOSPI"
            else "1001"
        )

        start_date = (
            date.today()
            - timedelta(days=200)
        )

        stock_rows = (
            self.stock_repository
            .get_daily_prices(
                stock_code=stock_code,
                start_date=start_date,
            )
        )

        sector_rows = (
            self.repository
            .get_sector_index_prices(
                sector_code=sector_code,
                market=market,
                start_date=start_date,
            )
        )

        market_rows = (
            self.repository
            .get_market_index_prices(
                index_code=market_index_code,
                start_date=start_date,
            )
        )

        if not stock_rows:
            raise ValueError(
                "종목 일봉이 없습니다: "
                f"{stock_code}"
            )

        if not sector_rows:
            raise ValueError(
                "업종지수가 없습니다: "
                f"{stock_code} "
                f"{market} "
                f"{sector_code}"
            )

        if not market_rows:
            raise ValueError(
                "시장지수가 없습니다: "
                f"{stock_code} "
                f"{market}"
            )

        as_of_date = min(
            stock_rows[-1].trade_date,
            sector_rows[-1].trade_date,
            market_rows[-1].trade_date,
        )

        stock_rows = [
            row
            for row in stock_rows
            if row.trade_date <= as_of_date
        ]

        sector_rows = [
            row
            for row in sector_rows
            if row.trade_date <= as_of_date
        ]

        market_rows = [
            row
            for row in market_rows
            if row.trade_date <= as_of_date
        ]

        if len(market_rows) < 61:
            raise ValueError(
                "시장지수 60거래일 부족: "
                f"{stock_code} "
                f"{len(market_rows)}건"
            )

        result = {
            "stock_code": stock.code,
            "stock_name": stock.name,
            "market": market,
            "sector_code": sector_code,
            "sector_name": stock.sector_name,
            "as_of_date": (
                as_of_date.isoformat()
            ),
            "stock_price_rows": (
                len(stock_rows)
            ),
            "sector_index_rows": (
                len(sector_rows)
            ),
            "market_index_rows": (
                len(market_rows)
            ),
        }

        for period in (
            5,
            20,
            60,
        ):
            period_start_date = (
                market_rows[
                    -(period + 1)
                ].trade_date
            )

            stock_return = (
                self._return_between_dates(
                    stock_rows,
                    start_date=(
                        period_start_date
                    ),
                    end_date=as_of_date,
                )
            )

            sector_return = (
                self._return_between_dates(
                    sector_rows,
                    start_date=(
                        period_start_date
                    ),
                    end_date=as_of_date,
                )
            )

            market_return = (
                self._return_between_dates(
                    market_rows,
                    start_date=(
                        period_start_date
                    ),
                    end_date=as_of_date,
                )
            )

            if (
                stock_return is None
                or sector_return is None
                or market_return is None
            ):
                raise ValueError(
                    "상대강도 계산 데이터 부족: "
                    f"{stock_code} "
                    f"{period}거래일"
                )

            result[
                f"start_date_{period}d"
            ] = (
                period_start_date
                .isoformat()
            )

            result[
                f"stock_return_{period}d_pct"
            ] = stock_return

            result[
                f"sector_return_{period}d_pct"
            ] = sector_return

            result[
                f"market_return_{period}d_pct"
            ] = market_return

            result[
                f"sector_excess_{period}d_pct"
            ] = round(
                stock_return
                - sector_return,
                4,
            )

            result[
                f"market_excess_{period}d_pct"
            ] = round(
                stock_return
                - market_return,
                4,
            )

        return result

    def calculate_top_relative_strength(
        self,
        *,
        limit: int = 100,
    ) -> dict:
        stock_codes = (
            self.get_current_universe_stock_codes(
                limit=limit,
            )
        )

        results: list[dict] = []
        errors: list[dict] = []

        for stock_code in stock_codes:
            try:
                results.append(
                    self
                    .calculate_stock_relative_strength(
                        stock_code
                    )
                )

            except ValueError as e:
                errors.append(
                    {
                        "stock_code":
                            stock_code,
                        "reason": str(e),
                    }
                )

        return {
            "success": (
                len(stock_codes) > 0
                and not errors
            ),
            "universe_count": (
                len(stock_codes)
            ),
            "calculated_count": (
                len(results)
            ),
            "failed_count": (
                len(errors)
            ),
            "periods": [
                5,
                20,
                60,
            ],
            "errors": errors,
            "results": results,
        }
        
    async def sync_top_universe_sector_indices(
        self,
        *,
        limit: int = 100,
        days: int = 365,
    ) -> tuple[
        int,
        int,
        int,
        int,
        int,
    ]:
        raw_universe = (
            await kis_master_client
            .get_market_cap_universe(
                limit=max(
                    limit * 2,
                    200,
                )
            )
        )

        metadata_rows: list[dict] = []

        sectors: dict[
            tuple[str, str],
            str,
        ] = {}

        selected = 0

        for raw in raw_universe:
            stock_code = str(
                raw.get(
                    "stock_code",
                    "",
                )
            ).strip()

            market = str(
                raw.get(
                    "market",
                    "",
                )
            ).strip()

            sector_code = str(
                raw.get(
                    "sector_large_code",
                    "",
                )
            ).strip()

            if (
                not stock_code
                or not market
                or not sector_code
            ):
                continue

            stock = (
                self.stock_repository
                .get_stock(
                    stock_code
                )
            )

            if (
                stock is None
                or not stock.is_active
            ):
                continue

            metadata_rows.append(
                {
                    "stock_code":
                        stock_code,
                    "market":
                        market,
                    "sector_code":
                        sector_code,
                }
            )

            sector_name = (
                str(
                    stock.sector_name
                    or sector_code
                ).strip()
            )

            sectors[
                (
                    market,
                    sector_code,
                )
            ] = sector_name

            selected += 1

            if selected >= limit:
                break

        metadata_count = (
            self.stock_repository
            .update_market_sector_metadata(
                metadata_rows
            )
        )

        total_rows = 0
        success_sectors = 0
        failed_sectors = 0

        sector_items = list(
            sectors.items()
        )

        total_sectors = len(
            sector_items
        )

        for index, (
            (
                market,
                sector_code,
            ),
            sector_name,
        ) in enumerate(
            sector_items,
            start=1,
        ):
            print(
                "[SECTOR] "
                f"{index}/{total_sectors} "
                f"{market} "
                f"{sector_code} "
                f"{sector_name}",
                flush=True,
            )

            try:
                count = (
                    await self
                    .sync_sector_index(
                        sector_code=(
                            sector_code
                        ),
                        sector_name=(
                            sector_name
                        ),
                        market=market,
                        days=days,
                    )
                )

                if count <= 0:
                    failed_sectors += 1

                    print(
                        "[SECTOR] "
                        f"{market} "
                        f"{sector_code} "
                        "데이터 없음",
                        flush=True,
                    )

                    continue

                success_sectors += 1
                total_rows += count

                print(
                    "[SECTOR] "
                    f"{market} "
                    f"{sector_code} "
                    f"{count}건 저장",
                    flush=True,
                )

            except (
                ConfigurationError,
                ExternalApiError,
                ValueError,
            ) as e:
                self.db.rollback()

                failed_sectors += 1

                print(
                    "[SECTOR] "
                    f"{market} "
                    f"{sector_code} "
                    f"실패: {e}",
                    flush=True,
                )

        return (
            metadata_count,
            total_sectors,
            success_sectors,
            total_rows,
            failed_sectors,
        )