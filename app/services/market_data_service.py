from collections import defaultdict
from datetime import date, datetime, timedelta
from statistics import median

from sqlalchemy.orm import Session
from app.clients.kis_master_client import (
    kis_master_client,
)
from app.clients.ecos_client import ecos_client
from app.clients.kis_client import kis_client
from app.clients.toss_client import toss_client
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

    async def sync_toss_market_indices(
        self,
        *,
        count: int = 200,
    ) -> int:
        symbols = [
            "KOSPI",
            "KOSDAQ",
        ]

        definitions = {
            "KOSPI": (
                "0001",
                "KOSPI",
                "KOSPI",
            ),
            "KOSDAQ": (
                "1001",
                "KOSDAQ",
                "KOSDAQ",
            ),
        }

        current_rows = (
            await toss_client
            .get_market_indicator_prices(
                symbols
            )
        )

        current_prices = {
            str(
                row.get(
                    "symbol",
                    "",
                )
            ).strip().upper():
                self._optional_float(
                    row.get(
                        "lastPrice"
                    )
                )
            for row in current_rows
            if isinstance(
                row,
                dict,
            )
        }

        total_count = 0

        for symbol in symbols:
            payload = (
                await toss_client
                .get_market_indicator_candles(
                    symbol,
                    interval="1d",
                    count=count,
                )
            )

            (
                index_code,
                index_name,
                market,
            ) = definitions[symbol]

            parsed_rows: list[
                dict
            ] = []

            for raw in payload.get(
                "candles",
                [],
            ):
                trade_date = (
                    self._parse_iso_date(
                        raw.get(
                            "timestamp"
                        )
                    )
                )

                close = (
                    self._optional_float(
                        raw.get(
                            "closePrice"
                        )
                    )
                )

                if (
                    trade_date is None
                    or close is None
                    or close <= 0
                ):
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
                                    "openPrice"
                                )
                            ),
                        "high":
                            self._optional_float(
                                raw.get(
                                    "highPrice"
                                )
                            ),
                        "low":
                            self._optional_float(
                                raw.get(
                                    "lowPrice"
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
                                    "volume"
                                )
                            ),
                        "trading_value":
                            None,
                        "source":
                            "TOSS",
                        "raw_json":
                            dict(raw),
                    }
                )

            today = date.today()

            live_price = (
                current_prices.get(
                    symbol
                )
            )

            if (
                live_price is not None
                and live_price > 0
            ):
                today_row = next(
                    (
                        row
                        for row
                        in parsed_rows
                        if row[
                            "trade_date"
                        ] == today
                    ),
                    None,
                )

                if today_row is not None:
                    today_row[
                        "close"
                    ] = live_price

                    today_row[
                        "raw_json"
                    ] = {
                        **today_row[
                            "raw_json"
                        ],
                        "livePrice":
                            live_price,
                    }

                else:
                    parsed_rows.append(
                        {
                            "index_code":
                                index_code,
                            "index_name":
                                index_name,
                            "market":
                                market,
                            "trade_date":
                                today,
                            "open":
                                None,
                            "high":
                                None,
                            "low":
                                None,
                            "close":
                                live_price,
                            "change":
                                None,
                            "change_rate":
                                None,
                            "volume":
                                None,
                            "trading_value":
                                None,
                            "source":
                                "TOSS",
                            "raw_json": {
                                "symbol":
                                    symbol,
                                "lastPrice":
                                    live_price,
                                "provisional":
                                    True,
                            },
                        }
                    )

            parsed_rows.sort(
                key=lambda row: (
                    row[
                        "trade_date"
                    ]
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

    async def sync_toss_market_investor_flows(
        self,
        *,
        count: int = 60,
    ) -> int:
        total_count = 0

        for market in (
            "KOSPI",
            "KOSDAQ",
        ):
            payload = (
                await toss_client
                .get_market_indicator_investor_trading(
                    market,
                    interval="1d",
                    count=count,
                )
            )

            rows: list[
                dict
            ] = []

            for raw in payload.get(
                "records",
                [],
            ):
                trade_date = (
                    self._parse_iso_date(
                        raw.get(
                            "date"
                        )
                    )
                )

                if trade_date is None:
                    continue

                individual = (
                    raw.get(
                        "individual"
                    )
                    if isinstance(
                        raw.get(
                            "individual"
                        ),
                        dict,
                    )
                    else {}
                )

                foreigner = (
                    raw.get(
                        "foreigner"
                    )
                    if isinstance(
                        raw.get(
                            "foreigner"
                        ),
                        dict,
                    )
                    else {}
                )

                institution = (
                    raw.get(
                        "institution"
                    )
                    if isinstance(
                        raw.get(
                            "institution"
                        ),
                        dict,
                    )
                    else {}
                )

                other_corporation = (
                    raw.get(
                        "otherCorporation"
                    )
                    if isinstance(
                        raw.get(
                            "otherCorporation"
                        ),
                        dict,
                    )
                    else {}
                )

                breakdown = (
                    institution.get(
                        "breakdown"
                    )
                )

                rows.append(
                    {
                        "market":
                            market,
                        "trade_date":
                            trade_date,

                        "individual_buy_amount":
                            to_decimal_or_none(
                                individual.get(
                                    "buyAmount"
                                )
                            ),

                        "individual_sell_amount":
                            to_decimal_or_none(
                                individual.get(
                                    "sellAmount"
                                )
                            ),

                        "foreign_buy_amount":
                            to_decimal_or_none(
                                foreigner.get(
                                    "buyAmount"
                                )
                            ),

                        "foreign_sell_amount":
                            to_decimal_or_none(
                                foreigner.get(
                                    "sellAmount"
                                )
                            ),

                        "institution_buy_amount":
                            to_decimal_or_none(
                                institution.get(
                                    "buyAmount"
                                )
                            ),

                        "institution_sell_amount":
                            to_decimal_or_none(
                                institution.get(
                                    "sellAmount"
                                )
                            ),

                        "other_corporation_buy_amount":
                            to_decimal_or_none(
                                other_corporation.get(
                                    "buyAmount"
                                )
                            ),

                        "other_corporation_sell_amount":
                            to_decimal_or_none(
                                other_corporation.get(
                                    "sellAmount"
                                )
                            ),

                        "institution_breakdown_json":
                            (
                                dict(
                                    breakdown
                                )
                                if isinstance(
                                    breakdown,
                                    dict,
                                )
                                else None
                            ),

                        "source":
                            "TOSS",

                        "raw_json":
                            dict(raw),
                    }
                )

            total_count += (
                self.repository
                .upsert_market_investor_flows(
                    rows
                )
            )

        return total_count

    async def sync_toss_usd_krw(
        self,
    ) -> int:
        raw = (
            await toss_client
            .get_exchange_rate(
                base_currency="USD",
                quote_currency="KRW",
            )
        )

        rate = (
            self._optional_float(
                raw.get(
                    "rate"
                )
            )
        )

        if (
            rate is None
            or rate <= 0
        ):
            return 0

        observed_date = (
            self._parse_iso_date(
                raw.get(
                    "validFrom"
                )
            )
            or date.today()
        )

        return (
            self.repository
            .upsert_macro_indicators(
                [
                    {
                        "indicator_code":
                            "USD_KRW",

                        "indicator_name":
                            "원/달러 환율",

                        "indicator_type":
                            "fx",

                        "frequency":
                            "D",

                        "observed_date":
                            observed_date,

                        "value":
                            rate,

                        "unit":
                            "KRW/USD",

                        "source":
                            "TOSS",

                        "metadata_json":
                            dict(raw),
                    }
                ]
            )
        )

    async def sync_toss_market_context(
        self,
        *,
        index_count: int = 200,
        investor_count: int = 60,
    ) -> dict:
        index_rows = (
            await self
            .sync_toss_market_indices(
                count=index_count,
            )
        )

        investor_rows = (
            await self
            .sync_toss_market_investor_flows(
                count=investor_count,
            )
        )

        fx_rows = (
            await self
            .sync_toss_usd_krw()
        )

        return {
            "indexRows":
                index_rows,
            "investorRows":
                investor_rows,
            "fxRows":
                fx_rows,
        }

    @staticmethod
    def _calculate_index_return(
        history: list[dict],
        sessions: int,
    ) -> float | None:
        if len(history) <= sessions:
            return None

        latest_close = history[-1].get(
            "close"
        )

        base_close = history[
            -(sessions + 1)
        ].get(
            "close"
        )

        if (
            latest_close is None
            or base_close is None
            or base_close <= 0
        ):
            return None

        return (
            (
                latest_close
                / base_close
                - 1.0
            )
            * 100.0
        )

    @staticmethod
    def _build_investor_period_summary(
        history: list[dict],
        sessions: int,
    ) -> dict:
        if len(history) < sessions:
            return {
                "sessionCount":
                    len(history),
                "individualNet":
                    None,
                "foreignNet":
                    None,
                "institutionNet":
                    None,
                "otherCorporationNet":
                    None,
            }

        selected = history[
            -sessions:
        ]

        def sum_net(
            key: str,
        ) -> float:
            return float(
                sum(
                    float(
                        row.get(
                            key,
                            {},
                        ).get(
                            "netAmount",
                            0.0,
                        )
                        or 0.0
                    )
                    for row in selected
                )
            )

        return {
            "sessionCount":
                sessions,
            "individualNet":
                sum_net(
                    "individual"
                ),
            "foreignNet":
                sum_net(
                    "foreign"
                ),
            "institutionNet":
                sum_net(
                    "institution"
                ),
            "otherCorporationNet":
                sum_net(
                    "otherCorporation"
                ),
        }

    @staticmethod
    def _build_fx_summary(
        history: list[dict],
    ) -> dict:
        if not history:
            return {
                "observedDate": None,
                "value": None,
                "previousObservedDate": None,
                "previousValue": None,
                "change": None,
                "changeRate": None,
                "unit": None,
                "source": None,
            }

        latest = history[-1]

        previous = (
            history[-2]
            if len(history) >= 2
            else None
        )

        latest_value = latest.get(
            "value"
        )

        previous_value = (
            previous.get(
                "value"
            )
            if previous
            else None
        )

        change = None
        change_rate = None

        if (
            latest_value is not None
            and previous_value is not None
        ):
            change = (
                float(latest_value)
                - float(previous_value)
            )

            if float(
                previous_value
            ) != 0:
                change_rate = (
                    change
                    / float(
                        previous_value
                    )
                    * 100.0
                )

        return {
            "observedDate":
                latest.get(
                    "observedDate"
                ),
            "value":
                latest_value,
            "previousObservedDate": (
                previous.get(
                    "observedDate"
                )
                if previous
                else None
            ),
            "previousValue":
                previous_value,
            "change":
                change,
            "changeRate":
                change_rate,
            "unit":
                latest.get(
                    "unit"
                ),
            "source":
                latest.get(
                    "source"
                ),
        }

    def get_toss_market_context_summary(
        self,
        *,
        days: int = 30,
    ) -> dict:
        end_date = date.today()

        start_date = (
            end_date
            - timedelta(
                days=days
            )
        )

        indices = {}

        for (
            market,
            index_code,
        ) in (
            (
                "KOSPI",
                "0001",
            ),
            (
                "KOSDAQ",
                "1001",
            ),
        ):
            rows = (
                self.repository
                .get_market_index_prices(
                    index_code=index_code,
                    start_date=start_date,
                    end_date=end_date,
                )
            )

            history = [
                {
                    "tradeDate":
                        row.trade_date.isoformat(),
                    "open":
                        row.open,
                    "high":
                        row.high,
                    "low":
                        row.low,
                    "close":
                        row.close,
                    "change":
                        row.change,
                    "changeRate":
                        row.change_rate,
                    "volume":
                        row.volume,
                    "source":
                        row.source,
                }
                for row in rows
            ]

            latest = (
                history[-1]
                if history
                else None
            )

            indices[
                market
            ] = {
                "latest":
                    latest,
                "history":
                    history,
            }

        investor_flows = {}

        for market in (
            "KOSPI",
            "KOSDAQ",
        ):
            rows = (
                self.repository
                .get_market_investor_flows(
                    market=market,
                    start_date=start_date,
                    end_date=end_date,
                )
            )

            history = []

            for row in rows:
                individual_buy = (
                    float(
                        row.individual_buy_amount
                    )
                    if row.individual_buy_amount
                    is not None
                    else None
                )

                individual_sell = (
                    float(
                        row.individual_sell_amount
                    )
                    if row.individual_sell_amount
                    is not None
                    else None
                )

                foreign_buy = (
                    float(
                        row.foreign_buy_amount
                    )
                    if row.foreign_buy_amount
                    is not None
                    else None
                )

                foreign_sell = (
                    float(
                        row.foreign_sell_amount
                    )
                    if row.foreign_sell_amount
                    is not None
                    else None
                )

                institution_buy = (
                    float(
                        row.institution_buy_amount
                    )
                    if row.institution_buy_amount
                    is not None
                    else None
                )

                institution_sell = (
                    float(
                        row.institution_sell_amount
                    )
                    if row.institution_sell_amount
                    is not None
                    else None
                )

                other_buy = (
                    float(
                        row.other_corporation_buy_amount
                    )
                    if row.other_corporation_buy_amount
                    is not None
                    else None
                )

                other_sell = (
                    float(
                        row.other_corporation_sell_amount
                    )
                    if row.other_corporation_sell_amount
                    is not None
                    else None
                )

                history.append(
                    {
                        "tradeDate":
                            row.trade_date.isoformat(),
                        "individual": {
                            "buyAmount":
                                individual_buy,
                            "sellAmount":
                                individual_sell,
                            "netAmount": (
                                individual_buy
                                - individual_sell
                                if (
                                    individual_buy
                                    is not None
                                    and individual_sell
                                    is not None
                                )
                                else None
                            ),
                        },
                        "foreign": {
                            "buyAmount":
                                foreign_buy,
                            "sellAmount":
                                foreign_sell,
                            "netAmount": (
                                foreign_buy
                                - foreign_sell
                                if (
                                    foreign_buy
                                    is not None
                                    and foreign_sell
                                    is not None
                                )
                                else None
                            ),
                        },
                        "institution": {
                            "buyAmount":
                                institution_buy,
                            "sellAmount":
                                institution_sell,
                            "netAmount": (
                                institution_buy
                                - institution_sell
                                if (
                                    institution_buy
                                    is not None
                                    and institution_sell
                                    is not None
                                )
                                else None
                            ),
                            "breakdown":
                                row.institution_breakdown_json,
                        },
                        "otherCorporation": {
                            "buyAmount":
                                other_buy,
                            "sellAmount":
                                other_sell,
                            "netAmount": (
                                other_buy
                                - other_sell
                                if (
                                    other_buy
                                    is not None
                                    and other_sell
                                    is not None
                                )
                                else None
                            ),
                        },
                        "source":
                            row.source,
                    }
                )

            investor_flows[
                market
            ] = {
                "latest": (
                    history[-1]
                    if history
                    else None
                ),
                "history":
                    history,
            }

        fx_rows = (
            self.repository
            .get_macro_indicators(
                indicator_code="USD_KRW",
                start_date=start_date,
                end_date=end_date,
            )
        )

        fx_history = [
            {
                "observedDate":
                    row.observed_date.isoformat(),
                "value":
                    row.value,
                "unit":
                    row.unit,
                "source":
                    row.source,
            }
            for row in fx_rows
        ]

        market_summary = {
            "indices": {},
            "investorFlows": {},
            "usdKrw": (
                self._build_fx_summary(
                    fx_history
                )
            ),
        }

        for market in (
            "KOSPI",
            "KOSDAQ",
        ):
            index_history = (
                indices.get(
                    market,
                    {},
                ).get(
                    "history",
                    [],
                )
            )

            latest_index = (
                index_history[-1]
                if index_history
                else None
            )

            market_summary[
                "indices"
            ][market] = {
                "tradeDate": (
                    latest_index.get(
                        "tradeDate"
                    )
                    if latest_index
                    else None
                ),
                "close": (
                    latest_index.get(
                        "close"
                    )
                    if latest_index
                    else None
                ),
                "dailyChangeRate": (
                    latest_index.get(
                        "changeRate"
                    )
                    if latest_index
                    else None
                ),
                "return5d": (
                    self._calculate_index_return(
                        index_history,
                        5,
                    )
                ),
                "return20d": (
                    self._calculate_index_return(
                        index_history,
                        20,
                    )
                ),
                "source": (
                    latest_index.get(
                        "source"
                    )
                    if latest_index
                    else None
                ),
            }

            investor_history = (
                investor_flows.get(
                    market,
                    {},
                ).get(
                    "history",
                    [],
                )
            )

            latest_flow = (
                investor_history[-1]
                if investor_history
                else None
            )

            market_summary[
                "investorFlows"
            ][market] = {
                "tradeDate": (
                    latest_flow.get(
                        "tradeDate"
                    )
                    if latest_flow
                    else None
                ),
                "latest": {
                    "individualNet": (
                        latest_flow.get(
                            "individual",
                            {},
                        ).get(
                            "netAmount"
                        )
                        if latest_flow
                        else None
                    ),
                    "foreignNet": (
                        latest_flow.get(
                            "foreign",
                            {},
                        ).get(
                            "netAmount"
                        )
                        if latest_flow
                        else None
                    ),
                    "institutionNet": (
                        latest_flow.get(
                            "institution",
                            {},
                        ).get(
                            "netAmount"
                        )
                        if latest_flow
                        else None
                    ),
                    "otherCorporationNet": (
                        latest_flow.get(
                            "otherCorporation",
                            {},
                        ).get(
                            "netAmount"
                        )
                        if latest_flow
                        else None
                    ),
                },
                "fiveDays": (
                    self._build_investor_period_summary(
                        investor_history,
                        5,
                    )
                ),
                "twentyDays": (
                    self._build_investor_period_summary(
                        investor_history,
                        20,
                    )
                ),
                "source": (
                    latest_flow.get(
                        "source"
                    )
                    if latest_flow
                    else None
                ),
            }

        return {
            "asOf":
                end_date.isoformat(),
            "days":
                days,
            "summary":
                market_summary,
            "indices":
                indices,
            "investorFlows":
                investor_flows,
            "usdKrw": {
                "latest": (
                    fx_history[-1]
                    if fx_history
                    else None
                ),
                "history":
                    fx_history,
            },
        }

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

            market_cap_by_code: dict[
                str,
                float,
            ] = {}

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

                market_cap_100m = (
                    self._optional_float(
                        raw.get(
                            "market_cap_100m"
                        )
                    )
                )

                if (
                    market_cap_100m
                    is not None
                    and market_cap_100m > 0.0
                ):
                    market_cap_by_code[
                        stock_code
                    ] = (
                        market_cap_100m
                        * 100_000_000.0
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
                            market_cap_override=(
                                market_cap_by_code
                                .get(
                                    stock_code
                                )
                            ),
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
        try:
            result = (
                await toss_client
                .get_investor_trading(
                    stock_code,
                    count=65,
                )
            )

            parsed_rows: list[
                dict
            ] = []

            for raw in result[
                "records"
            ]:
                try:
                    trade_date = (
                        date.fromisoformat(
                            str(
                                raw.get(
                                    "date"
                                )
                                or ""
                            )
                        )
                    )
                except ValueError:
                    continue

                individual = raw.get(
                    "individual"
                )

                foreigner = raw.get(
                    "foreigner"
                )

                institution = raw.get(
                    "institution"
                )

                if not (
                    isinstance(
                        individual,
                        dict,
                    )
                    and isinstance(
                        foreigner,
                        dict,
                    )
                    and isinstance(
                        institution,
                        dict,
                    )
                ):
                    continue

                if (
                    individual.get(
                        "netBuyVolume"
                    )
                    is None
                    or foreigner.get(
                        "netBuyVolume"
                    )
                    is None
                    or institution.get(
                        "netBuyVolume"
                    )
                    is None
                ):
                    continue

                foreign_holding = raw.get(
                    "foreignerHolding"
                )

                holding_rate = None

                if isinstance(
                    foreign_holding,
                    dict,
                ):
                    holding_rate = (
                        self._optional_float(
                            foreign_holding.get(
                                "holdingRate"
                            )
                        )
                    )

                parsed_rows.append(
                    {
                        "stock_code":
                            stock_code,

                        "trade_date":
                            trade_date,

                        "foreign_net_buy_volume":
                            to_int(
                                foreigner.get(
                                    "netBuyVolume"
                                )
                            ),

                        "institution_net_buy_volume":
                            to_int(
                                institution.get(
                                    "netBuyVolume"
                                )
                            ),

                        "individual_net_buy_volume":
                            to_int(
                                individual.get(
                                    "netBuyVolume"
                                )
                            ),

                        "foreign_holding_ratio":
                            holding_rate,

                        "source":
                            "TOSS",

                        "raw_json":
                            dict(raw),
                    }
                )

            if parsed_rows:
                print(
                    "[MARKET][TOSS-FLOW] "
                    f"{stock_code} "
                    f"rows={len(parsed_rows)}",
                    flush=True,
                )

                return (
                    self.repository
                    .upsert_toss_stock_investor_flows(
                        parsed_rows
                    )
                )

        except (
            ConfigurationError,
            ExternalApiError,
        ) as e:
            print(
                "[MARKET][TOSS-FLOW] "
                f"{stock_code} "
                f"fallback=KIS error={e}",
                flush=True,
            )

        raw_rows = (
            await kis_client
            .get_investor_flows(
                stock_code
            )
        )

        parsed_rows = []

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

    async def sync_stock_trading_trends(
        self,
        stock_code: str,
        *,
        count: int = 60,
    ) -> dict[str, int]:
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

        program_result = (
            await toss_client
            .get_program_trades(
                stock_code,
                count=count,
            )
        )

        short_result = (
            await toss_client
            .get_short_selling(
                stock_code,
                count=count,
            )
        )

        credit_result = (
            await toss_client
            .get_credit_trades(
                stock_code,
                count=count,
            )
        )

        lending_result = (
            await toss_client
            .get_securities_lending(
                stock_code,
                count=count,
            )
        )

        program_rows: list[dict] = []

        for raw in program_result["records"]:
            try:
                trade_date = date.fromisoformat(
                    str(
                        raw.get(
                            "date"
                        )
                        or ""
                    )
                )
            except ValueError:
                continue

            arbitrage = raw.get(
                "arbitrage"
            )

            non_arbitrage = raw.get(
                "nonArbitrage"
            )

            if not (
                isinstance(
                    arbitrage,
                    dict,
                )
                and isinstance(
                    non_arbitrage,
                    dict,
                )
            ):
                continue

            program_rows.append(
                {
                    "stock_code":
                        stock_code,
                    "trade_date":
                        trade_date,

                    "arbitrage_buy_volume":
                        to_int(
                            arbitrage.get(
                                "buyVolume"
                            )
                        ),
                    "arbitrage_sell_volume":
                        to_int(
                            arbitrage.get(
                                "sellVolume"
                            )
                        ),
                    "arbitrage_net_buy_volume":
                        to_int(
                            arbitrage.get(
                                "netBuyVolume"
                            )
                        ),

                    "non_arbitrage_buy_volume":
                        to_int(
                            non_arbitrage.get(
                                "buyVolume"
                            )
                        ),
                    "non_arbitrage_sell_volume":
                        to_int(
                            non_arbitrage.get(
                                "sellVolume"
                            )
                        ),
                    "non_arbitrage_net_buy_volume":
                        to_int(
                            non_arbitrage.get(
                                "netBuyVolume"
                            )
                        ),

                    "source":
                        "TOSS",
                    "raw_json":
                        dict(raw),
                }
            )

        short_rows: list[dict] = []

        for raw in short_result["records"]:
            try:
                trade_date = date.fromisoformat(
                    str(
                        raw.get(
                            "date"
                        )
                        or ""
                    )
                )
            except ValueError:
                continue

            short_rows.append(
                {
                    "stock_code":
                        stock_code,
                    "trade_date":
                        trade_date,

                    "short_selling_volume":
                        to_int(
                            raw.get(
                                "shortSellingVolume"
                            )
                        ),
                    "short_selling_amount":
                        to_int(
                            raw.get(
                                "shortSellingAmount"
                            )
                        ),
                    "short_selling_volume_rate":
                        self._optional_float(
                            raw.get(
                                "shortSellingVolumeRate"
                            )
                        ),
                    "short_selling_amount_rate":
                        self._optional_float(
                            raw.get(
                                "shortSellingAmountRate"
                            )
                        ),

                    "source":
                        "TOSS",
                    "raw_json":
                        dict(raw),
                }
            )

        credit_rows: list[dict] = []

        for raw in credit_result["records"]:
            try:
                trade_date = date.fromisoformat(
                    str(
                        raw.get(
                            "date"
                        )
                        or ""
                    )
                )
            except ValueError:
                continue

            margin_loan = raw.get(
                "marginLoan"
            )

            stock_loan = raw.get(
                "stockLoan"
            )

            if not (
                isinstance(
                    margin_loan,
                    dict,
                )
                and isinstance(
                    stock_loan,
                    dict,
                )
            ):
                continue

            credit_rows.append(
                {
                    "stock_code":
                        stock_code,
                    "trade_date":
                        trade_date,

                    "margin_loan_new_quantity":
                        to_int(
                            margin_loan.get(
                                "newQuantity"
                            )
                        ),
                    "margin_loan_return_quantity":
                        to_int(
                            margin_loan.get(
                                "returnQuantity"
                            )
                        ),
                    "margin_loan_balance_quantity":
                        to_int(
                            margin_loan.get(
                                "balanceQuantity"
                            )
                        ),
                    "margin_loan_balance_rate":
                        self._optional_float(
                            margin_loan.get(
                                "balanceRate"
                            )
                        ),
                    "margin_loan_trading_rate":
                        self._optional_float(
                            margin_loan.get(
                                "tradingRate"
                            )
                        ),

                    "stock_loan_new_quantity":
                        to_int(
                            stock_loan.get(
                                "newQuantity"
                            )
                        ),
                    "stock_loan_return_quantity":
                        to_int(
                            stock_loan.get(
                                "returnQuantity"
                            )
                        ),
                    "stock_loan_balance_quantity":
                        to_int(
                            stock_loan.get(
                                "balanceQuantity"
                            )
                        ),
                    "stock_loan_balance_rate":
                        self._optional_float(
                            stock_loan.get(
                                "balanceRate"
                            )
                        ),
                    "stock_loan_trading_rate":
                        self._optional_float(
                            stock_loan.get(
                                "tradingRate"
                            )
                        ),

                    "source":
                        "TOSS",
                    "raw_json":
                        dict(raw),
                }
            )

        lending_rows: list[dict] = []

        for raw in lending_result["records"]:
            try:
                trade_date = date.fromisoformat(
                    str(
                        raw.get(
                            "date"
                        )
                        or ""
                    )
                )
            except ValueError:
                continue

            lending_rows.append(
                {
                    "stock_code":
                        stock_code,
                    "trade_date":
                        trade_date,

                    "execution_quantity":
                        to_int(
                            raw.get(
                                "executionQuantity"
                            )
                        ),
                    "repayment_quantity":
                        to_int(
                            raw.get(
                                "repaymentQuantity"
                            )
                        ),
                    "balance_quantity":
                        to_int(
                            raw.get(
                                "balanceQuantity"
                            )
                        ),
                    "balance_amount":
                        to_int(
                            raw.get(
                                "balanceAmount"
                            )
                        ),

                    "source":
                        "TOSS",
                    "raw_json":
                        dict(raw),
                }
            )

        result = {
            "programTrades":
                self.repository
                .upsert_stock_program_trades(
                    program_rows
                ),

            "shortSelling":
                self.repository
                .upsert_stock_short_selling(
                    short_rows
                ),

            "creditTrades":
                self.repository
                .upsert_stock_credit_trades(
                    credit_rows
                ),

            "securitiesLending":
                self.repository
                .upsert_stock_securities_lending(
                    lending_rows
                ),
        }

        print(
            "[MARKET][TOSS-TRADING-TRENDS] "
            f"{stock_code} "
            f"{result}",
            flush=True,
        )

        return result

    async def sync_historical_stock_investor_flow(
        self,
        stock_code: str,
        *,
        start_date: date,
        end_date: date,
    ) -> int:
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

        raw_rows = (
            await kis_client
            .get_historical_investor_flows(
                stock_code,
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
        market_cap_override: float | None = None,
    ) -> int:
        stock_service = StockService(
            self.db
        )

        previous_valuation = (
            self.repository
            .get_latest_usable_valuation(
                stock_code
            )
        )

        kis_refreshed = False
        kis_error: str | None = None

        try:
            await (
                stock_service
                ._sync_current_price_from_kis(
                    stock_code,
                    market_override=(
                        market_override
                    ),
                )
            )
            kis_refreshed = True

        except (
            ConfigurationError,
            ExternalApiError,
            ValueError,
        ) as e:
            self.db.rollback()
            kis_error = str(e)

            print(
                "[VALUATION][KIS-FAIL] "
                f"{stock_code}: {e}",
                flush=True,
            )

            stock = (
                self.stock_repository
                .get_stock(
                    stock_code
                )
            )

            if (
                stock is None
                or stock.current_price is None
                or float(
                    stock.current_price
                ) <= 0.0
            ):
                try:
                    await (
                        stock_service
                        ._sync_current_price_from_toss(
                            stock_code
                        )
                    )
                except (
                    ConfigurationError,
                    ExternalApiError,
                    ValueError,
                ) as toss_error:
                    self.db.rollback()

                    print(
                        "[VALUATION][TOSS-FAIL] "
                        f"{stock_code}: "
                        f"{toss_error}",
                        flush=True,
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

        price = self._optional_float(
            stock.current_price
        )
        market_cap = self._optional_float(
            stock.market_cap
        )
        eps = self._optional_float(
            stock.eps
        )
        bps = self._optional_float(
            stock.bps
        )
        per = self._optional_float(
            stock.per
        )
        pbr = self._optional_float(
            stock.pbr
        )

        if price is not None and price <= 0.0:
            price = None

        if (
            market_cap is None
            or market_cap <= 0.0
        ):
            market_cap = self._optional_float(
                market_cap_override
            )

        if previous_valuation is not None:
            if price is None:
                price = self._optional_float(
                    previous_valuation.price
                )

            if (
                market_cap is None
                or market_cap <= 0.0
            ):
                market_cap = self._optional_float(
                    previous_valuation.market_cap
                )

            if eps is None:
                eps = self._optional_float(
                    previous_valuation.eps
                )

            if bps is None:
                bps = self._optional_float(
                    previous_valuation.bps
                )

        if per is not None and per <= 0.0:
            per = None

        if pbr is not None and pbr <= 0.0:
            pbr = None

        if (
            per is None
            and price is not None
            and eps is not None
            and eps > 0.0
        ):
            per = price / eps

        if (
            pbr is None
            and price is not None
            and bps is not None
            and bps > 0.0
        ):
            pbr = price / bps

        if previous_valuation is not None:
            previous_price = (
                self._optional_float(
                    previous_valuation.price
                )
            )

            if (
                per is None
                and price is not None
                and previous_price is not None
                and previous_price > 0.0
                and previous_valuation.per
                is not None
                and float(
                    previous_valuation.per
                ) > 0.0
            ):
                per = (
                    float(
                        previous_valuation.per
                    )
                    * price
                    / previous_price
                )

            if (
                pbr is None
                and price is not None
                and previous_price is not None
                and previous_price > 0.0
                and previous_valuation.pbr
                is not None
                and float(
                    previous_valuation.pbr
                ) > 0.0
            ):
                pbr = (
                    float(
                        previous_valuation.pbr
                    )
                    * price
                    / previous_price
                )

        if (
            per is None
            and previous_valuation
            is not None
            and previous_valuation.per
            is not None
            and float(
                previous_valuation.per
            ) > 0.0
        ):
            per = float(
                previous_valuation.per
            )

        if (
            pbr is None
            and previous_valuation
            is not None
            and previous_valuation.pbr
            is not None
            and float(
                previous_valuation.pbr
            ) > 0.0
        ):
            pbr = float(
                previous_valuation.pbr
            )

        if (
            market_cap is not None
            and market_cap > 0.0
            and (
                stock.market_cap is None
                or float(
                    stock.market_cap
                ) <= 0.0
            )
        ):
            stock.market_cap = market_cap

        if (
            market_override
            and stock.market
            != market_override
        ):
            stock.market = market_override

        self.db.commit()

        if (
            per is None
            and pbr is None
        ):
            print(
                "[VALUATION][EMPTY] "
                f"stock={stock_code} "
                f"price={price} "
                f"marketCap={market_cap} "
                f"stockPER={stock.per} "
                f"stockPBR={stock.pbr} "
                f"stockEPS={stock.eps} "
                f"stockBPS={stock.bps} "
                f"previousDate={previous_valuation.snapshot_date if previous_valuation is not None else None} "
                f"previousPER={previous_valuation.per if previous_valuation is not None else None} "
                f"previousPBR={previous_valuation.pbr if previous_valuation is not None else None} "
                f"kisError={kis_error}",
                flush=True,
            )

            raise ExternalApiError(
                "PER/PBR을 확보하지 못했습니다: "
                f"{stock_code}"
            )

        source = (
            "KIS"
            if kis_refreshed
            else "KIS_FALLBACK"
        )

        row = {
            "stock_code": stock_code,
            "snapshot_date": date.today(),
            "price": price,
            "market_cap": market_cap,
            "per": per,
            "pbr": pbr,
            "eps": eps,
            "bps": bps,
            "dividend_yield": (
                self._optional_float(
                    previous_valuation
                    .dividend_yield
                )
                if previous_valuation
                is not None
                else None
            ),
            "sector_code": (
                stock.sector_code
                or (
                    previous_valuation
                    .sector_code
                    if previous_valuation
                    is not None
                    else None
                )
            ),
            "sector_name": (
                stock.sector_name
                or (
                    previous_valuation
                    .sector_name
                    if previous_valuation
                    is not None
                    else None
                )
            ),
            "sector_per": None,
            "sector_pbr": None,
            "market_per": None,
            "market_pbr": None,
            "source": source,
            "raw_json": {
                "stock_code": stock.code,
                "sector_name": stock.sector_name,
                "kis_refreshed": kis_refreshed,
                "kis_error": kis_error,
                "used_previous_valuation": (
                    previous_valuation
                    is not None
                ),
            },
        }

        print(
            "[VALUATION][FINAL] "
            f"stock={stock_code} "
            f"source={source} "
            f"price={price} "
            f"marketCap={market_cap} "
            f"PER={per} "
            f"PBR={pbr} "
            f"EPS={eps} "
            f"BPS={bps} "
            f"previousDate={previous_valuation.snapshot_date if previous_valuation is not None else None}",
            flush=True,
        )

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
    def _parse_iso_date(
        value,
    ) -> date | None:
        text = str(
            value or ""
        ).strip()

        if not text:
            return None

        try:
            return datetime.fromisoformat(
                text.replace(
                    "Z",
                    "+00:00",
                )
            ).date()
        except ValueError:
            pass

        try:
            return date.fromisoformat(
                text[:10]
            )
        except ValueError:
            return None

    @staticmethod
    def _parse_yyyymmdd(
        value,
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
        return (
            self.stock_repository
            .get_stock_codes_for_market_context(
                limit=limit
            )
        )
    
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