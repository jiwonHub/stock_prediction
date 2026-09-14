from __future__ import annotations

import hashlib
import html
import re
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo
from email.utils import parsedate_to_datetime
from xml.etree import ElementTree

import httpx
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.clients.kis_client import dart_client
from app.models.future import (
    Disclosure,
    NewsArticle,
    RankingItem,
    RankingSnapshot,
    RecommendationPerformance,
    StockNews,
)
from app.models.market_data import MarketIndexPrice
from app.models.stock import Stock
from app.models.stock_price import StockPrice

class MarketContextService:
    RANKING_VERSION = "phase13-long-term-investment-v3"
    
    PRIMARY_HORIZON_DAYS = 20

    PERFORMANCE_HORIZONS = (
        20,
        60,
        120,
        240,
    )

    PERFORMANCE_RANK_LIMIT = 100

    MIN_SNAPSHOT_RANKS = 20

    KST = ZoneInfo("Asia/Seoul")

    def __init__(
        self,
        db: Session,
    ):
        self.db = db

    @staticmethod
    def _clean_text(
        value: str | None,
    ) -> str:
        if not value:
            return ""

        text = re.sub(
            r"<[^>]+>",
            " ",
            value,
        )

        text = html.unescape(text)

        return re.sub(
            r"\s+",
            " ",
            text,
        ).strip()

    @staticmethod
    def _sentiment_score(
        text: str,
    ) -> float:
        lowered = text.lower()

        positive = (
            "호실적",
            "상승",
            "급등",
            "수주",
            "흑자",
            "증가",
            "최대",
            "성장",
            "상향",
            "매수",
            "신고가",
            "개선",
            "돌파",
            "호재",
        )

        negative = (
            "적자",
            "하락",
            "급락",
            "감소",
            "하향",
            "매도",
            "유상증자",
            "감자",
            "소송",
            "제재",
            "부진",
            "악재",
            "횡령",
            "배임",
            "상장폐지",
        )

        score = 0.0

        for keyword in positive:
            if keyword in lowered:
                score += 15.0

        for keyword in negative:
            if keyword in lowered:
                score -= 15.0

        return max(
            -100.0,
            min(100.0, score),
        )

    @staticmethod
    def _importance_score(
        text: str,
    ) -> float:
        lowered = text.lower()
        score = 35.0

        high = (
            "실적",
            "영업이익",
            "매출",
            "수주",
            "계약",
            "합병",
            "분할",
            "유상증자",
            "무상증자",
            "자사주",
            "배당",
            "소송",
            "횡령",
            "배임",
            "상장폐지",
            "공급계약",
            "최대주주",
            "전환사채",
            "신주인수권",
        )

        medium = (
            "목표가",
            "투자의견",
            "급등",
            "급락",
            "신고가",
            "하락",
            "상승",
        )

        for keyword in high:
            if keyword in lowered:
                score += 18.0

        for keyword in medium:
            if keyword in lowered:
                score += 8.0

        return max(
            0.0,
            min(100.0, score),
        )

    @staticmethod
    def _disclosure_impact(
        report_name: str,
    ) -> tuple[float, float]:
        importance = (
            MarketContextService
            ._importance_score(
                report_name
            )
        )

        sentiment = (
            MarketContextService
            ._sentiment_score(
                report_name
            )
        )

        impact = min(
            100.0,
            importance * 0.7
            + abs(sentiment) * 0.3,
        )

        return importance, impact

    async def sync_news(
        self,
        *,
        stock_code: str | None = None,
        limit: int = 100,
    ) -> int:
        stock: Stock | None = None

        if stock_code:
            stock = self.db.get(
                Stock,
                stock_code,
            )

            if stock is None:
                return 0

            query = (
                f'"{stock.name}" '
                f"주식 증권"
            )
        else:
            query = (
                "코스피 코스닥 "
                "주식 증권"
            )

        params = {
            "q": query,
            "hl": "ko",
            "gl": "KR",
            "ceid": "KR:ko",
        }

        async with httpx.AsyncClient(
            timeout=20.0,
            follow_redirects=True,
        ) as client:
            response = await client.get(
                "https://news.google.com/"
                "rss/search",
                params=params,
            )

            response.raise_for_status()

        root = ElementTree.fromstring(
            response.content
        )

        items = root.findall(
            "./channel/item"
        )[:limit]

        saved = 0

        for item in items:
            title = self._clean_text(
                item.findtext("title")
            )

            url = (
                item.findtext("link")
                or ""
            ).strip()

            description = self._clean_text(
                item.findtext(
                    "description"
                )
            )

            guid = (
                item.findtext("guid")
                or ""
            ).strip() or None

            source_node = item.find(
                "source"
            )

            source = self._clean_text(
                source_node.text
                if source_node is not None
                else ""
            ) or "Google News"

            published_at = None

            pub_date = (
                item.findtext(
                    "pubDate"
                )
                or ""
            ).strip()

            if pub_date:
                try:
                    parsed = (
                        parsedate_to_datetime(
                            pub_date
                        )
                    )

                    published_at = (
                        parsed.replace(
                            tzinfo=None
                        )
                    )
                except (
                    TypeError,
                    ValueError,
                    OverflowError,
                ):
                    published_at = None

            if not title or not url:
                continue

            digest = hashlib.sha256(
                f"{title}|{url}".encode(
                    "utf-8"
                )
            ).hexdigest()

            article = self.db.scalar(
                select(
                    NewsArticle
                )
                .where(
                    NewsArticle.content_hash
                    == digest
                )
                .limit(1)
            )

            if article is None:
                article = NewsArticle(
                    source=source,
                    external_id=guid,
                    title=title,
                    url=url,
                    canonical_url=url,
                    published_at=published_at,
                    body_text=description,
                    language="ko",
                    content_hash=digest,
                )

                self.db.add(article)
                self.db.flush()

                saved += 1

            if stock is not None:
                exists = self.db.scalar(
                    select(
                        StockNews.id
                    )
                    .where(
                        StockNews.stock_code
                        == stock.code,
                        StockNews.article_id
                        == article.id,
                    )
                    .limit(1)
                )

                if exists is None:
                    self.db.add(
                        StockNews(
                            stock_code=(
                                stock.code
                            ),
                            article_id=(
                                article.id
                            ),
                            relevance_score=(
                                100.0
                            ),
                            matched_by=(
                                "stock-name"
                            ),
                        )
                    )

        self.db.commit()

        return saved

    def get_news(
        self,
        *,
        stock_code: str | None,
        limit: int,
    ) -> list[dict]:
        if stock_code:
            stmt = (
                select(NewsArticle)
                .join(
                    StockNews,
                    StockNews.article_id
                    == NewsArticle.id,
                )
                .where(
                    StockNews.stock_code
                    == stock_code
                )
                .order_by(
                    NewsArticle
                    .published_at
                    .desc()
                    .nullslast(),
                    NewsArticle.id.desc(),
                )
                .limit(limit)
            )
        else:
            stmt = (
                select(NewsArticle)
                .order_by(
                    NewsArticle
                    .published_at
                    .desc()
                    .nullslast(),
                    NewsArticle.id.desc(),
                )
                .limit(limit)
            )

        articles = list(
            self.db.scalars(
                stmt
            ).all()
        )

        result: list[dict] = []

        for article in articles:
            related_codes = list(
                self.db.scalars(
                    select(
                        StockNews.stock_code
                    ).where(
                        StockNews.article_id
                        == article.id
                    )
                ).all()
            )

            combined = (
                f"{article.title} "
                f"{article.body_text or ''}"
            )

            result.append(
                {
                    "id": str(
                        article.id
                    ),
                    "title": (
                        article.title
                    ),
                    "summary": (
                        self._clean_text(
                            article.body_text
                        )[:500]
                    ),
                    "source": (
                        article.source
                    ),
                    "publishedAt": (
                        article
                        .published_at
                        .isoformat()
                        if article.published_at
                        else None
                    ),
                    "sentimentScore": (
                        self._sentiment_score(
                            combined
                        )
                    ),
                    "importanceScore": (
                        self._importance_score(
                            combined
                        )
                    ),
                    "relatedStockCodes": (
                        related_codes
                    ),
                    "url": article.url,
                }
            )

        return result

    async def sync_disclosures(
        self,
        *,
        stock_code: str | None = None,
        limit: int = 100,
    ) -> int:
        corp_code: str | None = None

        if stock_code:
            stock = self.db.get(
                Stock,
                stock_code,
            )

            if (
                stock is None
                or not stock.corp_code
            ):
                return 0

            corp_code = (
                stock.corp_code
            )

        end_date = date.today()

        start_date = (
            end_date
            - timedelta(
                days=(
                    90
                    if stock_code
                    else 14
                )
            )
        )

        rows = (
            await dart_client
            .fetch_disclosures(
                corp_code=corp_code,
                begin_date=start_date,
                end_date=end_date,
                page_count=min(
                    100,
                    max(1, limit),
                ),
            )
        )

        saved = 0

        for row in rows[:limit]:
            receipt_no = str(
                row.get(
                    "rcept_no",
                    "",
                )
            ).strip()

            if not receipt_no:
                continue

            disclosure = (
                self.db.scalar(
                    select(
                        Disclosure
                    )
                    .where(
                        Disclosure
                        .receipt_no
                        == receipt_no
                    )
                    .limit(1)
                )
            )

            if disclosure is not None:
                continue

            row_corp_code = (
                str(
                    row.get(
                        "corp_code",
                        "",
                    )
                ).strip()
                or None
            )

            related_stock = None

            if row_corp_code:
                related_stock = (
                    self.db.scalar(
                        select(Stock)
                        .where(
                            Stock.corp_code
                            == row_corp_code
                        )
                        .limit(1)
                    )
                )

            receipt_date = None

            receipt_date_text = str(
                row.get(
                    "rcept_dt",
                    "",
                )
            ).strip()

            if receipt_date_text:
                try:
                    receipt_date = (
                        datetime.strptime(
                            receipt_date_text,
                            "%Y%m%d",
                        ).date()
                    )
                except ValueError:
                    receipt_date = None

            report_name = str(
                row.get(
                    "report_nm",
                    "",
                )
            ).strip()

            self.db.add(
                Disclosure(
                    stock_code=(
                        related_stock.code
                        if related_stock
                        else stock_code
                    ),
                    corp_code=(
                        row_corp_code
                    ),
                    receipt_no=(
                        receipt_no
                    ),
                    report_name=(
                        report_name
                        or "공시"
                    ),
                    filer_name=(
                        str(
                            row.get(
                                "flr_nm",
                                "",
                            )
                        ).strip()
                        or None
                    ),
                    receipt_date=(
                        receipt_date
                    ),
                    disclosure_type=(
                        str(
                            row.get(
                                "corp_cls",
                                "",
                            )
                        ).strip()
                        or None
                    ),
                    url=(
                        "https://dart.fss.or.kr/"
                        "dsaf001/main.do?"
                        f"rcpNo={receipt_no}"
                    ),
                    raw_json=dict(row),
                )
            )

            saved += 1

        self.db.commit()

        return saved
    
    async def sync_disclosures_history(
        self,
        *,
        stock_code: str,
        begin_date: date,
        end_date: date,
    ) -> dict:
        stock = self.db.get(
            Stock,
            stock_code,
        )

        if stock is None:
            raise ValueError(
                "등록되지 않은 종목입니다: "
                f"{stock_code}"
            )

        if not stock.corp_code:
            raise ValueError(
                "DART corp_code가 없습니다: "
                f"{stock_code}"
            )

        rows = (
            await dart_client
            .fetch_disclosures_all(
                corp_code=(
                    stock.corp_code
                ),
                begin_date=(
                    begin_date
                ),
                end_date=(
                    end_date
                ),
                page_count=100,
            )
        )

        saved = 0
        duplicate = 0
        invalid = 0

        for row in rows:
            receipt_no = str(
                row.get(
                    "rcept_no",
                    "",
                )
            ).strip()

            if not receipt_no:
                invalid += 1
                continue

            exists = self.db.scalar(
                select(
                    Disclosure.id
                )
                .where(
                    Disclosure.receipt_no
                    == receipt_no
                )
                .limit(1)
            )

            if exists is not None:
                duplicate += 1
                continue

            receipt_date = None

            receipt_date_text = str(
                row.get(
                    "rcept_dt",
                    "",
                )
            ).strip()

            if receipt_date_text:
                try:
                    receipt_date = (
                        datetime.strptime(
                            receipt_date_text,
                            "%Y%m%d",
                        )
                        .date()
                    )
                except ValueError:
                    receipt_date = None

            report_name = str(
                row.get(
                    "report_nm",
                    "",
                )
            ).strip()

            row_corp_code = str(
                row.get(
                    "corp_code",
                    "",
                )
            ).strip()

            self.db.add(
                Disclosure(
                    stock_code=(
                        stock.code
                    ),

                    corp_code=(
                        row_corp_code
                        or stock.corp_code
                    ),

                    receipt_no=(
                        receipt_no
                    ),

                    report_name=(
                        report_name
                        or "공시"
                    ),

                    filer_name=(
                        str(
                            row.get(
                                "flr_nm",
                                "",
                            )
                        ).strip()
                        or None
                    ),

                    receipt_date=(
                        receipt_date
                    ),

                    disclosure_type=(
                        str(
                            row.get(
                                "corp_cls",
                                "",
                            )
                        ).strip()
                        or None
                    ),

                    url=(
                        "https://dart.fss.or.kr/"
                        "dsaf001/main.do?"
                        f"rcpNo={receipt_no}"
                    ),

                    raw_json=dict(
                        row
                    ),
                )
            )

            saved += 1

        self.db.commit()

        return {
            "stock_code":
                stock_code,

            "fetched":
                len(
                    rows
                ),

            "saved":
                saved,

            "duplicate":
                duplicate,

            "invalid":
                invalid,
        }

    def get_disclosures(
        self,
        *,
        stock_code: str | None,
        limit: int,
    ) -> list[dict]:
        stmt = (
            select(
                Disclosure,
                Stock.name,
            )
            .outerjoin(
                Stock,
                Stock.code
                == Disclosure.stock_code,
            )
        )

        if stock_code:
            stmt = stmt.where(
                Disclosure.stock_code
                == stock_code
            )

        stmt = (
            stmt
            .order_by(
                Disclosure
                .receipt_date
                .desc()
                .nullslast(),
                Disclosure.id.desc(),
            )
            .limit(limit)
        )

        result: list[dict] = []

        for (
            disclosure,
            stock_name,
        ) in self.db.execute(
            stmt
        ).all():
            (
                importance,
                impact,
            ) = (
                self._disclosure_impact(
                    disclosure.report_name
                )
            )

            result.append(
                {
                    "id": str(
                        disclosure.id
                    ),
                    "stockCode": (
                        disclosure
                        .stock_code
                        or ""
                    ),
                    "stockName": (
                        stock_name
                        or disclosure
                        .filer_name
                        or ""
                    ),
                    "title": (
                        disclosure
                        .report_name
                    ),
                    "reportName": (
                        disclosure
                        .report_name
                    ),
                    "receivedAt": (
                        datetime.combine(
                            disclosure
                            .receipt_date,
                            datetime.min.time(),
                        ).isoformat()
                        if disclosure
                        .receipt_date
                        else None
                    ),
                    "importanceScore": (
                        importance
                    ),
                    "impactScore": (
                        impact
                    ),
                    "summary": (
                        disclosure
                        .filer_name
                        or ""
                    ),
                    "url": (
                        disclosure.url
                    ),
                }
            )

        return result

    def record_rankings(
        self,
        rankings: list,
        *,
        as_of_date: date | None = None,
    ) -> None:
        if not rankings:
            return

        available_ranks = {
            int(
                ranking.rank
            )
            for ranking
            in rankings
        }

        required_ranks = set(
            range(
                1,
                self.MIN_SNAPSHOT_RANKS
                + 1,
            )
        )

        if not required_ranks.issubset(
            available_ranks
        ):
            return

        snapshot_date = (
            as_of_date
            or datetime.now(
                self.KST
            ).date()
        )

        snapshot = self.db.scalar(
            select(
                RankingSnapshot
            )
            .where(
                RankingSnapshot
                .ranking_version
                == self.RANKING_VERSION,
                RankingSnapshot
                .as_of_date
                == snapshot_date,
                RankingSnapshot
                .horizon_days
                == self.PRIMARY_HORIZON_DAYS,
                RankingSnapshot
                .universe
                == "KRX",
            )
            .limit(1)
        )

        if snapshot is not None:
            print(
                "[RANKING][SNAPSHOT-WRITE][SKIP] "
                f"version={self.RANKING_VERSION} "
                f"date={snapshot_date} "
                f"snapshotId={snapshot.id}",
                flush=True,
            )
            return

        print(
            "[RANKING][SNAPSHOT-WRITE][START] "
            f"version={self.RANKING_VERSION} "
            f"date={snapshot_date} "
            f"count={len(rankings)} "
            f"quality={sum(1 for ranking in rankings if ranking.qualityScore is not None)} "
            f"value={sum(1 for ranking in rankings if ranking.valueScore is not None)} "
            f"samples={[{'stock': ranking.stockCode, 'quality': ranking.qualityScore, 'value': ranking.valueScore, 'total': ranking.totalScore} for ranking in rankings[:5]]}",
            flush=True,
        )

        snapshot = RankingSnapshot(
            ranking_version=(
                self.RANKING_VERSION
            ),
            as_of_date=snapshot_date,
            horizon_days=(
                self.PRIMARY_HORIZON_DAYS
            ),
            universe="KRX",
            weights_json={
                "quality": 0.25,
                "growth": 0.20,
                "value": 0.20,
                "financial_health": 0.15,
                "relative_strength": 0.10,
                "flow": 0.05,
                "ml": 0.05,
            },
            metadata_json={
                "ranker": (
                    self.RANKING_VERSION
                ),
                "score_policy": (
                    "multi_factor_investment_attractiveness"
                ),
                "performance_tracking": (
                    "top10_20_60_trading_days"
                ),
            },
        )

        self.db.add(snapshot)
        self.db.flush()

        for ranking in rankings:
            ranking_item = RankingItem(
                snapshot_id=snapshot.id,
                stock_code=(
                    ranking.stockCode
                ),
                rank=ranking.rank,
                total_score=(
                    ranking.totalScore
                ),
                financial_score=(
                    ranking.financialScore
                ),
                ml_score=(
                    ranking.mlScore
                ),
                valuation_score=(
                    ranking.valueScore
                    or 0.0
                ),
                momentum_score=0.0,
                predicted_return=None,
                upside_probability=None,
                score_components_json={
                    "sector_name": (
                        ranking.sectorName
                    ),
                    "sector_rank": (
                        ranking.sectorRank
                    ),
                    "sector_peer_count": (
                        ranking.sectorPeerCount
                    ),

                    "quality": (
                        ranking.qualityScore
                    ),
                    "growth": (
                        ranking.growthScore
                    ),
                    "value": (
                        ranking.valueScore
                    ),
                    "financial_health": (
                        ranking.financialHealthScore
                    ),
                    "relative_strength": (
                        ranking.relativeStrengthScore
                    ),
                    "flow": (
                        ranking.flowScore
                    ),
                    "ml": (
                        ranking.mlScore
                    ),
                    "data_coverage": (
                        ranking.dataCoverage
                    ),
                    "financial_reference": (
                        float(
                            ranking.financialScore
                        )
                    ),
                },
                rationale=(
                    "동종업종 대비 수익성·성장성·"
                    "밸류에이션·재무안정성/현금흐름을 "
                    "중심으로 상대성과·수급·단기 ML "
                    "신호를 보조 반영한 장기 투자 순위"
                ),
            )

            self.db.add(
                ranking_item
            )
            self.db.flush()

            if (
                ranking.rank
                > self.PERFORMANCE_RANK_LIMIT
            ):
                continue

            for horizon_days in (
                self.PERFORMANCE_HORIZONS
            ):
                performance = self.db.scalar(
                    select(
                        RecommendationPerformance
                    )
                    .where(
                        RecommendationPerformance
                        .ranking_item_id
                        == ranking_item.id,

                        RecommendationPerformance
                        .horizon_days
                        == horizon_days,
                    )
                    .limit(
                        1
                    )
                )

                if performance is None:
                    performance = (
                        RecommendationPerformance(
                            ranking_item_id=(
                                ranking_item.id
                            ),

                            stock_code=(
                                ranking.stockCode
                            ),

                            recommendation_date=(
                                snapshot_date
                            ),

                            horizon_days=(
                                horizon_days
                            ),
                        )
                    )

                    self.db.add(
                        performance
                    )

                performance.ranking_item_id = (
                    ranking_item.id
                )

                performance.entry_price = (
                    ranking.currentPrice
                    if ranking.currentPrice
                    > 0.0
                    else None
                )

                performance.target_date = None
                performance.exit_price = None

                performance.predicted_return = None
                performance.actual_return = None
                performance.benchmark_return = None
                performance.excess_return = None

                performance.direction_correct = None
                performance.evaluated_at = None

                performance.metadata_json = {
                    "ranking_version":
                        self.RANKING_VERSION,

                    "rank":
                        int(
                            ranking.rank
                        ),

                    "composite_score":
                        float(
                            ranking.totalScore
                        ),

                    "horizon_trading_days":
                        horizon_days,

                    "tracking":
                        (
                            "long_term_"
                            "ranking_backtest"
                        ),
                }

        self.db.commit()

    @staticmethod
    def _benchmark_index_code(
        market: str | None,
    ) -> str | None:
        normalized = (
            market
            or ""
        ).upper()

        if normalized == "KOSPI":
            return "0001"

        if normalized == "KOSDAQ":
            return "1001"

        return None

    def _benchmark_return_for_period(
        self,
        *,
        stock_code: str,
        entry_date: date,
        exit_date: date,
    ) -> tuple[
        float | None,
        str | None,
    ]:
        stock = self.db.scalar(
            select(
                Stock
            )
            .where(
                Stock.code
                == stock_code
            )
            .limit(1)
        )

        if stock is None:
            return (
                None,
                None,
            )

        index_code = (
            self._benchmark_index_code(
                stock.market
            )
        )

        if index_code is None:
            return (
                None,
                None,
            )

        entry_row = self.db.scalar(
            select(
                MarketIndexPrice
            )
            .where(
                MarketIndexPrice.index_code
                == index_code,
                MarketIndexPrice.trade_date
                <= entry_date,
            )
            .order_by(
                MarketIndexPrice.trade_date.desc()
            )
            .limit(1)
        )

        if (
            entry_row is None
            or entry_row.close is None
            or float(
                entry_row.close
            ) <= 0.0
        ):
            return (
                None,
                index_code,
            )

        exit_row = self.db.scalar(
            select(
                MarketIndexPrice
            )
            .where(
                MarketIndexPrice.index_code
                == index_code,
                MarketIndexPrice.trade_date
                > entry_row.trade_date,
                MarketIndexPrice.trade_date
                <= exit_date,
            )
            .order_by(
                MarketIndexPrice.trade_date.desc()
            )
            .limit(1)
        )

        if (
            exit_row is None
            or exit_row.close is None
        ):
            return (
                None,
                index_code,
            )

        benchmark_return = (
            float(
                exit_row.close
            )
            / float(
                entry_row.close
            )
            - 1.0
        ) * 100.0

        return (
            benchmark_return,
            index_code,
        )

    def evaluate_performance(
        self,
    ) -> None:
        today = datetime.now(
            self.KST
        ).date()

        pending = []

        for horizon_days in (
            self.PERFORMANCE_HORIZONS
        ):
            earliest_possible_date = (
                today
                - timedelta(
                    days=horizon_days
                )
            )

            horizon_pending = list(
                self.db.scalars(
                    select(
                        RecommendationPerformance
                    )
                    .join(
                        RankingItem,
                        RankingItem.id
                        == RecommendationPerformance
                        .ranking_item_id,
                    )
                    .join(
                        RankingSnapshot,
                        RankingSnapshot.id
                        == RankingItem.snapshot_id,
                    )
                    .where(
                        RankingSnapshot
                        .ranking_version
                        == self.RANKING_VERSION,

                        RecommendationPerformance
                        .horizon_days
                        == horizon_days,

                        RecommendationPerformance
                        .recommendation_date
                        <= earliest_possible_date,

                        RecommendationPerformance
                        .evaluated_at
                        .is_(None),
                    )
                    .order_by(
                        RecommendationPerformance
                        .recommendation_date
                        .asc(),

                        RankingItem.rank.asc(),
                    )
                    .limit(
                        2000
                    )
                ).all()
            )

            pending.extend(
                horizon_pending
            )

        changed = False

        for item in pending:
            if (
                not item.entry_price
                or item.entry_price <= 0
            ):
                continue

            prices = list(
                self.db.scalars(
                    select(
                        StockPrice
                    )
                    .where(
                        StockPrice.stock_code
                        == item.stock_code,
                        StockPrice.trade_date
                        > item.recommendation_date,
                    )
                    .order_by(
                        StockPrice.trade_date.asc()
                    )
                    .limit(
                        item.horizon_days
                    )
                ).all()
            )
            # 해당 horizon만큼의 실제 미래 거래일이
            # 모두 존재할 때만 평가합니다.
            if (
                len(prices)
                < item.horizon_days
            ):
                continue

            exit_price = float(
                prices[-1].close
            )

            actual_return = (
                exit_price
                / float(
                    item.entry_price
                )
                - 1.0
            ) * 100.0

            (
                benchmark_return,
                benchmark_index_code,
            ) = (
                self._benchmark_return_for_period(
                    stock_code=(
                        item.stock_code
                    ),
                    entry_date=(
                        item.recommendation_date
                    ),
                    exit_date=(
                        prices[-1].trade_date
                    ),
                )
            )

            excess_return = (
                actual_return
                - benchmark_return
                if benchmark_return
                is not None
                else None
            )

            item.target_date = (
                prices[-1].trade_date
            )

            item.exit_price = (
                exit_price
            )

            item.actual_return = (
                actual_return
            )

            item.predicted_return = None

            item.benchmark_return = (
                benchmark_return
            )

            item.excess_return = (
                excess_return
            )

            item.direction_correct = None

            item.evaluated_at = (
                datetime.utcnow()
            )

            metadata = dict(
                item.metadata_json
                or {}
            )

            metadata.update(
                {
                    "exit_trade_date": (
                        prices[-1]
                        .trade_date
                        .isoformat()
                    ),
                    "evaluated_trading_days": (
                        item.horizon_days
                    ),
                    "benchmark_index_code": (
                        benchmark_index_code
                    ),
                    "benchmark_return": (
                        benchmark_return
                    ),
                    "excess_return": (
                        excess_return
                    ),
                }
            )

            item.metadata_json = (
                metadata
            )

            changed = True

        if changed:
            self.db.commit()

    def get_performance(
        self,
        *,
        limit: int,
    ) -> dict:
        self.evaluate_performance()

        base_filter = (
            RankingSnapshot
            .ranking_version
            == self.RANKING_VERSION
        )

        total_count = int(
            self.db.scalar(
                select(
                    func.count(
                        RecommendationPerformance.id
                    )
                )
                .join(
                    RankingItem,
                    RankingItem.id
                    == RecommendationPerformance
                    .ranking_item_id,
                )
                .join(
                    RankingSnapshot,
                    RankingSnapshot.id
                    == RankingItem.snapshot_id,
                )
                .where(
                    base_filter
                )
            )
            or 0
        )

        evaluated_returns = [
            float(
                value
            )
            for value
            in self.db.scalars(
                select(
                    RecommendationPerformance
                    .actual_return
                )
                .join(
                    RankingItem,
                    RankingItem.id
                    == RecommendationPerformance
                    .ranking_item_id,
                )
                .join(
                    RankingSnapshot,
                    RankingSnapshot.id
                    == RankingItem.snapshot_id,
                )
                .where(
                    base_filter,
                    RecommendationPerformance
                    .actual_return
                    .is_not(None),
                )
            ).all()
            if value is not None
        ]

        evaluated_count = len(
            evaluated_returns
        )

        pending_count = (
            total_count
            - evaluated_count
        )

        evaluated_excess_returns = [
            float(
                value
            )
            for value
            in self.db.scalars(
                select(
                    RecommendationPerformance
                    .excess_return
                )
                .join(
                    RankingItem,
                    RankingItem.id
                    == RecommendationPerformance
                    .ranking_item_id,
                )
                .join(
                    RankingSnapshot,
                    RankingSnapshot.id
                    == RankingItem.snapshot_id,
                )
                .where(
                    base_filter,
                    RecommendationPerformance
                    .excess_return
                    .is_not(None),
                )
            ).all()
            if value is not None
        ]

        average_return = 0.0
        average_excess_return = 0.0
        win_rate = 0.0
        hit_rate = 0.0

        if evaluated_returns:
            average_return = (
                sum(
                    evaluated_returns
                )
                / evaluated_count
            )

            win_rate = (
                sum(
                    1
                    for value
                    in evaluated_returns
                    if value > 0.0
                )
                / evaluated_count
                * 100.0
            )

        if evaluated_excess_returns:
            average_excess_return = (
                sum(
                    evaluated_excess_returns
                )
                / len(
                    evaluated_excess_returns
                )
            )

            hit_rate = (
                sum(
                    1
                    for value
                    in evaluated_excess_returns
                    if value > 0.0
                )
                / len(
                    evaluated_excess_returns
                )
                * 100.0
            )

        stmt = (
            select(
                RecommendationPerformance,
                Stock.name,
                RankingItem.total_score,
                RankingItem.rank,
            )
            .join(
                Stock,
                Stock.code
                == RecommendationPerformance
                .stock_code,
            )
            .join(
                RankingItem,
                RankingItem.id
                == RecommendationPerformance
                .ranking_item_id,
            )
            .join(
                RankingSnapshot,
                RankingSnapshot.id
                == RankingItem.snapshot_id,
            )
            .where(
                base_filter
            )
            .order_by(
                RecommendationPerformance
                .recommendation_date
                .desc(),
                RankingItem.rank.asc(),
            )
            .limit(
                limit
            )
        )

        rows = (
            self.db.execute(
                stmt
            ).all()
        )

        records = []

        for (
            performance,
            stock_name,
            total_score,
            rank,
        ) in rows:
            records.append(
                {
                    "stockCode": (
                        performance
                        .stock_code
                    ),

                    "stockName": (
                        stock_name
                        or ""
                    ),

                    "rank": int(
                        rank
                    ),

                    "recommendedAt": (
                        datetime.combine(
                            performance
                            .recommendation_date,
                            datetime.min.time(),
                        ).isoformat()
                    ),

                    "totalScore": float(
                        total_score
                        or 0.0
                    ),

                    # Flutter 구버전 호환 필드.
                    # 실제 예측수익률로 사용하지 않습니다.
                    "predictedReturn":
                        0.0,

                    "actualReturn": (
                        float(
                            performance
                            .actual_return
                        )
                        if performance
                        .actual_return
                        is not None
                        else None
                    ),

                    "excessReturn": (
                        float(
                            performance
                            .excess_return
                        )
                        if performance
                        .excess_return
                        is not None
                        else None
                    ),

                    "directionCorrect":
                        None,

                    "status": (
                        "evaluated"
                        if performance
                        .actual_return
                        is not None
                        else "pending"
                    ),

                    "horizonTradingDays": (
                        performance
                        .horizon_days
                    ),
                }
            )

        return {
            "rankingVersion": (
                self.RANKING_VERSION
            ),

            "trackedTopRanks": (
                self.PERFORMANCE_RANK_LIMIT
            ),

            "horizonTradingDays":
                5,

            # 아래 통계는 limit과 관계없이
            # 전체 누적 추천 기준입니다.
            "totalRecommendations":
                total_count,

            "evaluatedRecommendations":
                evaluated_count,

            "pendingRecommendations":
                pending_count,

            # 모델은 방향 분류기가 아니라
            # 횡단면 랭커이므로 hitRate는
            # 기준지수 초과 비율로 사용합니다.
            "hitRate":
                hit_rate,

            "averageReturn5d": (
                average_return
            ),

            "averageExcessReturn5d": (
                average_excess_return
            ),

            "winRate": (
                win_rate
            ),

            # records만 요청 limit만큼 반환합니다.
            "records":
                records,
        }