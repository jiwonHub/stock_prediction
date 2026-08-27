from __future__ import annotations

import hashlib
import html
import re
from datetime import date, datetime, timedelta
from email.utils import parsedate_to_datetime
from xml.etree import ElementTree

import httpx
from sqlalchemy import select
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
from app.models.stock import Stock
from app.models.stock_price import StockPrice


class MarketContextService:
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
    ) -> None:
        if not rankings:
            return

        today = date.today()

        snapshot = self.db.scalar(
            select(
                RankingSnapshot
            )
            .where(
                RankingSnapshot
                .ranking_version
                == "phase4-v1",
                RankingSnapshot
                .as_of_date
                == today,
                RankingSnapshot
                .horizon_days
                == 5,
                RankingSnapshot
                .universe
                == "KRX",
            )
            .limit(1)
        )

        if snapshot is not None:
            return

        snapshot = RankingSnapshot(
            ranking_version=(
                "phase4-v1"
            ),
            as_of_date=today,
            horizon_days=5,
            universe="KRX",
            weights_json={
                "financial": 0.45,
                "ml": 0.55,
            },
        )

        self.db.add(snapshot)
        self.db.flush()

        for ranking in rankings:
            ranking_item = (
                RankingItem(
                    snapshot_id=(
                        snapshot.id
                    ),
                    stock_code=(
                        ranking.stockCode
                    ),
                    rank=(
                        ranking.rank
                    ),
                    total_score=(
                        ranking.totalScore
                    ),
                    financial_score=(
                        ranking
                        .financialScore
                    ),
                    ml_score=max(
                        0.0,
                        min(
                            100.0,
                            ranking
                            .upsideProbability,
                        ),
                    ),
                    predicted_return=(
                        ranking
                        .predictedReturn
                    ),
                    upside_probability=(
                        ranking
                        .upsideProbability
                    ),
                )
            )

            self.db.add(
                ranking_item
            )

            self.db.flush()

            self.db.add(
                RecommendationPerformance(
                    ranking_item_id=(
                        ranking_item.id
                    ),
                    stock_code=(
                        ranking.stockCode
                    ),
                    recommendation_date=(
                        today
                    ),
                    horizon_days=5,
                    entry_price=(
                        ranking.currentPrice
                        if ranking
                        .currentPrice
                        > 0.0
                        else None
                    ),
                    predicted_return=(
                        ranking
                        .predictedReturn
                    ),
                    target_date=(
                        today
                        + timedelta(
                            days=7
                        )
                    ),
                )
            )

        self.db.commit()

    def evaluate_performance(
        self,
    ) -> None:
        pending = list(
            self.db.scalars(
                select(
                    RecommendationPerformance
                )
                .where(
                    RecommendationPerformance
                    .evaluated_at
                    .is_(None)
                )
                .order_by(
                    RecommendationPerformance
                    .recommendation_date
                    .asc()
                )
                .limit(500)
            ).all()
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
                        > item
                        .recommendation_date,
                    )
                    .order_by(
                        StockPrice
                        .trade_date
                        .asc()
                    )
                    .limit(
                        item.horizon_days
                    )
                ).all()
            )

            if (
                len(prices)
                >= item.horizon_days
            ):
                exit_price = float(
                    prices[-1].close
                )

            elif (
                item.target_date
                is not None
                and date.today()
                >= item.target_date
            ):
                stock = self.db.get(
                    Stock,
                    item.stock_code,
                )

                if (
                    stock is None
                    or not stock
                    .current_price
                    or stock
                    .current_price
                    <= 0
                ):
                    continue

                exit_price = float(
                    stock.current_price
                )

            else:
                continue

            actual_return = (
                exit_price
                / float(
                    item.entry_price
                )
                - 1.0
            ) * 100.0

            predicted = float(
                item.predicted_return
                or 0.0
            )

            item.exit_price = (
                exit_price
            )

            item.actual_return = (
                actual_return
            )

            item.excess_return = (
                actual_return
            )

            item.direction_correct = (
                (
                    predicted >= 0.0
                    and actual_return
                    >= 0.0
                )
                or (
                    predicted < 0.0
                    and actual_return
                    < 0.0
                )
            )

            item.evaluated_at = (
                datetime.utcnow()
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

        stmt = (
            select(
                RecommendationPerformance,
                Stock.name,
                RankingItem.total_score,
            )
            .join(
                Stock,
                Stock.code
                == RecommendationPerformance
                .stock_code,
            )
            .outerjoin(
                RankingItem,
                RankingItem.id
                == RecommendationPerformance
                .ranking_item_id,
            )
            .order_by(
                RecommendationPerformance
                .recommendation_date
                .desc(),
                RecommendationPerformance
                .id
                .desc(),
            )
            .limit(limit)
        )

        rows = self.db.execute(
            stmt
        ).all()

        evaluated = [
            row[0]
            for row in rows
            if row[0].actual_return
            is not None
        ]

        hit_rate = 0.0
        average_return = 0.0
        average_excess = 0.0
        win_rate = 0.0

        if evaluated:
            hit_rate = (
                sum(
                    1
                    for item
                    in evaluated
                    if item
                    .direction_correct
                )
                / len(evaluated)
                * 100.0
            )

            average_return = (
                sum(
                    float(
                        item.actual_return
                        or 0.0
                    )
                    for item
                    in evaluated
                )
                / len(evaluated)
            )

            average_excess = (
                sum(
                    float(
                        item.excess_return
                        or 0.0
                    )
                    for item
                    in evaluated
                )
                / len(evaluated)
            )

            win_rate = (
                sum(
                    1
                    for item
                    in evaluated
                    if float(
                        item.actual_return
                        or 0.0
                    ) > 0.0
                )
                / len(evaluated)
                * 100.0
            )

        records = []

        for (
            performance,
            stock_name,
            total_score,
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
                    "recommendedAt": (
                        datetime.combine(
                            performance
                            .recommendation_date,
                            datetime
                            .min
                            .time(),
                        ).isoformat()
                    ),
                    "totalScore": float(
                        total_score
                        or 0.0
                    ),
                    "predictedReturn": (
                        float(
                            performance
                            .predicted_return
                            or 0.0
                        )
                    ),
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
                    "directionCorrect": (
                        performance
                        .direction_correct
                    ),
                }
            )

        return {
            "totalRecommendations": (
                len(rows)
            ),
            "hitRate": hit_rate,
            "averageReturn5d": (
                average_return
            ),
            "averageExcessReturn5d": (
                average_excess
            ),
            "winRate": win_rate,
            "records": records,
        }