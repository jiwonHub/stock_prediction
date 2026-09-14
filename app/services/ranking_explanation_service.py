from __future__ import annotations

from sqlalchemy.orm import Session
from app.repositories.stock_repository import (
    StockRepository,
)
from app.services.market_context_service import (
    MarketContextService,
)
from app.services.ml_ranking_inference_service import (
    MlRankingInferenceService,
)

class RankingExplanationService:
    def __init__(
        self,
        db: Session,
    ):
        self.db = db

        self.repository = (
            StockRepository(
                db
            )
        )

    FEATURE_META = {
        "return_1d": (
            "1일 수익률",
            "percent",
        ),
        "return_5d": (
            "5일 수익률",
            "percent",
        ),
        "return_20d": (
            "20일 수익률",
            "percent",
        ),
        "return_60d": (
            "60일 수익률",
            "percent",
        ),
        "sma5_ratio": (
            "5일 이동평균 대비 위치",
            "percent",
        ),
        "sma20_ratio": (
            "20일 이동평균 대비 위치",
            "percent",
        ),
        "sma60_ratio": (
            "60일 이동평균 대비 위치",
            "percent",
        ),
        "volatility_20d": (
            "20일 변동성",
            "percent",
        ),
        "rsi_14": (
            "RSI",
            "normalized_100",
        ),
        "volume_ratio_20d": (
            "20일 평균 대비 거래량",
            "percent",
        ),
        "intraday_range": (
            "일중 가격 변동폭",
            "percent",
        ),
        "close_position": (
            "당일 종가 위치",
            "normalized_percent",
        ),
        "ema_12_gap": (
            "12일 EMA 대비 위치",
            "percent",
        ),
        "ema_26_gap": (
            "26일 EMA 대비 위치",
            "percent",
        ),
        "macd_line_ratio": (
            "MACD",
            "percent",
        ),
        "macd_signal_ratio": (
            "MACD 시그널",
            "percent",
        ),
        "macd_hist_ratio": (
            "MACD 히스토그램",
            "percent",
        ),
        "bollinger_position_20": (
            "볼린저밴드 내 가격 위치",
            "normalized_percent",
        ),
        "bollinger_width_20": (
            "볼린저밴드 폭",
            "percent",
        ),
        "atr_14_ratio": (
            "ATR 변동폭",
            "percent",
        ),
        "stochastic_k_14": (
            "스토캐스틱 %K",
            "normalized_100",
        ),
        "stochastic_d_3": (
            "스토캐스틱 %D",
            "normalized_100",
        ),
        "adx_14": (
            "ADX 추세 강도",
            "normalized_100",
        ),
        "volume_zscore_20": (
            "거래량 이상도",
            "zscore",
        ),
        "volume_trend_5_20": (
            "단기·중기 거래량 추세",
            "percent",
        ),
        "price_vs_high_60": (
            "60일 고점 대비 위치",
            "percent",
        ),
        "price_vs_low_60": (
            "60일 저점 대비 위치",
            "percent",
        ),
    }

    @classmethod
    def _format_feature_value(
        cls,
        *,
        feature_name: str,
        value: float,
    ) -> str:
        _, value_type = (
            cls.FEATURE_META.get(
                feature_name,
                (
                    feature_name,
                    "number",
                ),
            )
        )

        if value_type == "percent":
            return (
                f"{value * 100:+.2f}%"
            )

        if value_type == "normalized_percent":
            return (
                f"{value * 100:.1f}%"
            )

        if value_type == "normalized_100":
            return (
                f"{value * 100:.1f}"
            )

        if value_type == "zscore":
            return (
                f"{value:+.2f}σ"
            )

        return (
            f"{value:.4f}"
        )

    @classmethod
    def _build_factor(
        cls,
        *,
        feature_name: str,
        value: float,
        contribution: float,
        total_absolute_contribution: float,
    ) -> dict:
        label, _ = (
            cls.FEATURE_META.get(
                feature_name,
                (
                    feature_name,
                    "number",
                ),
            )
        )

        contribution_share = (
            abs(
                contribution
            )
            / total_absolute_contribution
            * 100.0
            if total_absolute_contribution
            > 0.0
            else 0.0
        )

        return {
            "feature":
                feature_name,

            "label":
                label,

            "value":
                float(
                    value
                ),

            "value_text":
                cls._format_feature_value(
                    feature_name=feature_name,
                    value=float(
                        value
                    ),
                ),

            "contribution":
                float(
                    contribution
                ),

            "contribution_share":
                round(
                    contribution_share,
                    2,
                ),
        }

    @classmethod
    def build_explanation(
        cls,
        score_row: dict,
        *,
        top_k: int = 3,
    ) -> dict:
        features = (
            score_row.get(
                "features"
            )
            or {}
        )

        contributions = (
            score_row.get(
                "feature_contributions"
            )
            or {}
        )

        total_absolute_contribution = sum(
            abs(
                float(
                    contribution
                )
            )
            for contribution
            in contributions.values()
        )

        positive_items = sorted(
            (
                (
                    feature_name,
                    float(
                        contribution
                    ),
                )
                for (
                    feature_name,
                    contribution,
                )
                in contributions.items()
                if float(
                    contribution
                ) > 0.0
            ),
            key=lambda item: item[1],
            reverse=True,
        )

        negative_items = sorted(
            (
                (
                    feature_name,
                    float(
                        contribution
                    ),
                )
                for (
                    feature_name,
                    contribution,
                )
                in contributions.items()
                if float(
                    contribution
                ) < 0.0
            ),
            key=lambda item: item[1],
        )

        positive_factors = [
            cls._build_factor(
                feature_name=feature_name,
                value=float(
                    features.get(
                        feature_name,
                        0.0,
                    )
                ),
                contribution=contribution,
                total_absolute_contribution=(
                    total_absolute_contribution
                ),
            )
            for (
                feature_name,
                contribution,
            )
            in positive_items[:top_k]
        ]

        negative_factors = [
            cls._build_factor(
                feature_name=feature_name,
                value=float(
                    features.get(
                        feature_name,
                        0.0,
                    )
                ),
                contribution=contribution,
                total_absolute_contribution=(
                    total_absolute_contribution
                ),
            )
            for (
                feature_name,
                contribution,
            )
            in negative_items[:top_k]
        ]

        positive_text = ", ".join(
            (
                f"{factor['label']} "
                f"({factor['value_text']})"
            )
            for factor
            in positive_factors
        )

        negative_text = ", ".join(
            (
                f"{factor['label']} "
                f"({factor['value_text']})"
            )
            for factor
            in negative_factors
        )

        summary_parts = []

        if positive_text:
            summary_parts.append(
                "모델 점수를 가장 크게 "
                f"끌어올린 요인은 "
                f"{positive_text}입니다."
            )

        if negative_text:
            summary_parts.append(
                "반대로 "
                f"{negative_text}은(는) "
                "모델 점수를 낮추는 방향으로 "
                "작용했습니다."
            )

        composite_row = (
            score_row.get(
                "composite_row"
            )
            or {}
        )

        return {
            "stock_code":
                score_row.get(
                    "stock_code"
                ),

            "stock_name":
                score_row.get(
                    "stock_name"
                ),

            "rank":
                composite_row.get(
                    "rank"
                ),

            "total_score":
                composite_row.get(
                    "total_score"
                ),

            "quality_score":
                composite_row.get(
                    "quality_score"
                ),

            "growth_score":
                composite_row.get(
                    "growth_score"
                ),

            "value_score":
                composite_row.get(
                    "value_score"
                ),

            "financial_health_score":
                composite_row.get(
                    "financial_health_score"
                ),

            "relative_strength_score":
                composite_row.get(
                    "relative_strength_score"
                ),

            "flow_score":
                composite_row.get(
                    "flow_score"
                ),

            "data_coverage":
                composite_row.get(
                    "data_coverage"
                ),

            "ml_rank":
                score_row.get(
                    "ml_rank"
                ),

            "ml_score":
                score_row.get(
                    "ml_score"
                ),

            "summary":
                " ".join(
                    summary_parts
                ),

            "positive_factors":
                positive_factors,

            "negative_factors":
                negative_factors,

            "method":
                "long_term_composite_with_ml_support",
        }

    def get_current_explanation(
        self,
        stock_code: str,
        *,
        top_k: int = 3,
    ) -> dict:
        (
            snapshot,
            ranking_item,
        ) = (
            self.repository
            .get_latest_ranking_snapshot_item_by_stock_code(
                stock_code=stock_code,
                ranking_version=(
                    MarketContextService
                    .RANKING_VERSION
                ),
                horizon_days=(
                    MarketContextService
                    .PRIMARY_HORIZON_DAYS
                ),
                universe="KRX",
            )
        )

        if (
            snapshot is None
            or ranking_item is None
        ):
            raise ValueError(
                "현재 장기 투자 TOP100 대상이 아닙니다: "
                f"{stock_code}"
            )

        stock = (
            self.repository
            .get_stock(
                stock_code
            )
        )

        components = dict(
            ranking_item
            .score_components_json
            or {}
        )

        composite_row = {
            "stock_code":
                stock_code,

            "stock_name":
                (
                    stock.name
                    if stock is not None
                    else stock_code
                ),

            "rank":
                int(
                    ranking_item.rank
                ),

            "total_score":
                float(
                    ranking_item.total_score
                ),

            "quality_score":
                components.get(
                    "quality"
                ),

            "growth_score":
                components.get(
                    "growth"
                ),

            "value_score":
                components.get(
                    "value"
                ),

            "financial_health_score":
                components.get(
                    "financial_health"
                ),

            "relative_strength_score":
                components.get(
                    "relative_strength"
                ),

            "flow_score":
                components.get(
                    "flow"
                ),

            "data_coverage":
                components.get(
                    "data_coverage"
                ),
        }

        ml_score = (
            components.get(
                "ml"
            )
        )

        if ml_score is None:
            ml_score = (
                ranking_item.ml_score
            )

        # ML 설명은 TOP500 전체가 아니라
        # 상세화면에서 보고 있는 종목 하나만 계산합니다.
        try:
            score_row = dict(
                MlRankingInferenceService(
                    self.db
                )
                .predict_stock(
                    stock_code
                )
            )

        except Exception:
            # ML 설명 생성이 실패해도
            # 장기 종합분석 상세화면 전체가 죽으면 안 됩니다.
            score_row = {
                "stock_code":
                    stock_code,

                "stock_name":
                    composite_row[
                        "stock_name"
                    ],

                "features":
                    {},

                "feature_contributions":
                    {},
            }

        # ML 순위는 Snapshot에 별도로 저장하지 않으므로
        # 억지로 현재 Universe 전체를 다시 계산하지 않습니다.
        score_row[
            "ml_rank"
        ] = None

        # 실제 종합랭킹 계산 당시 사용한 ML 점수를 사용합니다.
        score_row[
            "ml_score"
        ] = (
            float(
                ml_score
            )
            if ml_score is not None
            else None
        )

        score_row[
            "composite_row"
        ] = composite_row

        return self.build_explanation(
            score_row,
            top_k=top_k,
        )