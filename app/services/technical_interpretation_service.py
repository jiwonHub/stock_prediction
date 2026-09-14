from __future__ import annotations


class TechnicalInterpretationService:
    @staticmethod
    def _item(
        *,
        state: str,
        label: str,
        description: str,
    ) -> dict:
        return {
            "state": state,
            "label": label,
            "description": description,
        }

    @classmethod
    def _build_trend(
        cls,
        *,
        ema_12_gap: float,
        ema_26_gap: float,
    ) -> dict:
        average_gap = (
            ema_12_gap
            + ema_26_gap
        ) / 2.0

        if (
            ema_12_gap <= -0.05
            and ema_26_gap <= -0.05
        ):
            return cls._item(
                state="STRONG_BEARISH",
                label="강한 단기 하락",
                description=(
                    "현재 가격이 12일·26일 EMA보다 모두 "
                    "5% 이상 아래에 있어 단기 하락 압력이 큽니다."
                ),
            )

        if (
            ema_12_gap >= 0.05
            and ema_26_gap >= 0.05
        ):
            return cls._item(
                state="STRONG_BULLISH",
                label="강한 단기 상승",
                description=(
                    "현재 가격이 12일·26일 EMA보다 모두 "
                    "5% 이상 위에 있어 단기 상승 흐름이 강합니다."
                ),
            )

        if average_gap <= -0.015:
            return cls._item(
                state="BEARISH",
                label="단기 하락",
                description=(
                    "현재 가격이 단기 EMA 아래에 있어 "
                    "단기 추세는 약세 쪽입니다."
                ),
            )

        if average_gap >= 0.015:
            return cls._item(
                state="BULLISH",
                label="단기 상승",
                description=(
                    "현재 가격이 단기 EMA 위에 있어 "
                    "단기 추세는 강세 쪽입니다."
                ),
            )

        return cls._item(
            state="SIDEWAYS",
            label="단기 중립",
            description=(
                "현재 가격이 단기 EMA와 크게 벌어지지 않아 "
                "뚜렷한 단기 방향성은 약합니다."
            ),
        )

    @classmethod
    def _build_momentum(
        cls,
        *,
        stochastic_k: float,
        stochastic_d: float,
        macd_hist_ratio: float,
    ) -> dict:
        if (
            stochastic_k <= 0.20
            and stochastic_d <= 0.20
        ):
            return cls._item(
                state="OVERSOLD",
                label="과매도",
                description=(
                    "스토캐스틱 %K와 %D가 모두 20 아래로 "
                    "내려와 단기 과매도 신호가 강합니다."
                ),
            )

        if (
            stochastic_k >= 0.80
            and stochastic_d >= 0.80
        ):
            return cls._item(
                state="OVERBOUGHT",
                label="과매수",
                description=(
                    "스토캐스틱 %K와 %D가 모두 80 위에 있어 "
                    "단기 과열 여부를 확인할 구간입니다."
                ),
            )

        if macd_hist_ratio >= 0.005:
            return cls._item(
                state="IMPROVING",
                label="모멘텀 개선",
                description=(
                    "MACD 히스토그램이 플러스 방향으로 벌어져 "
                    "단기 모멘텀이 개선되고 있습니다."
                ),
            )

        if macd_hist_ratio <= -0.005:
            return cls._item(
                state="WEAKENING",
                label="모멘텀 약화",
                description=(
                    "MACD 히스토그램이 마이너스 방향으로 벌어져 "
                    "단기 모멘텀이 약화되고 있습니다."
                ),
            )

        return cls._item(
            state="NEUTRAL",
            label="모멘텀 중립",
            description=(
                "스토캐스틱과 MACD 모멘텀에서 "
                "극단적인 신호가 나타나지 않습니다."
            ),
        )

    @classmethod
    def _build_trend_strength(
        cls,
        *,
        adx: float,
    ) -> dict:
        if adx >= 0.25:
            return cls._item(
                state="STRONG",
                label="강한 추세",
                description=(
                    f"ADX가 {adx * 100.0:.1f}로 "
                    "현재 방향의 추세 강도가 높은 편입니다."
                ),
            )

        if adx >= 0.20:
            return cls._item(
                state="MODERATE",
                label="보통 추세",
                description=(
                    f"ADX가 {adx * 100.0:.1f}로 "
                    "일정 수준의 추세 강도가 확인됩니다."
                ),
            )

        return cls._item(
            state="WEAK",
            label="약한 추세",
            description=(
                f"ADX가 {adx * 100.0:.1f}로 낮아 "
                "현재 방향이 강하게 이어진다고 보기는 어렵습니다."
            ),
        )

    @classmethod
    def _build_volatility(
        cls,
        *,
        atr_ratio: float,
    ) -> dict:
        if atr_ratio >= 0.08:
            return cls._item(
                state="VERY_HIGH",
                label="매우 높은 변동성",
                description=(
                    f"ATR 변동폭이 가격의 {atr_ratio * 100.0:.2f}% 수준으로 "
                    "단기 가격 변동이 매우 큰 구간입니다."
                ),
            )

        if atr_ratio >= 0.05:
            return cls._item(
                state="HIGH",
                label="높은 변동성",
                description=(
                    f"ATR 변동폭이 가격의 {atr_ratio * 100.0:.2f}% 수준으로 "
                    "평소보다 큰 가격 움직임에 유의할 구간입니다."
                ),
            )

        if atr_ratio <= 0.025:
            return cls._item(
                state="LOW",
                label="낮은 변동성",
                description=(
                    f"ATR 변동폭이 가격의 {atr_ratio * 100.0:.2f}% 수준으로 "
                    "단기 가격 움직임이 비교적 제한적입니다."
                ),
            )

        return cls._item(
            state="NORMAL",
            label="보통 변동성",
            description=(
                f"ATR 변동폭이 가격의 {atr_ratio * 100.0:.2f}% 수준입니다."
            ),
        )

    @classmethod
    def _build_volume(
        cls,
        *,
        volume_zscore: float,
        volume_trend: float,
    ) -> dict:
        if volume_zscore >= 1.5:
            return cls._item(
                state="SURGE",
                label="거래량 급증",
                description=(
                    f"최근 거래량이 20일 기준 {volume_zscore:+.2f}σ로 "
                    "통상 범위를 크게 웃돌고 있습니다."
                ),
            )

        if volume_zscore <= -1.5:
            return cls._item(
                state="DRY",
                label="거래량 위축",
                description=(
                    f"최근 거래량이 20일 기준 {volume_zscore:+.2f}σ로 "
                    "통상 범위보다 크게 줄어 있습니다."
                ),
            )

        if volume_trend >= 0.20:
            return cls._item(
                state="INCREASING",
                label="거래량 증가",
                description=(
                    "단기 거래량이 중기 평균보다 "
                    f"{volume_trend * 100.0:+.1f}% 많습니다."
                ),
            )

        if volume_trend <= -0.20:
            return cls._item(
                state="DECREASING",
                label="거래량 감소",
                description=(
                    "단기 거래량이 중기 평균보다 "
                    f"{volume_trend * 100.0:+.1f}% 적습니다."
                ),
            )

        return cls._item(
            state="NORMAL",
            label="거래량 보통",
            description=(
                "최근 거래량이 통상 범위에서 움직이고 있습니다."
            ),
        )

    @classmethod
    def _build_price_location(
        cls,
        *,
        bollinger_position: float,
    ) -> dict:
        if bollinger_position < 0.0:
            return cls._item(
                state="BELOW_LOWER_BAND",
                label="볼린저 하단 이탈",
                description=(
                    "현재 가격이 볼린저밴드 하단 아래에 있어 "
                    "단기 낙폭이 큰 구간입니다."
                ),
            )

        if bollinger_position <= 0.20:
            return cls._item(
                state="LOWER_BAND",
                label="볼린저 하단권",
                description=(
                    "현재 가격이 볼린저밴드 하단에 가까워 "
                    "단기 약세 또는 반등 가능성을 함께 볼 구간입니다."
                ),
            )

        if bollinger_position > 1.0:
            return cls._item(
                state="ABOVE_UPPER_BAND",
                label="볼린저 상단 돌파",
                description=(
                    "현재 가격이 볼린저밴드 상단을 넘어 "
                    "강한 상승 또는 단기 과열 여부를 확인할 구간입니다."
                ),
            )

        if bollinger_position >= 0.80:
            return cls._item(
                state="UPPER_BAND",
                label="볼린저 상단권",
                description=(
                    "현재 가격이 볼린저밴드 상단에 가까워 "
                    "상승 탄력과 단기 과열 여부를 함께 볼 구간입니다."
                ),
            )

        return cls._item(
            state="MIDDLE_BAND",
            label="볼린저 중립권",
            description=(
                "현재 가격이 볼린저밴드 중앙 영역에서 움직이고 있습니다."
            ),
        )

    @classmethod
    def _build_rebound_signal(
        cls,
        *,
        trend_state: str,
        momentum_state: str,
        price_location_state: str,
        macd_hist_ratio: float,
    ) -> dict:
        bearish = trend_state in {
            "BEARISH",
            "STRONG_BEARISH",
        }

        oversold = (
            momentum_state
            == "OVERSOLD"
        )

        lower_band = (
            price_location_state
            in {
                "BELOW_LOWER_BAND",
                "LOWER_BAND",
            }
        )

        if (
            bearish
            and oversold
            and lower_band
        ):
            return cls._item(
                state="STRONG",
                label="반등 관찰 강함",
                description=(
                    "하락 추세와 과매도, 볼린저 하단권이 동시에 나타나 "
                    "기술적 반등 가능성을 집중 관찰할 구간입니다."
                ),
            )

        if (
            bearish
            and lower_band
        ) or (
            oversold
            and macd_hist_ratio > 0.0
        ):
            return cls._item(
                state="WATCH",
                label="반등 관찰",
                description=(
                    "일부 낙폭과대 신호가 확인되어 "
                    "단기 반등 여부를 관찰할 수 있습니다."
                ),
            )

        return cls._item(
            state="NONE",
            label="뚜렷한 반등 신호 없음",
            description=(
                "현재 지표 조합에서는 강한 낙폭과대 "
                "반등 신호가 확인되지 않습니다."
            ),
        )

    @staticmethod
    def _headline(
        *,
        trend: dict,
        momentum: dict,
        trend_strength: dict,
        rebound_signal: dict,
    ) -> str:
        if rebound_signal["state"] == "STRONG":
            return (
                f"{trend['label']} 속 {momentum['label']} 구간으로, "
                "단기 기술적 반등 가능성을 집중 관찰할 구간입니다."
            )

        if rebound_signal["state"] == "WATCH":
            return (
                f"{trend['label']} 흐름에서 일부 낙폭과대 신호가 나타나 "
                "단기 반등 여부를 확인할 구간입니다."
            )

        if trend["state"] in {
            "BULLISH",
            "STRONG_BULLISH",
        }:
            return (
                f"{trend['label']} 흐름이지만 추세 강도는 "
                f"'{trend_strength['label']}' 상태입니다."
            )

        if trend["state"] in {
            "BEARISH",
            "STRONG_BEARISH",
        }:
            return (
                f"{trend['label']} 흐름이며 모멘텀은 "
                f"'{momentum['label']}' 상태입니다."
            )

        return (
            "단기 방향성은 중립이며 모멘텀은 "
            f"'{momentum['label']}' 상태입니다."
        )

    @classmethod
    def build(
        cls,
        *,
        technical: dict,
    ) -> dict:
        features = (
            technical.get(
                "features"
            )
            or {}
        )

        def value(
            name: str,
        ) -> float:
            return float(
                features.get(
                    name
                )
                or 0.0
            )

        trend = cls._build_trend(
            ema_12_gap=value(
                "ema_12_gap"
            ),
            ema_26_gap=value(
                "ema_26_gap"
            ),
        )

        momentum = cls._build_momentum(
            stochastic_k=value(
                "stochastic_k_14"
            ),
            stochastic_d=value(
                "stochastic_d_3"
            ),
            macd_hist_ratio=value(
                "macd_hist_ratio"
            ),
        )

        trend_strength = (
            cls._build_trend_strength(
                adx=value(
                    "adx_14"
                ),
            )
        )

        volatility = cls._build_volatility(
            atr_ratio=value(
                "atr_14_ratio"
            ),
        )

        volume = cls._build_volume(
            volume_zscore=value(
                "volume_zscore_20"
            ),
            volume_trend=value(
                "volume_trend_5_20"
            ),
        )

        price_location = (
            cls._build_price_location(
                bollinger_position=value(
                    "bollinger_position_20"
                ),
            )
        )

        rebound_signal = (
            cls._build_rebound_signal(
                trend_state=trend[
                    "state"
                ],
                momentum_state=momentum[
                    "state"
                ],
                price_location_state=(
                    price_location[
                        "state"
                    ]
                ),
                macd_hist_ratio=value(
                    "macd_hist_ratio"
                ),
            )
        )

        return {
            "headline": cls._headline(
                trend=trend,
                momentum=momentum,
                trend_strength=(
                    trend_strength
                ),
                rebound_signal=(
                    rebound_signal
                ),
            ),
            "trend": trend,
            "momentum": momentum,
            "trendStrength": (
                trend_strength
            ),
            "volatility": volatility,
            "volume": volume,
            "priceLocation": (
                price_location
            ),
            "reboundSignal": (
                rebound_signal
            ),
        }