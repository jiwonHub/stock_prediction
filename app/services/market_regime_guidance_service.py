from __future__ import annotations


class MarketRegimeGuidanceService:
    @staticmethod
    def _signal(
        *,
        signal_type: str,
        label: str,
        text: str,
    ) -> dict:
        return {
            "type": signal_type,
            "label": label,
            "text": text,
        }

    @classmethod
    def _trend_signal(
        cls,
        *,
        trend: str,
    ) -> dict:
        if trend == "BULL":
            return cls._signal(
                signal_type="POSITIVE",
                label="추세",
                text="시장 지수의 방향은 상승 쪽입니다.",
            )

        if trend == "BEAR":
            return cls._signal(
                signal_type="CAUTION",
                label="추세",
                text="시장 지수의 방향은 하락 쪽입니다.",
            )

        return cls._signal(
            signal_type="NEUTRAL",
            label="추세",
            text="시장 지수의 뚜렷한 방향성이 약합니다.",
        )

    @classmethod
    def _volatility_signal(
        cls,
        *,
        volatility: str,
    ) -> dict:
        if volatility == "HIGH":
            return cls._signal(
                signal_type="CAUTION",
                label="변동성",
                text=(
                    "시장 변동성이 높아 "
                    "가격 흔들림이 큰 구간입니다."
                ),
            )

        if volatility == "LOW":
            return cls._signal(
                signal_type="POSITIVE",
                label="변동성",
                text=(
                    "시장 변동성이 낮아 "
                    "가격 흐름이 비교적 안정적입니다."
                ),
            )

        return cls._signal(
            signal_type="NEUTRAL",
            label="변동성",
            text="시장 변동성은 통상 범위입니다.",
        )

    @classmethod
    def _risk_signal(
        cls,
        *,
        risk: str,
    ) -> dict:
        if risk == "RISK_ON":
            return cls._signal(
                signal_type="POSITIVE",
                label="위험선호",
                text=(
                    "시장 전반의 위험선호 신호가 "
                    "우세합니다."
                ),
            )

        if risk == "RISK_OFF":
            return cls._signal(
                signal_type="CAUTION",
                label="위험선호",
                text=(
                    "시장 전반의 위험회피 신호가 "
                    "우세합니다."
                ),
            )

        return cls._signal(
            signal_type="NEUTRAL",
            label="위험선호",
            text=(
                "위험선호와 위험회피 신호가 "
                "균형에 가깝습니다."
            ),
        )

    @staticmethod
    def _posture(
        *,
        trend: str,
        volatility: str,
        risk: str,
    ) -> tuple[
        str,
        str,
        str,
        str,
    ]:
        if (
            risk == "RISK_ON"
            and trend == "BULL"
        ):
            if volatility == "HIGH":
                return (
                    "POSITIVE",
                    "SELECTIVE",
                    "선별적 강세 대응",
                    (
                        "상승 추세와 위험선호가 확인되지만 "
                        "변동성이 높아 종목 선별이 "
                        "중요한 구간입니다."
                    ),
                )

            return (
                "POSITIVE",
                "PROACTIVE",
                "우호적 시장 환경",
                (
                    "상승 추세와 위험선호가 함께 나타나 "
                    "시장 환경이 상대적으로 우호적입니다."
                ),
            )

        if (
            risk == "RISK_OFF"
            and trend == "BEAR"
        ):
            return (
                "CAUTION",
                "DEFENSIVE",
                "방어적 대응 구간",
                (
                    "하락 추세와 위험회피가 함께 나타나 "
                    "손실 관리가 중요한 시장 환경입니다."
                ),
            )

        if risk == "RISK_OFF":
            return (
                "CAUTION",
                "CAUTIOUS",
                "보수적 대응 구간",
                (
                    "위험회피 신호가 우세해 "
                    "시장 방향이 개선되는지 "
                    "확인이 필요한 구간입니다."
                ),
            )

        if volatility == "HIGH":
            return (
                "CAUTION",
                "CAUTIOUS",
                "변동성 주의 구간",
                (
                    "시장 방향은 확정적이지 않지만 "
                    "변동성이 높아 보수적인 접근이 "
                    "필요한 구간입니다."
                ),
            )

        if trend == "BULL":
            return (
                "NEUTRAL",
                "SELECTIVE",
                "선별적 접근",
                (
                    "상승 흐름은 나타나지만 "
                    "위험선호 신호가 충분히 강하지 않아 "
                    "종목별 확인이 필요합니다."
                ),
            )

        if trend == "BEAR":
            return (
                "CAUTION",
                "CAUTIOUS",
                "약세 관찰 구간",
                (
                    "하락 흐름이 나타나지만 "
                    "강한 위험회피 국면은 아니어서 "
                    "추가 방향 확인이 필요합니다."
                ),
            )

        return (
            "NEUTRAL",
            "WAIT",
            "관망·선별 대응",
            (
                "추세와 위험선호가 중립에 가까워 "
                "강한 시장 방향보다 개별 종목 신호가 "
                "중요한 구간입니다."
            ),
        )

    @classmethod
    def build(
        cls,
        *,
        trend: str,
        volatility: str,
        risk: str,
    ) -> dict:
        (
            tone,
            posture,
            label,
            headline,
        ) = cls._posture(
            trend=trend,
            volatility=volatility,
            risk=risk,
        )

        return {
            "tone": tone,
            "posture": posture,
            "label": label,
            "headline": headline,
            "signals": [
                cls._trend_signal(
                    trend=trend,
                ),
                cls._volatility_signal(
                    volatility=volatility,
                ),
                cls._risk_signal(
                    risk=risk,
                ),
            ],
        }