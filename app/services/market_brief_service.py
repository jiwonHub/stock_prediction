from __future__ import annotations


class MarketBriefService:
    @staticmethod
    def _number(
        value,
    ) -> float | None:
        if value is None:
            return None

        try:
            return float(
                value
            )
        except (
            TypeError,
            ValueError,
        ):
            return None

    @staticmethod
    def _format_percent(
        value: float | None,
    ) -> str:
        if value is None:
            return "-"

        sign = (
            "+"
            if value > 0
            else ""
        )

        return (
            f"{sign}"
            f"{value:.2f}%"
        )

    @staticmethod
    def _format_amount(
        value: float | None,
    ) -> str:
        if value is None:
            return "-"

        sign = (
            "+"
            if value > 0
            else "-"
            if value < 0
            else ""
        )

        absolute = abs(
            value
        )

        if absolute >= 1_000_000_000_000:
            return (
                f"{sign}"
                f"{absolute / 1_000_000_000_000:.2f}"
                "조원"
            )

        if absolute >= 100_000_000:
            return (
                f"{sign}"
                f"{absolute / 100_000_000:.1f}"
                "억원"
            )

        return (
            f"{sign}"
            f"{absolute:,.0f}원"
        )

    @staticmethod
    def _direction(
        value: float | None,
    ) -> str:
        if value is None:
            return "UNKNOWN"

        if value > 0:
            return "POSITIVE"

        if value < 0:
            return "NEGATIVE"

        return "FLAT"

    @classmethod
    def _build_market_brief(
        cls,
        *,
        market: str,
        summary: dict,
    ) -> dict:
        indices = (
            summary.get(
                "indices",
                {}
            )
        )

        investor_flows = (
            summary.get(
                "investorFlows",
                {}
            )
        )

        regimes = (
            summary.get(
                "regimes",
                {}
            )
        )

        index = (
            indices.get(
                market,
                {}
            )
        )

        flow = (
            investor_flows.get(
                market,
                {}
            )
        )

        regime = (
            regimes.get(
                market,
                {}
            )
        )

        return_5d = (
            cls._number(
                index.get(
                    "return5d"
                )
            )
        )

        return_20d = (
            cls._number(
                index.get(
                    "return20d"
                )
            )
        )

        five_days = (
            flow.get(
                "fiveDays",
                {}
            )
        )

        foreign_5d = (
            cls._number(
                five_days.get(
                    "foreignNet"
                )
            )
        )

        institution_5d = (
            cls._number(
                five_days.get(
                    "institutionNet"
                )
            )
        )

        regime_state = (
            regime.get(
                "state"
            )
        )

        score = 0

        if return_5d is not None:
            if return_5d >= 1.0:
                score += 1
            elif return_5d <= -1.0:
                score -= 1

        if foreign_5d is not None:
            if foreign_5d > 0:
                score += 1
            elif foreign_5d < 0:
                score -= 1

        if institution_5d is not None:
            if institution_5d > 0:
                score += 1
            elif institution_5d < 0:
                score -= 1

        if regime_state == "FAVORABLE":
            score += 1

        elif regime_state == "RISKY":
            score -= 1

        if score >= 2:
            tone = "POSITIVE"

        elif score <= -2:
            tone = "CAUTION"

        else:
            tone = "NEUTRAL"

        foreign_buying = (
            foreign_5d is not None
            and foreign_5d > 0
        )

        foreign_selling = (
            foreign_5d is not None
            and foreign_5d < 0
        )

        institution_buying = (
            institution_5d is not None
            and institution_5d > 0
        )

        institution_selling = (
            institution_5d is not None
            and institution_5d < 0
        )

        if (
            return_5d is not None
            and return_5d > 0
            and foreign_buying
            and institution_buying
        ):
            headline = (
                f"{market} 단기 상승과 "
                "외국인·기관 매수가 "
                "함께 나타나고 있습니다."
            )

        elif (
            return_5d is not None
            and return_5d < 0
            and foreign_selling
            and institution_selling
        ):
            headline = (
                f"{market} 단기 약세와 "
                "외국인·기관 매도가 "
                "동반되고 있습니다."
            )

        elif (
            return_5d is not None
            and return_5d > 0
        ):
            headline = (
                f"{market}는 단기 상승 "
                "흐름을 보이고 있습니다."
            )

        elif (
            return_5d is not None
            and return_5d < 0
        ):
            headline = (
                f"{market}는 단기 조정 "
                "흐름을 보이고 있습니다."
            )

        else:
            headline = (
                f"{market}는 뚜렷한 "
                "단기 방향성을 탐색하고 "
                "있습니다."
            )

        regime_label = (
            regime.get(
                "label"
            )
            or "확인 중"
        )

        description = (
            f"5거래일 수익률 "
            f"{cls._format_percent(return_5d)}, "
            f"20거래일 수익률 "
            f"{cls._format_percent(return_20d)}입니다. "
            f"최근 5거래일 외국인은 "
            f"{cls._format_amount(foreign_5d)}, "
            f"기관은 "
            f"{cls._format_amount(institution_5d)}이며, "
            f"시장 환경은 "
            f"'{regime_label}'으로 "
            "분류되어 있습니다."
        )

        signals = [
            {
                "type":
                    "RETURN_5D",
                "label":
                    "5거래일 수익률",
                "value":
                    return_5d,
                "unit":
                    "%",
                "direction":
                    cls._direction(
                        return_5d
                    ),
            },
            {
                "type":
                    "RETURN_20D",
                "label":
                    "20거래일 수익률",
                "value":
                    return_20d,
                "unit":
                    "%",
                "direction":
                    cls._direction(
                        return_20d
                    ),
            },
            {
                "type":
                    "FOREIGN_5D",
                "label":
                    "외국인 5거래일",
                "value":
                    foreign_5d,
                "unit":
                    "KRW",
                "direction":
                    cls._direction(
                        foreign_5d
                    ),
            },
            {
                "type":
                    "INSTITUTION_5D",
                "label":
                    "기관 5거래일",
                "value":
                    institution_5d,
                "unit":
                    "KRW",
                "direction":
                    cls._direction(
                        institution_5d
                    ),
            },
        ]

        return {
            "market":
                market,
            "tone":
                tone,
            "headline":
                headline,
            "description":
                description,
            "regimeLabel":
                regime_label,
            "signals":
                signals,
        }

    @classmethod
    def _build_fx_brief(
        cls,
        *,
        summary: dict,
    ) -> dict:
        fx = (
            summary.get(
                "usdKrw",
                {}
            )
        )

        value = (
            cls._number(
                fx.get(
                    "value"
                )
            )
        )

        change = (
            cls._number(
                fx.get(
                    "change"
                )
            )
        )

        change_rate = (
            cls._number(
                fx.get(
                    "changeRate"
                )
            )
        )

        if value is None:
            return {
                "available": False,
                "headline":
                    "USD/KRW 환율 데이터가 없습니다.",
                "description":
                    "환율 데이터가 아직 준비되지 않았습니다.",
            }

        headline = (
            "USD/KRW "
            f"{value:,.1f}원"
        )

        if (
            change is None
            or change_rate is None
        ):
            description = (
                "전일 비교 데이터가 "
                "아직 충분히 축적되지 않았습니다."
            )

        elif change > 0:
            description = (
                "원/달러 환율이 전일 대비 "
                f"{change:+.1f}원 "
                f"({change_rate:+.2f}%) "
                "상승했습니다."
            )

        elif change < 0:
            description = (
                "원/달러 환율이 전일 대비 "
                f"{change:+.1f}원 "
                f"({change_rate:+.2f}%) "
                "하락했습니다."
            )

        else:
            description = (
                "원/달러 환율은 전일과 "
                "동일한 수준입니다."
            )

        return {
            "available": True,
            "headline":
                headline,
            "description":
                description,
            "value":
                value,
            "change":
                change,
            "changeRate":
                change_rate,
        }

    @classmethod
    def build(
        cls,
        summary: dict,
    ) -> dict:
        kospi = (
            cls._build_market_brief(
                market="KOSPI",
                summary=summary,
            )
        )

        kosdaq = (
            cls._build_market_brief(
                market="KOSDAQ",
                summary=summary,
            )
        )

        fx = (
            cls._build_fx_brief(
                summary=summary
            )
        )

        tones = {
            kospi[
                "tone"
            ],
            kosdaq[
                "tone"
            ],
        }

        if tones == {
            "POSITIVE"
        }:
            tone = "POSITIVE"

            headline = (
                "양 시장 모두 단기 흐름과 수급이 우호적입니다. "
                "무리한 추격보다 강한 종목을 선별할 구간입니다."
            )

        elif tones == {
            "CAUTION"
        }:
            tone = "CAUTION"

            headline = (
                "양 시장 모두 단기 약세와 수급 부담이 겹쳐 "
                "추격 매수보다 보수적 종목 선별이 필요한 구간입니다."
            )

        elif (
            "POSITIVE" in tones
            and "CAUTION" in tones
        ):
            tone = "MIXED"

            headline = (
                "코스피와 코스닥 신호가 엇갈려 "
                "지수 방향보다 종목별 신호가 중요한 구간입니다."
            )

        elif "CAUTION" in tones:
            tone = "CAUTION"

            headline = (
                "시장 전반은 혼조지만 일부 시장의 약세와 수급 부담이 있어 "
                "보수적인 종목 선별이 필요합니다."
            )

        elif "POSITIVE" in tones:
            tone = "NEUTRAL"

            headline = (
                "일부 시장 흐름은 개선됐지만 "
                "시장 전체 방향은 아직 중립에 가깝습니다."
            )

        else:
            tone = "NEUTRAL"

            headline = (
                "시장 방향이 뚜렷하지 않아 "
                "지수 추격보다 종목별 신호 확인이 중요한 구간입니다."
            )

        fx_change_rate = (
            cls._number(
                fx.get(
                    "changeRate"
                )
            )
            if fx.get(
                "available"
            )
            else None
        )

        if (
            fx_change_rate is not None
            and fx_change_rate >= 0.5
        ):
            headline += (
                " 원/달러 환율 상승도 부담 요인입니다."
            )

        elif (
            fx_change_rate is not None
            and fx_change_rate <= -0.5
        ):
            headline += (
                " 원/달러 환율 하락은 부담 완화 요인입니다."
            )

        return {
            "tone":
                tone,
            "headline":
                headline,
            "markets": {
                "KOSPI":
                    kospi,
                "KOSDAQ":
                    kosdaq,
            },
            "fx":
                fx,
        }