from __future__ import annotations


class BenchmarkInterpretationService:
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

    @staticmethod
    def _value(
        period: dict | None,
    ) -> float | None:
        if not period:
            return None

        value = period.get(
            "excessReturn"
        )

        if value is None:
            return None

        return float(
            value
        )

    @staticmethod
    def _format_excess(
        value: float,
    ) -> str:
        return f"{value:+.2f}%p"

    @classmethod
    def _build_period(
        cls,
        *,
        title: str,
        period: dict | None,
        benchmark_name: str,
    ) -> dict:
        value = cls._value(
            period
        )

        if value is None:
            return cls._item(
                state="UNAVAILABLE",
                label=f"{title} 비교 불가",
                description=(
                    f"{title} {benchmark_name} 대비 상대성과를 "
                    "계산할 데이터가 충분하지 않습니다."
                ),
            )

        if value >= 3.0:
            state = "STRONG_OUTPERFORM"
            label = "시장 대비 강한 우위"

        elif value >= 1.0:
            state = "OUTPERFORM"
            label = "시장 대비 우위"

        elif value <= -3.0:
            state = "STRONG_UNDERPERFORM"
            label = "시장 대비 큰 열위"

        elif value <= -1.0:
            state = "UNDERPERFORM"
            label = "시장 대비 열위"

        else:
            state = "INLINE"
            label = "시장과 유사"

        return cls._item(
            state=state,
            label=label,
            description=(
                f"{title} {benchmark_name} 대비 초과수익률은 "
                f"{cls._format_excess(value)}입니다."
            ),
        )

    @classmethod
    def _build_trend(
        cls,
        *,
        one_day: float | None,
        five_days: float | None,
        twenty_days: float | None,
        benchmark_name: str,
    ) -> dict:
        values = [
            value
            for value in [
                one_day,
                five_days,
                twenty_days,
            ]
            if value is not None
        ]

        if not values:
            return cls._item(
                state="UNAVAILABLE",
                label="상대성과 흐름 확인 불가",
                description=(
                    f"{benchmark_name} 대비 상대성과 추세를 "
                    "판단할 데이터가 충분하지 않습니다."
                ),
            )

        if (
            one_day is not None
            and five_days is not None
            and twenty_days is not None
            and one_day > 0.0
            and five_days > 0.0
            and twenty_days > 0.0
        ):
            return cls._item(
                state="PERSISTENT_OUTPERFORM",
                label="시장 대비 우위 지속",
                description=(
                    "1·5·20거래일 모두 시장 대비 초과수익이 "
                    "플러스로 상대 우위가 이어지고 있습니다."
                ),
            )

        if (
            one_day is not None
            and five_days is not None
            and twenty_days is not None
            and one_day < 0.0
            and five_days < 0.0
            and twenty_days < 0.0
        ):
            return cls._item(
                state="PERSISTENT_UNDERPERFORM",
                label="시장 대비 열위 지속",
                description=(
                    "1·5·20거래일 모두 시장 대비 초과수익이 "
                    "마이너스로 상대 약세가 이어지고 있습니다."
                ),
            )

        if (
            five_days is not None
            and twenty_days is not None
            and five_days > 0.0
            and twenty_days <= 0.0
        ):
            return cls._item(
                state="IMPROVING",
                label="상대성과 개선",
                description=(
                    "20거래일 기준 상대성과는 아직 약하지만 "
                    "최근 5거래일에는 시장 대비 우위로 전환됐습니다."
                ),
            )

        if (
            five_days is not None
            and twenty_days is not None
            and five_days < 0.0
            and twenty_days >= 0.0
        ):
            return cls._item(
                state="DETERIORATING",
                label="상대성과 악화",
                description=(
                    "20거래일 기준 상대성과와 달리 최근 5거래일에는 "
                    "시장 대비 열위로 전환됐습니다."
                ),
            )

        if (
            one_day is not None
            and five_days is not None
            and one_day > 0.0
            and five_days <= 0.0
        ):
            return cls._item(
                state="SHORT_TERM_RECOVERY",
                label="단기 상대성과 반등",
                description=(
                    "최근 5거래일 상대성과는 약하지만 "
                    "최근 1거래일에는 시장 대비 우위로 돌아섰습니다."
                ),
            )

        if (
            one_day is not None
            and five_days is not None
            and one_day < 0.0
            and five_days >= 0.0
        ):
            return cls._item(
                state="SHORT_TERM_WEAKENING",
                label="단기 상대성과 둔화",
                description=(
                    "최근 5거래일 상대성과는 양호하지만 "
                    "최근 1거래일에는 시장 대비 약해졌습니다."
                ),
            )

        return cls._item(
            state="MIXED",
            label="상대성과 혼조",
            description=(
                "1·5·20거래일 시장 대비 성과가 서로 엇갈려 "
                "뚜렷한 상대 강도 방향은 약합니다."
            ),
        )

    @staticmethod
    def _tone(
        *,
        trend_state: str,
    ) -> str:
        if trend_state in {
            "PERSISTENT_OUTPERFORM",
            "IMPROVING",
            "SHORT_TERM_RECOVERY",
        }:
            return "POSITIVE"

        if trend_state in {
            "PERSISTENT_UNDERPERFORM",
            "DETERIORATING",
            "SHORT_TERM_WEAKENING",
        }:
            return "CAUTION"

        return "NEUTRAL"

    @staticmethod
    def _headline(
        *,
        trend: dict,
        benchmark_name: str,
    ) -> str:
        state = trend[
            "state"
        ]

        if state == "PERSISTENT_OUTPERFORM":
            return (
                f"최근 1·5·20거래일 모두 {benchmark_name} 대비 "
                "상대 우위를 유지하고 있습니다."
            )

        if state == "PERSISTENT_UNDERPERFORM":
            return (
                f"최근 1·5·20거래일 모두 {benchmark_name} 대비 "
                "상대 열위가 이어지고 있습니다."
            )

        if state == "IMPROVING":
            return (
                "중기 상대성과는 약하지만 최근 5거래일에는 "
                f"{benchmark_name} 대비 개선되고 있습니다."
            )

        if state == "DETERIORATING":
            return (
                "중기 상대성과와 달리 최근 5거래일에는 "
                f"{benchmark_name} 대비 약해지고 있습니다."
            )

        if state == "SHORT_TERM_RECOVERY":
            return (
                f"최근 1거래일에는 {benchmark_name} 대비 "
                "상대성과가 반등했습니다."
            )

        if state == "SHORT_TERM_WEAKENING":
            return (
                "최근 5거래일 우위에도 최근 1거래일에는 "
                f"{benchmark_name} 대비 상대성과가 둔화됐습니다."
            )

        return (
            f"기간별 {benchmark_name} 대비 상대성과가 엇갈려 "
            "뚜렷한 우위나 열위가 확인되지 않습니다."
        )

    @classmethod
    def build(
        cls,
        *,
        benchmark_performance: dict,
    ) -> dict:
        if not benchmark_performance.get(
            "available"
        ):
            unavailable = cls._item(
                state="UNAVAILABLE",
                label="시장 비교 데이터 없음",
                description=(
                    "현재 비교 가능한 시장 지수 데이터가 없습니다."
                ),
            )

            return {
                "available": False,
                "tone": "NEUTRAL",
                "headline": (
                    "현재 비교 가능한 시장 지수 데이터가 없습니다."
                ),
                "trend": unavailable,
                "oneDay": unavailable,
                "fiveDays": unavailable,
                "twentyDays": unavailable,
            }

        benchmark_name = (
            benchmark_performance.get(
                "benchmark"
            )
            or benchmark_performance.get(
                "market"
            )
            or "시장"
        )

        one_day_period = (
            benchmark_performance.get(
                "oneDay"
            )
        )

        five_days_period = (
            benchmark_performance.get(
                "fiveDays"
            )
        )

        twenty_days_period = (
            benchmark_performance.get(
                "twentyDays"
            )
        )

        one_day_value = cls._value(
            one_day_period
        )

        five_days_value = cls._value(
            five_days_period
        )

        twenty_days_value = cls._value(
            twenty_days_period
        )

        trend = cls._build_trend(
            one_day=one_day_value,
            five_days=five_days_value,
            twenty_days=twenty_days_value,
            benchmark_name=benchmark_name,
        )

        return {
            "available": True,
            "tone": cls._tone(
                trend_state=trend[
                    "state"
                ],
            ),
            "headline": cls._headline(
                trend=trend,
                benchmark_name=benchmark_name,
            ),
            "trend": trend,
            "oneDay": cls._build_period(
                title="최근 1거래일",
                period=one_day_period,
                benchmark_name=benchmark_name,
            ),
            "fiveDays": cls._build_period(
                title="최근 5거래일",
                period=five_days_period,
                benchmark_name=benchmark_name,
            ),
            "twentyDays": cls._build_period(
                title="최근 20거래일",
                period=twenty_days_period,
                benchmark_name=benchmark_name,
            ),
        }