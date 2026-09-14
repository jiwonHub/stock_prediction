from __future__ import annotations


class InvestorFlowInterpretationService:
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
    def _format_volume(
        value: int,
    ) -> str:
        return f"{value:+,}주"

    @staticmethod
    def _sign(
        value: int,
    ) -> int:
        if value > 0:
            return 1

        if value < 0:
            return -1

        return 0

    @classmethod
    def _build_short_term(
        cls,
        *,
        five_days: dict,
    ) -> dict:
        value = int(
            five_days.get(
                "foreignInstitutionNetBuyVolume"
            )
            or 0
        )

        if value > 0:
            return cls._item(
                state="BUYING",
                label="단기 순매수",
                description=(
                    "최근 5거래일 외국인·기관 합산 수급이 "
                    f"{cls._format_volume(value)} 순매수입니다."
                ),
            )

        if value < 0:
            return cls._item(
                state="SELLING",
                label="단기 순매도",
                description=(
                    "최근 5거래일 외국인·기관 합산 수급이 "
                    f"{cls._format_volume(value)} 순매도입니다."
                ),
            )

        return cls._item(
            state="NEUTRAL",
            label="단기 수급 중립",
            description=(
                "최근 5거래일 외국인·기관 합산 수급에서 "
                "뚜렷한 방향성이 나타나지 않습니다."
            ),
        )

    @classmethod
    def _build_participant_alignment(
        cls,
        *,
        twenty_days: dict,
    ) -> dict:
        foreign = int(
            twenty_days.get(
                "foreignNetBuyVolume"
            )
            or 0
        )

        institution = int(
            twenty_days.get(
                "institutionNetBuyVolume"
            )
            or 0
        )

        if (
            foreign > 0
            and institution > 0
        ):
            return cls._item(
                state="BOTH_BUY",
                label="외국인·기관 동반 매수",
                description=(
                    "최근 20거래일 외국인과 기관이 모두 순매수했습니다. "
                    f"외국인 {cls._format_volume(foreign)}, "
                    f"기관 {cls._format_volume(institution)}입니다."
                ),
            )

        if (
            foreign < 0
            and institution < 0
        ):
            return cls._item(
                state="BOTH_SELL",
                label="외국인·기관 동반 매도",
                description=(
                    "최근 20거래일 외국인과 기관이 모두 순매도했습니다. "
                    f"외국인 {cls._format_volume(foreign)}, "
                    f"기관 {cls._format_volume(institution)}입니다."
                ),
            )

        if (
            foreign == 0
            and institution == 0
        ):
            return cls._item(
                state="NEUTRAL",
                label="주체별 수급 중립",
                description=(
                    "최근 20거래일 외국인과 기관 수급에서 "
                    "뚜렷한 방향성이 나타나지 않습니다."
                ),
            )

        return cls._item(
            state="MIXED",
            label="외국인·기관 수급 엇갈림",
            description=(
                "최근 20거래일 외국인과 기관의 매매 방향이 엇갈립니다. "
                f"외국인 {cls._format_volume(foreign)}, "
                f"기관 {cls._format_volume(institution)}입니다."
            ),
        )

    @classmethod
    def _build_trend(
        cls,
        *,
        five_days: dict,
        twenty_days: dict,
        sixty_days: dict,
    ) -> dict:
        value_5d = int(
            five_days.get(
                "foreignInstitutionNetBuyVolume"
            )
            or 0
        )

        value_20d = int(
            twenty_days.get(
                "foreignInstitutionNetBuyVolume"
            )
            or 0
        )

        value_60d = int(
            sixty_days.get(
                "foreignInstitutionNetBuyVolume"
            )
            or 0
        )

        sign_20d = cls._sign(
            value_20d
        )

        sign_60d = cls._sign(
            value_60d
        )

        description = (
            "외국인·기관 합산 기준 "
            f"20일 {cls._format_volume(value_20d)}, "
            f"60일 {cls._format_volume(value_60d)}이며, "
            f"최근 5일은 {cls._format_volume(value_5d)}입니다."
        )

        if (
            sign_20d > 0
            and sign_60d > 0
        ):
            return cls._item(
                state="PERSISTENT_BUY",
                label="중기 순매수 지속",
                description=description,
            )

        if (
            sign_20d < 0
            and sign_60d < 0
        ):
            return cls._item(
                state="PERSISTENT_SELL",
                label="중기 순매도 지속",
                description=description,
            )

        if (
            sign_20d > 0
            and sign_60d <= 0
        ):
            return cls._item(
                state="IMPROVING",
                label="중기 수급 개선",
                description=description,
            )

        if (
            sign_20d < 0
            and sign_60d >= 0
        ):
            return cls._item(
                state="DETERIORATING",
                label="중기 수급 악화",
                description=description,
            )

        if (
            sign_20d == 0
            and sign_60d == 0
        ):
            return cls._item(
                state="NEUTRAL",
                label="중기 수급 중립",
                description=description,
            )

        return cls._item(
            state="MIXED",
            label="20·60일 수급 혼조",
            description=description,
        )

    @classmethod
    def _build_foreign_holding(
        cls,
        *,
        foreign_holding_ratio: float | None,
    ) -> dict:
        if foreign_holding_ratio is None:
            return cls._item(
                state="UNAVAILABLE",
                label="외국인 보유비중 확인 불가",
                description=(
                    "현재 외국인 보유비중 데이터가 없습니다."
                ),
            )

        value = float(
            foreign_holding_ratio
        )

        return cls._item(
            state="INFO",
            label=(
                "외국인 보유비중 "
                f"{value * 100.0:.2f}%"
            ),
            description=(
                "현재 기준 외국인 보유비중입니다. "
                "보유비중 자체만으로 매수·매도 신호를 판단하지 않습니다."
            ),
        )

    @staticmethod
    def _tone(
        *,
        trend_state: str,
    ) -> str:
        if trend_state in {
            "PERSISTENT_BUY",
            "IMPROVING",
        }:
            return "POSITIVE"

        if trend_state in {
            "PERSISTENT_SELL",
            "DETERIORATING",
        }:
            return "CAUTION"

        return "NEUTRAL"

    @staticmethod
    def _headline(
        *,
        trend: dict,
        participant_alignment: dict,
    ) -> str:
        trend_state = trend[
            "state"
        ]

        alignment_state = (
            participant_alignment[
                "state"
            ]
        )

        if trend_state == "PERSISTENT_BUY":
            if alignment_state == "BOTH_BUY":
                return (
                    "20거래일 외국인·기관 동반 순매수와 "
                    "60거래일 누적 순매수가 함께 이어지고 있습니다."
                )

            return (
                "외국인·기관 합산 수급이 20·60거래일 모두 "
                "순매수입니다."
            )

        if trend_state == "PERSISTENT_SELL":
            if alignment_state == "BOTH_SELL":
                return (
                    "20거래일 외국인·기관 동반 순매도와 "
                    "60거래일 누적 순매도가 함께 나타나고 있습니다."
                )

            return (
                "외국인·기관 합산 수급이 20·60거래일 모두 "
                "순매도입니다."
            )

        if trend_state == "IMPROVING":
            return (
                "60거래일 누적 수급은 약하지만 20거래일 수급이 "
                "순매수로 개선됐습니다."
            )

        if trend_state == "DETERIORATING":
            return (
                "60거래일 누적 흐름과 달리 20거래일 수급이 "
                "순매도로 약해졌습니다."
            )

        if trend_state == "NEUTRAL":
            return (
                "20·60거래일 외국인·기관 합산 수급은 "
                "뚜렷한 매수·매도 우위가 없습니다."
            )

        return (
            "20거래일과 60거래일 외국인·기관 수급 방향이 "
            "서로 엇갈립니다."
        )

    @classmethod
    def build(
        cls,
        *,
        investor_flow: dict,
    ) -> dict:
        if not investor_flow.get(
            "available"
        ):
            unavailable = cls._item(
                state="UNAVAILABLE",
                label="수급 데이터 없음",
                description=(
                    "현재 확인 가능한 투자자 수급 데이터가 없습니다."
                ),
            )

            return {
                "available": False,
                "tone": "NEUTRAL",
                "headline": (
                    "현재 확인 가능한 투자자 수급 데이터가 없습니다."
                ),
                "shortTerm": unavailable,
                "participantAlignment": unavailable,
                "trend": unavailable,
                "foreignHolding": unavailable,
            }

        five_days = (
            investor_flow.get(
                "fiveDays"
            )
            or {}
        )

        twenty_days = (
            investor_flow.get(
                "twentyDays"
            )
            or {}
        )

        sixty_days = (
            investor_flow.get(
                "sixtyDays"
            )
            or {}
        )

        short_term = (
            cls._build_short_term(
                five_days=five_days,
            )
        )

        participant_alignment = (
            cls._build_participant_alignment(
                twenty_days=twenty_days,
            )
        )

        trend = cls._build_trend(
            five_days=five_days,
            twenty_days=twenty_days,
            sixty_days=sixty_days,
        )

        foreign_holding = (
            cls._build_foreign_holding(
                foreign_holding_ratio=(
                    investor_flow.get(
                        "foreignHoldingRatio"
                    )
                ),
            )
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
                participant_alignment=(
                    participant_alignment
                ),
            ),
            "shortTerm": short_term,
            "participantAlignment":
                participant_alignment,
            "trend": trend,
            "foreignHolding":
                foreign_holding,
        }