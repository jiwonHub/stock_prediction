from __future__ import annotations

import re


DISCLOSURE_EVENT_TYPES = (
    "EARNINGS",
    "PERIODIC_REPORT",
    "CONTRACT_ORDER",
    "FINANCING_DILUTION",
    "SECURITIES_ISSUANCE",
    "SHAREHOLDER_RETURN",
    "TREASURY_STOCK",
    "MNA_RESTRUCTURING",
    "INVESTMENT_CAPEX",
    "IR_GUIDANCE",
    "MANAGEMENT_DECISION",
    "CREDIT_GUARANTEE",
    "INSIDER_OWNERSHIP",
    "BOARD_GOVERNANCE",
    "RELATED_PARTY",
    "OWNERSHIP_GOVERNANCE",
    "LEGAL_RISK",
    "LISTING_TRADING",
    "CAPITAL_STRUCTURE",
    "OTHER",
)


_EVENT_RULES = (
    (
        "LEGAL_RISK",
        (
            "횡령",
            "배임",
            "소송등의제기",
            "소송",
            "벌금",
            "과징금",
            "영업정지",
            "부도",
            "회생절차",
            "파산",
            "상장폐지",
            "불성실공시",
            "관리종목",
            "감사의견거절",
            "감사의견부적정",
        ),
        -0.90,
        0.95,
    ),
    (
        "CREDIT_GUARANTEE",
        (
            "타인에대한채무보증결정",
            "채무보증결정",
            "채무인수결정",
            "담보제공결정",
        ),
        -0.45,
        0.80,
    ),
    (
        "CONTRACT_ORDER",
        (
            "단일판매ㆍ공급계약",
            "단일판매공급계약",
            "공급계약체결",
            "판매계약체결",
            "수주계약",
            "수주",
        ),
        0.70,
        0.90,
    ),
    (
        "FINANCING_DILUTION",
        (
            "유상증자",
            "전환사채권발행",
            "전환사채",
            "신주인수권부사채",
            "교환사채",
            "제3자배정",
            "신주인수권",
            "단기차입금증가",
            "차입금증가",
        ),
        -0.70,
        0.85,
    ),
    (
        "SHAREHOLDER_RETURN",
        (
            "현금ㆍ현물배당",
            "현금현물배당",
            "배당결정",
            "자기주식취득결정",
            "자기주식취득신탁계약체결",
            "자기주식취득",
            "주식소각결정",
            "주식소각",
        ),
        0.80,
        0.90,
    ),
    (
        "TREASURY_STOCK",
        (
            "자기주식처분결정",
            "자기주식처분",
        ),
        -0.35,
        0.80,
    ),
    (
        "MNA_RESTRUCTURING",
        (
            "합병결정",
            "회사합병",
            "분할결정",
            "회사분할",
            "분할합병",
            "주식교환",
            "주식이전",
            "영업양수",
            "영업양도",
            "타법인주식및출자증권취득",
            "타법인주식및출자증권처분",
        ),
        0.0,
        0.90,
    ),
    (
        "INVESTMENT_CAPEX",
        (
            "신규시설투자",
            "시설투자",
            "유형자산취득",
            "유형자산양도",
            "투자결정",
        ),
        0.0,
        0.75,
    ),
    (
        "CAPITAL_STRUCTURE",
        (
            "무상증자",
            "주식분할",
            "주식병합",
            "액면분할",
            "액면병합",
            "감자결정",
            "감자",
        ),
        0.0,
        0.80,
    ),
    (
        "EARNINGS",
        (
            "영업(잠정)실적",
            "영업잠정실적",
            "잠정실적",
            "매출액또는손익구조",
            "손익구조",
            "실적공시",
        ),
        0.0,
        0.90,
    ),
    (
        "SECURITIES_ISSUANCE",
        (
            "증권발행실적보고서",
            "증권신고서",
            "투자설명서",
            "일괄신고추가서류",
            "일괄신고서",
            "발행조건확정",
        ),
        0.0,
        0.45,
    ),
    (
        "IR_GUIDANCE",
        (
            "기업설명회(IR)개최",
            "기업설명회개최",
            "IR개최",
        ),
        0.0,
        0.55,
    ),
    (
        "INSIDER_OWNERSHIP",
        (
            "임원ㆍ주요주주특정증권등소유상황보고서",
            "임원주요주주특정증권등소유상황보고서",
            "주식등의대량보유상황보고서",
            "최대주주등소유주식변동",
            "최대주주변경",
        ),
        0.0,
        0.65,
    ),
    (
        "BOARD_GOVERNANCE",
        (
            "의결권대리행사권유참고서류",
            "주주총회",
            "주주명부폐쇄기간",
            "주주명부폐쇄",
            "대표이사변경",
            "독립이사의선임",
            "독립이사의선임ㆍ해임",
            "독립이사",
            "기업지배구조",
        ),
        0.0,
        0.55,
    ),
    (
        "RELATED_PARTY",
        (
            "동일인등출자계열회사와의상품ㆍ용역거래",
            "동일인등출자계열회사와의상품용역거래",
            "특수관계인과의내부거래",
            "특수관계인에대한출자",
            "특수관계인으로부터기타유가증권매수",
            "특수관계인으로부터유가증권매수",
            "특수관계인에대한유가증권매도",
            "특수관계인과의수익증권거래",
            "특수관계인과의보험거래",
        ),
        0.0,
        0.50,
    ),
    (
        "OWNERSHIP_GOVERNANCE",
        (
            "대표이사",
            "임원변경",
            "지배구조",
        ),
        0.0,
        0.60,
    ),
    (
        "MANAGEMENT_DECISION",
        (
            "투자판단관련주요경영사항",
            "기타경영사항",
        ),
        0.0,
        0.80,
    ),
    (
        "PERIODIC_REPORT",
        (
            "사업보고서",
            "반기보고서",
            "분기보고서",
            "감사보고서",
            "연결감사보고서",
        ),
        0.0,
        0.60,
    ),
    (
        "LISTING_TRADING",
        (
            "매매거래정지해제",
            "매매거래정지",
            "신규상장",
            "추가상장",
            "변경상장",
            "상장예비심사",
        ),
        0.0,
        0.60,
    ),
)


def _compact(
    value: str,
) -> str:
    text = (
        value
        or ""
    ).strip().lower()

    return re.sub(
        r"[\s·ㆍ\-\(\)\[\]\{\}/]",
        "",
        text,
    )


def classify_disclosure_event(
    report_name: str,
) -> dict:
    original = (
        report_name
        or ""
    ).strip()

    compact = _compact(
        original
    )

    is_correction = (
        "정정" in compact
    )

    for (
        event_type,
        keywords,
        direction_prior,
        importance_prior,
    ) in _EVENT_RULES:
        for keyword in keywords:
            normalized_keyword = (
                _compact(
                    keyword
                )
            )

            if (
                normalized_keyword
                not in compact
            ):
                continue

            final_direction = (
                direction_prior
            )

            if (
                event_type
                == "CAPITAL_STRUCTURE"
            ):
                if "감자" in compact:
                    final_direction = -0.75

                elif (
                    "무상증자" in compact
                    or "주식분할" in compact
                    or "액면분할" in compact
                ):
                    final_direction = 0.25

            if (
                event_type
                == "LISTING_TRADING"
            ):
                if (
                    "매매거래정지해제"
                    in compact
                ):
                    final_direction = 0.30

                elif (
                    "매매거래정지"
                    in compact
                ):
                    final_direction = -0.80

            return {
                "event_type":
                    event_type,

                "direction_prior":
                    float(
                        final_direction
                    ),

                "importance_prior":
                    float(
                        importance_prior
                    ),

                "matched_keyword":
                    keyword,

                "is_correction":
                    is_correction,

                "normalized_name":
                    compact,
            }

    return {
        "event_type":
            "OTHER",

        "direction_prior":
            0.0,

        "importance_prior":
            0.30,

        "matched_keyword":
            None,

        "is_correction":
            is_correction,

        "normalized_name":
            compact,
    }