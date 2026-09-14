from __future__ import annotations


class StockAnalysisBriefService:
    @staticmethod
    def _format_percent_point(
        value: float,
    ) -> str:
        return f"{value:+.2f}%p"

    @staticmethod
    def _format_volume(
        value: int,
    ) -> str:
        return f"{value:+,}주"

    @staticmethod
    def _format_metric_value(
        *,
        value: float,
        unit: str,
    ) -> str:
        if unit == "%":
            return f"{value:.1f}%"

        if unit == "배":
            return f"{value:.2f}배"

        return f"{value:.2f}{unit}"

    @staticmethod
    def _format_metric_gap(
        *,
        value: float,
        unit: str,
    ) -> str:
        if unit == "%":
            return f"{value:+.1f}%p"

        if unit == "배":
            return f"{value:+.2f}배"

        return f"{value:+.2f}{unit}"

    @staticmethod
    def _build_ranking_point(
        *,
        ranking: dict,
    ) -> dict | None:
        rank = (
            ranking.get(
                "rank"
            )
        )

        total_score = (
            ranking.get(
                "total_score"
            )
        )

        coverage = (
            ranking.get(
                "data_coverage"
            )
        )

        ml_rank = (
            ranking.get(
                "ml_rank"
            )
        )

        ml_score = (
            ranking.get(
                "ml_score"
            )
        )

        if (
            rank is None
            or total_score is None
        ):
            return None

        rank_value = int(
            rank
        )

        score_value = float(
            total_score
        )

        if rank_value <= 20:
            point_type = "POSITIVE"
        elif rank_value <= 60:
            point_type = "NEUTRAL"
        else:
            point_type = "CAUTION"

        text = (
            "장기 투자매력도 "
            f"TOP100 중 {rank_value}위, "
            f"종합점수 {score_value:.1f}점"
        )

        if coverage is not None:
            text += (
                ", 데이터 충족도 "
                f"{float(coverage):.1f}%"
            )

        text += "."

        if (
            ml_rank is not None
            and ml_score is not None
        ):
            text += (
                " 단기 ML 보조신호는 "
                f"{int(ml_rank)}위"
                f"(상대점수 "
                f"{float(ml_score):.1f})입니다."
            )

        return {
            "type":
                point_type,

            "text":
                text,
        }
    
    @classmethod
    def _build_financial_metric_points(
        cls,
        *,
        financial_peer_metrics: dict,
    ) -> list[dict]:
        if not financial_peer_metrics.get(
            "available"
        ):
            return []

        candidates = []

        for metric in (
            financial_peer_metrics.get(
                "metrics"
            )
            or []
        ):
            if not metric.get(
                "available"
            ):
                continue

            value = metric.get(
                "value"
            )

            median = metric.get(
                "peerMedian"
            )

            rank = metric.get(
                "peerRank"
            )

            peer_count = metric.get(
                "peerCount"
            )

            top_percent = metric.get(
                "topPercent"
            )

            if (
                value is None
                or median is None
                or rank is None
                or not peer_count
                or top_percent is None
            ):
                continue

            label = str(
                metric.get(
                    "label"
                )
                or metric.get(
                    "key"
                )
                or "재무지표"
            )

            unit = str(
                metric.get(
                    "unit"
                )
                or ""
            )

            state = str(
                metric.get(
                    "state"
                )
                or "NEUTRAL"
            ).upper()

            gap = (
                float(
                    value
                )
                - float(
                    median
                )
            )

            if state in {
                "STRONG",
                "GOOD",
            }:
                point_type = (
                    "POSITIVE"
                )

                priority = float(
                    top_percent
                )

            elif state in {
                "WEAK",
                "POOR",
            }:
                point_type = (
                    "CAUTION"
                )

                priority = (
                    100.0
                    - float(
                        top_percent
                    )
                )

            else:
                continue

            text = (
                f"{label} "
                f"{cls._format_metric_value(value=float(value), unit=unit)}로 "
                "업종 중앙값 "
                f"{cls._format_metric_value(value=float(median), unit=unit)} 대비 "
                f"{cls._format_metric_gap(value=gap, unit=unit)}이며, "
                f"동일 업종 {int(peer_count)}개 중 "
                f"{int(rank)}위"
                f"(상위 {float(top_percent):.1f}%)입니다."
            )

            candidates.append(
                {
                    "type":
                        point_type,

                    "text":
                        text,

                    "priority":
                        priority,
                }
            )

        positives = sorted(
            (
                item
                for item
                in candidates
                if item[
                    "type"
                ]
                == "POSITIVE"
            ),
            key=lambda item:
                item[
                    "priority"
                ],
        )[
            :3
        ]

        cautions = sorted(
            (
                item
                for item
                in candidates
                if item[
                    "type"
                ]
                == "CAUTION"
            ),
            key=lambda item:
                item[
                    "priority"
                ],
        )[
            :2
        ]

        return [
            {
                "type":
                    item[
                        "type"
                    ],

                "text":
                    item[
                        "text"
                    ],
            }
            for item
            in (
                positives
                + cautions
            )
        ]

    @staticmethod
    def _build_technical_point(
        *,
        technical_interpretation: dict,
    ) -> dict | None:
        if not technical_interpretation:
            return None

        trend = (
            technical_interpretation.get(
                "trend"
            )
            or {}
        )

        rebound = (
            technical_interpretation.get(
                "reboundSignal"
            )
            or {}
        )

        trend_state = str(
            trend.get(
                "state"
            )
            or ""
        ).upper()

        rebound_state = str(
            rebound.get(
                "state"
            )
            or ""
        ).upper()

        if rebound_state in {
            "STRONG",
            "WATCH",
        }:
            point_type = "POSITIVE"
        elif trend_state in {
            "STRONG_BEARISH",
            "BEARISH",
        }:
            point_type = "CAUTION"
        elif trend_state in {
            "STRONG_BULLISH",
            "BULLISH",
        }:
            point_type = "POSITIVE"
        else:
            point_type = "NEUTRAL"

        headline = str(
            technical_interpretation.get(
                "headline"
            )
            or ""
        ).strip()

        if not headline:
            return None

        return {
            "type": point_type,
            "text": (
                "기술적으로는 "
                f"{headline}"
            ),
        }

    @staticmethod
    def _build_long_term_financial_points(
        *,
        long_term_financial: dict,
    ) -> list[dict]:
        if not long_term_financial.get(
            "available"
        ):
            return []

        year_count = int(
            long_term_financial.get(
                "yearCount"
            )
            or 0
        )

        points: list[
            dict
        ] = []

        revenue_cagr = (
            long_term_financial.get(
                "revenueCagr"
            )
        )

        operating_income_cagr = (
            long_term_financial.get(
                "operatingIncomeCagr"
            )
        )

        if revenue_cagr is not None:
            revenue_value = float(
                revenue_cagr
            )

            if revenue_value >= 5.0:
                point_type = (
                    "POSITIVE"
                )
            elif revenue_value < 0.0:
                point_type = (
                    "CAUTION"
                )
            else:
                point_type = (
                    "NEUTRAL"
                )

            text = (
                f"최근 {year_count}개 연도 "
                f"매출 CAGR은 "
                f"{revenue_value:+.1f}%"
            )

            if (
                operating_income_cagr
                is not None
            ):
                text += (
                    ", 영업이익 CAGR은 "
                    f"{float(operating_income_cagr):+.1f}%"
                )

            text += "입니다."

            points.append(
                {
                    "type":
                        point_type,

                    "text":
                        text,
                }
            )

        average_roe = (
            long_term_financial.get(
                "averageRoe"
            )
        )

        roe_positive_ratio = (
            long_term_financial.get(
                "roePositiveYearRatio"
            )
        )

        roe_std_dev = (
            long_term_financial.get(
                "roeStdDev"
            )
        )

        if (
            average_roe is not None
            and roe_positive_ratio
            is not None
        ):
            average_roe_value = float(
                average_roe
            )

            positive_ratio_value = float(
                roe_positive_ratio
            )

            if (
                average_roe_value >= 10.0
                and positive_ratio_value >= 80.0
            ):
                point_type = (
                    "POSITIVE"
                )
            elif (
                average_roe_value < 5.0
                or positive_ratio_value < 60.0
            ):
                point_type = (
                    "CAUTION"
                )
            else:
                point_type = (
                    "NEUTRAL"
                )

            text = (
                f"장기 평균 ROE는 "
                f"{average_roe_value:.1f}%이며 "
                f"ROE가 플러스였던 연도는 "
                f"{positive_ratio_value:.0f}%입니다."
            )

            if roe_std_dev is not None:
                text += (
                    " 연도별 ROE 표준편차는 "
                    f"{float(roe_std_dev):.1f}%p입니다."
                )

            points.append(
                {
                    "type":
                        point_type,

                    "text":
                        text,
                }
            )

        margin_change = (
            long_term_financial.get(
                "operatingMarginChange"
            )
        )

        debt_change = (
            long_term_financial.get(
                "debtRatioChange"
            )
        )

        if (
            margin_change is not None
            or debt_change is not None
        ):
            margin_state = str(
                long_term_financial.get(
                    "operatingMarginTrend"
                )
                or ""
            )

            debt_state = str(
                long_term_financial.get(
                    "debtRatioTrend"
                )
                or ""
            )

            positive_count = sum(
                state == "IMPROVING"
                for state
                in (
                    margin_state,
                    debt_state,
                )
            )

            caution_count = sum(
                state == "DETERIORATING"
                for state
                in (
                    margin_state,
                    debt_state,
                )
            )

            if (
                positive_count
                > caution_count
            ):
                point_type = (
                    "POSITIVE"
                )
            elif (
                caution_count
                > positive_count
            ):
                point_type = (
                    "CAUTION"
                )
            else:
                point_type = (
                    "NEUTRAL"
                )

            parts = []

            if margin_change is not None:
                parts.append(
                    "영업이익률 "
                    f"{float(margin_change):+.1f}%p"
                )

            if debt_change is not None:
                parts.append(
                    "부채비율 "
                    f"{float(debt_change):+.1f}%p"
                )

            points.append(
                {
                    "type":
                        point_type,

                    "text":
                        (
                            f"{year_count}개 연도 기준 "
                            + ", ".join(
                                parts
                            )
                            + " 변화했습니다."
                        ),
                }
            )

        ocf_ratio = (
            long_term_financial.get(
                "positiveOperatingCashFlowRatio"
            )
        )

        fcf_ratio = (
            long_term_financial.get(
                "positiveFreeCashFlowRatio"
            )
        )

        if (
            ocf_ratio is not None
            or fcf_ratio is not None
        ):
            ratios = [
                float(
                    value
                )
                for value
                in (
                    ocf_ratio,
                    fcf_ratio,
                )
                if value is not None
            ]

            minimum_ratio = (
                min(
                    ratios
                )
                if ratios
                else 0.0
            )

            if minimum_ratio >= 80.0:
                point_type = (
                    "POSITIVE"
                )
            elif minimum_ratio < 50.0:
                point_type = (
                    "CAUTION"
                )
            else:
                point_type = (
                    "NEUTRAL"
                )

            parts = []

            if ocf_ratio is not None:
                parts.append(
                    "영업현금흐름 플러스 연도 "
                    f"{float(ocf_ratio):.0f}%"
                )

            if fcf_ratio is not None:
                parts.append(
                    "FCF 플러스 연도 "
                    f"{float(fcf_ratio):.0f}%"
                )

            points.append(
                {
                    "type":
                        point_type,

                    "text":
                        "장기 현금흐름은 "
                        + ", ".join(
                            parts
                        )
                        + "입니다.",
                }
            )

        return points

    @staticmethod
    def _build_valuation_points(
        *,
        valuation: dict,
    ) -> list[dict]:
        if not valuation.get(
            "available"
        ):
            return []

        points = []

        for key, label in (
            (
                "per",
                "PER",
            ),
            (
                "pbr",
                "PBR",
            ),
        ):
            multiple = (
                valuation.get(
                    key
                )
                or {}
            )

            value = multiple.get(
                "value"
            )

            sector = multiple.get(
                "sectorBenchmark"
            )

            relative = multiple.get(
                "sectorRelativePercent"
            )

            if (
                value is None
                or sector is None
                or relative is None
            ):
                continue

            relative_value = float(
                relative
            )

            if relative_value <= -10.0:
                point_type = (
                    "POSITIVE"
                )

                comparison = (
                    f"업종 {float(sector):.2f}배보다 "
                    f"{abs(relative_value):.1f}% 낮습니다."
                )

            elif relative_value >= 20.0:
                point_type = (
                    "CAUTION"
                )

                comparison = (
                    f"업종 {float(sector):.2f}배보다 "
                    f"{relative_value:.1f}% 높습니다."
                )

            else:
                point_type = (
                    "NEUTRAL"
                )

                comparison = (
                    f"업종 {float(sector):.2f}배와 "
                    f"{relative_value:+.1f}% 차이입니다."
                )

            points.append(
                {
                    "type":
                        point_type,

                    "text":
                        (
                            f"{label} "
                            f"{float(value):.2f}배로 "
                            f"{comparison}"
                        ),
                }
            )

        return points

    @classmethod
    def _build_benchmark_point(
        cls,
        *,
        benchmark_performance: dict,
    ) -> dict | None:
        if not benchmark_performance.get(
            "available"
        ):
            return None

        period = (
            benchmark_performance.get(
                "twentyDays"
            )
            or {}
        )

        stock_return = (
            period.get(
                "stockReturn"
            )
        )

        benchmark_return = (
            period.get(
                "benchmarkReturn"
            )
        )

        excess_return = (
            period.get(
                "excessReturn"
            )
        )

        if (
            stock_return is None
            or benchmark_return is None
            or excess_return is None
        ):
            return None

        excess_value = float(
            excess_return
        )

        benchmark_name = str(
            benchmark_performance.get(
                "benchmark"
            )
            or "시장"
        )

        if excess_value >= 3.0:
            point_type = (
                "POSITIVE"
            )

        elif excess_value <= -3.0:
            point_type = (
                "CAUTION"
            )

        else:
            point_type = (
                "NEUTRAL"
            )

        return {
            "type":
                point_type,

            "text":
                (
                    "최근 20거래일 수익률은 "
                    f"{float(stock_return):+.2f}%로 "
                    f"{benchmark_name} "
                    f"{float(benchmark_return):+.2f}% 대비 "
                    f"{cls._format_percent_point(excess_value)}입니다."
                ),
        }

    @classmethod
    def _build_investor_flow_point(
        cls,
        *,
        investor_flow: dict,
    ) -> dict | None:
        if not investor_flow.get(
            "available"
        ):
            return None

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

        flow_20d = int(
            twenty_days.get(
                "foreignInstitutionNetBuyVolume"
            )
            or 0
        )

        flow_60d = int(
            sixty_days.get(
                "foreignInstitutionNetBuyVolume"
            )
            or 0
        )

        if (
            flow_20d > 0
            and flow_60d > 0
        ):
            point_type = (
                "POSITIVE"
            )

        elif (
            flow_20d < 0
            and flow_60d < 0
        ):
            point_type = (
                "CAUTION"
            )

        else:
            point_type = (
                "NEUTRAL"
            )

        return {
            "type":
                point_type,

            "text":
                (
                    "외국인·기관 합산 수급은 "
                    "20거래일 "
                    f"{cls._format_volume(flow_20d)}, "
                    "60거래일 "
                    f"{cls._format_volume(flow_60d)}입니다."
                ),
        }
    @staticmethod
    def _build_label_and_tone(
        *,
        ranking: dict,
        positive_count: int,
        caution_count: int,
    ) -> tuple[
        str,
        str,
    ]:
        rank = int(
            ranking.get(
                "rank"
            )
            or 999
        )

        if (
            rank <= 20
            and positive_count
            >= caution_count
        ):
            return (
                "POSITIVE",
                "장기 투자매력도 상위",
            )

        if (
            caution_count
            >= positive_count + 2
        ):
            return (
                "CAUTION",
                "강점보다 위험요인 우세",
            )

        if (
            positive_count
            >= caution_count + 2
        ):
            return (
                "POSITIVE",
                "재무·가치 근거 우세",
            )

        return (
            "NEUTRAL",
            "강점과 위험요인 혼재",
        )

    @staticmethod
    def _build_market_regime_point(
        *,
        market_regime: dict,
    ) -> dict | None:
        if not market_regime.get(
            "available"
        ):
            return None

        state = (
            market_regime.get(
                "state"
            )
            or ""
        ).upper()

        label = (
            market_regime.get(
                "label"
            )
            or state
            or "확인 필요"
        )

        if state in {
            "RISK_ON",
            "BULLISH",
        }:
            point_type = "POSITIVE"
        elif state in {
            "RISK_OFF",
            "BEARISH",
        }:
            point_type = "CAUTION"
        else:
            point_type = "NEUTRAL"

        return {
            "type": point_type,
            "text": (
                f"시장 환경 {label}"
            ),
        }

    @classmethod
    def build(
        cls,
        *,
        ranking: dict,
        financial: dict | None,
        financial_peer_comparison: dict,
        financial_peer_metrics: dict,
        valuation: dict,
        long_term_financial: dict,
        investor_flow: dict,
        benchmark_performance: dict,
    ) -> dict:
        ranking_point = (
            cls._build_ranking_point(
                ranking=ranking,
            )
        )

        financial_points = (
            cls._build_financial_metric_points(
                financial_peer_metrics=(
                    financial_peer_metrics
                ),
            )
        )

        valuation_points = (
            cls._build_valuation_points(
                valuation=valuation,
            )
        )

        long_term_financial_points = (
            cls._build_long_term_financial_points(
                long_term_financial=(
                    long_term_financial
                ),
            )
        )

        investor_flow_point = (
            cls._build_investor_flow_point(
                investor_flow=(
                    investor_flow
                ),
            )
        )

        benchmark_point = (
            cls._build_benchmark_point(
                benchmark_performance=(
                    benchmark_performance
                ),
            )
        )

        evidence_points = [
            *long_term_financial_points,
            *financial_points,
            *valuation_points,

            *(
                [
                    investor_flow_point
                ]
                if investor_flow_point
                is not None
                else []
            ),

            *(
                [
                    benchmark_point
                ]
                if benchmark_point
                is not None
                else []
            ),
        ]

        positive_points = [
            point
            for point
            in evidence_points
            if point[
                "type"
            ]
            == "POSITIVE"
        ]

        caution_points = [
            point
            for point
            in evidence_points
            if point[
                "type"
            ]
            == "CAUTION"
        ]

        (
            tone,
            label,
        ) = (
            cls._build_label_and_tone(
                ranking=ranking,
                positive_count=len(
                    positive_points
                ),
                caution_count=len(
                    caution_points
                ),
            )
        )

        if positive_points:
            headline = (
                positive_points[
                    0
                ][
                    "text"
                ]
            )

        elif caution_points:
            headline = (
                caution_points[
                    0
                ][
                    "text"
                ]
            )

        elif ranking_point is not None:
            headline = (
                ranking_point[
                    "text"
                ]
            )

        else:
            headline = (
                "현재 확보된 정량 근거를 "
                "기준으로 분석했습니다."
            )

        if caution_points:
            description = (
                "주요 위험요인: "
                + " ".join(
                    point[
                        "text"
                    ]
                    for point
                    in caution_points[
                        :2
                    ]
                )
            )

        elif len(
            positive_points
        ) >= 2:
            description = (
                "추가 강점: "
                + " ".join(
                    point[
                        "text"
                    ]
                    for point
                    in positive_points[
                        1:3
                    ]
                )
            )

        elif positive_points:
            description = (
                positive_points[
                    0
                ][
                    "text"
                ]
            )

        elif ranking_point is not None:
            description = (
                ranking_point[
                    "text"
                ]
            )

        else:
            description = ""

        points = [
            *(
                [
                    ranking_point
                ]
                if ranking_point
                is not None
                else []
            ),

            *evidence_points,
        ]

        return {
            "tone":
                tone,

            "label":
                label,

            "headline":
                headline,

            "description":
                description,

            "points":
                points,

            "financialPeer":
                financial_peer_comparison,
        }