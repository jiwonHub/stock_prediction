from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.core.exceptions import ConfigurationError, ExternalApiError
from app.schemas.ml import MlBatchResponse, StockPredictionResponse
from app.services.ml_prediction_service import MlPredictionService
from app.services.historical_ml_training_service import (
    HistoricalMlTrainingService,
)
from app.services.historical_ml_oof_service import (
    HistoricalMlOofService,
)
from app.services.historical_ml_final_model_service import (
    HistoricalMlFinalModelService,
)
from app.services.historical_ml_ablation_service import (
    HistoricalMlAblationService,
)
from app.services.composite_ranking_service import (
    CompositeRankingService,
)
from app.services.ml_ranking_inference_service import (
    MlRankingInferenceService,
)

router = APIRouter(
    prefix="/admin/ml",
    tags=["admin-ml"],
)


@router.post(
    "/historical/final-model/refit",
)
def refit_historical_final_model(
    confirm: bool = Query(
        default=False,
    ),
    db: Session = Depends(
        get_db
    ),
):
    if not confirm:
        raise HTTPException(
            status_code=400,
            detail=(
                "Phase 7 최종 Production 모델 "
                "재학습입니다. 실행하려면 "
                "confirm=true를 지정하세요."
            ),
        )

    try:
        return (
            HistoricalMlFinalModelService(
                db
            )
            .train_final_model()
        )

    except (
        ValueError,
        RuntimeError,
    ) as e:
        raise HTTPException(
            status_code=400,
            detail=str(e),
        ) from e


@router.get(
    "/historical/final-model/composite-smoke",
)
def smoke_test_final_composite_ranking(
    limit: int = Query(
        default=100,
        ge=10,
        le=100,
    ),
    db: Session = Depends(
        get_db
    ),
):
    try:
        result = (
            CompositeRankingService(
                db
            )
            .score_current_universe(
                limit=limit,
            )
        )

    except (
        FileNotFoundError,
        RuntimeError,
        ValueError,
    ) as e:
        raise HTTPException(
            status_code=400,
            detail=str(e),
        ) from e

    weights = result[
        "weights"
    ]

    configured_ml_weight = float(
        weights.get(
            "ml",
            0.0,
        )
    )

    expected_ml_weight = float(
        HistoricalMlFinalModelService
        .PRODUCTION_ML_WEIGHT
    )

    if abs(
        configured_ml_weight
        - expected_ml_weight
    ) > 1e-12:
        raise HTTPException(
            status_code=400,
            detail=(
                "Composite Ranking ML 가중치가 "
                "최종 Production 기준과 "
                "일치하지 않습니다: "
                f"actual={configured_ml_weight}, "
                f"expected={expected_ml_weight}"
            ),
        )

    scores = result[
        "scores"
    ]

    if not scores:
        raise HTTPException(
            status_code=400,
            detail=(
                "Composite Ranking 결과가 "
                "비어 있습니다."
            ),
        )

    analyzed_rows = []
    formula_mismatch_count = 0
    missing_ml_count = 0

    for row in scores:
        coverage = row.get(
            "factor_coverage",
            {},
        )

        weighted_score = 0.0
        available_weight = 0.0

        for factor_name, factor_weight in (
            weights.items()
        ):
            factor_score = row.get(
                f"{factor_name}_score"
            )

            factor_coverage = (
                float(
                    coverage.get(
                        factor_name,
                        0.0,
                    )
                )
                / 100.0
            )

            if (
                factor_score is None
                or factor_coverage <= 0.0
            ):
                continue

            available_weight += (
                float(
                    factor_weight
                )
                * factor_coverage
            )

            weighted_score += (
                float(
                    factor_score
                )
                * float(
                    factor_weight
                )
                * factor_coverage
            )

        if available_weight <= 0.0:
            raise HTTPException(
                status_code=400,
                detail=(
                    "Composite Ranking의 "
                    "available weight가 0입니다: "
                    f"{row.get('stock_code')}"
                ),
            )

        expected_total_score = (
            weighted_score
            / available_weight
        )

        actual_total_score = float(
            row[
                "total_score"
            ]
        )

        formula_error = abs(
            expected_total_score
            - actual_total_score
        )

        if formula_error > 0.001:
            formula_mismatch_count += 1

        ml_score = row.get(
            "ml_score"
        )

        ml_coverage = (
            float(
                coverage.get(
                    "ml",
                    0.0,
                )
            )
            / 100.0
        )

        if (
            ml_score is None
            or ml_coverage <= 0.0
        ):
            missing_ml_count += 1

        ml_available_weight = (
            configured_ml_weight
            * ml_coverage
        )

        without_ml_available_weight = (
            available_weight
            - ml_available_weight
        )

        without_ml_weighted_score = (
            weighted_score
            - (
                float(
                    ml_score or 0.0
                )
                * configured_ml_weight
                * ml_coverage
            )
        )

        score_without_ml = (
            without_ml_weighted_score
            / without_ml_available_weight
            if without_ml_available_weight
            > 0.0
            else None
        )

        effective_ml_weight_pct = (
            (
                ml_available_weight
                / available_weight
                * 100.0
            )
            if available_weight > 0.0
            else 0.0
        )

        analyzed_rows.append(
            {
                "current_rank":
                    int(
                        row[
                            "rank"
                        ]
                    ),

                "stock_code":
                    row[
                        "stock_code"
                    ],

                "stock_name":
                    row[
                        "stock_name"
                    ],

                "total_score":
                    actual_total_score,

                "expected_total_score":
                    round(
                        expected_total_score,
                        6,
                    ),

                "formula_error":
                    round(
                        formula_error,
                        8,
                    ),

                "ml_score":
                    ml_score,

                "configured_ml_weight_pct":
                    round(
                        configured_ml_weight
                        * 100.0,
                        4,
                    ),

                "effective_ml_weight_pct":
                    round(
                        effective_ml_weight_pct,
                        4,
                    ),

                "score_without_ml":
                    (
                        round(
                            score_without_ml,
                            6,
                        )
                        if score_without_ml
                        is not None
                        else None
                    ),

                "total_score_delta_from_ml":
                    (
                        round(
                            actual_total_score
                            - score_without_ml,
                            6,
                        )
                        if score_without_ml
                        is not None
                        else None
                    ),
            }
        )

    without_ml_sorted = sorted(
        analyzed_rows,
        key=lambda row: (
            row[
                "score_without_ml"
            ]
            if row[
                "score_without_ml"
            ]
            is not None
            else float(
                "-inf"
            ),
            row[
                "stock_code"
            ],
        ),
        reverse=True,
    )

    without_ml_rank = {
        row[
            "stock_code"
        ]: index
        for index, row
        in enumerate(
            without_ml_sorted,
            start=1,
        )
    }

    for row in analyzed_rows:
        previous_rank = (
            without_ml_rank[
                row[
                    "stock_code"
                ]
            ]
        )

        row[
            "rank_without_ml_within_returned_universe"
        ] = previous_rank

        row[
            "rank_improvement_from_ml"
        ] = (
            previous_rank
            - row[
                "current_rank"
            ]
        )

    ranking_changed_count = sum(
        1
        for row in analyzed_rows
        if row[
            "rank_improvement_from_ml"
        ] != 0
    )

    max_absolute_rank_change = max(
        abs(
            row[
                "rank_improvement_from_ml"
            ]
        )
        for row in analyzed_rows
    )

    if (
        missing_ml_count > 0
        or formula_mismatch_count > 0
    ):
        raise HTTPException(
            status_code=400,
            detail={
                "message":
                    "Composite Ranking ML 통합 검증 실패",

                "missing_ml_count":
                    missing_ml_count,

                "formula_mismatch_count":
                    formula_mismatch_count,
            },
        )

    return {
        "status":
            "pass",

        "phase":
            "7.3",

        "strategy":
            result[
                "strategy"
            ],

        "weights":
            weights,

        "configured_ml_weight_pct":
            configured_ml_weight
            * 100.0,

        "candidate_count":
            result[
                "candidate_count"
            ],

        "eligible_count":
            result[
                "eligible_count"
            ],

        "universe_count":
            result[
                "universe_count"
            ],

        "ml_coverage_count":
            result[
                "coverage"
            ][
                "ml"
            ],

        "selected_ml_missing_count":
            missing_ml_count,

        "formula_mismatch_count":
            formula_mismatch_count,

        "ranking_changed_count":
            ranking_changed_count,

        "max_absolute_rank_change":
            max_absolute_rank_change,

        "top10":
            analyzed_rows[
                :10
            ],
    }


@router.get(
    "/historical/final-model/inference-smoke",
)
def smoke_test_final_model_inference(
    limit: int = Query(
        default=100,
        ge=10,
        le=100,
    ),
    db: Session = Depends(
        get_db
    ),
):
    try:
        service = (
            MlRankingInferenceService(
                db
            )
        )

        result = (
            service
            .score_current_universe(
                limit=limit,
                include_explanations=False,
            )
        )

    except (
        FileNotFoundError,
        RuntimeError,
        ValueError,
    ) as e:
        raise HTTPException(
            status_code=400,
            detail=str(e),
        ) from e

    scores = result[
        "scores"
    ]

    top10 = [
        {
            "ml_rank":
                row[
                    "ml_rank"
                ],

            "stock_code":
                row[
                    "stock_code"
                ],

            "stock_name":
                row[
                    "stock_name"
                ],

            "feature_date":
                row[
                    "feature_date"
                ].isoformat(),

            "raw_probability_pct":
                row[
                    "raw_probability_pct"
                ],

            "ml_score":
                row[
                    "ml_score"
                ],
        }
        for row
        in scores[:10]
    ]

    return {
        "status":
            "pass",

        "phase":
            "7.2",

        "model_name":
            result[
                "model_name"
            ],

        "model_version":
            result[
                "model_version"
            ],

        "refit_version":
            service.artifact.get(
                "refit_version"
            ),

        "artifact_created_at":
            service.artifact.get(
                "created_at"
            ),

        "feature_count":
            result[
                "feature_count"
            ],

        "final_selection":
            service.artifact.get(
                "final_selection"
            ),

        "requested_universe":
            result[
                "requested_universe"
            ],

        "scored_count":
            result[
                "scored_count"
            ],

        "skipped_count":
            result[
                "skipped_count"
            ],

        "feature_date_min":
            result[
                "feature_date_min"
            ],

        "feature_date_max":
            result[
                "feature_date_max"
            ],

        "raw_probability_min_pct":
            result[
                "raw_probability_min_pct"
            ],

        "raw_probability_mean_pct":
            result[
                "raw_probability_mean_pct"
            ],

        "raw_probability_max_pct":
            result[
                "raw_probability_max_pct"
            ],

        "top10":
            top10,

        "skipped":
            result[
                "skipped"
            ],
    }


@router.get(
    "/historical/split/inspect",
)
def inspect_historical_ml_split(
    feature_version: str = Query(
        default=(
            HistoricalMlTrainingService
            .DEFAULT_FEATURE_VERSION
        ),
        min_length=1,
        max_length=40,
    ),
    horizon: int = Query(
        default=5,
    ),
    train_ratio: float = Query(
        default=0.70,
        ge=0.50,
        le=0.85,
    ),
    valid_ratio: float = Query(
        default=0.15,
        ge=0.05,
        le=0.30,
    ),
    db: Session = Depends(
        get_db
    ),
):
    try:
        return (
            HistoricalMlTrainingService(
                db
            )
            .inspect_temporal_split(
                feature_version=(
                    feature_version
                ),
                horizon=horizon,
                train_ratio=(
                    train_ratio
                ),
                valid_ratio=(
                    valid_ratio
                ),
            )
        )

    except ValueError as e:
        raise HTTPException(
            status_code=400,
            detail=str(e),
        ) from e


@router.post(
    "/historical/oof/compare-horizons",
)
def compare_historical_ml_horizons(
    horizons: str = Query(
        default="5,20",
        min_length=1,
        max_length=20,
    ),
    feature_version: str = Query(
        default=(
            HistoricalMlFinalModelService
            .FEATURE_VERSION
        ),
        min_length=1,
        max_length=40,
    ),
    folds: int = Query(
        default=4,
        ge=3,
        le=8,
    ),
    initial_train_ratio: float = Query(
        default=0.55,
        ge=0.40,
        le=0.70,
    ),
    db: Session = Depends(
        get_db
    ),
):
    try:
        parsed_horizons = list(
            dict.fromkeys(
                int(
                    value.strip()
                )
                for value
                in horizons.split(",")
                if value.strip()
            )
        )

    except ValueError as e:
        raise HTTPException(
            status_code=400,
            detail=(
                "horizons는 1,5,20 형식으로 "
                "입력해야 합니다."
            ),
        ) from e

    if (
        not parsed_horizons
        or any(
            horizon not in (
                HistoricalMlTrainingService
                .SUPPORTED_HORIZONS
            )
            for horizon
            in parsed_horizons
        )
    ):
        raise HTTPException(
            status_code=400,
            detail=(
                "horizon은 1, 5, 20만 "
                "지원합니다."
            ),
        )

    service = HistoricalMlOofService(
        db
    )

    def find_portfolio_summary(
        backtest: dict,
        *,
        portfolio_size: int,
        transaction_cost_bps: float,
    ) -> dict | None:
        for scenario in backtest.get(
            "scenarios",
            [],
        ):
            if (
                scenario.get(
                    "portfolio_size"
                )
                == portfolio_size
                and float(
                    scenario.get(
                        "transaction_cost_bps",
                        -1.0,
                    )
                )
                == transaction_cost_bps
            ):
                return dict(
                    scenario[
                        "summary"
                    ]
                )

        return None

    def find_buffer_summary(
        backtest: dict,
        *,
        portfolio_size: int,
        exit_rank: int,
        transaction_cost_bps: float,
    ) -> dict | None:
        for scenario in backtest.get(
            "scenarios",
            [],
        ):
            if (
                scenario.get(
                    "portfolio_size"
                )
                == portfolio_size
                and scenario.get(
                    "exit_rank"
                )
                == exit_rank
                and float(
                    scenario.get(
                        "transaction_cost_bps",
                        -1.0,
                    )
                )
                == transaction_cost_bps
            ):
                return dict(
                    scenario[
                        "summary"
                    ]
                )

        return None

    comparison = []

    for horizon in parsed_horizons:
        try:
            result = service.run(
                feature_version=(
                    feature_version
                ),
                horizon=horizon,
                folds=folds,
                initial_train_ratio=(
                    initial_train_ratio
                ),
            )

        except ValueError as e:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"horizon={horizon}: "
                    f"{e}"
                ),
            ) from e

        probability_metrics = dict(
            result[
                "oof_probability_metrics"
            ]
        )

        ranking_metrics = dict(
            result[
                "oof_ranking_metrics"
            ]
        )

        comparison.append(
            {
                "horizon":
                    horizon,

                "oofRows":
                    result[
                        "oof_rows"
                    ],

                "rankingDates":
                    ranking_metrics[
                        "ranking_dates"
                    ],

                "averageStocksPerDate":
                    ranking_metrics[
                        "average_stocks_per_date"
                    ],

                "probabilityMetrics": {
                    "rocAuc":
                        probability_metrics[
                            "roc_auc"
                        ],

                    "logLoss":
                        probability_metrics[
                            "log_loss"
                        ],

                    "brierScore":
                        probability_metrics[
                            "brier_score"
                        ],

                    "ece10":
                        probability_metrics[
                            "ece_10"
                        ],
                },

                "rankingMetrics": {
                    "spearmanIcMean":
                        ranking_metrics[
                            "spearman_ic_mean"
                        ],

                    "spearmanIcPositiveRatePct":
                        ranking_metrics[
                            "spearman_ic_positive_rate_pct"
                        ],

                    "top10ExcessMeanPct":
                        ranking_metrics[
                            "top10_excess_mean_pct"
                        ],

                    "top10ExcessPositiveRatePct":
                        ranking_metrics[
                            "top10_excess_positive_rate_pct"
                        ],

                    "top20ExcessMeanPct":
                        ranking_metrics[
                            "top20_excess_mean_pct"
                        ],

                    "longShort10MeanPct":
                        ranking_metrics[
                            "long_short_10_mean_pct"
                        ],

                    "nonOverlappingSummary":
                        ranking_metrics[
                            "non_overlapping_summary"
                        ],
                },

                "top10Cost20bps":
                    find_portfolio_summary(
                        ranking_metrics[
                            "portfolio_backtest"
                        ],
                        portfolio_size=10,
                        transaction_cost_bps=20.0,
                    ),

                "top10Exit20Cost20bps":
                    find_buffer_summary(
                        ranking_metrics[
                            "turnover_buffer_backtest"
                        ],
                        portfolio_size=10,
                        exit_rank=20,
                        transaction_cost_bps=20.0,
                    ),

                "testDatasetUsed":
                    result[
                        "test_dataset_used"
                    ],

                "lockedTestFirstDate":
                    result[
                        "locked_test_first_date"
                    ],
            }
        )

    return {
        "status":
            "completed",

        "featureVersion":
            feature_version,

        "productionModel": {
            "modelName":
                HistoricalMlFinalModelService
                .MODEL_NAME,

            "modelVersion":
                HistoricalMlFinalModelService
                .MODEL_VERSION,

            "currentHorizon":
                HistoricalMlFinalModelService
                .HORIZON,

            "featureStrategy":
                HistoricalMlFinalModelService
                .FEATURE_STRATEGY,

            "intendedUse":
                HistoricalMlFinalModelService
                .INTENDED_USE,

            "lockedTestMetrics":
                dict(
                    HistoricalMlFinalModelService
                    .LOCKED_TEST_METRICS
                ),
        },

        "comparison":
            comparison,
    }
@router.post(
    "/historical/oof/feature-permutation",
)
def inspect_historical_feature_permutation(
    feature_version: str = Query(
        default=(
            HistoricalMlFinalModelService
            .FEATURE_VERSION
        ),
        min_length=1,
        max_length=40,
    ),
    folds: int = Query(
        default=4,
        ge=3,
        le=8,
    ),
    initial_train_ratio: float = Query(
        default=0.55,
        ge=0.40,
        le=0.70,
    ),
    db: Session = Depends(
        get_db
    ),
):
    try:
        return (
            HistoricalMlAblationService(
                db
            )
            .run_oof_feature_permutation(
                feature_version=(
                    feature_version
                ),
                horizon=(
                    HistoricalMlFinalModelService
                    .HORIZON
                ),
                folds=folds,
                initial_train_ratio=(
                    initial_train_ratio
                ),
            )
        )

    except ValueError as e:
        raise HTTPException(
            status_code=400,
            detail=str(e),
        ) from e


@router.post(
    "/historical/oof/objective-compare",
)
def compare_historical_objective(
    experiment: str = Query(
        ...,
        min_length=1,
        max_length=60,
    ),
    feature_version: str = Query(
        default=(
            HistoricalMlFinalModelService
            .FEATURE_VERSION
        ),
        min_length=1,
        max_length=40,
    ),
    folds: int = Query(
        default=4,
        ge=3,
        le=8,
    ),
    initial_train_ratio: float = Query(
        default=0.55,
        ge=0.40,
        le=0.70,
    ),
    db: Session = Depends(
        get_db
    ),
):
    try:
        return (
            HistoricalMlAblationService(
                db
            )
            .run_oof_objective_experiment(
                feature_version=(
                    feature_version
                ),
                experiment=(
                    experiment
                ),
                horizon=(
                    HistoricalMlFinalModelService
                    .HORIZON
                ),
                folds=folds,
                initial_train_ratio=(
                    initial_train_ratio
                ),
            )
        )

    except ValueError as e:
        raise HTTPException(
            status_code=400,
            detail=str(e),
        ) from e


@router.post(
    "/historical/oof/model-tuning-compare",
)
def compare_historical_model_tuning(
    experiment: str = Query(
        ...,
        min_length=1,
        max_length=60,
    ),
    feature_version: str = Query(
        default=(
            HistoricalMlFinalModelService
            .FEATURE_VERSION
        ),
        min_length=1,
        max_length=40,
    ),
    folds: int = Query(
        default=4,
        ge=3,
        le=8,
    ),
    initial_train_ratio: float = Query(
        default=0.55,
        ge=0.40,
        le=0.70,
    ),
    db: Session = Depends(
        get_db
    ),
):
    try:
        return (
            HistoricalMlAblationService(
                db
            )
            .run_oof_model_tuning_experiment(
                feature_version=(
                    feature_version
                ),
                experiment=(
                    experiment
                ),
                horizon=(
                    HistoricalMlFinalModelService
                    .HORIZON
                ),
                folds=folds,
                initial_train_ratio=(
                    initial_train_ratio
                ),
            )
        )

    except ValueError as e:
        raise HTTPException(
            status_code=400,
            detail=str(e),
        ) from e


@router.post(
    "/historical/oof/feature-subset-compare",
)
def compare_historical_feature_subset(
    experiment: str = Query(
        ...,
        min_length=1,
        max_length=40,
    ),
    feature_version: str = Query(
        default=(
            HistoricalMlFinalModelService
            .FEATURE_VERSION
        ),
        min_length=1,
        max_length=40,
    ),
    folds: int = Query(
        default=4,
        ge=3,
        le=8,
    ),
    initial_train_ratio: float = Query(
        default=0.55,
        ge=0.40,
        le=0.70,
    ),
    db: Session = Depends(
        get_db
    ),
):
    try:
        return (
            HistoricalMlAblationService(
                db
            )
            .run_oof_feature_subset_experiment(
                feature_version=(
                    feature_version
                ),
                experiment=(
                    experiment
                ),
                horizon=(
                    HistoricalMlFinalModelService
                    .HORIZON
                ),
                folds=folds,
                initial_train_ratio=(
                    initial_train_ratio
                ),
            )
        )

    except ValueError as e:
        raise HTTPException(
            status_code=400,
            detail=str(e),
        ) from e


@router.post(
    "/historical/locked-test/final-compare",
)
def compare_locked_test_final_candidate(
    confirm: bool = Query(
        default=False,
    ),
    feature_version: str = Query(
        default=(
            HistoricalMlFinalModelService
            .FEATURE_VERSION
        ),
        min_length=1,
        max_length=40,
    ),
    db: Session = Depends(
        get_db
    ),
):
    if not confirm:
        raise HTTPException(
            status_code=400,
            detail=(
                "Locked Test 최종 검증입니다. "
                "실행하려면 confirm=true를 지정하세요."
            ),
        )

    try:
        return (
            HistoricalMlAblationService(
                db
            )
            .run_locked_test_final_comparison(
                feature_version=(
                    feature_version
                ),
                horizon=(
                    HistoricalMlFinalModelService
                    .HORIZON
                ),
            )
        )

    except ValueError as e:
        raise HTTPException(
            status_code=400,
            detail=str(e),
        ) from e


@router.post(
    "/historical/oof/candidate-ml-weight-sweep",
)
def compare_candidate_historical_ml_weights(
    feature_version: str = Query(
        default=(
            HistoricalMlFinalModelService
            .FEATURE_VERSION
        ),
        min_length=1,
        max_length=40,
    ),
    folds: int = Query(
        default=4,
        ge=3,
        le=8,
    ),
    initial_train_ratio: float = Query(
        default=0.55,
        ge=0.40,
        le=0.70,
    ),
    db: Session = Depends(
        get_db
    ),
):
    try:
        result = (
            HistoricalMlOofService(
                db
            )
            .run(
                feature_version=(
                    feature_version
                ),
                horizon=(
                    HistoricalMlFinalModelService
                    .HORIZON
                ),
                folds=folds,
                initial_train_ratio=(
                    initial_train_ratio
                ),
                excluded_features={
                    "rsi_14",
                },
                classifier_params=(
                    dict(
                        HistoricalMlOofService
                        .FINAL_PARAMS
                    )
                ),
                model_mode=
                    "classifier",
            )
        )

    except ValueError as e:
        raise HTTPException(
            status_code=400,
            detail=str(e),
        ) from e

    sweep = result[
        "ml_weight_sweep"
    ]

    return {
        "status":
            "completed",

        "phase":
            "6.3-5",

        "candidate": {
            "feature_count": 26,

            "removed_features": [
                "rsi_14",
            ],

            "objective":
                "binary:logistic",

            "classifier_params":
                dict(
                    HistoricalMlOofService
                    .FINAL_PARAMS
                ),
        },

        "featureVersion":
            feature_version,

        "horizon":
            HistoricalMlFinalModelService
            .HORIZON,

        "currentProductionMlWeightPct":
            (
                CompositeRankingService
                .WEIGHTS[
                    "ml"
                ]
                * 100.0
            ),

        "testDatasetUsed":
            result[
                "test_dataset_used"
            ],

        "lockedTestFirstDate":
            result[
                "locked_test_first_date"
            ],

        "historicalFinancialCoverage":
            result[
                "historical_financial_coverage"
            ],

        "historicalFlowCoverage":
            result[
                "historical_flow_coverage"
            ],

        "proxyDefinition":
            sweep[
                "proxy_definition"
            ],

        "comparison":
            sweep[
                "results"
            ],
    }


@router.post(
    "/historical/oof/ml-weight-sweep",
)
def compare_historical_ml_weights(
    feature_version: str = Query(
        default=(
            HistoricalMlFinalModelService
            .FEATURE_VERSION
        ),
        min_length=1,
        max_length=40,
    ),
    folds: int = Query(
        default=4,
        ge=3,
        le=8,
    ),
    initial_train_ratio: float = Query(
        default=0.55,
        ge=0.40,
        le=0.70,
    ),
    db: Session = Depends(
        get_db
    ),
):
    try:
        result = (
            HistoricalMlOofService(
                db
            )
            .run(
                feature_version=(
                    feature_version
                ),
                horizon=(
                    HistoricalMlFinalModelService
                    .HORIZON
                ),
                folds=folds,
                initial_train_ratio=(
                    initial_train_ratio
                ),
            )
        )

    except ValueError as e:
        raise HTTPException(
            status_code=400,
            detail=str(e),
        ) from e

    sweep = result[
        "ml_weight_sweep"
    ]

    return {
        "status":
            "completed",

        "phase":
            "6.2",

        "featureVersion":
            feature_version,

        "horizon":
            HistoricalMlFinalModelService
            .HORIZON,

        "productionMlWeightPct":
            (
                CompositeRankingService
                .WEIGHTS[
                    "ml"
                ]
                * 100.0
            ),

        "testDatasetUsed":
            result[
                "test_dataset_used"
            ],

        "lockedTestFirstDate":
            result[
                "locked_test_first_date"
            ],

        "historicalFinancialCoverage":
            result[
                "historical_financial_coverage"
            ],

        "historicalFlowCoverage":
            result[
                "historical_flow_coverage"
            ],

        "proxyDefinition":
            sweep[
                "proxy_definition"
            ],

        "comparison":
            sweep[
                "results"
            ],
    }


@router.post(
    "/predict/batch",
    response_model=MlBatchResponse,
)

async def predict_stock_models_batch(
    limit: int = Query(
        default=100,
        ge=1,
        le=100,
    ),
    horizon_days: int = Query(
        default=5,
        ge=1,
        le=20,
    ),
    db: Session = Depends(get_db),
):
    service = MlPredictionService(
        db
    )

    stock_codes = (
        service.repository
        .get_stock_codes_with_predictions(
            horizon_days=horizon_days,
            limit=limit,
        )
    )

    predicted = 0
    skipped = 0
    total = len(stock_codes)

    for (
        index,
        stock_code,
    ) in enumerate(
        stock_codes,
        start=1,
    ):
        print(
            f"[PREDICT] "
            f"{index}/{total} "
            f"{stock_code}",
            flush=True,
        )

        try:
            await service.predict_existing(
                stock_code,
                horizon_days=horizon_days,
            )

            predicted += 1

        except (
            ValueError,
            ConfigurationError,
            ExternalApiError,
            SQLAlchemyError,
        ) as e:
            db.rollback()

            skipped += 1

            print(
                f"[PREDICT] "
                f"{stock_code} "
                f"건너뜀: {e}",
                flush=True,
            )

    return MlBatchResponse(
        trainedCount=predicted,
        skippedCount=skipped,
        message=(
            f"기존 모델 예측 "
            f"{predicted}개 완료, "
            f"{skipped}개 건너뜀"
        ),
    )

@router.post(
    "/train/batch",
    response_model=MlBatchResponse,
)
async def train_stock_models_batch(
    limit: int = Query(default=10, ge=1, le=100),
    horizon_days: int = Query(default=5, ge=1, le=20),
    days: int = Query(default=1200, ge=180, le=3650),
    sync_prices: bool = Query(default=True),
    db: Session = Depends(get_db),
):
    service = MlPredictionService(db)
    stock_codes = service.repository.get_stock_codes_for_ml(limit=limit)

    trained = 0
    skipped = 0

    for stock_code in stock_codes:
        try:
            await service.train(
                stock_code,
                horizon_days=horizon_days,
                days=days,
                sync_prices=sync_prices,
            )
            trained += 1
        except (
            ValueError,
            ConfigurationError,
            ExternalApiError,
            SQLAlchemyError,
        ):
            db.rollback()
            skipped += 1

    return MlBatchResponse(
        trainedCount=trained,
        skippedCount=skipped,
        message=f"ML 모델 {trained}개 학습 완료, {skipped}개 건너뜀",
    )


@router.post(
    "/train/{stock_code}",
    response_model=StockPredictionResponse,
)
async def train_stock_model(
    stock_code: str,
    horizon_days: int = Query(default=5, ge=1, le=20),
    days: int = Query(default=1200, ge=180, le=3650),
    sync_prices: bool = Query(default=True),
    db: Session = Depends(get_db),
):
    try:
        return await MlPredictionService(db).train(
            stock_code,
            horizon_days=horizon_days,
            days=days,
            sync_prices=sync_prices,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except ConfigurationError as e:
        raise HTTPException(status_code=500, detail=str(e)) from e
    except ExternalApiError as e:
        raise HTTPException(status_code=502, detail=str(e)) from e
