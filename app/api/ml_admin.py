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

router = APIRouter(
    prefix="/admin/ml",
    tags=["admin-ml"],
)


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
