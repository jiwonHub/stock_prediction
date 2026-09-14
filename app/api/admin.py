from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session
import asyncio
import time
from datetime import date, timedelta
from app.core.database import get_db
from app.core.exceptions import ConfigurationError, ExternalApiError
from app.schemas.stock import StockResponse, SyncResponse
from app.services.market_data_service import (
    MarketDataService,
)
from app.services.market_context_service import (
    MarketContextService,
)
from app.services.stock_service import StockService
from app.services.feature_engine_service import (
    FeatureEngineService,
)
from app.services.feature_quality_service import (
    FeatureQualityService,
)
from app.services.feature_matrix_service import (
    FeatureMatrixService,
)
from app.services.historical_feature_service import (
    HistoricalFeatureService,
)
from app.services.historical_target_service import (
    HistoricalTargetService,
)
from app.services.historical_dataset_service import (
    HistoricalDatasetService,
)
from app.services.technical_analysis_service import (
    TechnicalAnalysisService,
)
from app.services.feature_redundancy_service import (
    FeatureRedundancyService,
)
from app.services.market_regime_service import (
    MarketRegimeService,
)
from app.services.sector_regime_service import (
    SectorRegimeService,
)
from app.services.disclosure_taxonomy_service import (
    DisclosureTaxonomyService,
)
from app.services.disclosure_analysis_service import (
    DisclosureAnalysisService,
)
from app.services.disclosure_feature_service import (
    DisclosureFeatureService,
)

router = APIRouter(
    prefix="/admin/sync",
    tags=["admin-sync"],
)


@router.post(
    "/stocks",
    response_model=SyncResponse,
)
async def sync_stocks(
    db: Session = Depends(get_db),
):
    service = StockService(db)

    try:
        count = await service.sync_stocks_from_dart()
    except ConfigurationError as e:
        raise HTTPException(
            status_code=500,
            detail=str(e),
        ) from e
    except ExternalApiError as e:
        raise HTTPException(
            status_code=502,
            detail=str(e),
        ) from e

    return SyncResponse(
        count=count,
        message=f"DART 상장 종목 {count}건 저장 완료",
    )

@router.post(
    "/prices",
    response_model=SyncResponse,
)
async def sync_prices(
    limit: int = Query(
        default=100,
        ge=1,
        le=100,
    ),
    db: Session = Depends(get_db),
):
    service = StockService(db)

    try:
        success, skipped = (
            await service.sync_ranked_current_prices(
                limit=limit,
            )
        )
    except ConfigurationError as e:
        raise HTTPException(
            status_code=500,
            detail=str(e),
        ) from e
    except ExternalApiError as e:
        raise HTTPException(
            status_code=502,
            detail=str(e),
        ) from e

    return SyncResponse(
        count=success,
        message=(
            f"TOP {limit} 현재가 "
            f"{success}건 갱신, "
            f"{skipped}건 건너뜀"
        ),
    )

@router.post(
    "/price/{stock_code}",
    response_model=StockResponse,
)
async def sync_price(
    stock_code: str,
    db: Session = Depends(get_db),
):
    service = StockService(db)

    try:
        return await service.sync_current_price(
            stock_code
        )
    except ConfigurationError as e:
        raise HTTPException(
            status_code=500,
            detail=str(e),
        ) from e
    except ExternalApiError as e:
        raise HTTPException(
            status_code=502,
            detail=str(e),
        ) from e


@router.post(
    "/chart/{stock_code}",
    response_model=SyncResponse,
)
async def sync_chart(
    stock_code: str,
    days: int = Query(
        default=365,
        ge=7,
        le=3650,
    ),
    db: Session = Depends(get_db),
):
    service = StockService(db)

    try:
        count = await service.sync_daily_prices(
            stock_code,
            days=days,
        )
    except ValueError as e:
        raise HTTPException(
            status_code=404,
            detail=str(e),
        ) from e
    except ConfigurationError as e:
        raise HTTPException(
            status_code=500,
            detail=str(e),
        ) from e
    except ExternalApiError as e:
        raise HTTPException(
            status_code=502,
            detail=str(e),
        ) from e

    return SyncResponse(
        count=count,
        message=f"{stock_code} 일봉 {count}건 저장 완료",
    )


@router.post(
    "/financials/{stock_code}",
    response_model=SyncResponse,
)
async def sync_financials(
    stock_code: str,
    year: int = Query(
        ge=2015,
        le=2100,
    ),
    report_code: str = Query(
        default="11011",
        pattern="^(11011|11012|11013|11014)$",
    ),
    db: Session = Depends(get_db),
):
    service = StockService(db)

    try:
        count, fs_div = await service.sync_financials(
            stock_code,
            business_year=str(year),
            report_code=report_code,
        )
    except ValueError as e:
        raise HTTPException(
            status_code=404,
            detail=str(e),
        ) from e
    except ConfigurationError as e:
        raise HTTPException(
            status_code=500,
            detail=str(e),
        ) from e
    except ExternalApiError as e:
        raise HTTPException(
            status_code=502,
            detail=str(e),
        ) from e

    return SyncResponse(
        count=count,
        message=(
            f"{stock_code} {year} {report_code} "
            f"{fs_div} 재무제표 {count}건 저장 완료"
        ),
    )

@router.post(
    "/market-context",
    response_model=SyncResponse,
)
async def sync_market_context(
    days: int = Query(
        default=365,
        ge=30,
        le=3650,
    ),
    db: Session = Depends(get_db),
):
    service = MarketDataService(
        db
    )

    try:
        (
            index_count,
            macro_count,
        ) = (
            await service
            .sync_market_context(
                days=days
            )
        )

    except ConfigurationError as e:
        raise HTTPException(
            status_code=500,
            detail=str(e),
        ) from e

    except ExternalApiError as e:
        raise HTTPException(
            status_code=502,
            detail=str(e),
        ) from e

    total = (
        index_count
        + macro_count
    )

    return SyncResponse(
        count=total,
        message=(
            "시장 데이터 동기화 완료: "
            f"지수 {index_count}건, "
            f"거시경제 {macro_count}건"
        ),
    )


@router.post(
    "/market-context/stock/{stock_code}",
    response_model=SyncResponse,
)
async def sync_stock_market_context(
    stock_code: str,
    db: Session = Depends(get_db),
):
    service = MarketDataService(
        db
    )

    try:
        (
            investor_count,
            valuation_count,
        ) = (
            await service
            .sync_stock_context(
                stock_code
            )
        )

    except ValueError as e:
        raise HTTPException(
            status_code=404,
            detail=str(e),
        ) from e

    except ConfigurationError as e:
        raise HTTPException(
            status_code=500,
            detail=str(e),
        ) from e

    except ExternalApiError as e:
        raise HTTPException(
            status_code=502,
            detail=str(e),
        ) from e

    total = (
        investor_count
        + valuation_count
    )

    return SyncResponse(
        count=total,
        message=(
            f"{stock_code} 시장 데이터 "
            "동기화 완료: "
            f"투자자 수급 "
            f"{investor_count}건, "
            f"밸류에이션 "
            f"{valuation_count}건"
        ),
    )


@router.post(
    "/market-context/ranked",
    response_model=SyncResponse,
)
async def sync_ranked_market_context(
    limit: int = Query(
        default=100,
        ge=1,
        le=100,
    ),
    db: Session = Depends(get_db),
):
    service = MarketDataService(
        db
    )

    try:
        (
            success_stocks,
            investor_count,
            valuation_count,
            investor_failures,
            valuation_failures,
        ) = (
            await service
            .sync_ranked_stock_context(
                limit=limit
            )
        )

    except ConfigurationError as e:
        raise HTTPException(
            status_code=500,
            detail=str(e),
        ) from e

    except ExternalApiError as e:
        raise HTTPException(
            status_code=502,
            detail=str(e),
        ) from e

    total_rows = (
        investor_count
        + valuation_count
    )

    return SyncResponse(
        count=total_rows,
        message=(
            "TOP "
            f"{limit} 시장 데이터 "
            "동기화 완료: "
            f"완전 성공 {success_stocks}종목, "
            f"수급 {investor_count}건, "
            f"밸류에이션 {valuation_count}건, "
            f"수급 실패 {investor_failures}건, "
            f"밸류에이션 실패 "
            f"{valuation_failures}건"
        ),
    )

@router.post(
    "/market-context/valuation-benchmarks",
    response_model=SyncResponse,
)
async def refresh_valuation_benchmarks(
    db: Session = Depends(get_db),
):
    service = MarketDataService(
        db
    )

    (
        updated_count,
        sector_count,
        market_count,
    ) = (
        service
        .refresh_valuation_benchmarks()
    )

    return SyncResponse(
        count=updated_count,
        message=(
            "밸류에이션 benchmark 계산 완료: "
            f"{updated_count}종목, "
            f"{sector_count}개 업종, "
            f"{market_count}개 시장"
        ),
    )


@router.post(
    "/sector-index/{sector_code}",
    response_model=SyncResponse,
)
async def sync_sector_index(
    sector_code: str,
    sector_name: str = Query(
        min_length=1,
    ),
    market: str = Query(
        default="KRX",
    ),
    days: int = Query(
        default=365,
        ge=30,
        le=3650,
    ),
    db: Session = Depends(get_db),
):
    service = MarketDataService(
        db
    )

    try:
        count = (
            await service
            .sync_sector_index(
                sector_code=sector_code,
                sector_name=sector_name,
                market=market,
                days=days,
            )
        )

    except ConfigurationError as e:
        raise HTTPException(
            status_code=500,
            detail=str(e),
        ) from e

    except ExternalApiError as e:
        raise HTTPException(
            status_code=502,
            detail=str(e),
        ) from e

    return SyncResponse(
        count=count,
        message=(
            f"{sector_name} "
            f"업종지수 {count}건 "
            "저장 완료"
        ),
    )

@router.post(
    "/market-context/sector-indices",
    response_model=SyncResponse,
)
async def sync_top_sector_indices(
    limit: int = Query(
        default=100,
        ge=1,
        le=500,
    ),
    days: int = Query(
        default=365,
        ge=1,
        le=3650,
    ),
    db: Session = Depends(get_db),
):
    service = MarketDataService(
        db
    )

    try:
        (
            metadata_count,
            total_sectors,
            success_sectors,
            index_rows,
            failed_sectors,
        ) = (
            await service
            .sync_top_universe_sector_indices(
                limit=limit,
                days=days,
            )
        )

    except ConfigurationError as e:
        raise HTTPException(
            status_code=500,
            detail=str(e),
        ) from e

    except ExternalApiError as e:
        raise HTTPException(
            status_code=502,
            detail=str(e),
        ) from e

    return SyncResponse(
        count=index_rows,
        message=(
            "TOP Universe 업종 동기화 완료: "
            f"종목 메타데이터 "
            f"{metadata_count}건, "
            f"업종 {total_sectors}개, "
            f"성공 {success_sectors}개, "
            f"실패 {failed_sectors}개, "
            f"업종지수 {index_rows}건"
        ),
    )

@router.post(
    "/market-context/sector-indices/stream",
)
async def stream_top_sector_indices(
    limit: int = Query(
        default=100,
        ge=1,
        le=500,
    ),
    days: int = Query(
        default=365,
        ge=1,
        le=3650,
    ),
    db: Session = Depends(get_db),
):
    service = MarketDataService(
        db
    )

    async def stream():
        started_at = (
            time.monotonic()
        )

        task = asyncio.create_task(
            service
            .sync_top_universe_sector_indices(
                limit=limit,
                days=days,
            )
        )

        while not task.done():
            elapsed = int(
                time.monotonic()
                - started_at
            )

            yield (
                "[RUNNING] "
                f"{elapsed}초 경과\n"
            )

            await asyncio.sleep(
                1.0
            )

        try:
            (
                metadata_count,
                total_sectors,
                success_sectors,
                index_rows,
                failed_sectors,
            ) = await task

            yield (
                "[DONE] "
                f"메타데이터 "
                f"{metadata_count}건, "
                f"업종 "
                f"{total_sectors}개, "
                f"성공 "
                f"{success_sectors}개, "
                f"실패 "
                f"{failed_sectors}개, "
                f"업종지수 "
                f"{index_rows}건\n"
            )

        except Exception as e:
            yield (
                "[ERROR] "
                f"{type(e).__name__}: "
                f"{e}\n"
            )

    return StreamingResponse(
        stream(),
        media_type=(
            "text/plain; "
            "charset=utf-8"
        ),
        headers={
            "Cache-Control":
                "no-cache",
            "X-Accel-Buffering":
                "no",
        },
    )

@router.post(
    "/market-context/investor-flows/historical/stream",
)
async def stream_historical_investor_flows(
    feature_version: str = Query(
        default=HistoricalFeatureService.FEATURE_VERSION,
        min_length=1,
        max_length=40,
    ),
    horizon: int = Query(
        default=5,
        ge=1,
        le=20,
    ),
    offset: int = Query(
        default=0,
        ge=0,
    ),
    limit: int = Query(
        default=10,
        ge=1,
        le=50,
    ),
    lookback_days: int = Query(
        default=45,
        ge=30,
        le=120,
    ),
    db: Session = Depends(get_db),
):
    market_service = MarketDataService(db)
    dataset_service = HistoricalDatasetService(db)
    stock_service = StockService(db)

    async def generate():
        ranges = dataset_service.get_stock_date_ranges(
            feature_version=feature_version,
            horizon=horizon,
        )

        total_universe = len(ranges)

        selected = ranges[
            offset:offset + limit
        ]

        if not selected:
            yield (
                "[ERROR] 처리할 Historical 종목이 없습니다. "
                f"전체 {total_universe}종목, "
                f"offset={offset}\n"
            )
            return

        success = 0
        failed = 0
        total_rows = 0

        yield (
            "[START] Historical Investor Flow "
            f"전체 {total_universe}종목 중 "
            f"offset {offset}, "
            f"{len(selected)}종목 처리\n"
        )

        try:
            for index, (
                stock_code,
                first_feature_date,
                last_feature_date,
            ) in enumerate(
                selected,
                start=1,
            ):
                stock = stock_service.repository.get_stock(
                    stock_code
                )

                stock_name = (
                    stock.name
                    if stock is not None
                    else stock_code
                )

                start_date = (
                    first_feature_date
                    - timedelta(
                        days=lookback_days,
                    )
                )

                end_date = last_feature_date

                yield (
                    f"[{index}/{len(selected)}] "
                    f"{stock_code} "
                    f"{stock_name} "
                    f"{start_date} ~ "
                    f"{end_date} 시작\n"
                )

                try:
                    count = await (
                        market_service
                        .sync_historical_stock_investor_flow(
                            stock_code,
                            start_date=start_date,
                            end_date=end_date,
                        )
                    )

                    success += 1
                    total_rows += count

                    yield (
                        f"[{index}/{len(selected)}] "
                        f"{stock_code} 완료: "
                        f"{count}건\n"
                    )

                except (
                    ConfigurationError,
                    ExternalApiError,
                    ValueError,
                ) as e:
                    db.rollback()
                    failed += 1

                    yield (
                        f"[{index}/{len(selected)}] "
                        f"{stock_code} 실패: "
                        f"{e}\n"
                    )

        except asyncio.CancelledError:
            db.rollback()
            raise

        yield (
            "[DONE] "
            f"성공 {success}종목, "
            f"실패 {failed}종목, "
            f"수급 {total_rows}건, "
            f"다음 offset={offset + len(selected)}\n"
        )

    return StreamingResponse(
        generate(),
        media_type="text/plain; charset=utf-8",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@router.post(
    "/market-context/stock-prices/stream",
)
async def stream_top_stock_prices(
    limit: int = Query(
        default=100,
        ge=1,
        le=100,
    ),
    days: int = Query(
        default=120,
        ge=60,
        le=3650,
    ),
    db: Session = Depends(get_db),
):
    market_service = MarketDataService(
        db
    )

    stock_service = StockService(
        db
    )

    async def generate():
        stock_codes = (
            market_service
            .get_current_universe_stock_codes(
                limit=limit,
            )
        )

        total = len(
            stock_codes
        )

        if total == 0:
            yield (
                "[ERROR] 오늘 TOP Universe가 없습니다. "
                "market-context/ranked를 먼저 실행하세요.\n"
            )
            return

        success = 0
        failed = 0
        total_rows = 0

        yield (
            "[START] "
            f"TOP {total} 종목 "
            f"일봉 {days}일 동기화\n"
        )

        try:
            for index, stock_code in enumerate(
                stock_codes,
                start=1,
            ):
                stock = (
                    stock_service.repository
                    .get_stock(
                        stock_code
                    )
                )

                stock_name = (
                    stock.name
                    if stock is not None
                    else stock_code
                )

                yield (
                    f"[{index}/{total}] "
                    f"{stock_code} "
                    f"{stock_name} 시작\n"
                )

                try:
                    count = (
                        await stock_service
                        .sync_daily_prices(
                            stock_code,
                            days=days,
                        )
                    )

                    success += 1
                    total_rows += count

                    yield (
                        f"[{index}/{total}] "
                        f"{stock_code} 완료: "
                        f"{count}건\n"
                    )

                except (
                    ConfigurationError,
                    ExternalApiError,
                    ValueError,
                ) as e:
                    db.rollback()
                    failed += 1

                    yield (
                        f"[{index}/{total}] "
                        f"{stock_code} 실패: "
                        f"{e}\n"
                    )

        except asyncio.CancelledError:
            db.rollback()
            raise

        yield (
            "[DONE] "
            f"성공 {success}종목, "
            f"실패 {failed}종목, "
            f"일봉 {total_rows}건\n"
        )

    return StreamingResponse(
        generate(),
        media_type=(
            "text/plain; charset=utf-8"
        ),
        headers={
            "Cache-Control":
                "no-cache",
            "X-Accel-Buffering":
                "no",
        },
    )

@router.get(
    "/market-context/relative-strength",
)
def get_top_relative_strength(
    limit: int = Query(
        default=100,
        ge=1,
        le=100,
    ),
    include_rows: bool = Query(
        default=False,
    ),
    db: Session = Depends(get_db),
):
    service = MarketDataService(db)

    result = (
        service
        .calculate_top_relative_strength(
            limit=limit,
        )
    )

    rows = result.pop(
        "results"
    )

    result["rows"] = (
        rows
        if include_rows
        else rows[:5]
    )

    return result

@router.post(
    "/features/stream",
)
async def sync_feature_snapshots(
    limit: int = Query(
        default=100,
        ge=1,
        le=100,
    ),
    feature_version: str = Query(
        default=FeatureEngineService.FEATURE_VERSION,
        min_length=1,
        max_length=40,
    ),
    db: Session = Depends(
        get_db
    ),
):
    service = (
        FeatureEngineService(
            db
        )
    )

    async def generate():
        stock_codes = (
            service
            .get_current_universe_stock_codes(
                limit=limit,
            )
        )

        total = len(
            stock_codes
        )

        if total == 0:
            yield (
                "[ERROR] "
                "현재 TOP Universe가 없습니다.\n"
            )
            return

        success = 0
        failed = 0
        saved = 0
        missing_total = 0

        yield (
            "[START] "
            f"TOP {total} "
            "Feature Snapshot 생성\n"
        )

        for index, stock_code in enumerate(
            stock_codes,
            start=1,
        ):
            yield (
                f"[{index}/{total}] "
                f"{stock_code} 시작\n"
            )

            try:
                (
                    row,
                    missing,
                ) = (
                    service
                    .build_current_snapshot_row(
                        stock_code,
                        feature_version=(
                            feature_version
                        ),
                    )
                )

                saved += (
                    service
                    .feature_repository
                    .upsert_snapshots(
                        [
                            row
                        ]
                    )
                )

                success += 1
                missing_total += (
                    len(
                        missing
                    )
                )

                missing_preview = (
                    ", ".join(
                        missing[:5]
                    )
                    if missing
                    else "-"
                )

                yield (
                    f"[{index}/{total}] "
                    f"{stock_code} 완료: "
                    f"{len(row['features'])} features, "
                    f"missing={len(missing)} "
                    f"[{missing_preview}]\n"
                )

            except Exception as e:
                db.rollback()
                failed += 1

                yield (
                    f"[{index}/{total}] "
                    f"{stock_code} 실패: "
                    f"{type(e).__name__}: "
                    f"{e}\n"
                )

            await asyncio.sleep(
                0
            )

        yield (
            "[DONE] "
            f"저장 {saved}건, "
            f"성공 {success}, "
            f"실패 {failed}, "
            f"전체 missing "
            f"{missing_total}\n"
        )

    return StreamingResponse(
        generate(),
        media_type=(
            "text/plain; charset=utf-8"
        ),
        headers={
            "Cache-Control":
                "no-cache",

            "X-Accel-Buffering":
                "no",
        },
    )

@router.get(
    "/features/quality",
)
def get_feature_quality(
    feature_version: str = Query(
        default=FeatureEngineService.FEATURE_VERSION,
        min_length=1,
        max_length=40,
    ),
    expected_snapshots: int = Query(
        default=100,
        ge=1,
        le=1000,
    ),
    include_all_features: bool = Query(
        default=False,
    ),
    db: Session = Depends(
        get_db
    ),
):
    service = (
        FeatureQualityService(
            db
        )
    )

    return (
        service.inspect_latest(
            feature_version=(
                feature_version
            ),
            expected_snapshots=(
                expected_snapshots
            ),
            include_all_features=(
                include_all_features
            ),
        )
    )

@router.get(
    "/features/matrix",
)
def get_feature_matrix(
    feature_version: str = Query(
        default=FeatureEngineService.FEATURE_VERSION,
        min_length=1,
        max_length=40,
    ),
    limit: int = Query(
        default=100,
        ge=1,
        le=100,
    ),
    preview_rows: int = Query(
        default=3,
        ge=0,
        le=10,
    ),
    db: Session = Depends(
        get_db
    ),
):
    service = (
        FeatureMatrixService(
            db
        )
    )

    try:
        return (
            service.inspect_latest(
                feature_version=(
                    feature_version
                ),
                limit=limit,
                preview_rows=(
                    preview_rows
                ),
            )
        )

    except ValueError as e:
        raise HTTPException(
            status_code=400,
            detail=str(e),
        ) from e
    
@router.post(
    "/features/historical/stream",
)
async def sync_historical_features(
    limit: int = Query(
        default=100,
        ge=1,
        le=100,
    ),
    feature_version: str = Query(
        default=HistoricalFeatureService.FEATURE_VERSION,
        min_length=1,
        max_length=40,
    ),
    db: Session = Depends(
        get_db
    ),
):
    service = (
        HistoricalFeatureService(
            db
        )
    )

    async def generate():
        stock_codes = (
            service
            .get_current_universe_stock_codes(
                limit=limit,
            )
        )

        total = len(
            stock_codes
        )

        success = 0
        failed = 0
        total_built = 0
        total_saved = 0
        total_skipped = 0

        yield (
            "[START] "
            f"TOP {total} Historical Feature 생성 "
            f"({len(HistoricalFeatureService.HISTORICAL_FEATURE_NAMES)} features)\n"
        )

        for index, stock_code in enumerate(
            stock_codes,
            start=1,
        ):
            yield (
                f"[{index}/{total}] "
                f"{stock_code} 시작\n"
            )

            try:
                (
                    rows,
                    meta,
                ) = (
                    service
                    .build_stock_rows(
                        stock_code,
                        feature_version=(
                            feature_version
                        ),
                    )
                )

                saved = (
                    service
                    .save_rows(
                        rows
                    )
                )

                success += 1
                total_built += (
                    meta[
                        "built"
                    ]
                )
                total_saved += saved
                total_skipped += (
                    meta[
                        "skipped"
                    ]
                )

                yield (
                    f"[{index}/{total}] "
                    f"{stock_code} 완료: "
                    f"생성 {meta['built']}건, "
                    f"저장 {saved}건, "
                    f"skip {meta['skipped']}건, "
                    f"{meta['first_feature_date']} "
                    "~ "
                    f"{meta['last_feature_date']}\n"
                )

            except Exception as e:
                db.rollback()

                failed += 1

                yield (
                    f"[{index}/{total}] "
                    f"{stock_code} 실패: "
                    f"{type(e).__name__}: "
                    f"{e}\n"
                )

            await asyncio.sleep(
                0
            )

        yield (
            "[DONE] "
            f"성공 {success}종목, "
            f"실패 {failed}종목, "
            f"생성 {total_built}건, "
            f"저장 {total_saved}건, "
            f"skip {total_skipped}건\n"
        )

    return StreamingResponse(
        generate(),
        media_type=(
            "text/plain; charset=utf-8"
        ),
        headers={
            "Cache-Control":
                "no-cache",

            "X-Accel-Buffering":
                "no",
        },
    )

@router.post(
    "/features/historical/targets/stream",
)
async def sync_historical_targets(
    limit: int = Query(
        default=100,
        ge=1,
        le=100,
    ),
    feature_version: str = Query(
        default=HistoricalFeatureService.FEATURE_VERSION,
        min_length=1,
        max_length=40,
    ),
    db: Session = Depends(
        get_db
    ),
):
    service = (
        HistoricalTargetService(
            db
        )
    )

    async def generate():
        stock_codes = (
            service
            .get_current_universe_stock_codes(
                limit=limit,
            )
        )

        total = len(
            stock_codes
        )

        success = 0
        failed = 0
        total_updated = 0

        total_filled_1d = 0
        total_filled_5d = 0
        total_filled_20d = 0

        total_pending_1d = 0
        total_pending_5d = 0
        total_pending_20d = 0

        yield (
            "[START] "
            f"TOP {total} Historical Target 생성 "
            "(1d / 5d / 20d)\n"
        )

        for index, stock_code in enumerate(
            stock_codes,
            start=1,
        ):
            yield (
                f"[{index}/{total}] "
                f"{stock_code} 시작\n"
            )

            try:
                result = (
                    service
                    .build_stock_targets(
                        stock_code=(
                            stock_code
                        ),
                        feature_version=(
                            feature_version
                        ),
                    )
                )

                success += 1

                total_updated += (
                    result[
                        "updated"
                    ]
                )

                total_filled_1d += (
                    result[
                        "filled_1d"
                    ]
                )

                total_filled_5d += (
                    result[
                        "filled_5d"
                    ]
                )

                total_filled_20d += (
                    result[
                        "filled_20d"
                    ]
                )

                total_pending_1d += (
                    result[
                        "pending_1d"
                    ]
                )

                total_pending_5d += (
                    result[
                        "pending_5d"
                    ]
                )

                total_pending_20d += (
                    result[
                        "pending_20d"
                    ]
                )

                yield (
                    f"[{index}/{total}] "
                    f"{stock_code} 완료: "
                    f"업데이트 {result['updated']}건, "
                    f"1d={result['filled_1d']}, "
                    f"5d={result['filled_5d']}, "
                    f"20d={result['filled_20d']}, "
                    "pending="
                    f"{result['pending_1d']}/"
                    f"{result['pending_5d']}/"
                    f"{result['pending_20d']}\n"
                )

            except Exception as e:
                db.rollback()

                failed += 1

                yield (
                    f"[{index}/{total}] "
                    f"{stock_code} 실패: "
                    f"{type(e).__name__}: "
                    f"{e}\n"
                )

            await asyncio.sleep(
                0
            )

        yield (
            "[DONE] "
            f"성공 {success}종목, "
            f"실패 {failed}종목, "
            f"업데이트 {total_updated}건, "
            "filled="
            f"{total_filled_1d}/"
            f"{total_filled_5d}/"
            f"{total_filled_20d}, "
            "pending="
            f"{total_pending_1d}/"
            f"{total_pending_5d}/"
            f"{total_pending_20d}\n"
        )

    return StreamingResponse(
        generate(),
        media_type=(
            "text/plain; charset=utf-8"
        ),
        headers={
            "Cache-Control":
                "no-cache",

            "X-Accel-Buffering":
                "no",
        },
    )

@router.get(
    "/features/historical/dataset",
)
def inspect_historical_dataset(
    horizon: int = Query(
        default=5,
        ge=1,
        le=20,
    ),
    feature_version: str = Query(
        default=HistoricalFeatureService.FEATURE_VERSION,
        min_length=1,
        max_length=40,
    ),
    preview_rows: int = Query(
        default=3,
        ge=0,
        le=10,
    ),
    db: Session = Depends(
        get_db
    ),
):
    if horizon not in {
        1,
        5,
        20,
    }:
        raise HTTPException(
            status_code=400,
            detail=(
                "horizon은 "
                "1, 5, 20만 지원합니다."
            ),
        )

    service = (
        HistoricalDatasetService(
            db
        )
    )

    try:
        return (
            service
            .inspect_dataset(
                feature_version=(
                    feature_version
                ),
                horizon=horizon,
                preview_rows=(
                    preview_rows
                ),
            )
        )

    except ValueError as e:
        raise HTTPException(
            status_code=400,
            detail=str(e),
        ) from e
    
@router.get(
    "/technical/inspect",
)
def inspect_technical_features(
    stock_code: str = Query(
        default="005930",
        min_length=6,
        max_length=6,
    ),
    db: Session = Depends(
        get_db
    ),
):
    service = (
        TechnicalAnalysisService(
            db
        )
    )

    try:
        return (
            service.inspect_stock(
                stock_code=(
                    stock_code
                ),
            )
        )

    except ValueError as e:
        raise HTTPException(
            status_code=400,
            detail=str(e),
        ) from e
    
@router.get(
    "/features/historical/redundancy",
)
def inspect_feature_redundancy(
    feature_version: str = Query(
        default=HistoricalFeatureService.FEATURE_VERSION,
        min_length=1,
        max_length=40,
    ),
    horizon: int = Query(
        default=5,
    ),
    correlation_threshold: float = Query(
        default=0.95,
        ge=0.5,
        le=1.0,
    ),
    db: Session = Depends(
        get_db
    ),
):
    if horizon not in {
        1,
        5,
        20,
    }:
        raise HTTPException(
            status_code=400,
            detail=(
                "horizon은 "
                "1, 5, 20만 지원합니다."
            ),
        )

    service = (
        FeatureRedundancyService(
            db
        )
    )

    try:
        return (
            service.inspect(
                feature_version=(
                    feature_version
                ),
                horizon=horizon,
                correlation_threshold=(
                    correlation_threshold
                ),
            )
        )

    except ValueError as e:
        raise HTTPException(
            status_code=400,
            detail=str(e),
        ) from e
    
@router.get(
    "/market-regime/inspect",
)
def inspect_market_regime(
    market: str = Query(
        default="KOSPI",
        min_length=5,
        max_length=6,
    ),
    db: Session = Depends(
        get_db
    ),
):
    service = (
        MarketRegimeService(
            db
        )
    )

    try:
        return (
            service.inspect_market(
                market=market,
            )
        )

    except ValueError as e:
        raise HTTPException(
            status_code=400,
            detail=str(e),
        ) from e

@router.get(
    "/sector-regime/inspect",
)
def inspect_sector_regime(
    stock_code: str = Query(
        default="005930",
        min_length=6,
        max_length=6,
    ),
    db: Session = Depends(
        get_db
    ),
):
    service = (
        SectorRegimeService(
            db
        )
    )

    try:
        return (
            service.inspect_stock(
                stock_code=(
                    stock_code
                ),
            )
        )

    except ValueError as e:
        raise HTTPException(
            status_code=400,
            detail=str(e),
        ) from e
    
@router.post(
    "/disclosures/historical/stream",
)
async def sync_historical_disclosures(
    limit: int = Query(
        default=100,
        ge=1,
        le=100,
    ),
    days: int = Query(
        default=1200,
        ge=365,
        le=3650,
    ),
    db: Session = Depends(
        get_db
    ),
):
    universe_service = (
        MarketDataService(
            db
        )
    )

    disclosure_service = (
        MarketContextService(
            db
        )
    )

    async def generate():
        stock_codes = (
            universe_service
            .get_current_universe_stock_codes(
                limit=limit,
            )
        )

        total = len(
            stock_codes
        )

        if total == 0:
            yield (
                "[ERROR] "
                "현재 TOP Universe가 없습니다.\n"
            )
            return

        end_date = date.today()

        begin_date = (
            end_date
            - timedelta(
                days=days
            )
        )

        success = 0
        failed = 0
        fetched_total = 0
        saved_total = 0
        duplicate_total = 0
        invalid_total = 0

        yield (
            "[START] "
            f"TOP {total} Historical Disclosure "
            f"{begin_date} ~ {end_date}\n"
        )

        for index, stock_code in enumerate(
            stock_codes,
            start=1,
        ):
            yield (
                f"[{index}/{total}] "
                f"{stock_code} 시작\n"
            )

            try:
                result = (
                    await disclosure_service
                    .sync_disclosures_history(
                        stock_code=(
                            stock_code
                        ),
                        begin_date=(
                            begin_date
                        ),
                        end_date=(
                            end_date
                        ),
                    )
                )

                success += 1

                fetched_total += (
                    result[
                        "fetched"
                    ]
                )

                saved_total += (
                    result[
                        "saved"
                    ]
                )

                duplicate_total += (
                    result[
                        "duplicate"
                    ]
                )

                invalid_total += (
                    result[
                        "invalid"
                    ]
                )

                yield (
                    f"[{index}/{total}] "
                    f"{stock_code} 완료: "
                    f"조회 {result['fetched']}건, "
                    f"저장 {result['saved']}건, "
                    f"중복 {result['duplicate']}건, "
                    f"invalid {result['invalid']}건\n"
                )

            except Exception as e:
                db.rollback()

                failed += 1

                yield (
                    f"[{index}/{total}] "
                    f"{stock_code} 실패: "
                    f"{type(e).__name__}: "
                    f"{e}\n"
                )

            await asyncio.sleep(
                0
            )

        yield (
            "[DONE] "
            f"성공 {success}종목, "
            f"실패 {failed}종목, "
            f"조회 {fetched_total}건, "
            f"저장 {saved_total}건, "
            f"중복 {duplicate_total}건, "
            f"invalid {invalid_total}건\n"
        )

    return StreamingResponse(
        generate(),
        media_type=(
            "text/plain; charset=utf-8"
        ),
        headers={
            "Cache-Control":
                "no-cache",

            "X-Accel-Buffering":
                "no",
        },
    )

@router.get(
    "/disclosures/taxonomy/inspect",
)
def inspect_disclosure_taxonomy(
    limit: int = Query(
        default=100,
        ge=1,
        le=100,
    ),
    sample_per_type: int = Query(
        default=5,
        ge=1,
        le=20,
    ),
    db: Session = Depends(
        get_db
    ),
):
    universe_service = (
        MarketDataService(
            db
        )
    )

    stock_codes = (
        universe_service
        .get_current_universe_stock_codes(
            limit=limit,
        )
    )

    if not stock_codes:
        raise HTTPException(
            status_code=400,
            detail=(
                "현재 TOP Universe가 없습니다."
            ),
        )

    service = (
        DisclosureTaxonomyService(
            db
        )
    )

    try:
        return service.inspect(
            stock_codes=(
                stock_codes
            ),
            sample_per_type=(
                sample_per_type
            ),
        )

    except ValueError as e:
        raise HTTPException(
            status_code=400,
            detail=str(e),
        ) from e
    
@router.post(
    "/disclosures/analysis/stream",
)
async def sync_disclosure_analysis(
    limit: int = Query(
        default=100,
        ge=1,
        le=100,
    ),
    db: Session = Depends(
        get_db
    ),
):
    universe_service = (
        MarketDataService(
            db
        )
    )

    analysis_service = (
        DisclosureAnalysisService(
            db
        )
    )

    async def generate():
        stock_codes = (
            universe_service
            .get_current_universe_stock_codes(
                limit=limit,
            )
        )

        total = len(
            stock_codes
        )

        if total == 0:
            yield (
                "[ERROR] "
                "현재 TOP Universe가 없습니다.\n"
            )
            return

        total_processed = 0
        total_created = 0
        total_updated = 0
        total_other = 0

        success = 0
        failed = 0

        yield (
            "[START] "
            f"TOP {total} Disclosure Analysis "
            f"{analysis_service.MODEL_NAME}/"
            f"{analysis_service.MODEL_VERSION}\n"
        )

        for index, stock_code in enumerate(
            stock_codes,
            start=1,
        ):
            yield (
                f"[{index}/{total}] "
                f"{stock_code} 시작\n"
            )

            try:
                result = (
                    analysis_service
                    .analyze_stock(
                        stock_code=(
                            stock_code
                        )
                    )
                )

                success += 1

                total_processed += (
                    result[
                        "processed"
                    ]
                )

                total_created += (
                    result[
                        "created"
                    ]
                )

                total_updated += (
                    result[
                        "updated"
                    ]
                )

                total_other += (
                    result[
                        "other"
                    ]
                )

                yield (
                    f"[{index}/{total}] "
                    f"{stock_code} 완료: "
                    f"처리 {result['processed']}건, "
                    f"신규 {result['created']}건, "
                    f"갱신 {result['updated']}건, "
                    f"OTHER {result['other']}건\n"
                )

            except Exception as e:
                db.rollback()

                failed += 1

                yield (
                    f"[{index}/{total}] "
                    f"{stock_code} 실패: "
                    f"{type(e).__name__}: "
                    f"{e}\n"
                )

            await asyncio.sleep(
                0
            )

        yield (
            "[DONE] "
            f"성공 {success}종목, "
            f"실패 {failed}종목, "
            f"처리 {total_processed}건, "
            f"신규 {total_created}건, "
            f"갱신 {total_updated}건, "
            f"OTHER {total_other}건\n"
        )

    return StreamingResponse(
        generate(),
        media_type=(
            "text/plain; charset=utf-8"
        ),
        headers={
            "Cache-Control":
                "no-cache",

            "X-Accel-Buffering":
                "no",
        },
    )

@router.get(
    "/disclosures/features/inspect",
)
def inspect_disclosure_features(
    stock_code: str = Query(
        ...,
        min_length=6,
        max_length=6,
    ),
    as_of: date | None = Query(
        default=None,
    ),
    db: Session = Depends(
        get_db
    ),
):
    target_date = (
        as_of
        or date.today()
    )

    service = (
        DisclosureFeatureService(
            db
        )
    )

    return service.inspect(
        stock_code=(
            stock_code
        ),
        as_of_date=(
            target_date
        ),
    )