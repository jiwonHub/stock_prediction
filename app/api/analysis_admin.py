from datetime import date

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.core.exceptions import ConfigurationError, ExternalApiError
from app.schemas.financial import FinancialAnalysisResponse, FinancialBatchResponse
from app.services.financial_analysis_service import FinancialAnalysisService
from app.services.stock_service import StockService


router = APIRouter(
    prefix="/admin/analyze",
    tags=["admin-analysis"],
)


@router.post(
    "/financials/batch",
    response_model=FinancialBatchResponse,
)
async def analyze_financials_batch(
    year: int | None = Query(default=None, ge=2015, le=2100),
    report_code: str = Query(default="11011", pattern="^(11011|11012|11013|11014)$"),
    limit: int = Query(default=100, ge=1, le=500),
    sync_missing: bool = Query(default=True),
    db: Session = Depends(get_db),
):
    business_year = str(year or (date.today().year - 1))
    service = FinancialAnalysisService(db)

    stock_codes = (
        service.repository.get_stock_codes_for_analysis(limit=limit)
        if sync_missing
        else service.repository.get_stock_codes_with_financials(
            business_year=business_year,
            report_code=report_code,
            limit=limit,
        )
    )

    analyzed = 0
    skipped = 0
    stock_service = StockService(db)

    for stock_code in stock_codes:
        try:
            rows = service.repository.get_financial_rows(
                stock_code=stock_code,
                business_year=business_year,
                report_code=report_code,
            )
            if not rows and sync_missing:
                await stock_service.sync_financials(
                    stock_code,
                    business_year=business_year,
                    report_code=report_code,
                )

            service.analyze(
                stock_code=stock_code,
                business_year=business_year,
                report_code=report_code,
            )
            analyzed += 1
        except (
            ValueError,
            ConfigurationError,
            ExternalApiError,
            SQLAlchemyError,
        ):
            db.rollback()
            skipped += 1

    return FinancialBatchResponse(
        analyzedCount=analyzed,
        skippedCount=skipped,
        message=f"재무분석 {analyzed}건 완료, {skipped}건 건너뜀",
    )


@router.post(
    "/financials/{stock_code}",
    response_model=FinancialAnalysisResponse,
)
async def analyze_financials(
    stock_code: str,
    year: int | None = Query(default=None, ge=2015, le=2100),
    report_code: str = Query(default="11011", pattern="^(11011|11012|11013|11014)$"),
    refresh: bool = Query(default=False),
    db: Session = Depends(get_db),
):
    business_year = str(year or (date.today().year - 1))
    stock_service = StockService(db)
    analysis_service = FinancialAnalysisService(db)

    try:
        rows = analysis_service.repository.get_financial_rows(
            stock_code=stock_code,
            business_year=business_year,
            report_code=report_code,
        )

        if refresh or not rows:
            await stock_service.sync_financials(
                stock_code,
                business_year=business_year,
                report_code=report_code,
            )

        return analysis_service.analyze(
            stock_code=stock_code,
            business_year=business_year,
            report_code=report_code,
        )
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    except ConfigurationError as e:
        raise HTTPException(status_code=500, detail=str(e)) from e
    except ExternalApiError as e:
        raise HTTPException(status_code=502, detail=str(e)) from e
