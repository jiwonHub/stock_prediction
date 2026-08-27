from datetime import date

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.core.exceptions import ConfigurationError, ExternalApiError
from app.schemas.financial import FinancialAnalysisResponse
from app.services.financial_analysis_service import FinancialAnalysisService
from app.services.stock_service import StockService


router = APIRouter(
    prefix="/stocks",
    tags=["financial-analysis"],
)


@router.get(
    "/{stock_code}/financial-analysis",
    response_model=FinancialAnalysisResponse,
)
async def get_financial_analysis(
    stock_code: str,
    year: int | None = Query(default=None, ge=2015, le=2100),
    report_code: str = Query(default="11011", pattern="^(11011|11012|11013|11014)$"),
    auto_sync: bool = Query(default=True),
    db: Session = Depends(get_db),
):
    business_year = str(year or (date.today().year - 1))
    service = FinancialAnalysisService(db)

    try:
        metric = service.repository.get_financial_metric(
            stock_code=stock_code,
            business_year=business_year,
            report_code=report_code,
        )
        if metric is not None:
            return service.get_analysis(
                stock_code=stock_code,
                business_year=business_year,
                report_code=report_code,
            )

        rows = service.repository.get_financial_rows(
            stock_code=stock_code,
            business_year=business_year,
            report_code=report_code,
        )

        if not rows and auto_sync:
            await StockService(db).sync_financials(
                stock_code,
                business_year=business_year,
                report_code=report_code,
            )

        return service.analyze(
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
