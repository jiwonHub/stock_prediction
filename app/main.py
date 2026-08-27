from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

import app.models  # noqa: F401 - register all DB models

from app.api.admin import router as admin_router
from app.api.analysis_admin import router as analysis_admin_router
from app.api.financials import router as financials_router
from app.api.future_context import router as future_context_router
from app.api.ml import router as ml_router
from app.api.ml_admin import router as ml_admin_router
from app.api.rankings import router as rankings_router
from app.api.stocks import router as stocks_router
from app.core.config import settings
from app.core.database import Base, engine
from app.core.schema_bootstrap import ensure_runtime_schema


@asynccontextmanager
async def lifespan(_: FastAPI):
    Base.metadata.create_all(bind=engine)
    ensure_runtime_schema(engine)
    yield


app = FastAPI(
    title=settings.app_name,
    version="0.4.1-final-db-baseline",
    lifespan=lifespan,
)

origins = (
    ["*"]
    if settings.cors_origins.strip() == "*"
    else [value.strip() for value in settings.cors_origins.split(",") if value.strip()]
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=origins != ["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(stocks_router, prefix="/v1")
app.include_router(financials_router, prefix="/v1")
app.include_router(future_context_router, prefix="/v1")
app.include_router(ml_router, prefix="/v1")
app.include_router(rankings_router, prefix="/v1")
app.include_router(admin_router, prefix="/v1")
app.include_router(analysis_admin_router, prefix="/v1")
app.include_router(ml_admin_router, prefix="/v1")


@app.get("/health")
def health():
    return {
        "status": "ok",
        "phase": 4,
        "schema": "2026-08-runtime-hotfix-v2",
    }
