from __future__ import annotations

from datetime import date, datetime

from sqlalchemy import (
    BigInteger,
    Boolean,
    Date,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from app.core.database import Base


class SchemaVersion(Base):
    __tablename__ = "schema_versions"
    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    version: Mapped[str] = mapped_column(String(40), unique=True)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    applied_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class StockIntradayPrice(Base):
    __tablename__ = "stock_intraday_prices"
    __table_args__ = (UniqueConstraint("stock_code", "captured_at", name="uq_intraday_code_time"),)
    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    stock_code: Mapped[str] = mapped_column(ForeignKey("stocks.code", ondelete="CASCADE"), index=True)
    captured_at: Mapped[datetime] = mapped_column(DateTime, index=True)
    price: Mapped[float] = mapped_column(Float)
    open: Mapped[float | None] = mapped_column(Float, nullable=True)
    high: Mapped[float | None] = mapped_column(Float, nullable=True)
    low: Mapped[float | None] = mapped_column(Float, nullable=True)
    volume: Mapped[int | None] = mapped_column(BigInteger, nullable=True)
    accumulated_volume: Mapped[int | None] = mapped_column(BigInteger, nullable=True)
    raw_json: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class DataSyncRun(Base):
    __tablename__ = "data_sync_runs"
    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    source: Mapped[str] = mapped_column(String(40))
    sync_type: Mapped[str] = mapped_column(String(80))
    stock_code: Mapped[str | None] = mapped_column(ForeignKey("stocks.code", ondelete="SET NULL"), nullable=True)
    started_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    status: Mapped[str] = mapped_column(String(30), default="running")
    requested_count: Mapped[int] = mapped_column(Integer, default=0)
    success_count: Mapped[int] = mapped_column(Integer, default=0)
    failure_count: Mapped[int] = mapped_column(Integer, default=0)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    metadata_json: Mapped[dict | None] = mapped_column(JSONB, nullable=True)


class StockFeatureSnapshot(Base):
    __tablename__ = "stock_feature_snapshots"
    __table_args__ = (UniqueConstraint("stock_code", "feature_date", "feature_version", name="uq_feature_snapshot"),)
    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    stock_code: Mapped[str] = mapped_column(ForeignKey("stocks.code", ondelete="CASCADE"), index=True)
    feature_date: Mapped[date] = mapped_column(Date, index=True)
    feature_version: Mapped[str] = mapped_column(String(40), default="v1")
    features: Mapped[dict] = mapped_column(JSONB)
    target_return_1d: Mapped[float | None] = mapped_column(Float, nullable=True)
    target_return_5d: Mapped[float | None] = mapped_column(Float, nullable=True)
    target_return_20d: Mapped[float | None] = mapped_column(Float, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class ModelRun(Base):
    __tablename__ = "model_runs"
    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    model_version: Mapped[str] = mapped_column(String(80), index=True)
    model_type: Mapped[str] = mapped_column(String(100))
    horizon_days: Mapped[int] = mapped_column(Integer)
    scope: Mapped[str] = mapped_column(String(40), default="stock")
    stock_code: Mapped[str | None] = mapped_column(ForeignKey("stocks.code", ondelete="SET NULL"), nullable=True, index=True)
    feature_version: Mapped[str | None] = mapped_column(String(40), nullable=True)
    train_start_date: Mapped[date | None] = mapped_column(Date, nullable=True)
    train_end_date: Mapped[date | None] = mapped_column(Date, nullable=True)
    validation_start_date: Mapped[date | None] = mapped_column(Date, nullable=True)
    validation_end_date: Mapped[date | None] = mapped_column(Date, nullable=True)
    training_rows: Mapped[int] = mapped_column(Integer, default=0)
    validation_mae: Mapped[float | None] = mapped_column(Float, nullable=True)
    validation_rmse: Mapped[float | None] = mapped_column(Float, nullable=True)
    validation_accuracy: Mapped[float | None] = mapped_column(Float, nullable=True)
    validation_auc: Mapped[float | None] = mapped_column(Float, nullable=True)
    backtest_return: Mapped[float | None] = mapped_column(Float, nullable=True)
    backtest_mdd: Mapped[float | None] = mapped_column(Float, nullable=True)
    backtest_sharpe: Mapped[float | None] = mapped_column(Float, nullable=True)
    model_path: Mapped[str | None] = mapped_column(String(500), nullable=True)
    parameters_json: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    feature_importance_json: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    status: Mapped[str] = mapped_column(String(30), default="completed")
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    trained_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class StockPredictionHistory(Base):
    __tablename__ = "stock_prediction_history"
    __table_args__ = (UniqueConstraint("stock_code", "model_version", "horizon_days", "prediction_date", name="uq_prediction_history"),)
    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    stock_code: Mapped[str] = mapped_column(ForeignKey("stocks.code", ondelete="CASCADE"), index=True)
    model_run_id: Mapped[int | None] = mapped_column(ForeignKey("model_runs.id", ondelete="SET NULL"), nullable=True)
    model_version: Mapped[str] = mapped_column(String(80))
    horizon_days: Mapped[int] = mapped_column(Integer)
    prediction_date: Mapped[date] = mapped_column(Date, index=True)
    feature_date: Mapped[date] = mapped_column(Date)
    target_date: Mapped[date | None] = mapped_column(Date, nullable=True)
    base_price: Mapped[float | None] = mapped_column(Float, nullable=True)
    predicted_return: Mapped[float] = mapped_column(Float)
    predicted_price: Mapped[float | None] = mapped_column(Float, nullable=True)
    upside_probability: Mapped[float | None] = mapped_column(Float, nullable=True)
    confidence_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    ml_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    actual_return: Mapped[float | None] = mapped_column(Float, nullable=True)
    actual_price: Mapped[float | None] = mapped_column(Float, nullable=True)
    direction_correct: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    evaluated_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    payload_json: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class NewsArticle(Base):
    __tablename__ = "news_articles"
    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    source: Mapped[str] = mapped_column(String(80))
    external_id: Mapped[str | None] = mapped_column(Text, nullable=True)
    title: Mapped[str] = mapped_column(Text)
    url: Mapped[str] = mapped_column(Text)
    canonical_url: Mapped[str | None] = mapped_column(Text, nullable=True)
    published_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True, index=True)
    author: Mapped[str | None] = mapped_column(String(160), nullable=True)
    body_text: Mapped[str | None] = mapped_column(Text, nullable=True)
    language: Mapped[str | None] = mapped_column(String(20), default="ko")
    raw_json: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    content_hash: Mapped[str | None] = mapped_column(String(128), nullable=True, index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class StockNews(Base):
    __tablename__ = "stock_news"
    __table_args__ = (UniqueConstraint("stock_code", "article_id", name="uq_stock_news"),)
    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    stock_code: Mapped[str] = mapped_column(ForeignKey("stocks.code", ondelete="CASCADE"), index=True)
    article_id: Mapped[int] = mapped_column(ForeignKey("news_articles.id", ondelete="CASCADE"), index=True)
    relevance_score: Mapped[float] = mapped_column(Float, default=0.0)
    matched_by: Mapped[str | None] = mapped_column(String(40), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class NewsAnalysis(Base):
    __tablename__ = "news_analyses"
    __table_args__ = (UniqueConstraint("stock_code", "article_id", "model_name", "model_version", name="uq_news_analysis"),)
    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    stock_code: Mapped[str] = mapped_column(ForeignKey("stocks.code", ondelete="CASCADE"), index=True)
    article_id: Mapped[int] = mapped_column(ForeignKey("news_articles.id", ondelete="CASCADE"))
    model_name: Mapped[str] = mapped_column(String(100), default="rule-based")
    model_version: Mapped[str] = mapped_column(String(80), default="v1")
    sentiment_score: Mapped[float] = mapped_column(Float, default=0.0)
    importance_score: Mapped[float] = mapped_column(Float, default=0.0)
    relevance_score: Mapped[float] = mapped_column(Float, default=0.0)
    impact_score: Mapped[float] = mapped_column(Float, default=0.0, index=True)
    event_type: Mapped[str | None] = mapped_column(String(80), nullable=True)
    summary: Mapped[str | None] = mapped_column(Text, nullable=True)
    rationale: Mapped[str | None] = mapped_column(Text, nullable=True)
    entities_json: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    analyzed_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class Disclosure(Base):
    __tablename__ = "disclosures"
    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    stock_code: Mapped[str | None] = mapped_column(ForeignKey("stocks.code", ondelete="CASCADE"), nullable=True, index=True)
    corp_code: Mapped[str | None] = mapped_column(String(8), nullable=True)
    receipt_no: Mapped[str] = mapped_column(String(30), unique=True)
    report_name: Mapped[str] = mapped_column(Text)
    filer_name: Mapped[str | None] = mapped_column(String(160), nullable=True)
    receipt_date: Mapped[date | None] = mapped_column(Date, nullable=True, index=True)
    disclosure_type: Mapped[str | None] = mapped_column(String(80), nullable=True)
    url: Mapped[str | None] = mapped_column(Text, nullable=True)
    raw_json: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class RankingSnapshot(Base):
    __tablename__ = "ranking_snapshots"
    __table_args__ = (UniqueConstraint("ranking_version", "as_of_date", "horizon_days", "universe", name="uq_ranking_snapshot"),)
    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    ranking_version: Mapped[str] = mapped_column(String(80))
    as_of_date: Mapped[date] = mapped_column(Date, index=True)
    horizon_days: Mapped[int] = mapped_column(Integer, default=5)
    universe: Mapped[str] = mapped_column(String(80), default="KRX")
    weights_json: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    metadata_json: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class RankingItem(Base):
    __tablename__ = "ranking_items"
    __table_args__ = (
        UniqueConstraint("snapshot_id", "stock_code", name="uq_ranking_item"),
        UniqueConstraint("snapshot_id", "rank", name="uq_ranking_position"),
    )
    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    snapshot_id: Mapped[int] = mapped_column(ForeignKey("ranking_snapshots.id", ondelete="CASCADE"), index=True)
    stock_code: Mapped[str] = mapped_column(ForeignKey("stocks.code", ondelete="CASCADE"), index=True)
    rank: Mapped[int] = mapped_column(Integer)
    total_score: Mapped[float] = mapped_column(Float)
    financial_score: Mapped[float] = mapped_column(Float, default=0.0)
    ml_score: Mapped[float] = mapped_column(Float, default=0.0)
    news_score: Mapped[float] = mapped_column(Float, default=0.0)
    disclosure_score: Mapped[float] = mapped_column(Float, default=0.0)
    valuation_score: Mapped[float] = mapped_column(Float, default=0.0)
    momentum_score: Mapped[float] = mapped_column(Float, default=0.0)
    predicted_return: Mapped[float | None] = mapped_column(Float, nullable=True)
    upside_probability: Mapped[float | None] = mapped_column(Float, nullable=True)
    rationale: Mapped[str | None] = mapped_column(Text, nullable=True)
    score_components_json: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class RecommendationPerformance(Base):
    __tablename__ = "recommendation_performance"
    __table_args__ = (UniqueConstraint("stock_code", "recommendation_date", "horizon_days", name="uq_recommendation_perf"),)
    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    ranking_item_id: Mapped[int | None] = mapped_column(ForeignKey("ranking_items.id", ondelete="SET NULL"), nullable=True)
    stock_code: Mapped[str] = mapped_column(ForeignKey("stocks.code", ondelete="CASCADE"), index=True)
    recommendation_date: Mapped[date] = mapped_column(Date, index=True)
    horizon_days: Mapped[int] = mapped_column(Integer)
    entry_price: Mapped[float | None] = mapped_column(Float, nullable=True)
    target_date: Mapped[date | None] = mapped_column(Date, nullable=True)
    exit_price: Mapped[float | None] = mapped_column(Float, nullable=True)
    predicted_return: Mapped[float | None] = mapped_column(Float, nullable=True)
    actual_return: Mapped[float | None] = mapped_column(Float, nullable=True)
    benchmark_return: Mapped[float | None] = mapped_column(Float, nullable=True)
    excess_return: Mapped[float | None] = mapped_column(Float, nullable=True)
    direction_correct: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    max_favorable_excursion: Mapped[float | None] = mapped_column(Float, nullable=True)
    max_adverse_excursion: Mapped[float | None] = mapped_column(Float, nullable=True)
    evaluated_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    metadata_json: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class AppSetting(Base):
    __tablename__ = "app_settings"
    key: Mapped[str] = mapped_column(String(120), primary_key=True)
    value_json: Mapped[dict] = mapped_column(JSONB)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
