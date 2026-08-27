BEGIN;

-- =========================================================
-- Stock AI - FINAL DATABASE BASELINE
-- One-time schema baseline for current + planned phases.
-- Safe to re-run: CREATE IF NOT EXISTS / ADD COLUMN IF NOT EXISTS.
-- =========================================================

CREATE TABLE IF NOT EXISTS schema_versions (
    id SERIAL PRIMARY KEY,
    version VARCHAR(40) NOT NULL UNIQUE,
    description TEXT,
    applied_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

-- -------------------------
-- Core master data
-- -------------------------
CREATE TABLE IF NOT EXISTS stocks (
    code VARCHAR(6) PRIMARY KEY,
    corp_code VARCHAR(8) UNIQUE,
    name VARCHAR(120) NOT NULL,
    english_name VARCHAR(200),
    market VARCHAR(40) NOT NULL DEFAULT 'KRX',
    current_price DOUBLE PRECISION,
    change DOUBLE PRECISION,
    change_rate DOUBLE PRECISION,
    market_cap DOUBLE PRECISION,
    per DOUBLE PRECISION,
    pbr DOUBLE PRECISION,
    eps DOUBLE PRECISION,
    bps DOUBLE PRECISION,
    sector_name VARCHAR(120),
    sector_code VARCHAR(40),
    industry_name VARCHAR(160),
    listing_date DATE,
    shares_outstanding BIGINT,
    is_active BOOLEAN NOT NULL DEFAULT TRUE,
    modified_at TIMESTAMP,
    last_price_at TIMESTAMP,
    last_financial_at TIMESTAMP,
    last_news_at TIMESTAMP,
    last_prediction_at TIMESTAMP,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

ALTER TABLE stocks ADD COLUMN IF NOT EXISTS sector_code VARCHAR(40);
ALTER TABLE stocks ADD COLUMN IF NOT EXISTS industry_name VARCHAR(160);
ALTER TABLE stocks ADD COLUMN IF NOT EXISTS listing_date DATE;
ALTER TABLE stocks ADD COLUMN IF NOT EXISTS shares_outstanding BIGINT;
ALTER TABLE stocks ADD COLUMN IF NOT EXISTS is_active BOOLEAN NOT NULL DEFAULT TRUE;
ALTER TABLE stocks ADD COLUMN IF NOT EXISTS last_financial_at TIMESTAMP;
ALTER TABLE stocks ADD COLUMN IF NOT EXISTS last_news_at TIMESTAMP;
ALTER TABLE stocks ADD COLUMN IF NOT EXISTS last_prediction_at TIMESTAMP;

CREATE INDEX IF NOT EXISTS ix_stocks_name ON stocks(name);
CREATE INDEX IF NOT EXISTS ix_stocks_corp_code ON stocks(corp_code);
CREATE INDEX IF NOT EXISTS ix_stocks_market_cap ON stocks(market_cap DESC);
CREATE INDEX IF NOT EXISTS ix_stocks_sector_name ON stocks(sector_name);
CREATE INDEX IF NOT EXISTS ix_stocks_is_active ON stocks(is_active);

-- -------------------------
-- Market prices
-- -------------------------
CREATE TABLE IF NOT EXISTS stock_prices (
    id SERIAL PRIMARY KEY,
    stock_code VARCHAR(6) NOT NULL REFERENCES stocks(code) ON DELETE CASCADE,
    trade_date DATE NOT NULL,
    open DOUBLE PRECISION NOT NULL,
    high DOUBLE PRECISION NOT NULL,
    low DOUBLE PRECISION NOT NULL,
    close DOUBLE PRECISION NOT NULL,
    adjusted_close DOUBLE PRECISION,
    volume BIGINT NOT NULL DEFAULT 0,
    trading_value NUMERIC(30,2),
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT uq_stock_price_code_date UNIQUE (stock_code, trade_date)
);

ALTER TABLE stock_prices ADD COLUMN IF NOT EXISTS adjusted_close DOUBLE PRECISION;
ALTER TABLE stock_prices ADD COLUMN IF NOT EXISTS trading_value NUMERIC(30,2);
CREATE INDEX IF NOT EXISTS ix_stock_prices_stock_code ON stock_prices(stock_code);
CREATE INDEX IF NOT EXISTS ix_stock_prices_trade_date ON stock_prices(trade_date DESC);
CREATE INDEX IF NOT EXISTS ix_stock_prices_code_date ON stock_prices(stock_code, trade_date DESC);

CREATE TABLE IF NOT EXISTS stock_intraday_prices (
    id BIGSERIAL PRIMARY KEY,
    stock_code VARCHAR(6) NOT NULL REFERENCES stocks(code) ON DELETE CASCADE,
    captured_at TIMESTAMP NOT NULL,
    price DOUBLE PRECISION NOT NULL,
    open DOUBLE PRECISION,
    high DOUBLE PRECISION,
    low DOUBLE PRECISION,
    volume BIGINT,
    accumulated_volume BIGINT,
    raw_json JSONB,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT uq_intraday_code_time UNIQUE (stock_code, captured_at)
);
CREATE INDEX IF NOT EXISTS ix_intraday_code_time ON stock_intraday_prices(stock_code, captured_at DESC);

-- -------------------------
-- DART financial source data
-- -------------------------
CREATE TABLE IF NOT EXISTS financial_statements (
    id SERIAL PRIMARY KEY,
    stock_code VARCHAR(6) NOT NULL REFERENCES stocks(code) ON DELETE CASCADE,
    corp_code VARCHAR(8) NOT NULL,
    business_year VARCHAR(4) NOT NULL,
    report_code VARCHAR(5) NOT NULL,
    fs_div VARCHAR(10) NOT NULL DEFAULT 'CFS',
    fs_nm VARCHAR(80),
    sj_div VARCHAR(10) NOT NULL DEFAULT '',
    sj_nm VARCHAR(80),
    account_id VARCHAR(180) NOT NULL DEFAULT '',
    account_nm VARCHAR(180) NOT NULL DEFAULT '',
    account_detail VARCHAR(500) NOT NULL DEFAULT '',
    currency VARCHAR(20),
    thstrm_amount NUMERIC(30,2),
    frmtrm_amount NUMERIC(30,2),
    bfefrmtrm_amount NUMERIC(30,2),
    thstrm_nm VARCHAR(120),
    frmtrm_nm VARCHAR(120),
    bfefrmtrm_nm VARCHAR(120),
    raw_json TEXT,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

ALTER TABLE financial_statements ADD COLUMN IF NOT EXISTS account_detail VARCHAR(500) NOT NULL DEFAULT '';

DO $$
BEGIN
    IF EXISTS (
        SELECT 1 FROM pg_constraint WHERE conname = 'uq_financial_statement_row'
    ) THEN
        ALTER TABLE financial_statements DROP CONSTRAINT uq_financial_statement_row;
    END IF;
EXCEPTION WHEN undefined_table THEN NULL;
END $$;

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint WHERE conname = 'uq_financial_statement_row'
    ) THEN
        ALTER TABLE financial_statements
        ADD CONSTRAINT uq_financial_statement_row UNIQUE (
            stock_code, business_year, report_code, fs_div,
            sj_div, account_id, account_nm, account_detail
        );
    END IF;
END $$;

CREATE INDEX IF NOT EXISTS ix_financial_statements_stock_code ON financial_statements(stock_code);
CREATE INDEX IF NOT EXISTS ix_financial_statements_business_year ON financial_statements(business_year);
CREATE INDEX IF NOT EXISTS ix_financial_statements_lookup ON financial_statements(stock_code, business_year, report_code, fs_div);

-- -------------------------
-- Fundamental metrics / scoring
-- -------------------------
CREATE TABLE IF NOT EXISTS financial_metrics (
    id SERIAL PRIMARY KEY,
    stock_code VARCHAR(6) NOT NULL REFERENCES stocks(code) ON DELETE CASCADE,
    business_year VARCHAR(4) NOT NULL,
    report_code VARCHAR(5) NOT NULL,
    fs_div VARCHAR(10) NOT NULL DEFAULT 'CFS',
    revenue NUMERIC(30,2),
    operating_income NUMERIC(30,2),
    net_income NUMERIC(30,2),
    total_assets NUMERIC(30,2),
    total_liabilities NUMERIC(30,2),
    total_equity NUMERIC(30,2),
    operating_cash_flow NUMERIC(30,2),
    previous_revenue NUMERIC(30,2),
    previous_operating_income NUMERIC(30,2),
    previous_net_income NUMERIC(30,2),
    revenue_growth DOUBLE PRECISION,
    operating_income_growth DOUBLE PRECISION,
    net_income_growth DOUBLE PRECISION,
    operating_margin DOUBLE PRECISION,
    net_margin DOUBLE PRECISION,
    roe DOUBLE PRECISION,
    roa DOUBLE PRECISION,
    debt_ratio DOUBLE PRECISION,
    equity_ratio DOUBLE PRECISION,
    operating_cash_flow_margin DOUBLE PRECISION,
    cash_flow_quality DOUBLE PRECISION,
    growth_score DOUBLE PRECISION NOT NULL DEFAULT 0,
    profitability_score DOUBLE PRECISION NOT NULL DEFAULT 0,
    stability_score DOUBLE PRECISION NOT NULL DEFAULT 0,
    cash_flow_score DOUBLE PRECISION NOT NULL DEFAULT 0,
    valuation_score DOUBLE PRECISION NOT NULL DEFAULT 0,
    quality_score DOUBLE PRECISION NOT NULL DEFAULT 0,
    sector_relative_score DOUBLE PRECISION NOT NULL DEFAULT 0,
    financial_score DOUBLE PRECISION NOT NULL DEFAULT 0,
    data_completeness DOUBLE PRECISION NOT NULL DEFAULT 0,
    analysis_note TEXT,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT uq_financial_metric_snapshot UNIQUE (stock_code, business_year, report_code, fs_div)
);

ALTER TABLE financial_metrics ADD COLUMN IF NOT EXISTS valuation_score DOUBLE PRECISION NOT NULL DEFAULT 0;
ALTER TABLE financial_metrics ADD COLUMN IF NOT EXISTS quality_score DOUBLE PRECISION NOT NULL DEFAULT 0;
ALTER TABLE financial_metrics ADD COLUMN IF NOT EXISTS sector_relative_score DOUBLE PRECISION NOT NULL DEFAULT 0;
CREATE INDEX IF NOT EXISTS ix_financial_metrics_stock_code ON financial_metrics(stock_code);
CREATE INDEX IF NOT EXISTS ix_financial_metrics_business_year ON financial_metrics(business_year);
CREATE INDEX IF NOT EXISTS ix_financial_metrics_financial_score ON financial_metrics(financial_score DESC);

-- -------------------------
-- Data sync audit
-- -------------------------
CREATE TABLE IF NOT EXISTS data_sync_runs (
    id BIGSERIAL PRIMARY KEY,
    source VARCHAR(40) NOT NULL,
    sync_type VARCHAR(80) NOT NULL,
    stock_code VARCHAR(6) REFERENCES stocks(code) ON DELETE SET NULL,
    started_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    finished_at TIMESTAMP,
    status VARCHAR(30) NOT NULL DEFAULT 'running',
    requested_count INTEGER NOT NULL DEFAULT 0,
    success_count INTEGER NOT NULL DEFAULT 0,
    failure_count INTEGER NOT NULL DEFAULT 0,
    error_message TEXT,
    metadata_json JSONB
);
CREATE INDEX IF NOT EXISTS ix_sync_runs_started_at ON data_sync_runs(started_at DESC);
CREATE INDEX IF NOT EXISTS ix_sync_runs_stock_code ON data_sync_runs(stock_code);

-- -------------------------
-- Flexible ML feature store
-- JSONB avoids future DB migrations when feature set changes.
-- -------------------------
CREATE TABLE IF NOT EXISTS stock_feature_snapshots (
    id BIGSERIAL PRIMARY KEY,
    stock_code VARCHAR(6) NOT NULL REFERENCES stocks(code) ON DELETE CASCADE,
    feature_date DATE NOT NULL,
    feature_version VARCHAR(40) NOT NULL DEFAULT 'v1',
    features JSONB NOT NULL,
    target_return_1d DOUBLE PRECISION,
    target_return_5d DOUBLE PRECISION,
    target_return_20d DOUBLE PRECISION,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT uq_feature_snapshot UNIQUE (stock_code, feature_date, feature_version)
);
CREATE INDEX IF NOT EXISTS ix_feature_snapshots_code_date ON stock_feature_snapshots(stock_code, feature_date DESC);
CREATE INDEX IF NOT EXISTS ix_feature_snapshots_version ON stock_feature_snapshots(feature_version);

CREATE TABLE IF NOT EXISTS model_runs (
    id BIGSERIAL PRIMARY KEY,
    model_version VARCHAR(80) NOT NULL,
    model_type VARCHAR(100) NOT NULL,
    horizon_days INTEGER NOT NULL,
    scope VARCHAR(40) NOT NULL DEFAULT 'stock',
    stock_code VARCHAR(6) REFERENCES stocks(code) ON DELETE SET NULL,
    feature_version VARCHAR(40),
    train_start_date DATE,
    train_end_date DATE,
    validation_start_date DATE,
    validation_end_date DATE,
    training_rows INTEGER NOT NULL DEFAULT 0,
    validation_mae DOUBLE PRECISION,
    validation_rmse DOUBLE PRECISION,
    validation_accuracy DOUBLE PRECISION,
    validation_auc DOUBLE PRECISION,
    backtest_return DOUBLE PRECISION,
    backtest_mdd DOUBLE PRECISION,
    backtest_sharpe DOUBLE PRECISION,
    model_path VARCHAR(500),
    parameters_json JSONB,
    feature_importance_json JSONB,
    status VARCHAR(30) NOT NULL DEFAULT 'completed',
    error_message TEXT,
    trained_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS ix_model_runs_stock_code ON model_runs(stock_code);
CREATE INDEX IF NOT EXISTS ix_model_runs_trained_at ON model_runs(trained_at DESC);
CREATE INDEX IF NOT EXISTS ix_model_runs_version ON model_runs(model_version);

-- Latest prediction cache (Phase 4 compatible)
CREATE TABLE IF NOT EXISTS stock_predictions (
    id SERIAL PRIMARY KEY,
    stock_code VARCHAR(6) NOT NULL REFERENCES stocks(code) ON DELETE CASCADE,
    horizon_days INTEGER NOT NULL DEFAULT 5,
    model_version VARCHAR(40) NOT NULL DEFAULT 'phase4-v1',
    model_type VARCHAR(80) NOT NULL DEFAULT 'gradient-boosting',
    model_path VARCHAR(500),
    model_run_id BIGINT REFERENCES model_runs(id) ON DELETE SET NULL,
    prediction_date DATE,
    target_date DATE,
    predicted_return DOUBLE PRECISION NOT NULL DEFAULT 0,
    predicted_price DOUBLE PRECISION,
    prediction_lower DOUBLE PRECISION,
    prediction_upper DOUBLE PRECISION,
    upside_probability DOUBLE PRECISION NOT NULL DEFAULT 0,
    confidence_score DOUBLE PRECISION,
    ml_score DOUBLE PRECISION NOT NULL DEFAULT 0,
    training_rows INTEGER NOT NULL DEFAULT 0,
    validation_mae DOUBLE PRECISION,
    validation_accuracy DOUBLE PRECISION,
    feature_date DATE NOT NULL,
    feature_json JSONB,
    trained_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT uq_stock_prediction_horizon UNIQUE (stock_code, horizon_days)
);

ALTER TABLE stock_predictions ADD COLUMN IF NOT EXISTS model_run_id BIGINT;
ALTER TABLE stock_predictions ADD COLUMN IF NOT EXISTS prediction_date DATE;
ALTER TABLE stock_predictions ADD COLUMN IF NOT EXISTS target_date DATE;
ALTER TABLE stock_predictions ADD COLUMN IF NOT EXISTS predicted_price DOUBLE PRECISION;
ALTER TABLE stock_predictions ADD COLUMN IF NOT EXISTS prediction_lower DOUBLE PRECISION;
ALTER TABLE stock_predictions ADD COLUMN IF NOT EXISTS prediction_upper DOUBLE PRECISION;
ALTER TABLE stock_predictions ADD COLUMN IF NOT EXISTS confidence_score DOUBLE PRECISION;
ALTER TABLE stock_predictions ADD COLUMN IF NOT EXISTS feature_json JSONB;

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint WHERE conname = 'fk_stock_predictions_model_run'
    ) THEN
        ALTER TABLE stock_predictions
        ADD CONSTRAINT fk_stock_predictions_model_run
        FOREIGN KEY (model_run_id) REFERENCES model_runs(id) ON DELETE SET NULL;
    END IF;
END $$;

CREATE INDEX IF NOT EXISTS ix_stock_predictions_stock_code ON stock_predictions(stock_code);
CREATE INDEX IF NOT EXISTS ix_stock_predictions_horizon_days ON stock_predictions(horizon_days);
CREATE INDEX IF NOT EXISTS ix_stock_predictions_ml_score ON stock_predictions(ml_score DESC);
CREATE INDEX IF NOT EXISTS ix_stock_predictions_feature_date ON stock_predictions(feature_date DESC);

-- Immutable prediction history for validation/performance tracking.
CREATE TABLE IF NOT EXISTS stock_prediction_history (
    id BIGSERIAL PRIMARY KEY,
    stock_code VARCHAR(6) NOT NULL REFERENCES stocks(code) ON DELETE CASCADE,
    model_run_id BIGINT REFERENCES model_runs(id) ON DELETE SET NULL,
    model_version VARCHAR(80) NOT NULL,
    horizon_days INTEGER NOT NULL,
    prediction_date DATE NOT NULL,
    feature_date DATE NOT NULL,
    target_date DATE,
    base_price DOUBLE PRECISION,
    predicted_return DOUBLE PRECISION NOT NULL,
    predicted_price DOUBLE PRECISION,
    upside_probability DOUBLE PRECISION,
    confidence_score DOUBLE PRECISION,
    ml_score DOUBLE PRECISION,
    actual_return DOUBLE PRECISION,
    actual_price DOUBLE PRECISION,
    direction_correct BOOLEAN,
    evaluated_at TIMESTAMP,
    payload_json JSONB,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT uq_prediction_history UNIQUE (stock_code, model_version, horizon_days, prediction_date)
);
CREATE INDEX IF NOT EXISTS ix_prediction_history_code_date ON stock_prediction_history(stock_code, prediction_date DESC);
CREATE INDEX IF NOT EXISTS ix_prediction_history_target_date ON stock_prediction_history(target_date);
CREATE INDEX IF NOT EXISTS ix_prediction_history_evaluated_at ON stock_prediction_history(evaluated_at);

-- -------------------------
-- News / disclosure / NLP
-- -------------------------
CREATE TABLE IF NOT EXISTS news_articles (
    id BIGSERIAL PRIMARY KEY,
    source VARCHAR(80) NOT NULL,
    external_id VARCHAR(200),
    title TEXT NOT NULL,
    url TEXT NOT NULL,
    canonical_url TEXT,
    published_at TIMESTAMP,
    author VARCHAR(160),
    body_text TEXT,
    language VARCHAR(20) DEFAULT 'ko',
    raw_json JSONB,
    content_hash VARCHAR(128),
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE UNIQUE INDEX IF NOT EXISTS uq_news_canonical_url ON news_articles(canonical_url) WHERE canonical_url IS NOT NULL;
CREATE UNIQUE INDEX IF NOT EXISTS uq_news_source_external ON news_articles(source, external_id) WHERE external_id IS NOT NULL;
CREATE INDEX IF NOT EXISTS ix_news_published_at ON news_articles(published_at DESC);
CREATE INDEX IF NOT EXISTS ix_news_content_hash ON news_articles(content_hash);

CREATE TABLE IF NOT EXISTS stock_news (
    id BIGSERIAL PRIMARY KEY,
    stock_code VARCHAR(6) NOT NULL REFERENCES stocks(code) ON DELETE CASCADE,
    article_id BIGINT NOT NULL REFERENCES news_articles(id) ON DELETE CASCADE,
    relevance_score DOUBLE PRECISION NOT NULL DEFAULT 0,
    matched_by VARCHAR(40),
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT uq_stock_news UNIQUE (stock_code, article_id)
);
CREATE INDEX IF NOT EXISTS ix_stock_news_code ON stock_news(stock_code);
CREATE INDEX IF NOT EXISTS ix_stock_news_article ON stock_news(article_id);

CREATE TABLE IF NOT EXISTS news_analyses (
    id BIGSERIAL PRIMARY KEY,
    stock_code VARCHAR(6) NOT NULL REFERENCES stocks(code) ON DELETE CASCADE,
    article_id BIGINT NOT NULL REFERENCES news_articles(id) ON DELETE CASCADE,
    model_name VARCHAR(100) NOT NULL DEFAULT 'rule-based',
    model_version VARCHAR(80) NOT NULL DEFAULT 'v1',
    sentiment_score DOUBLE PRECISION NOT NULL DEFAULT 0,
    importance_score DOUBLE PRECISION NOT NULL DEFAULT 0,
    relevance_score DOUBLE PRECISION NOT NULL DEFAULT 0,
    impact_score DOUBLE PRECISION NOT NULL DEFAULT 0,
    event_type VARCHAR(80),
    summary TEXT,
    rationale TEXT,
    entities_json JSONB,
    analyzed_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT uq_news_analysis UNIQUE (stock_code, article_id, model_name, model_version)
);
CREATE INDEX IF NOT EXISTS ix_news_analyses_code ON news_analyses(stock_code);
CREATE INDEX IF NOT EXISTS ix_news_analyses_impact ON news_analyses(impact_score DESC);

CREATE TABLE IF NOT EXISTS disclosures (
    id BIGSERIAL PRIMARY KEY,
    stock_code VARCHAR(6) REFERENCES stocks(code) ON DELETE CASCADE,
    corp_code VARCHAR(8),
    receipt_no VARCHAR(30) NOT NULL UNIQUE,
    report_name TEXT NOT NULL,
    filer_name VARCHAR(160),
    receipt_date DATE,
    disclosure_type VARCHAR(80),
    url TEXT,
    raw_json JSONB,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS ix_disclosures_stock_code ON disclosures(stock_code);
CREATE INDEX IF NOT EXISTS ix_disclosures_receipt_date ON disclosures(receipt_date DESC);

-- -------------------------
-- Ranking / recommendation snapshots
-- -------------------------
CREATE TABLE IF NOT EXISTS ranking_snapshots (
    id BIGSERIAL PRIMARY KEY,
    ranking_version VARCHAR(80) NOT NULL,
    as_of_date DATE NOT NULL,
    horizon_days INTEGER NOT NULL DEFAULT 5,
    universe VARCHAR(80) NOT NULL DEFAULT 'KRX',
    weights_json JSONB,
    metadata_json JSONB,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT uq_ranking_snapshot UNIQUE (ranking_version, as_of_date, horizon_days, universe)
);
CREATE INDEX IF NOT EXISTS ix_ranking_snapshots_date ON ranking_snapshots(as_of_date DESC);

CREATE TABLE IF NOT EXISTS ranking_items (
    id BIGSERIAL PRIMARY KEY,
    snapshot_id BIGINT NOT NULL REFERENCES ranking_snapshots(id) ON DELETE CASCADE,
    stock_code VARCHAR(6) NOT NULL REFERENCES stocks(code) ON DELETE CASCADE,
    rank INTEGER NOT NULL,
    total_score DOUBLE PRECISION NOT NULL,
    financial_score DOUBLE PRECISION NOT NULL DEFAULT 0,
    ml_score DOUBLE PRECISION NOT NULL DEFAULT 0,
    news_score DOUBLE PRECISION NOT NULL DEFAULT 0,
    disclosure_score DOUBLE PRECISION NOT NULL DEFAULT 0,
    valuation_score DOUBLE PRECISION NOT NULL DEFAULT 0,
    momentum_score DOUBLE PRECISION NOT NULL DEFAULT 0,
    predicted_return DOUBLE PRECISION,
    upside_probability DOUBLE PRECISION,
    rationale TEXT,
    score_components_json JSONB,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT uq_ranking_item UNIQUE (snapshot_id, stock_code),
    CONSTRAINT uq_ranking_position UNIQUE (snapshot_id, rank)
);
CREATE INDEX IF NOT EXISTS ix_ranking_items_snapshot_rank ON ranking_items(snapshot_id, rank);
CREATE INDEX IF NOT EXISTS ix_ranking_items_stock_code ON ranking_items(stock_code);

-- -------------------------
-- Recommendation performance / backtest tracking
-- -------------------------
CREATE TABLE IF NOT EXISTS recommendation_performance (
    id BIGSERIAL PRIMARY KEY,
    ranking_item_id BIGINT REFERENCES ranking_items(id) ON DELETE SET NULL,
    stock_code VARCHAR(6) NOT NULL REFERENCES stocks(code) ON DELETE CASCADE,
    recommendation_date DATE NOT NULL,
    horizon_days INTEGER NOT NULL,
    entry_price DOUBLE PRECISION,
    target_date DATE,
    exit_price DOUBLE PRECISION,
    predicted_return DOUBLE PRECISION,
    actual_return DOUBLE PRECISION,
    benchmark_return DOUBLE PRECISION,
    excess_return DOUBLE PRECISION,
    direction_correct BOOLEAN,
    max_favorable_excursion DOUBLE PRECISION,
    max_adverse_excursion DOUBLE PRECISION,
    evaluated_at TIMESTAMP,
    metadata_json JSONB,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT uq_recommendation_perf UNIQUE (stock_code, recommendation_date, horizon_days)
);
CREATE INDEX IF NOT EXISTS ix_recommendation_perf_date ON recommendation_performance(recommendation_date DESC);
CREATE INDEX IF NOT EXISTS ix_recommendation_perf_stock ON recommendation_performance(stock_code);

-- -------------------------
-- Application-wide key/value settings (future-proof, no schema change needed)
-- -------------------------
CREATE TABLE IF NOT EXISTS app_settings (
    key VARCHAR(120) PRIMARY KEY,
    value_json JSONB NOT NULL,
    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

INSERT INTO schema_versions(version, description)
VALUES ('2026-08-final-baseline-v1', 'Unified DB baseline: market, financials, ML, news, disclosures, rankings, recommendation performance')
ON CONFLICT (version) DO NOTHING;

COMMIT;
