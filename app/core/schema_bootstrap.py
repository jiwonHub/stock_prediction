from sqlalchemy import text
from sqlalchemy.engine import Engine


_RUNTIME_ALTERS = [
    # news
    "ALTER TABLE news_articles ALTER COLUMN external_id TYPE TEXT",
    
    # stocks
    "ALTER TABLE stocks ADD COLUMN IF NOT EXISTS sector_code VARCHAR(40)",
    "ALTER TABLE stocks ADD COLUMN IF NOT EXISTS industry_name VARCHAR(160)",
    "ALTER TABLE stocks ADD COLUMN IF NOT EXISTS listing_date DATE",
    "ALTER TABLE stocks ADD COLUMN IF NOT EXISTS shares_outstanding BIGINT",
    "ALTER TABLE stocks ADD COLUMN IF NOT EXISTS is_active BOOLEAN NOT NULL DEFAULT TRUE",
    "ALTER TABLE stocks ADD COLUMN IF NOT EXISTS last_financial_at TIMESTAMP",
    "ALTER TABLE stocks ADD COLUMN IF NOT EXISTS last_news_at TIMESTAMP",
    "ALTER TABLE stocks ADD COLUMN IF NOT EXISTS last_prediction_at TIMESTAMP",

    # stock_prices
    "ALTER TABLE stock_prices ADD COLUMN IF NOT EXISTS adjusted_close DOUBLE PRECISION",
    "ALTER TABLE stock_prices ADD COLUMN IF NOT EXISTS trading_value NUMERIC(30,2)",

    # financial_statements / financial_metrics
    "ALTER TABLE financial_statements ADD COLUMN IF NOT EXISTS account_detail VARCHAR(500) NOT NULL DEFAULT ''",
    "ALTER TABLE financial_metrics ADD COLUMN IF NOT EXISTS valuation_score DOUBLE PRECISION NOT NULL DEFAULT 0",
    "ALTER TABLE financial_metrics ADD COLUMN IF NOT EXISTS quality_score DOUBLE PRECISION NOT NULL DEFAULT 0",
    "ALTER TABLE financial_metrics ADD COLUMN IF NOT EXISTS sector_relative_score DOUBLE PRECISION NOT NULL DEFAULT 0",

    # stock_predictions
    "ALTER TABLE stock_predictions ADD COLUMN IF NOT EXISTS model_path VARCHAR(500)",
    "ALTER TABLE stock_predictions ADD COLUMN IF NOT EXISTS model_run_id BIGINT",
    "ALTER TABLE stock_predictions ADD COLUMN IF NOT EXISTS prediction_date DATE",
    "ALTER TABLE stock_predictions ADD COLUMN IF NOT EXISTS target_date DATE",
    "ALTER TABLE stock_predictions ADD COLUMN IF NOT EXISTS predicted_price DOUBLE PRECISION",
    "ALTER TABLE stock_predictions ADD COLUMN IF NOT EXISTS prediction_lower DOUBLE PRECISION",
    "ALTER TABLE stock_predictions ADD COLUMN IF NOT EXISTS prediction_upper DOUBLE PRECISION",
    "ALTER TABLE stock_predictions ADD COLUMN IF NOT EXISTS confidence_score DOUBLE PRECISION",
    "ALTER TABLE stock_predictions ADD COLUMN IF NOT EXISTS feature_json JSONB",
    "ALTER TABLE stock_predictions ADD COLUMN IF NOT EXISTS updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP",
]

_RUNTIME_INDEXES = [
    "CREATE INDEX IF NOT EXISTS ix_stocks_is_active ON stocks(is_active)",
    "CREATE INDEX IF NOT EXISTS ix_stock_prices_code_date ON stock_prices(stock_code, trade_date DESC)",
    "CREATE INDEX IF NOT EXISTS ix_financial_metrics_financial_score ON financial_metrics(financial_score DESC)",
    "CREATE INDEX IF NOT EXISTS ix_stock_predictions_ml_score ON stock_predictions(ml_score DESC)",
]


def ensure_runtime_schema(engine: Engine) -> None:
    with engine.begin() as conn:
        for statement in _RUNTIME_ALTERS:
            conn.execute(text(statement))

        for statement in _RUNTIME_INDEXES:
            conn.execute(text(statement))

        conn.execute(
            text(
                """
                INSERT INTO schema_versions(
                    version,
                    description,
                    applied_at
                )
                VALUES (
                    :version,
                    :description,
                    CURRENT_TIMESTAMP
                )
                ON CONFLICT (version) DO NOTHING
                """
            ),
            {
                "version": "2026-08-runtime-hotfix-v2",
                "description": "Automatic legacy DB compatibility + batch pipeline hotfix",
            },
        )