from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "Stock AI API"
    app_host: str = "127.0.0.1"
    app_port: int = 8000

    database_url: str = (
        "postgresql+psycopg://stock:stock@127.0.0.1:5432/stock_ai"
    )
    dart_api_key: str = ""

    logo_dev_token: str = ""

    ecos_api_key: str = ""
    ecos_base_url: str = "https://ecos.bok.or.kr/api"

    kis_app_key: str = ""
    kis_app_secret: str = ""
    kis_base_url: str = "https://openapi.koreainvestment.com:9443"
    kis_market_div_code: str = "J"
    kis_min_interval_seconds: float = 0.20
    kis_token_cache_path: str = ".cache/kis_token.json"

    toss_client_id: str = ""
    toss_client_secret: str = ""
    toss_base_url: str = "https://openapi.tossinvest.com"
    toss_token_cache_path: str = ".cache/toss_token.json"
    toss_token_refresh_margin_seconds: int = 300
    toss_market_data_min_interval_seconds: float = 0.08
    toss_trading_trend_min_interval_seconds: float = 0.11
    toss_market_indicator_min_interval_seconds: float = 0.11
    toss_market_indicator_chart_min_interval_seconds: float = 0.21
    toss_market_info_min_interval_seconds: float = 0.35
    toss_request_timeout_seconds: float = 15.0

    price_cache_seconds: int = 10
    live_quote_cache_seconds: float = 1.0
    live_quote_poll_seconds: float = 1.0

    cors_origins: str = "*"

    automation_cron_secret: str = ""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
