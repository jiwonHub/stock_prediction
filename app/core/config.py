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

    kis_app_key: str = ""
    kis_app_secret: str = ""
    kis_base_url: str = "https://openapi.koreainvestment.com:9443"
    kis_market_div_code: str = "J"
    kis_min_interval_seconds: float = 0.20
    kis_token_cache_path: str = ".cache/kis_token.json"

    price_cache_seconds: int = 10

    cors_origins: str = "*"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
