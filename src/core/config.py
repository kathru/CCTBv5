from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # Application
    app_env: str = "development"
    app_port: int = 8001
    log_level: str = "INFO"

    # PostgreSQL
    database_url: str = "postgresql+asyncpg://cctb:cctb_secret@localhost:5432/cctbv5"

    # Redis
    redis_url: str = "redis://localhost:6379/0"

    # OKX
    okx_api_key: str = ""
    okx_secret_key: str = ""
    okx_passphrase: str = ""
    okx_paper_trading: bool = True

    # Discord
    discord_webhook_url: str = ""

    # News
    finnhub_token: str = ""
    eodhd_token: str = ""

    # Risk limits
    max_total_exposure: float = 0.5
    max_daily_drawdown: float = 0.03


settings = Settings()
