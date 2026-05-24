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

    # OKX — Live credentials
    okx_api_key: str = ""
    okx_secret_key: str = ""
    okx_passphrase: str = ""
    okx_paper_trading: bool = True

    # OKX — Demo/Paper credentials (conta demo separada da OKX)
    # Se preenchidas, usadas automaticamente quando okx_paper_trading=True
    okx_demo_api_key: str = ""
    okx_demo_secret_key: str = ""
    okx_demo_passphrase: str = ""

    # Discord
    discord_webhook_url: str = ""
    bot_instance: str = "localhost"   # identificador da instância (ex: "Oracle", "localhost")

    # Anthropic (Daily Agent LLM)
    anthropic_api_key: str = ""

    # News
    finnhub_token: str = ""
    eodhd_token: str = ""

    # Risk limits
    max_total_exposure: float = 0.5
    max_daily_drawdown: float = 0.03

    # Modo monitor: avalia sinais e exibe dashboard, mas NÃO executa ordens.
    # Ativar no localhost para não contaminar o Oracle (que roda fulltime).
    monitor_only: bool = False


settings = Settings()
