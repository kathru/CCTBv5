"""
Structured logging setup — structlog with JSON output.

Call configure_logging() once at application startup (in main.py).

Every log entry automatically contains:
  - timestamp   (ISO8601 UTC)
  - level       (debug/info/warning/error/critical)
  - logger      (module name)
  - event       (the message)

Optional fields added by context (bind to logger):
  - strategy_id
  - symbol
  - event_type
  - order_id
  - latency_ms

In development (APP_ENV=development):
  - Console output with colors (human-readable)
  - File output in JSON (machine-readable)

In production (APP_ENV=production):
  - JSON only (stdout + file)
"""

import logging
import logging.handlers
import sys
from pathlib import Path

import structlog

LOG_DIR = Path("logs")
LOG_FILE = LOG_DIR / "cctbv5.log"
MAX_BYTES = 10 * 1024 * 1024   # 10 MB per file
BACKUP_COUNT = 5                # keep 5 rotated files


def configure_logging(
    log_level: str = "INFO",
    app_env: str = "development",
    log_dir: Path = LOG_DIR,
) -> None:
    """
    Configure structlog + standard logging.
    Call once at startup before any other code runs.
    """
    log_dir.mkdir(exist_ok=True)
    level = getattr(logging, log_level.upper(), logging.INFO)

    # ── Shared processors (applied to every log entry) ────────
    shared_processors = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        structlog.processors.StackInfoRenderer(),
    ]

    # ── structlog configuration ───────────────────────────────
    structlog.configure(
        processors=shared_processors + [
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    # ── File handler — always JSON ────────────────────────────
    file_formatter = structlog.stdlib.ProcessorFormatter(
        processors=[
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            structlog.processors.dict_tracebacks,
            structlog.processors.JSONRenderer(),
        ],
        foreign_pre_chain=shared_processors,
    )
    file_handler = logging.handlers.RotatingFileHandler(
        filename=str(LOG_FILE),
        maxBytes=MAX_BYTES,
        backupCount=BACKUP_COUNT,
        encoding="utf-8",
    )
    file_handler.setFormatter(file_formatter)
    file_handler.setLevel(level)

    # ── Console handler — colored in dev, JSON in prod ────────
    if app_env == "development":
        console_formatter = structlog.stdlib.ProcessorFormatter(
            processors=[
                structlog.stdlib.ProcessorFormatter.remove_processors_meta,
                structlog.dev.ConsoleRenderer(colors=True),
            ],
            foreign_pre_chain=shared_processors,
        )
    else:
        console_formatter = structlog.stdlib.ProcessorFormatter(
            processors=[
                structlog.stdlib.ProcessorFormatter.remove_processors_meta,
                structlog.processors.JSONRenderer(),
            ],
            foreign_pre_chain=shared_processors,
        )

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(console_formatter)
    console_handler.setLevel(level)

    # ── Root logger ───────────────────────────────────────────
    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)
    root_logger.setLevel(level)

    # Silence noisy third-party loggers
    logging.getLogger("asyncpg").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)

    log = structlog.get_logger("cctbv5.logging")
    log.info(
        "Logging configured",
        level=log_level,
        env=app_env,
        log_file=str(LOG_FILE),
    )


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    """
    Get a structured logger. Use instead of logging.getLogger().

    Usage:
        logger = get_logger(__name__)
        logger.info("Order submitted", order_id="abc", symbol="BTC-USDT")

    Bind persistent context:
        log = logger.bind(strategy_id="v4_momentum", symbol="BTC-USDT")
        log.info("Signal generated", score=0.75)
    """
    return structlog.get_logger(name)
