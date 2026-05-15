"""CCTBv5 — entry point."""
import uvicorn
from src.api.app import create_app
from src.core.config import settings
from src.core.logging import configure_logging


def main() -> None:
    # Configure structured logging FIRST — before anything else
    configure_logging(
        log_level=settings.log_level,
        app_env=settings.app_env,
    )

    app = create_app()
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=settings.app_port,
        log_config=None,    # disable uvicorn's default logging (we handle it)
    )


if __name__ == "__main__":
    main()
