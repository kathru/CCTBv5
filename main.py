"""CCTBv5 — entry point."""
import asyncio
import uvicorn
from src.api.app import create_app
from src.core.config import settings


def main() -> None:
    app = create_app()
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=settings.app_port,
        log_level=settings.log_level.lower(),
    )


if __name__ == "__main__":
    main()
