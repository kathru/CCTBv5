import asyncio
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from ..core.config import settings
from ..core.version import get_version, get_version_info
from ..persistence import Cache, Database
from .routers import metrics, orders, positions, watchdog

DASHBOARD_DIR = Path(__file__).parent.parent / "dashboard"


def create_app() -> FastAPI:

    db = Database(dsn=settings.database_url)
    cache = Cache(redis_url=settings.redis_url)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        await db.connect()
        await cache.connect()

        # Start trading loop in background (non-blocking)
        trading_task = None
        if settings.app_env != "test":
            try:
                from ..core.trading_loop import TradingLoop
                loop = TradingLoop(db=db, cache=cache, app_state=app.state)
                trading_task = asyncio.create_task(
                    loop.start(), name="trading_loop"
                )
                app.state.trading_loop = loop
            except Exception as exc:
                import logging
                logging.getLogger(__name__).warning(
                    "TradingLoop not started: %s", exc
                )

        yield

        # Shutdown
        if trading_task and not trading_task.done():
            trading_task.cancel()
            try:
                await trading_task
            except (asyncio.CancelledError, Exception):
                pass

        await db.disconnect()
        await cache.disconnect()

    app = FastAPI(title="CCTBv5", version="5.0.0", lifespan=lifespan)

    app.state.db = db
    app.state.cache = cache
    app.state.trading_loop = None

    # Routers
    app.include_router(positions.router)
    app.include_router(orders.router)
    app.include_router(metrics.router)
    app.include_router(watchdog.router)

    # Initialize watchdog placeholders (populated by trading engine)
    app.state.ws_watchdog = None
    app.state.heartbeat_watchdog = None
    app.state.resource_watchdog = None

    # Static files
    static_dir = DASHBOARD_DIR / "static"
    static_dir.mkdir(parents=True, exist_ok=True)
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

    @app.get("/health")
    async def health() -> dict:
        db_ok = db.is_connected
        cache_ok = await cache.ping()
        loop = app.state.trading_loop
        return {
            "status": "ok" if db_ok and cache_ok else "degraded",
            "version": get_version(),
            "postgres": db_ok,
            "redis": cache_ok,
            "trading_loop": loop is not None,
        }

    @app.get("/version")
    async def version() -> dict:
        return get_version_info()

    @app.get("/", response_class=HTMLResponse)
    async def dashboard():
        html_file = DASHBOARD_DIR / "templates" / "index.html"
        if html_file.exists():
            return HTMLResponse(html_file.read_text(encoding="utf-8"))
        return HTMLResponse("<h1>CCTBv5 Dashboard</h1><p>Template not found.</p>")

    return app
