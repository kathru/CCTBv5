from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from ..core.config import settings
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
        yield
        await db.disconnect()
        await cache.disconnect()

    app = FastAPI(title="CCTBv5", version="5.0.0", lifespan=lifespan)

    app.state.db = db
    app.state.cache = cache

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
        return {
            "status": "ok" if db_ok and cache_ok else "degraded",
            "version": "5.0.0",
            "postgres": db_ok,
            "redis": cache_ok,
        }

    @app.get("/", response_class=HTMLResponse)
    async def dashboard():
        html_file = DASHBOARD_DIR / "templates" / "index.html"
        if html_file.exists():
            return HTMLResponse(html_file.read_text(encoding="utf-8"))
        return HTMLResponse("<h1>CCTBv5 Dashboard</h1><p>Template not found.</p>")

    return app
