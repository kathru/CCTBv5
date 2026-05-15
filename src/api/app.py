from contextlib import asynccontextmanager
from fastapi import FastAPI

from ..core.config import settings
from ..persistence import Database, Cache


def create_app() -> FastAPI:

    db = Database(dsn=settings.database_url)
    cache = Cache(redis_url=settings.redis_url)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # Startup
        await db.connect()
        await cache.connect()
        yield
        # Shutdown
        await db.disconnect()
        await cache.disconnect()

    app = FastAPI(title="CCTBv5", version="5.0.0", lifespan=lifespan)

    # Attach to app state for use in routes
    app.state.db = db
    app.state.cache = cache

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

    return app
