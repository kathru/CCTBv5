"""
PostgreSQL connection pool — async via asyncpg.

Single pool shared across the application.
Initialize once on startup, close on shutdown.

Usage:
    db = Database(settings.database_url)
    await db.connect()
    ...
    await db.disconnect()

    # In a repository:
    async with db.acquire() as conn:
        rows = await conn.fetch("SELECT ...")
"""

import asyncpg
import logging
from contextlib import asynccontextmanager
from typing import AsyncGenerator

logger = logging.getLogger(__name__)


class Database:
    """Async PostgreSQL connection pool wrapper."""

    def __init__(
        self,
        dsn: str,
        min_size: int = 2,
        max_size: int = 10,
    ) -> None:
        self._dsn = dsn
        self._min_size = min_size
        self._max_size = max_size
        self._pool: asyncpg.Pool | None = None

    async def connect(self) -> None:
        """Initialize the connection pool."""
        self._pool = await asyncpg.create_pool(
            dsn=self._dsn,
            min_size=self._min_size,
            max_size=self._max_size,
        )
        logger.info(
            "PostgreSQL pool ready min=%d max=%d",
            self._min_size,
            self._max_size,
        )

    async def disconnect(self) -> None:
        """Close all connections in the pool."""
        if self._pool:
            await self._pool.close()
            self._pool = None
            logger.info("PostgreSQL pool closed")

    @asynccontextmanager
    async def acquire(self) -> AsyncGenerator[asyncpg.Connection, None]:
        """Acquire a connection from the pool."""
        if self._pool is None:
            raise RuntimeError("Database not connected — call connect() first")
        async with self._pool.acquire() as conn:
            yield conn

    async def execute(self, query: str, *args) -> str:
        """Execute a query that returns no rows."""
        async with self.acquire() as conn:
            return await conn.execute(query, *args)

    async def fetch(self, query: str, *args) -> list[asyncpg.Record]:
        """Execute a query and return all rows."""
        async with self.acquire() as conn:
            return await conn.fetch(query, *args)

    async def fetchrow(self, query: str, *args) -> asyncpg.Record | None:
        """Execute a query and return one row."""
        async with self.acquire() as conn:
            return await conn.fetchrow(query, *args)

    async def fetchval(self, query: str, *args):
        """Execute a query and return a single value."""
        async with self.acquire() as conn:
            return await conn.fetchval(query, *args)

    async def apply_schema(self, schema_path: str) -> None:
        """Run a SQL file against the database (used on startup)."""
        with open(schema_path) as f:
            sql = f.read()
        await self.execute(sql)
        logger.info("Schema applied from %s", schema_path)

    @property
    def is_connected(self) -> bool:
        return self._pool is not None
