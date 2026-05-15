from fastapi import FastAPI


def create_app() -> FastAPI:
    app = FastAPI(title="CCTBv5", version="5.0.0")

    @app.get("/health")
    async def health() -> dict:
        return {"status": "ok", "version": "5.0.0"}

    return app
