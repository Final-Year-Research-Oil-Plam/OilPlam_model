"""FastAPI application entrypoint and router registration."""

from fastapi import FastAPI

try:
    from app.config import API_TITLE, API_VERSION
    from app.routers.router import health_router, palm_router
except ModuleNotFoundError:
    from app.config import API_TITLE, API_VERSION
    from app.routers.router import health_router, palm_router

app = FastAPI(title=API_TITLE, version=API_VERSION)

# Split routers by concern: health checks and palm inference endpoints.
app.include_router(health_router)
app.include_router(palm_router)


@app.get("/api/v1/info")
def api_info():
    """Return static API metadata for quick service identification."""
    return {"title": API_TITLE, "version": API_VERSION}
