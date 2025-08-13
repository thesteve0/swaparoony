from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from .api.routes.face_swap import router as face_swap_router
from .api.dependencies import get_face_swap_service
from .core.config import settings
from .core.exceptions import ModelLoadError


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Initialize models
    try:
        service = get_face_swap_service()
        print("Face swap service initialized successfully")
    except ModelLoadError as e:
        print(f"Failed to initialize face swap service: {e}")
        raise

    yield

    # Shutdown: Clean up if needed
    print("Shutting down face swap service")


def create_app() -> FastAPI:
    app = FastAPI(
        title="Swaparoony Face Swap API",
        description="Trade show face swapping application",
        version="1.0.0",
        lifespan=lifespan,
    )

    # CORS middleware for web frontend
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Configure appropriately for production
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Include routers
    app.include_router(face_swap_router, prefix="/api/v1", tags=["face-swap"])

    @app.get("/")
    async def root():
        return {
            "message": "Swaparoony Face Swap API",
            "version": "1.0.0",
            "status": "running",
        }

    return app


app = create_app()

if __name__ == "__main__":
    import uvicorn

    uvicorn.run("src.swaparoony.main:app", host="0.0.0.0", port=8000, reload=True)
