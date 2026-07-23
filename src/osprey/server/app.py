from contextlib import asynccontextmanager
import threading

from fastapi import FastAPI

from osprey.server.api.v1.endpoints import router
from osprey.server.core.context import get_pipeline


@asynccontextmanager
async def lifespan(app: FastAPI):
    pipeline = get_pipeline()
    thread = threading.Thread(target=pipeline.start, daemon=True)
    thread.start()
    pipeline._ready.wait(timeout=30)
    yield
    pipeline.stop()


app = FastAPI(title="Osprey DeepStream API", lifespan=lifespan)
app.include_router(router, prefix="/api/v1", tags=["deepstream"])


def main() -> None:
    """Entry point for the ``osprey-server`` console script."""
    import os

    import uvicorn

    uvicorn.run(
        "osprey.server.app:app",
        host=os.environ.get("OSPREY_HOST", "0.0.0.0"),
        port=int(os.environ.get("OSPREY_PORT", "8000")),
    )


if __name__ == "__main__":
    main()
