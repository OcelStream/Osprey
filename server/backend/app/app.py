from contextlib import asynccontextmanager
from fastapi import FastAPI
from backend.app.api.v1.endpoints import router
from backend.app.core.context import pipeline
import threading


@asynccontextmanager
async def lifespan(app: FastAPI):
    thread = threading.Thread(target=pipeline.start, daemon=True)
    thread.start()
    pipeline._ready.wait(timeout=30)
    yield
    pipeline.stop()


app = FastAPI(title="DeepStream API", lifespan=lifespan)
app.include_router(router, prefix="/api/v1", tags=["deepstream"])
