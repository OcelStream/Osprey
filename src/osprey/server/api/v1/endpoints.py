from fastapi import APIRouter, HTTPException
from osprey.server.models import StreamRequest
from osprey.server.core.context import get_pipeline
import logging

logger = logging.getLogger(__name__)
router = APIRouter()

# ----------------- Endpoints -----------------
@router.post("/add")
def add_stream(req: StreamRequest):
    logger.info("Adding stream %s (uri=%s, %dx%d)",
                req.stream_id, req.uri, req.rtsp_output_width, req.rtsp_output_height)
    try:
        uuid = get_pipeline().add_source(
            req.uri, req.rtsp_output_width, req.rtsp_output_height, req.stream_id
        )
        return {"message": "Stream added", "uuid": uuid, "rtsp": f"rtsp://localhost:8554/ds-test{uuid}"}
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.delete("/remove/{stream_id}")
def remove_stream(stream_id: str):
    pipeline = get_pipeline()
    active = pipeline.get_active_streams()
    if not any(s["stream_id"] == stream_id for s in active):
        raise HTTPException(status_code=404, detail="Stream not found")
    pipeline.remove_source(stream_id)
    return {"message": "Stream removed", "uuid": stream_id}


@router.get("/streams")
def list_streams():
    return get_pipeline().get_active_streams()


@router.get("/health/ready")
def ready():
    """Readiness probe — returns 200 only after the pipeline has reached PLAYING state."""
    if get_pipeline()._ready.is_set():
        return {"status": "ready"}
    raise HTTPException(status_code=503, detail="Pipeline not ready yet")
