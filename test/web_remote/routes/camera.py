"""Camera selection, system status and MJPEG stream (TEST ONLY)."""

from fastapi import APIRouter
from fastapi.responses import JSONResponse, StreamingResponse

from test.web_remote.runtime import runtime
from test.web_remote.camera_manager import MIN_CHANNEL, MAX_CHANNEL


router = APIRouter(prefix="/api")


@router.get("/status")
def get_status():
    return runtime.system_status()


@router.get("/thresholds")
def get_thresholds():
    return {
        "ok": True,
        "loaded": runtime.loaded_thresholds(),
        "loaded_at": runtime.system_status()["thresholds_loaded_at"],
        "error": runtime.thresholds_reload_error,
    }


@router.post("/thresholds/reload")
def reload_thresholds():
    try:
        values = runtime.reload_thresholds()
    except Exception as error:
        runtime.thresholds_reload_error = str(error)
        return {"ok": False, "message": "Reload failed: %s" % error}
    return {"ok": True, "message": "Thresholds reloaded from .env", "loaded": values}


@router.post("/camera/laptop")
def camera_laptop():
    ok, message = runtime.set_camera_laptop()
    return {"ok": ok, "message": message, **runtime.system_status()}


@router.post("/camera/channel/{channel}")
def camera_channel(channel: int):
    if not (MIN_CHANNEL <= channel <= MAX_CHANNEL):
        return JSONResponse(
            status_code=400,
            content={"ok": False, "message": "Channel must be %d..%d" % (MIN_CHANNEL, MAX_CHANNEL)},
        )
    ok, message = runtime.set_camera_channel(channel)
    return {"ok": ok, "message": message, **runtime.system_status()}


@router.post("/camera/stop")
def camera_stop():
    if runtime.camera_locked():
        return {"ok": False, "message": "Camera locked: %s in progress" % runtime.current_mode()}
    runtime.camera.stop()
    return {"ok": True, "message": "Camera stopped", **runtime.system_status()}


def _mjpeg_generator():
    last_id = -1
    boundary = b"--frame\r\n"
    while not runtime.shutting_down:
        jpeg, frame_id = runtime.wait_for_frame(last_id, timeout=1.0)
        if jpeg is None:
            continue
        last_id = frame_id
        yield (
            boundary
            + b"Content-Type: image/jpeg\r\n"
            + b"Content-Length: " + str(len(jpeg)).encode() + b"\r\n\r\n"
            + jpeg
            + b"\r\n"
        )


@router.get("/video/stream")
def video_stream():
    return StreamingResponse(
        _mjpeg_generator(),
        media_type="multipart/x-mixed-replace; boundary=frame",
        headers={
            "Cache-Control": "no-cache, no-store, must-revalidate",
            "Pragma": "no-cache",
        },
    )


@router.get("/video/snapshot")
def video_snapshot():
    jpeg, _ = runtime.wait_for_frame(-1, timeout=1.0)
    if jpeg is None:
        return JSONResponse(status_code=503, content={"ok": False, "message": "No frame"})
    from fastapi.responses import Response
    return Response(content=jpeg, media_type="image/jpeg")
