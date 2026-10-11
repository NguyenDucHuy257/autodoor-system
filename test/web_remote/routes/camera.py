"""Camera selection, system status and MJPEG stream (TEST ONLY)."""

import asyncio
import json
import math
import time
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse, StreamingResponse
from app.performance import performance

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
    ok, message = runtime.stop_camera()
    return {"ok": ok, "message": message, **runtime.system_status()}


def _mjpeg_generator():
    last_id = -1
    boundary = b"--frame\r\n"
    performance.count("mjpeg_clients_opened")
    try:
        while not runtime.shutting_down:
            jpeg, frame_id = runtime.wait_for_frame(last_id, timeout=1.0)
            if jpeg is None:
                performance.count("mjpeg_wait_timeouts")
                continue
            if last_id >= 0:
                performance.count("mjpeg_frames_skipped", max(0, frame_id - last_id - 1))
            last_id = frame_id
            performance.count("mjpeg_frames")
            performance.count("mjpeg_bytes", len(jpeg))
            yield (
                boundary
                + b"Content-Type: image/jpeg\r\n"
                + b"Content-Length: " + str(len(jpeg)).encode() + b"\r\n\r\n"
                + jpeg
                + b"\r\n"
            )
    finally:
        performance.count("mjpeg_clients_closed")


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


# At most four delivered frames await feedback; no per-frame stop-and-wait.
MAX_IN_FLIGHT = 4
MAX_DELIVERY_AGE = 1.0


@router.websocket("/video/live")
async def video_live(socket: WebSocket, kind: str = "preview"):
    if kind not in ("preview", "ai"):
        await socket.close(code=1008)
        return
    await socket.accept()
    outstanding = {}
    feedback = asyncio.Event()
    disconnected = asyncio.Event()
    prefix = "client_ai" if kind == "ai" else "client_preview"

    async def receive_feedback():
        try:
            while True:
                reply = await socket.receive_text()
                try:
                    data = json.loads(reply)
                    if not isinstance(data, dict):
                        continue
                    sent = outstanding.pop(data.get("sequence"), None)
                    if sent is None:
                        continue
                    feedback.set()
                    performance.observe(prefix + "_delivery_ack_ms", (time.perf_counter()-sent)*1000)
                    rendered = data.get("rendered") is True
                    performance.count(("client_rendered_frames" if kind == "preview" else "client_ai_rendered_frames") if rendered else prefix + "_frames_dropped")
                    for key in ("decode_ms", "render_ms"):
                        value = float(data.get(key, 0))
                        if math.isfinite(value) and 0 <= value <= 60000:
                            performance.observe(prefix + "_" + key, value)
                except (ValueError, TypeError, KeyError):
                    continue
        except (WebSocketDisconnect, RuntimeError):
            disconnected.set()
            feedback.set()

    receiver = asyncio.create_task(receive_feedback())
    last_id, sequence = -1, 0
    try:
        while not runtime.shutting_down and not disconnected.is_set():
            # Slow clients cannot accumulate more than MAX_IN_FLIGHT frames.
            # Close an expired transport rather than replay queued stale images.
            if outstanding and time.perf_counter()-min(outstanding.values()) > MAX_DELIVERY_AGE:
                performance.count(prefix + "_slow_client_reconnects")
                await socket.close(code=1013)
                break
            if len(outstanding) >= MAX_IN_FLIGHT:
                feedback.clear()
                try:
                    await asyncio.wait_for(feedback.wait(), timeout=0.05)
                except asyncio.TimeoutError:
                    pass
                continue
            jpeg, frame_id, generation, faces = await asyncio.to_thread(runtime.wait_for_delivery, last_id, 0.1, kind)
            if disconnected.is_set() or frame_id == last_id:
                continue
            if last_id >= 0:
                performance.count(prefix + "_server_frames_skipped", max(0, frame_id-last_id-1))
            last_id = frame_id
            sequence += 1
            outstanding[sequence] = time.perf_counter()
            metadata = {"sequence": sequence, "generation": generation, "clear": jpeg is None, "faces": faces}
            async def send_frame():
                await socket.send_json(metadata)
                if jpeg is not None:
                    with performance.measure("websocket_" + kind + "_send_ms"):
                        await socket.send_bytes(jpeg)
                    performance.count("websocket_" + kind + "_bytes", len(jpeg))
            await asyncio.wait_for(send_frame(), timeout=MAX_DELIVERY_AGE)
            performance.count("websocket_" + kind + "_frames_sent")
    except (WebSocketDisconnect, asyncio.TimeoutError, RuntimeError):
        pass
    finally:
        receiver.cancel()
        await asyncio.gather(receiver, return_exceptions=True)
        try:
            await socket.close()
        except RuntimeError:
            pass
