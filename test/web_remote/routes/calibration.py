"""Calibration page + API (TEST ONLY)."""

from fastapi import APIRouter, Request
from pydantic import BaseModel

from test.web_remote.runtime import runtime, BusyError, MODE_CALIBRATION
from test.web_remote.calibration_session import CalibrationSession
from test.web_remote.templating import templates


router = APIRouter()


@router.get("/calibration")
def calibration_page(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="calibration.html",
        context={"page": "calibration", "title": "Camera Calibration"},
    )


@router.post("/api/calibration/start")
def calibration_start():
    if runtime.camera.source == "NONE":
        return {"ok": False, "message": "Select a camera first"}
    try:
        session = CalibrationSession()
        runtime.start_session(session)
    except BusyError as error:
        return {"ok": False, "message": str(error)}
    except Exception as error:
        return {"ok": False, "message": "%s: %s" % (type(error).__name__, error)}
    return {"ok": True, "message": "Calibration started", "status": session.status()}


@router.post("/api/calibration/ready")
def calibration_ready():
    session = runtime.current_session()
    if session is None or session.mode != MODE_CALIBRATION:
        return {"ok": False, "message": "Calibration is not running"}
    ok, message = session.ready()
    return {"ok": ok, "message": message}


@router.post("/api/calibration/cancel")
def calibration_cancel():
    stopped = runtime.stop_session(MODE_CALIBRATION)
    return {"ok": stopped, "message": "Calibration cancelled" if stopped else "Calibration is not running"}


@router.get("/api/calibration/status")
def calibration_status():
    session = runtime.session_for(MODE_CALIBRATION)
    return {
        "system": runtime.system_status(),
        "calibration": session.status() if session is not None else None,
    }


class SaveRequest(BaseModel):
    values: dict[str, float] | None = None


@router.post("/api/calibration/save")
def calibration_save(body: SaveRequest | None = None):
    session = runtime.session_for(MODE_CALIBRATION)
    if session is None:
        return {"ok": False, "message": "No calibration result"}
    values = body.values if body is not None else None
    ok, message = session.save(values)
    return {
        "ok": ok,
        "message": message,
        "saved_keys": session.saved_keys,
        "saved_values": session.saved_values,
        "loaded_thresholds": runtime.loaded_thresholds(),
    }
