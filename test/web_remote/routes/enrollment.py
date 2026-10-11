"""Enrollment page + API (TEST ONLY). Uses an isolated AdaFace gallery."""

from fastapi import APIRouter, Request
from pydantic import BaseModel

from test.web_remote.runtime import runtime, BusyError, MODE_ENROLLMENT
from test.web_remote.enrollment_session import EnrollmentSession
from test.web_remote.templating import templates


router = APIRouter()


class EnrollmentStartRequest(BaseModel):
    full_name: str
    room_number: str | None = None


@router.get("/enrollment")
def enrollment_page(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="enrollment.html",
        context={"page": "enrollment", "title": "Member Enrollment"},
    )


@router.post("/api/enrollment/start")
def enrollment_start(body: EnrollmentStartRequest):
    if not body.full_name or not body.full_name.strip():
        return {"ok": False, "message": "Họ tên không được để trống"}
    if runtime.camera.source == "NONE":
        return {"ok": False, "message": "Select a camera first"}

    # Check the mode before preparing local enrollment.
    active = runtime.current_session()
    if active is not None:
        return {"ok": False, "message": "System busy: %s" % active.mode}

    try:
        session = EnrollmentSession(
            full_name=body.full_name,
            room_number=body.room_number,
        )
    except Exception as error:
        return {"ok": False, "message": "%s: %s" % (type(error).__name__, error)}

    try:
        runtime.start_session(session)
    except BusyError as error:
        session.stop()
        return {"ok": False, "message": str(error)}

    return {"ok": True, "message": "Enrollment started", "status": session.status()}


@router.get("/api/enrollment/status")
def enrollment_status():
    session = runtime.session_for(MODE_ENROLLMENT)
    return {
        "system": runtime.system_status(),
        "enrollment": session.status() if session is not None else None,
    }


@router.post("/api/enrollment/cancel")
def enrollment_cancel():
    stopped = runtime.stop_session(MODE_ENROLLMENT)
    return {"ok": stopped, "message": "Enrollment cancelled" if stopped else "Enrollment is not running"}
