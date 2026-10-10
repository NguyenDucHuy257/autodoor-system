"""Realtime recognition debug page + API (TEST ONLY)."""

from fastapi import APIRouter, Request

from test.web_remote.runtime import runtime, BusyError, MODE_RECOGNITION
from test.web_remote.recognition_session import RecognitionSession
from test.web_remote.templating import templates


router = APIRouter()


@router.get("/recognition")
def recognition_page(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="recognition.html",
        context={"page": "recognition", "title": "Realtime Debug"},
    )


@router.post("/api/recognition/start")
def recognition_start():
    active = runtime.current_session()
    if active is not None:
        return {"ok": False, "message": "System busy: %s" % active.mode}

    try:
        # loads templates from DB (SessionLocal) - may fail if DB is down
        session = RecognitionSession()
    except Exception as error:
        return {"ok": False, "message": "%s: %s" % (type(error).__name__, error)}

    try:
        runtime.start_session(session)
    except BusyError as error:
        return {"ok": False, "message": str(error)}

    return {
        "ok": True,
        "message": "Recognition started (%d embeddings, %d members)" % (
            len(session.person_database), session.member_count
        ),
    }


@router.post("/api/recognition/stop")
def recognition_stop():
    stopped = runtime.stop_session(MODE_RECOGNITION)
    return {"ok": stopped, "message": "Recognition stopped" if stopped else "Recognition is not running"}


@router.get("/api/recognition/status")
def recognition_status():
    session = runtime.session_for(MODE_RECOGNITION)
    return {
        "system": runtime.system_status(),
        "recognition": session.status() if session is not None else None,
    }
