"""
AutoDoor LAN web test tool (TEST ONLY - not part of production source).

Run from the project root:

    python -m uvicorn test.web_remote.app:app --host 0.0.0.0 --port 8000

Then open http://<PC-LAN-IP>:8000 from a phone on the same Wi-Fi.
"""

import logging
import signal
import socket
import sys
from contextlib import asynccontextmanager
from test.web_remote.logging_setup import configure_logging, redirect_native_output

configure_logging()

# EnrollmentService prints Vietnamese text (e.g. "Ảnh không đạt chuẩn!").
# On Windows, a non-UTF-8 console / redirected stdout would raise
# UnicodeEncodeError inside the worker thread. Make stdout tolerant here
# (web tool only - app/ is untouched).
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles

from test.web_remote.runtime import runtime
from test.web_remote.templating import templates, STATIC_DIR
from test.web_remote.routes import camera, calibration, enrollment, recognition


log = logging.getLogger("web_remote")


def _lan_ip() -> str:
    try:
        probe = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        probe.connect(("10.255.255.255", 1))
        ip = probe.getsockname()[0]
        probe.close()
        return ip
    except Exception:
        return "127.0.0.1"


def _chain_signal_handlers() -> None:
    """
    uvicorn waits for open connections (our MJPEG streams) before running the
    lifespan shutdown. Flag the runtime on Ctrl+C so the stream generators exit.
    Best effort only (main thread required).
    """
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            previous = signal.getsignal(sig)

            def handler(signum, frame, previous=previous):
                runtime._stop_event.set()
                if callable(previous):
                    previous(signum, frame)

            signal.signal(sig, handler)
        except Exception:
            pass


@asynccontextmanager
async def lifespan(app: FastAPI):
    configure_logging()
    redirect_native_output()
    runtime.start()
    _chain_signal_handlers()
    log.info("AutoDoor Test Remote ready: http://%s:8000  (LAN)", _lan_ip())
    try:
        yield
    finally:
        log.info("Shutting down: stopping sessions and cameras")
        runtime.shutdown()


app = FastAPI(title="AutoDoor Test Remote", lifespan=lifespan, docs_url="/docs")

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

app.include_router(camera.router)
app.include_router(calibration.router)
app.include_router(enrollment.router)
app.include_router(recognition.router)


@app.get("/")
def index(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={"page": "index", "title": "AutoDoor Test Remote", "lan_ip": _lan_ip()},
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("test.web_remote.app:app", host="0.0.0.0", port=8000)
