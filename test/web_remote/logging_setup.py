"""File-only logs for the standalone test server, including native AI stderr."""
from datetime import datetime
import logging
import os
from pathlib import Path
import sys

LOG_PATH = Path(__file__).resolve().parents[2] / "logs" / ("web_remote_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f") + ".log")
_handler = None
_native_redirected = False


def configure_logging():
    global _handler
    if _handler is None:
        LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        _handler = logging.FileHandler(LOG_PATH, encoding="utf-8")
        _handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    for name in (None, "uvicorn", "uvicorn.error", "uvicorn.access", "web_remote"):
        logger = logging.getLogger(name)
        for handler in list(logger.handlers):
            if isinstance(handler, logging.StreamHandler) and not isinstance(handler, logging.FileHandler):
                logger.removeHandler(handler)
        if name is None or not logger.propagate:
            if _handler not in logger.handlers:
                logger.addHandler(_handler)
        logger.setLevel(logging.INFO)


def redirect_native_output():
    """Called only at server startup; import/offline tests keep their stdout."""
    global _native_redirected
    if _native_redirected:
        return
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.flush()
        except (OSError, ValueError):
            pass
    descriptor = os.open(LOG_PATH, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o666)
    try:
        os.dup2(descriptor, 1)
        os.dup2(descriptor, 2)
    finally:
        os.close(descriptor)
    _native_redirected = True
