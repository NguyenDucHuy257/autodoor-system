"""
Shared runtime for the LAN web test tool.

TEST ONLY - lives in test/web_remote.

Owns:
  * the single CameraManager
  * the AI components (imported from app/, never re-implemented)
  * ONE worker thread that grabs frames, runs the active mode session,
    and keeps the latest JPEG for the MJPEG stream
  * mode management: IDLE / CALIBRATION / ENROLLMENT / RECOGNITION
"""

import logging
import threading
import time
from pathlib import Path

import cv2
import numpy as np
from dotenv import dotenv_values

import app.config as app_config
import app.services.enrollment_service as enrollment_service_module

from app.recognition.yunet_detector import YuNetDetector
from app.recognition.sface_embedder import SFaceEmbedder
from app.recognition.face_matcher import FaceMatcher
from app.recognition.face_quality import FaceQualityGate

from app.config import (
    FACE_MIN_CONFIDENCE,
    FACE_MIN_RATIO,
    FACE_MIN_SHARPNESS,
    FACE_MIN_BRIGHTNESS,
    FACE_MAX_BRIGHTNESS,
)

from test.web_remote.camera_manager import CameraManager


log = logging.getLogger("web_remote")


# ----------------------------------------------------------------------
# paths / constants
# ----------------------------------------------------------------------

ROOT_DIR = Path(__file__).resolve().parents[2]

MODELS_DIR = ROOT_DIR / "models"
YUNET_MODEL_PATH = MODELS_DIR / "face_detection_yunet_2023mar.onnx"
SFACE_MODEL_PATH = MODELS_DIR / "face_recognition_sface_2021dec.onnx"

ENV_PATH = ROOT_DIR / ".env"

# same value used by test/test_realtime.py and detect_realtime.py
MATCH_THRESHOLD = 0.4

PREVIEW_MAX_WIDTH = 960
JPEG_QUALITY = 70
MAX_LOOP_FPS = 30

THRESHOLD_KEYS = (
    "YAW_THRESHOLD",
    "DOWN_THRESHOLD",
    "FAR_FACE_RATIO",
    "FACE_MIN_CONFIDENCE",
    "FACE_MIN_RATIO",
    "FACE_MIN_SHARPNESS",
    "FACE_MIN_BRIGHTNESS",
    "FACE_MAX_BRIGHTNESS",
)

MODE_IDLE = "IDLE"
MODE_CALIBRATION = "CALIBRATION"
MODE_ENROLLMENT = "ENROLLMENT"
MODE_RECOGNITION = "RECOGNITION"


class BusyError(RuntimeError):
    pass


def quality_breakdown(gate: FaceQualityGate, quality) -> list[dict]:
    """
    Per-condition view of FaceQualityGate.evaluate() (same strict comparisons),
    so the UI can show WHICH metric fails and against WHICH loaded threshold.
    Does not change the gate's decision; quality.passed stays authoritative.
    """
    if quality.face_ratio == 0 and quality.confidence == 0:
        return [{
            "metric": "face_roi",
            "value": None,
            "op": "",
            "threshold": None,
            "ok": False,
            "note": "face crop invalid: less than 70% of the box is inside the frame",
        }]

    checks = [
        ("confidence", quality.confidence, ">", gate.min_confidence),
        ("face_ratio", quality.face_ratio, ">", gate.min_face_ratio),
        ("sharpness", quality.sharpness, ">", gate.min_sharpness),
        ("brightness_min", quality.brightness, ">", gate.min_brightness),
        ("brightness_max", quality.brightness, "<", gate.max_brightness),
    ]
    result = []
    for metric, value, op, threshold in checks:
        ok = value > threshold if op == ">" else value < threshold
        result.append({
            "metric": metric,
            "value": round(float(value), 4),
            "op": op,
            "threshold": round(float(threshold), 4),
            "ok": bool(ok),
        })
    return result


# ----------------------------------------------------------------------
# AI components (reused from app/)
# ----------------------------------------------------------------------

class AIComponents:

    def __init__(self):
        self.detector = YuNetDetector(
            model_path=str(YUNET_MODEL_PATH),
        )

        self.embedder = SFaceEmbedder(
            model_path=str(SFACE_MODEL_PATH),
        )

        self.matcher = FaceMatcher(
            match_threshold=MATCH_THRESHOLD,
        )

        # production thresholds from .env (same as the test scripts)
        self.quality_gate = FaceQualityGate(
            min_confidence=FACE_MIN_CONFIDENCE,
            min_face_ratio=FACE_MIN_RATIO,
            min_sharpness=FACE_MIN_SHARPNESS,
            min_brightness=FACE_MIN_BRIGHTNESS,
            max_brightness=FACE_MAX_BRIGHTNESS,
        )

        # permissive gate for calibration: metrics only, "passed" is ignored
        # (same trick as calibrate_camera.py)
        self.calibration_gate = FaceQualityGate(
            min_confidence=0.0,
            min_face_ratio=0.0,
            min_sharpness=0.0,
            min_brightness=0.0,
            max_brightness=255.0,
        )


# ----------------------------------------------------------------------
# session base
# ----------------------------------------------------------------------

class BaseSession:
    """
    One running mode. process() is only ever called from the worker thread.
    status() / stop() may be called from API threads.
    """

    mode = MODE_IDLE
    locks_camera = False

    def __init__(self):
        self.finished = False
        self.error: str | None = None
        self.lock = threading.Lock()

    def process(self, frame: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def status(self) -> dict:
        raise NotImplementedError

    def stop(self) -> None:
        self.finished = True

    def fail(self, message: str) -> None:
        """Called by the worker when process() raised. Subclasses clean up here."""
        self.error = message
        self.finished = True


# ----------------------------------------------------------------------
# runtime
# ----------------------------------------------------------------------

class Runtime:

    def __init__(self):
        self.camera = CameraManager()
        self.ai: AIComponents | None = None

        self._mode_lock = threading.Lock()
        self._session: BaseSession | None = None
        self._last_sessions: dict[str, BaseSession] = {}

        self._frame_cond = threading.Condition()
        self._jpeg: bytes | None = None
        self._frame_id = 0

        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

        self.fps = 0.0
        self.frame_size: tuple[int, int] | None = None
        self.started_at = time.time()

        # hot reload of .env thresholds (no restart needed)
        self._reload_lock = threading.Lock()
        self._env_mtime_seen: float | None = None
        self.thresholds_loaded_at = time.time()
        self.thresholds_reload_error: str | None = None

    # ------------------------------------------------------------------
    # lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        if self.ai is None:
            log.info("Loading AI models from %s", MODELS_DIR)
            self.ai = AIComponents()

        try:
            self._env_mtime_seen = ENV_PATH.stat().st_mtime
        except OSError:
            self._env_mtime_seen = None

        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._loop,
            name="web-remote-worker",
            daemon=True,
        )
        self._thread.start()

    def shutdown(self) -> None:
        self._stop_event.set()

        with self._mode_lock:
            session = self._session
            self._session = None

        if session is not None:
            try:
                session.stop()
            except Exception:
                log.exception("session stop failed")

        if self._thread is not None:
            self._thread.join(timeout=3)
            self._thread = None

        self.camera.stop()

        # wake any MJPEG client so its generator can exit
        with self._frame_cond:
            self._frame_cond.notify_all()

    @property
    def shutting_down(self) -> bool:
        return self._stop_event.is_set()

    # ------------------------------------------------------------------
    # mode management
    # ------------------------------------------------------------------

    def current_session(self) -> BaseSession | None:
        with self._mode_lock:
            session = self._session
            if session is not None and session.finished:
                self._session = None
                return None
            return session

    def current_mode(self) -> str:
        session = self.current_session()
        return session.mode if session is not None else MODE_IDLE

    def camera_locked(self) -> bool:
        session = self.current_session()
        return session is not None and session.locks_camera

    def start_session(self, session: BaseSession) -> None:
        with self._mode_lock:
            active = self._session
            if active is not None and not active.finished:
                raise BusyError(f"System busy: {active.mode}")
            self._session = session
            self._last_sessions[session.mode] = session

    def stop_session(self, mode: str) -> bool:
        with self._mode_lock:
            session = self._session
            if session is None or session.mode != mode:
                return False
            self._session = None

        session.stop()
        return True

    def session_for(self, mode: str) -> BaseSession | None:
        """Active session of this mode, or the last finished one (for result display)."""
        session = self.current_session()
        if session is not None and session.mode == mode:
            return session
        return self._last_sessions.get(mode)

    # ------------------------------------------------------------------
    # camera control (checks the mode lock)
    # ------------------------------------------------------------------

    def set_camera_laptop(self) -> tuple[bool, str]:
        if self.camera_locked():
            return False, f"Camera locked: {self.current_mode()} in progress"
        return self.camera.use_laptop()

    def set_camera_channel(self, channel: int) -> tuple[bool, str]:
        if self.camera_locked():
            return False, f"Camera locked: {self.current_mode()} in progress"
        return self.camera.use_rtsp(channel)

    # ------------------------------------------------------------------
    # thresholds: hot reload from .env
    # ------------------------------------------------------------------

    def loaded_thresholds(self) -> dict | None:
        gate = self.ai.quality_gate if self.ai else None
        if gate is None:
            return None
        return {
            "YAW_THRESHOLD": enrollment_service_module.YAW_THRESHOLD,
            "DOWN_THRESHOLD": enrollment_service_module.DOWN_THRESHOLD,
            "FAR_FACE_RATIO": enrollment_service_module.FAR_FACE_RATIO,
            "FACE_MIN_CONFIDENCE": gate.min_confidence,
            "FACE_MIN_RATIO": gate.min_face_ratio,
            "FACE_MIN_SHARPNESS": gate.min_sharpness,
            "FACE_MIN_BRIGHTNESS": gate.min_brightness,
            "FACE_MAX_BRIGHTNESS": gate.max_brightness,
        }

    def reload_thresholds(self) -> dict:
        """
        Re-read the 8 calibration keys from .env and apply them IN PLACE:
          * FaceQualityGate instance used by enrollment + recognition
          * YAW/DOWN/FAR globals that EnrollmentService._classify_pose reads
            (module globals of app.services.enrollment_service and app.config)
        No file in app/ is modified; only the running process state.
        """
        with self._reload_lock:
            values = dotenv_values(ENV_PATH)
            parsed = {}
            for key in THRESHOLD_KEYS:
                raw = values.get(key)
                if raw is None or str(raw).strip() == "":
                    raise ValueError("%s missing in %s" % (key, ENV_PATH))
                parsed[key] = float(str(raw).strip().strip("'\""))

            gate = self.ai.quality_gate
            gate.min_confidence = parsed["FACE_MIN_CONFIDENCE"]
            gate.min_face_ratio = parsed["FACE_MIN_RATIO"]
            gate.min_sharpness = parsed["FACE_MIN_SHARPNESS"]
            gate.min_brightness = parsed["FACE_MIN_BRIGHTNESS"]
            gate.max_brightness = parsed["FACE_MAX_BRIGHTNESS"]

            for module in (enrollment_service_module, app_config):
                module.YAW_THRESHOLD = parsed["YAW_THRESHOLD"]
                module.DOWN_THRESHOLD = parsed["DOWN_THRESHOLD"]
                module.FAR_FACE_RATIO = parsed["FAR_FACE_RATIO"]
            for key in ("FACE_MIN_CONFIDENCE", "FACE_MIN_RATIO", "FACE_MIN_SHARPNESS",
                        "FACE_MIN_BRIGHTNESS", "FACE_MAX_BRIGHTNESS"):
                setattr(app_config, key, parsed[key])

            try:
                self._env_mtime_seen = ENV_PATH.stat().st_mtime
            except OSError:
                self._env_mtime_seen = None
            self.thresholds_loaded_at = time.time()
            self.thresholds_reload_error = None
            log.info("Thresholds reloaded from .env: %s", parsed)
            return parsed

    def check_env_changed(self) -> None:
        """Auto reload when .env was modified by anything (SAVE button, editor, script)."""
        try:
            mtime = ENV_PATH.stat().st_mtime
        except OSError:
            return
        if self._env_mtime_seen is None:
            self._env_mtime_seen = mtime
            return
        if mtime == self._env_mtime_seen:
            return
        try:
            self.reload_thresholds()
        except Exception as error:
            self.thresholds_reload_error = str(error)
            self._env_mtime_seen = mtime
            log.warning("Threshold reload failed: %s", error)

    def system_status(self) -> dict:
        if self.ai is not None:
            self.check_env_changed()
        cam = self.camera.describe()
        return {
            "mode": self.current_mode(),
            "source": cam["source"],
            "channel": cam["channel"],
            "camera_error": cam["error"],
            "camera_locked": self.camera_locked(),
            "fps": round(self.fps, 1),
            "frame_size": list(self.frame_size) if self.frame_size else None,
            "server_started_at": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(self.started_at)),
            "thresholds_loaded_at": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(self.thresholds_loaded_at)),
            "thresholds_reload_error": self.thresholds_reload_error,
            "loaded_thresholds": self.loaded_thresholds(),
        }

    # ------------------------------------------------------------------
    # MJPEG support
    # ------------------------------------------------------------------

    def wait_for_frame(self, last_id: int, timeout: float = 1.0) -> tuple[bytes | None, int]:
        with self._frame_cond:
            if self._frame_id == last_id:
                self._frame_cond.wait(timeout=timeout)
            if self._frame_id == last_id or self._jpeg is None:
                return None, last_id
            return self._jpeg, self._frame_id

    def _publish(self, display: np.ndarray) -> None:
        h, w = display.shape[:2]
        if w > PREVIEW_MAX_WIDTH:
            scale = PREVIEW_MAX_WIDTH / w
            display = cv2.resize(
                display,
                (PREVIEW_MAX_WIDTH, int(h * scale)),
                interpolation=cv2.INTER_AREA,
            )

        ok, buf = cv2.imencode(
            ".jpg",
            display,
            [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY],
        )
        if not ok:
            return

        with self._frame_cond:
            self._jpeg = buf.tobytes()
            self._frame_id += 1
            self._frame_cond.notify_all()

    def _placeholder(self) -> np.ndarray:
        img = np.zeros((270, 480, 3), dtype=np.uint8)
        cam = self.camera.describe()
        label = cam["source"]
        if cam["channel"] is not None:
            label = "%s CH%s" % (label, cam["channel"])
        cv2.putText(img, "NO SIGNAL", (140, 120), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 2)
        cv2.putText(img, label, (150, 170), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)
        return img

    # ------------------------------------------------------------------
    # worker loop
    # ------------------------------------------------------------------

    def _loop(self) -> None:
        min_interval = 1.0 / MAX_LOOP_FPS
        fps_window_start = time.perf_counter()
        fps_frames = 0
        last_placeholder = 0.0

        while not self._stop_event.is_set():
            loop_started = time.perf_counter()

            frame = None
            try:
                frame = self.camera.get_frame()
            except Exception:
                log.exception("camera read failed")

            if frame is None:
                now = time.perf_counter()
                if now - last_placeholder > 0.5:
                    self._publish(self._placeholder())
                    last_placeholder = now
                time.sleep(0.03)
                continue

            self.frame_size = (int(frame.shape[1]), int(frame.shape[0]))

            session = self.current_session()
            display = frame

            if session is not None:
                try:
                    display = session.process(frame)
                except Exception as error:
                    log.exception("session.process failed")
                    try:
                        session.fail("%s: %s" % (type(error).__name__, error))
                    except Exception:
                        log.exception("session.fail failed")
                        session.finished = True

                if session.finished:
                    with self._mode_lock:
                        if self._session is session:
                            self._session = None

            if display is None:
                display = frame

            self._publish(display)

            # fps
            fps_frames += 1
            now = time.perf_counter()
            if now - fps_window_start >= 1.0:
                self.fps = fps_frames / (now - fps_window_start)
                fps_frames = 0
                fps_window_start = now

            # pacing (RTSP get_latest_frame returns immediately)
            elapsed = time.perf_counter() - loop_started
            if elapsed < min_interval:
                time.sleep(min_interval - elapsed)


runtime = Runtime()
