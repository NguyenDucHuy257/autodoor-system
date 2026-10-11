"""
Shared runtime for the LAN web test tool.

TEST ONLY - lives in test/web_remote.

Owns:
  * the single CameraManager
  * the AI components (imported from app/, never re-implemented)
  * separate capture, preview, detection, recognition and recording workers
  * mode management: IDLE / CALIBRATION / ENROLLMENT / RECOGNITION
"""

import logging
import threading
import time
from pathlib import Path

import cv2
import numpy as np
from app.performance import performance
from dotenv import dotenv_values

import app.config as app_config
import app.services.enrollment_service as enrollment_service_module

from test.web_remote.scrfd_adapter import WebSCRFDDetector
from test.web_remote.adaface_engine import AdaEngine, DemoGallery
from test.web_remote.recorder import EventRecorder

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
SCRFD_MODEL_PATH = MODELS_DIR / "scrfd" / "det_10g.onnx"

ENV_PATH = ROOT_DIR / ".env"

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
        self.detector = WebSCRFDDetector(
            model_path=str(SCRFD_MODEL_PATH),
        )

        self.engine = None
        self.gallery = None
        self.model_error = None

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
        # Measure the shared instances, including calls made inside EnrollmentService.
        self.detector.detect = performance.wrap("detection_ms", self.detector.detect)
        self.quality_gate.evaluate = performance.wrap("quality_ms", self.quality_gate.evaluate)
        self.calibration_gate.evaluate = performance.wrap("quality_ms", self.calibration_gate.evaluate)


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

        self._mode_lock = threading.RLock()
        self._session: BaseSession | None = None
        self._last_sessions: dict[str, BaseSession] = {}

        self._frame_cond = threading.Condition()
        self._jpeg: bytes | None = None
        self._frame_id = 0
        self._ai_jpeg = None
        self._ai_faces = []
        self._ai_frame_id = 0
        self._ai_timestamp = 0.0
        self._jpeg_generation = self._ai_generation = 0
        self._jobs = threading.Condition()
        self._detect_job = None
        self._reco_job = None
        self._workers = []
        self.recorder = EventRecorder()

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
        performance.start(
            ROOT_DIR / "logs", opencv_version=cv2.__version__,
            opencv_threads=cv2.getNumThreads(),
            cuda_dnn_targets=list(cv2.dnn.getAvailableTargets(cv2.dnn.DNN_BACKEND_CUDA)),
            loop_fps_limit=MAX_LOOP_FPS, preview_max_width=PREVIEW_MAX_WIDTH,
            jpeg_quality=JPEG_QUALITY, detector="SCRFD-10G", embedder="AdaFace",
        )
        log.info("Performance log: %s", performance.log_path)
        performance.update(mode=MODE_IDLE, source=self.camera.source, channel=self.camera.channel)
        if self.ai is None:
            log.info("Loading AI models from %s", MODELS_DIR)
            try:
                with performance.measure("model_load_ms"):
                    self.ai = AIComponents()
            except Exception:
                performance.event("startup_failed", stage="model_load")
                performance.stop()
                raise

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
        self.camera.start()
        self.recorder.start()
        for name, target in (("ui-detector", self._detect_loop), ("ui-recognizer", self._recognize_loop), ("ui-model-loader", self._load_engine)):
            thread = threading.Thread(target=target, name=name, daemon=True)
            self._workers.append(thread)
            thread.start()

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

        with self._jobs:
            self._jobs.notify_all()
        self.camera.shutdown()
        for worker in self._workers:
            worker.join(timeout=5)
        self.recorder.stop()

        # wake any MJPEG client so its generator can exit
        with self._frame_cond:
            self._frame_cond.notify_all()
        performance.stop()

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
            self._clear_ai()
            self._session = session
            self._last_sessions[session.mode] = session
        performance.update(mode=session.mode)
        performance.event("session_start", mode=session.mode)

    def stop_session(self, mode: str) -> bool:
        with self._mode_lock:
            session = self._session
            if session is None or session.mode != mode:
                return False
            self._session = None
            self._clear_ai()
            self.recorder.submit(None, None, None)

        session.stop()
        performance.event("session_stop", mode=mode)
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

    def _switch(self, action):
        with self._mode_lock:
            if self.camera_locked():
                return False, f"Camera locked: {self.current_mode()} in progress"
            result = action()
            with self._frame_cond:
                self._jpeg = None
                self._frame_id += 1
                self._frame_cond.notify_all()
            self._clear_ai()
            self.recorder.submit(None, None, None)
            return result

    def set_camera_laptop(self):
        return self._switch(self.camera.use_laptop)

    def set_camera_channel(self, channel):
        return self._switch(lambda: self.camera.use_rtsp(channel))

    def stop_camera(self):
        return self._switch(self.camera.stop)

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
            "generation": cam["generation"],
            "switching": cam["switching"],
            "model_ready": bool(self.ai and self.ai.engine),
            "model_error": self.ai.model_error if self.ai else None,
            "detector": "SCRFD-10G",
            "detector_providers": self.ai.detector.providers if self.ai else [],
            "model_providers": self.ai.engine.providers if self.ai and self.ai.engine else [],
            "recording": str(self.recorder.path) if self.recorder.writer else None,
            "recording_error": self.recorder.error,
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

    def wait_for_frame(self, last_id, timeout=1.0, kind="preview"):
        with self._frame_cond:
            def current():
                return (self._ai_jpeg, self._ai_frame_id) if kind == "ai" else (self._jpeg, self._frame_id)
            jpeg, frame_id = current()
            if frame_id == last_id:
                self._frame_cond.wait(timeout)
                jpeg, frame_id = current()
            return (jpeg, frame_id) if frame_id != last_id else (None, last_id)

    def wait_for_delivery(self, last_id, timeout=0.5, kind="preview"):
        with self._frame_cond:
            jpeg, frame_id = self.wait_for_frame(last_id, timeout, kind)
            generation = self._ai_generation if kind == "ai" else self._jpeg_generation
            if generation != self.camera.current_generation() or (kind == "ai" and time.monotonic()-self._ai_timestamp > 1):
                jpeg = None
            return jpeg, frame_id, generation, (self._ai_faces if kind == "ai" and jpeg is not None else [])

    def _clear_ai(self):
        with self._jobs:
            self._detect_job = self._reco_job = None
        with self._frame_cond:
            self._ai_jpeg = None
            self._ai_faces = []
            self._ai_frame_id += 1
            self._frame_cond.notify_all()

    def _publish(self, display, kind="preview", packet=None, session=None):
        started = time.perf_counter()
        h, w = display.shape[:2]
        max_width = 320 if kind == "preview" and self.current_mode() == MODE_RECOGNITION else PREVIEW_MAX_WIDTH
        if w > max_width:
            display = cv2.resize(display, (max_width, int(h*max_width/w)), interpolation=cv2.INTER_AREA)
        with performance.measure("jpeg_encode_ms"):
            ok, buf = cv2.imencode(".jpg", display, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
        if not ok:
            return
        with self._mode_lock:
            if packet and packet.generation != self.camera.current_generation():
                return
            if kind == "ai" and not self._valid(packet, session, allow_finished=True):
                performance.count("ai_results_dropped")
                return
            with self._frame_cond:
                if kind == "ai":
                    self._ai_generation = packet.generation
                    self._ai_faces = list(getattr(session, "faces", [])) if session.mode == MODE_RECOGNITION else []
                    self._ai_jpeg = buf.tobytes()
                    self._ai_frame_id += 1
                    self._ai_timestamp = packet.timestamp
                else:
                    self._jpeg_generation = packet.generation if packet else self.camera.current_generation()
                    self._jpeg = buf.tobytes()
                    self._frame_id += 1
                self._frame_cond.notify_all()
            if kind == "ai" and session.mode == MODE_RECOGNITION:
                faces = self._ai_faces
                presence = packet.timestamp if faces else session.presence(packet.generation)
                self.recorder.submit(packet, (packet.generation, id(session)), presence, display, faces)
        performance.count("ai_published_frames" if kind == "ai" else "published_frames")
        performance.observe("publish_ms", (time.perf_counter()-started)*1000)

    def _valid(self, packet, session, allow_finished=False):
        return (packet is not None and session is self._session
                and (allow_finished or not session.finished)
                and packet.generation == self.camera.current_generation()
                and time.monotonic()-packet.timestamp <= 1.0)

    def _load_engine(self):
        try:
            engine = AdaEngine()
            self.ai.gallery = DemoGallery(engine.model_id)
            self.ai.engine = engine
        except Exception as error:
            self.ai.model_error = str(error)
            log.exception("AdaFace initialization failed; raw preview remains available")

    def _take_job(self, name):
        with self._jobs:
            self._jobs.wait_for(lambda: getattr(self, name) is not None or self.shutting_down, timeout=0.5)
            job = getattr(self, name)
            setattr(self, name, None)
            return job

    def _detect_loop(self):
        while not self.shutting_down:
            job = self._take_job("_detect_job")
            if job is None:
                continue
            packet, session = job
            if not self._valid(packet, session):
                performance.count("ai_jobs_dropped")
                continue
            try:
                if session.mode == MODE_RECOGNITION:
                    faces = session.detect(packet)
                    if self._valid(packet, session):
                        with self._jobs:
                            if self._reco_job is not None:
                                performance.count("ai_jobs_dropped")
                            self._reco_job = (packet, session, faces)
                            self._jobs.notify_all()
                else:
                    with performance.measure("processing_ms"):
                        display = session.process_packet(packet) if hasattr(session, "process_packet") else session.process(packet.frame.copy())
                    if display is not None:
                        self._publish(display, "ai", packet, session)
            except Exception as error:
                log.exception("Detection/session failed")
                session.fail(str(error))

    def _recognize_loop(self):
        while not self.shutting_down:
            job = self._take_job("_reco_job")
            if job is None:
                continue
            packet, session, faces = job
            if not self._valid(packet, session):
                performance.count("ai_jobs_dropped")
                continue
            try:
                with performance.measure("processing_ms"):
                    display = session.process_faces(packet, faces)
                if display is not None:
                    self._publish(display, "ai", packet, session)
            except Exception as error:
                log.exception("Recognition failed")
                session.fail(str(error))

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

    def _loop(self):
        last_packet = None
        last_placeholder = 0.0
        window, count = time.monotonic(), 0
        while not self.shutting_down:
            packet = self.camera.packet()
            now = time.monotonic()
            if packet is None:
                if now-last_placeholder > 0.5:
                    self._publish(self._placeholder())
                    last_placeholder = now
            elif (packet.generation, packet.frame_id) != last_packet:
                last_packet = (packet.generation, packet.frame_id)
                self.frame_size = (packet.frame.shape[1], packet.frame.shape[0])
                self._publish(packet.frame, packet=packet)
                session = self.current_session()
                with self._mode_lock:
                    if session is self._session and session is not None and not session.finished:
                        with self._jobs:
                            if self._detect_job is not None:
                                performance.count("ai_jobs_dropped")
                            self._detect_job = (packet, session)
                            self._jobs.notify_all()
                performance.observe("frame_age_at_publish_ms", (time.monotonic()-packet.timestamp)*1000)
                count += 1
            with self._frame_cond:
                if self._ai_jpeg is not None and now-self._ai_timestamp > 1.0:
                    self._ai_jpeg = None
                    self._ai_faces = []
                    self._ai_frame_id += 1
                    self._frame_cond.notify_all()
            if now-window >= 1:
                self.fps = count/(now-window)
                window, count = now, 0
            self._stop_event.wait(1/MAX_LOOP_FPS)


runtime = Runtime()
