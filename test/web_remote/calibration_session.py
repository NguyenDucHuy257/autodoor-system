"""
Web version of calibrate_camera.py (TEST ONLY).

Reuses YuNetDetector + FaceQualityGate from app/ (via runtime.ai).
Math comes from calibration_adapter (mirror of calibrate_camera.py).

Phases per stage:  WAIT -> (user READY) -> PREPARE (5s countdown) -> CAPTURE (30 samples)
Then next stage waits for READY again. After the last stage: DONE.
Nothing is written to .env until the user calls save().
"""

import threading
import time

import cv2

from test.web_remote.runtime import (
    BaseSession,
    MODE_CALIBRATION,
    ENV_PATH,
    runtime,
)
from test.web_remote.calibration_adapter import (
    STAGES,
    STAGE_GUIDES,
    PREPARE_SECONDS,
    SAMPLES_PER_STAGE,
    SAMPLE_INTERVAL_SECONDS,
    extract_pose_metrics,
    calculate_medians,
    calculate_thresholds,
    save_thresholds_to_env,
    relax_thresholds,
    validate_threshold_values,
)


PHASE_WAIT = "WAIT"
PHASE_PREPARE = "PREPARE"
PHASE_CAPTURE = "CAPTURE"
PHASE_DONE = "DONE"
PHASE_CANCELLED = "CANCELLED"
PHASE_ERROR = "ERROR"


class CalibrationSession(BaseSession):

    mode = MODE_CALIBRATION
    # thresholds are camera specific: no camera switch while calibrating
    locks_camera = True

    def __init__(self):
        super().__init__()

        self.camera_info = runtime.camera.describe()

        self.stage_index = 0
        self.phase = PHASE_WAIT
        self.phase_started_at: float | None = None
        self.last_sample_at = 0.0

        self.stage_samples: list[dict] = []
        self.all_samples: dict[str, list[dict]] = {stage: [] for stage in STAGES}

        self.medians: dict | None = None
        self.thresholds: dict | None = None            # strict = calibrate_camera.py formula
        self.thresholds_relaxed: dict | None = None    # web-only relaxed profile

        self.saved = False
        self.saved_keys: list[str] = []
        self.saved_values: dict | None = None

        self.face_count = 0
        self.live_metrics: dict | None = None

        self._save_lock = threading.Lock()

    # ------------------------------------------------------------------
    # user actions (API threads)
    # ------------------------------------------------------------------

    def ready(self) -> tuple[bool, str]:
        with self.lock:
            if self.finished:
                return False, "Calibration is not running"
            if self.phase != PHASE_WAIT:
                return False, "Not waiting: current phase is %s" % self.phase
            self.phase = PHASE_PREPARE
            self.phase_started_at = time.monotonic()
            return True, "Prepare %s" % self.current_stage

    def stop(self) -> None:
        with self.lock:
            if self.phase not in (PHASE_DONE, PHASE_ERROR):
                self.phase = PHASE_CANCELLED
            self.finished = True

    def fail(self, message: str) -> None:
        with self.lock:
            self.error = message
            self.phase = PHASE_ERROR
            self.finished = True

    def save(self, values: dict | None = None) -> tuple[bool, str]:
        """
        Write thresholds to .env. `values` = user-edited values from the UI
        (defaults to the relaxed profile). Then hot-reload into the runtime.
        """
        with self._save_lock:
            if self.phase != PHASE_DONE or self.thresholds is None:
                return False, "Calibration not complete"
            try:
                to_write = validate_threshold_values(values) if values else dict(self.thresholds_relaxed)
                written = save_thresholds_to_env(ENV_PATH, to_write)
            except Exception as error:
                return False, "Save failed: %s" % error

            self.saved = True
            self.saved_keys = written
            self.saved_values = {k: to_write[k] for k in written}

            try:
                runtime.reload_thresholds()
            except Exception as error:
                return True, "Saved %d keys, but reload failed: %s" % (len(written), error)

            return True, "Saved %d keys to .env and reloaded (no restart needed)" % len(written)

    # ------------------------------------------------------------------
    # worker thread
    # ------------------------------------------------------------------

    @property
    def current_stage(self) -> str:
        if self.stage_index >= len(STAGES):
            return STAGES[-1]
        return STAGES[self.stage_index]

    def process(self, frame):
        if self.finished:
            return frame

        ai = runtime.ai
        display = frame.copy()

        faces = ai.detector.detect(frame)
        self.face_count = len(faces)

        face = faces[0] if len(faces) == 1 else None

        # preview: green box + landmarks (as calibrate_camera.py)
        if face is not None:
            cv2.rectangle(
                display,
                (face.x, face.y),
                (face.x + face.width, face.y + face.height),
                (0, 255, 0),
                2,
            )
            for point in face.landmarks:
                cv2.circle(display, (int(point[0]), int(point[1])), 3, (0, 0, 255), -1)

            metrics = extract_pose_metrics(face)
            quality = ai.calibration_gate.evaluate(frame=frame, face=face)
            if metrics is not None:
                self.live_metrics = {
                    "yaw": round(metrics["yaw"], 4),
                    "pitch": round(metrics["pitch"], 4),
                    "face_ratio": round(quality.face_ratio, 4),
                    "confidence": round(quality.confidence, 4),
                    "sharpness": round(quality.sharpness, 2),
                    "brightness": round(quality.brightness, 2),
                }
        else:
            self.live_metrics = None

        with self.lock:
            if self.phase == PHASE_PREPARE:
                elapsed = time.monotonic() - self.phase_started_at
                if elapsed >= PREPARE_SECONDS:
                    self.phase = PHASE_CAPTURE
                    self.stage_samples = []
                    self.last_sample_at = 0.0

            elif self.phase == PHASE_CAPTURE:
                if face is not None:
                    now = time.monotonic()
                    if now - self.last_sample_at >= SAMPLE_INTERVAL_SECONDS:
                        metrics = extract_pose_metrics(face)
                        quality = ai.calibration_gate.evaluate(frame=frame, face=face)
                        if metrics is not None and quality.face_ratio > 0:
                            self.stage_samples.append({
                                "yaw": metrics["yaw"],
                                "pitch": metrics["pitch"],
                                "face_ratio": quality.face_ratio,
                                "confidence": quality.confidence,
                                "sharpness": quality.sharpness,
                                "brightness": quality.brightness,
                            })
                            self.last_sample_at = now

                if len(self.stage_samples) >= SAMPLES_PER_STAGE:
                    self.all_samples[self.current_stage] = list(self.stage_samples)
                    self.stage_index += 1
                    self.stage_samples = []
                    self.phase_started_at = None

                    if self.stage_index >= len(STAGES):
                        self._complete()
                    else:
                        self.phase = PHASE_WAIT

        return display

    def _complete(self) -> None:
        # called with self.lock held
        try:
            self.medians = calculate_medians(self.all_samples)
            self.thresholds = calculate_thresholds(
                medians=self.medians,
                all_samples=self.all_samples,
            )
            self.thresholds_relaxed = relax_thresholds(self.thresholds)
            self.phase = PHASE_DONE
        except Exception as error:
            self.error = str(error)
            self.phase = PHASE_ERROR
        finally:
            self.finished = True

    # ------------------------------------------------------------------
    # status (API threads)
    # ------------------------------------------------------------------

    def status(self) -> dict:
        with self.lock:
            countdown = None
            if self.phase == PHASE_PREPARE and self.phase_started_at is not None:
                countdown = max(0.0, PREPARE_SECONDS - (time.monotonic() - self.phase_started_at))

            stage = self.current_stage if self.stage_index < len(STAGES) else None

            return {
                "active": not self.finished,
                "phase": self.phase,
                "stage": stage,
                "stage_index": min(self.stage_index, len(STAGES)),
                "stage_total": len(STAGES),
                "stages": list(STAGES),
                "guide": STAGE_GUIDES.get(stage, "") if stage else "",
                "countdown": round(countdown, 1) if countdown is not None else None,
                "samples": len(self.stage_samples),
                "samples_per_stage": SAMPLES_PER_STAGE,
                "face_count": self.face_count,
                "live_metrics": self.live_metrics,
                "medians": self.medians,
                "thresholds": self.thresholds,
                "thresholds_relaxed": self.thresholds_relaxed,
                "saved": self.saved,
                "saved_keys": self.saved_keys,
                "saved_values": self.saved_values,
                "env_path": str(ENV_PATH),
                "camera": self.camera_info,
                "error": self.error,
            }
