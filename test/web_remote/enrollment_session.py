"""
Web version of test/test_enrollment_service.py (TEST ONLY).

Uses EnrollmentService directly:
  start_enrollment() / process_frame() / get_next_required_pose() / finish_enrollment()

Cooldown: after each captured pose, process_frame() is NOT called for
POSE_COOLDOWN_SECONDS so the user has time to turn their head.
"""

import threading
import time

import cv2

from app.database.database import SessionLocal
from app.database.enrollment_repository import EnrollmentRepository
from app.services.enrollment_service import EnrollmentService

from test.web_remote.runtime import (
    BaseSession,
    MODE_ENROLLMENT,
    runtime,
    quality_breakdown,
)


POSE_COOLDOWN_SECONDS = 1.5

STATE_CAPTURING = "CAPTURING"
STATE_COOLDOWN = "COOLDOWN"
STATE_SUCCESS = "SUCCESS"
STATE_CANCELLED = "CANCELLED"
STATE_ERROR = "ERROR"


class EnrollmentSession(BaseSession):

    mode = MODE_ENROLLMENT
    locks_camera = True

    def __init__(self, full_name: str, room_number: str | None):
        super().__init__()

        self.full_name = full_name.strip()
        self.room_number = room_number.strip() if room_number and room_number.strip() else None

        self.camera_info = runtime.camera.describe()

        self.db_session = SessionLocal()
        self.repository = EnrollmentRepository(session=self.db_session)

        ai = runtime.ai
        self.service = EnrollmentService(
            repository=self.repository,
            detector=ai.detector,
            embedder=ai.embedder,
            quality_gate=ai.quality_gate,
        )

        # raises ValueError if the name is empty
        self.service.start_enrollment(
            full_name=self.full_name,
            room_number=self.room_number,
        )

        self.required_slots = list(EnrollmentService.REQUIRED_SLOTS)
        self.state = STATE_CAPTURING
        self.next_capture_allowed_at = 0.0
        self.last_captured_pose: str | None = None
        self.last_captured_at: float | None = None

        self.result: dict | None = None
        self.debug: dict = {}

        self._db_lock = threading.Lock()

    # ------------------------------------------------------------------
    # API threads
    # ------------------------------------------------------------------

    def stop(self) -> None:
        with self.lock:
            if self.state not in (STATE_SUCCESS, STATE_ERROR):
                self.state = STATE_CANCELLED
            self.finished = True
        self._close_db()

    def fail(self, message: str) -> None:
        with self.lock:
            self.error = message
            self.state = STATE_ERROR
            self.finished = True
        self._close_db()

    def _close_db(self) -> None:
        with self._db_lock:
            if self.db_session is not None:
                try:
                    self.db_session.close()
                finally:
                    self.db_session = None

    # ------------------------------------------------------------------
    # worker thread
    # ------------------------------------------------------------------

    def process(self, frame):
        if self.finished:
            return frame

        ai = runtime.ai
        display = frame.copy()
        frame_height, frame_width = frame.shape[:2]

        # ---- live feedback (detector + quality gate, same instances as the service)
        faces = ai.detector.detect(frame)
        debug = {
            "face_count": len(faces),
            "quality": None,
            "detected_pose": None,
        }

        if len(faces) == 1:
            face = faces[0]
            quality = ai.quality_gate.evaluate(frame=frame, face=face)
            debug["quality"] = {
                "passed": bool(quality.passed),
                "confidence": round(float(quality.confidence), 4),
                "face_ratio": round(float(quality.face_ratio), 4),
                "sharpness": round(float(quality.sharpness), 2),
                "brightness": round(float(quality.brightness), 2),
                "checks": quality_breakdown(ai.quality_gate, quality),
            }
            # reuse the service's own classifier for debugging (no duplicate logic)
            try:
                debug["detected_pose"] = self.service._classify_pose(
                    face=face,
                    frame_width=frame_width,
                )
            except Exception:
                debug["detected_pose"] = None

            color = (0, 255, 0) if quality.passed else (0, 0, 255)
            cv2.rectangle(
                display,
                (face.x, face.y),
                (face.x + face.width, face.y + face.height),
                color,
                2,
            )
        else:
            for face in faces:
                cv2.rectangle(
                    display,
                    (face.x, face.y),
                    (face.x + face.width, face.y + face.height),
                    (0, 0, 255),
                    2,
                )

        self.debug = debug

        now = time.monotonic()

        with self.lock:
            if self.finished:
                return display

            if now < self.next_capture_allowed_at:
                # cooldown: camera stays live, no process_frame
                self.state = STATE_COOLDOWN
                return display

            self.state = STATE_CAPTURING

        before_count = len(self.service.collected_frames)
        result = self.service.process_frame(frame)
        after_count = len(self.service.collected_frames)

        if result is None:
            return display

        with self.lock:
            if after_count > before_count:
                self.last_captured_pose = result["collected"][-1]
                self.last_captured_at = time.time()

                if not result["complete"]:
                    self.next_capture_allowed_at = time.monotonic() + POSE_COOLDOWN_SECONDS
                    self.state = STATE_COOLDOWN

        if result["complete"]:
            self._finish()

        return display

    def _finish(self) -> None:
        try:
            with self._db_lock:
                if self.db_session is None:
                    raise RuntimeError("Enrollment cancelled before finish")
                member = self.service.finish_enrollment()

                # copy ORM data BEFORE closing the session (avoid DetachedInstanceError)
                data = {
                    "id": member.id,
                    "code": member.code,
                    "name": member.full_name,
                    "room": member.room_number,
                }

            with self.lock:
                self.result = data
                self.state = STATE_SUCCESS
                self.finished = True

        except Exception as error:
            with self.lock:
                self.error = "%s: %s" % (type(error).__name__, error)
                self.state = STATE_ERROR
                self.finished = True
        finally:
            self._close_db()

    # ------------------------------------------------------------------
    # status
    # ------------------------------------------------------------------

    def status(self) -> dict:
        with self.lock:
            collected = list(self.service.collected_frames.keys())
            next_pose = self.service.get_next_required_pose() if self.state != STATE_SUCCESS else None

            cooldown = 0.0
            if self.state == STATE_COOLDOWN:
                cooldown = max(0.0, self.next_capture_allowed_at - time.monotonic())

            return {
                "active": not self.finished,
                "state": self.state,
                "name": self.full_name,
                "room": self.room_number,
                "camera": self.camera_info,
                "next_pose": next_pose,
                "collected": collected,
                "required": self.required_slots,
                "progress": len(collected),
                "total": len(self.required_slots),
                "cooldown": round(cooldown, 1),
                "cooldown_seconds": POSE_COOLDOWN_SECONDS,
                "last_captured_pose": self.last_captured_pose,
                "debug": self.debug,
                "result": self.result,
                "error": self.error,
            }
