"""Test-only five-pose AdaFace enrollment into an isolated local gallery."""
import time
import threading
from app.performance import performance
from app.services.enrollment_service import EnrollmentService
from test.web_remote.runtime import BaseSession, MODE_ENROLLMENT, runtime, quality_breakdown
from test.web_remote.overlay import draw_faces

POSE_COOLDOWN_SECONDS = 1.5


class EnrollmentSession(BaseSession):
    mode = MODE_ENROLLMENT
    locks_camera = True

    def __init__(self, full_name, room_number):
        super().__init__()
        if runtime.ai.engine is None:
            raise RuntimeError(runtime.ai.model_error or "AdaFace is loading; try again shortly")
        self.cancelled = threading.Event()
        self.full_name = full_name.strip()
        if not self.full_name:
            raise ValueError("Name required")
        self.room_number = room_number.strip() if room_number else None
        self.camera_info = runtime.camera.describe()
        self.required_slots = list(EnrollmentService.REQUIRED_SLOTS)
        self.classifier = EnrollmentService.__new__(EnrollmentService)
        self.embeddings = {}
        self.state = "CAPTURING"
        self.next_capture_allowed_at = 0.0
        self.last_captured_pose = None
        self.debug = {}
        self.result = None

    def stop(self):
        # Same lock as final save: cancellation cannot claim success after commit.
        self.cancelled.set()
        with self.lock:
            if self.state not in ("SUCCESS", "ERROR"):
                self.state = "CANCELLED"
            self.finished = True

    def fail(self, message):
        with self.lock:
            self.error, self.state, self.finished = message, "ERROR", True

    def process_packet(self, packet):
        self.packet = packet
        return self.process(packet.frame.copy())

    def process(self, frame):
        with self.lock:
            if self.finished or (hasattr(self, "packet") and not runtime._valid(self.packet, self)):
                return frame
            faces = runtime.ai.detector.detect(frame)
            self.debug = dict(face_count=len(faces), quality=None, detected_pose=None)
            if len(faces) != 1:
                return frame.copy()
            face = faces[0]
            quality = runtime.ai.quality_gate.evaluate(frame, face)
            pose = self.classifier._classify_pose(face, frame.shape[1])
            self.debug.update(detected_pose=pose, quality=dict(passed=quality.passed,
                confidence=quality.confidence, face_ratio=quality.face_ratio,
                sharpness=quality.sharpness, brightness=quality.brightness,
                checks=quality_breakdown(runtime.ai.quality_gate, quality)))
            now = time.monotonic()
            expected = next((p for p in self.required_slots if p not in self.embeddings), None)
            self.state = "COOLDOWN" if now < self.next_capture_allowed_at else "CAPTURING"
            info = {}
            if self.state == "CAPTURING" and quality.passed and pose == expected and runtime._valid(self.packet, self):
                embedding, info = runtime.ai.engine.analyze(frame, face)
                if self.cancelled.is_set() or not runtime._valid(self.packet, self):
                    return frame.copy()
                self.embeddings[pose] = embedding
                self.last_captured_pose = pose
                self.next_capture_allowed_at = time.monotonic()+POSE_COOLDOWN_SECONDS
                if len(self.embeddings) == len(self.required_slots):
                    with performance.measure("enrollment_finish_ms"):
                        self.result = runtime.ai.gallery.save(self.full_name, self.room_number, self.embeddings)
                    self.state, self.finished = "SUCCESS", True
            result = dict(bbox=dict(x=face.x, y=face.y, width=face.width, height=face.height),
                          verdict="ENROLL " + self.state, confidence=face.confidence,
                          adaface=info, quality=self.debug["quality"], error=None)
            return draw_faces(frame, [result], 0)

    def status(self):
        with self.lock:
            return dict(active=not self.finished, state=self.state, name=self.full_name,
                room=self.room_number, camera=self.camera_info,
                next_pose=next((p for p in self.required_slots if p not in self.embeddings), None),
                collected=list(self.embeddings), required=self.required_slots,
                progress=len(self.embeddings), total=len(self.required_slots),
                cooldown=max(0, self.next_capture_allowed_at-time.monotonic()),
                cooldown_seconds=POSE_COOLDOWN_SECONDS, last_captured_pose=self.last_captured_pose,
                debug=self.debug, result=self.result, error=self.error)
