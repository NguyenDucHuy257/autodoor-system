"""SCRFD jobs and AdaFace results, drawn on the exact inference source frame."""
import time
import numpy as np
from app.performance import performance
from test.web_remote.runtime import BaseSession, MODE_RECOGNITION, runtime
from test.web_remote.adaface_engine import MATCH_THRESHOLD
from test.web_remote.overlay import draw_faces, face_thumbnail


class RecognitionSession(BaseSession):
    mode = MODE_RECOGNITION
    locks_camera = False

    def __init__(self):
        super().__init__()
        if runtime.ai.engine is None:
            raise RuntimeError(runtime.ai.model_error or "AdaFace model is loading; try again shortly")
        self.person_database, self.matrix = runtime.ai.gallery.load()
        self.member_count = len({p["code"] for p in self.person_database})
        self.faces = []
        self.detection_ms = 0.0
        self.last_presence = None
        self.presence_generation = None
        self.last_update = None

    def detect(self, packet):
        started = time.perf_counter()
        faces = runtime.ai.detector.detect(packet.frame)
        with self.lock:
            self.detection_ms = (time.perf_counter()-started)*1000
            if faces and not self.finished and packet.generation == runtime.camera.current_generation():
                self.last_presence = packet.timestamp
                self.presence_generation = packet.generation
        return faces

    def process_faces(self, packet, detections):
        results = []
        for index, face in enumerate(detections, 1):
            if not runtime._valid(packet, self):
                return None
            quality = runtime.ai.quality_gate.evaluate(packet.frame, face)
            embedding, info, error = None, {}, None
            if quality.passed:
                try:
                    embedding, info = runtime.ai.engine.analyze(packet.frame, face)
                except Exception as exc:
                    error = "AdaFace " + type(exc).__name__
            best, score = None, None
            if embedding is not None and len(self.matrix):
                with performance.measure("matching_ms"):
                    scores = self.matrix @ embedding
                    at = int(np.argmax(scores))
                    score = float(scores[at])
                    best = self.person_database[at]
            verdict = "REJECTED" if not quality.passed else "NO_EMBEDDING" if embedding is None else "RECOGNIZED" if score is not None and score >= MATCH_THRESHOLD else "UNKNOWN"
            results.append(dict(index=index, verdict=verdict,
                identity=best if verdict == "RECOGNIZED" else None, similarity=score,
                bbox=dict(x=face.x, y=face.y, width=face.width, height=face.height),
                confidence=face.confidence, adaface=info, error=error,
                quality=dict(passed=quality.passed, sharpness=quality.sharpness,
                             brightness=quality.brightness, face_ratio=quality.face_ratio)))
            results[-1]["crop"] = face_thumbnail(packet.frame, results[-1]["bbox"])
        with self.lock:
            if not runtime._valid(packet, self):
                return None
            self.faces = results
            self.last_update = time.monotonic()
        return draw_faces(packet.frame, results, (time.monotonic()-packet.timestamp)*1000)

    def presence(self, generation):
        with self.lock:
            return self.last_presence if generation == self.presence_generation and not self.finished else None

    def status(self):
        with self.lock:
            return dict(active=not self.finished, faces=self.faces, face_count=len(self.faces),
                        detection_ms=round(self.detection_ms, 2), db_embedding_count=len(self.matrix),
                        db_member_count=self.member_count, match_threshold=MATCH_THRESHOLD,
                        model="AdaFace", providers=runtime.ai.engine.providers, error=self.error)

    def process(self, frame):
        raise RuntimeError("Recognition requires a timestamped frame job")
