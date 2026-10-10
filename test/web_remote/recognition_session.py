"""
Web version of test/test_realtime.py as a DEBUG page (TEST ONLY).

Uses YuNetDetector, SFaceEmbedder, FaceMatcher, FaceQualityGate from app/
and Member / FaceEmbedding / SessionLocal from app.database.

Per face:
  * quality gate
  * SFace embedding (also for quality FAIL faces, for debugging)
  * best match over every stored template
  * verdict:
        quality PASS and similarity >= MATCH_THRESHOLD  -> RECOGNIZED
        quality PASS and similarity <  MATCH_THRESHOLD  -> UNKNOWN
        quality FAIL                                     -> REJECTED (candidate shown, never as identity)
Video only gets a RED box per face; all numbers go to the HTML panel.
"""

import time

import cv2
import numpy as np
from sqlalchemy import select

from app.database.database import SessionLocal
from app.database.models import Member, FaceEmbedding

from test.web_remote.runtime import (
    BaseSession,
    MODE_RECOGNITION,
    MATCH_THRESHOLD,
    runtime,
    quality_breakdown,
)


VERDICT_RECOGNIZED = "RECOGNIZED"
VERDICT_UNKNOWN = "UNKNOWN"
VERDICT_REJECTED = "REJECTED"
VERDICT_NO_EMBEDDING = "NO_EMBEDDING"


def load_person_database() -> list[dict]:
    """Same query as test/test_realtime.py; ORM data copied before session.close()."""
    session = SessionLocal()
    try:
        stmt = (
            select(Member, FaceEmbedding)
            .join(FaceEmbedding, FaceEmbedding.member_id == Member.id)
            .where(Member.active.is_(True))
        )
        rows = session.execute(stmt).all()

        person_database = []
        for member, face_embedding in rows:
            embedding = np.frombuffer(face_embedding.embedding, dtype=np.float32)
            person_database.append({
                "member_id": member.id,
                "code": member.code,
                "name": member.full_name,
                "room": member.room_number,
                "pose": face_embedding.pose,
                "embedding": embedding,
            })
        return person_database
    finally:
        session.close()


class RecognitionSession(BaseSession):

    mode = MODE_RECOGNITION
    # channel can be switched live while debugging
    locks_camera = False

    def __init__(self):
        super().__init__()

        self.person_database = load_person_database()
        self.member_count = len({p["member_id"] for p in self.person_database})

        self.faces: list[dict] = []
        self.yunet_ms = 0.0
        self.last_update = 0.0

    def process(self, frame):
        if self.finished:
            return frame

        ai = runtime.ai
        display = frame.copy()
        frame_height, frame_width = frame.shape[:2]

        detect_started = time.perf_counter()
        faces = ai.detector.detect(frame)
        yunet_ms = (time.perf_counter() - detect_started) * 1000.0

        results = []

        for index, face in enumerate(faces, start=1):
            quality = ai.quality_gate.evaluate(frame=frame, face=face)

            # SFace runs even for quality FAIL (debug only)
            sface_started = time.perf_counter()
            embedding = None
            sface_error = None
            try:
                embedding = ai.embedder.extract(frame=frame, face=face)
            except Exception as error:
                sface_error = str(error)
            sface_ms = (time.perf_counter() - sface_started) * 1000.0

            best_person = None
            best_score = None

            if embedding is not None:
                for person in self.person_database:
                    score = ai.matcher.similarity(person["embedding"], embedding)
                    if best_score is None or score > best_score:
                        best_score = score
                        best_person = person

            if embedding is None:
                verdict = VERDICT_NO_EMBEDDING
            elif not quality.passed:
                verdict = VERDICT_REJECTED
            elif best_score is not None and best_score >= ai.matcher.match_threshold:
                verdict = VERDICT_RECOGNIZED
            else:
                verdict = VERDICT_UNKNOWN

            candidate = None
            if best_person is not None:
                candidate = {
                    "name": best_person["name"],
                    "code": best_person["code"],
                    "room": best_person["room"],
                    "pose": best_person["pose"],
                    "similarity": round(float(best_score), 4),
                }

            # RED box only - no text on the video
            cv2.rectangle(
                display,
                (face.x, face.y),
                (face.x + face.width, face.y + face.height),
                (0, 0, 255),
                2,
            )

            results.append({
                "index": index,
                "verdict": verdict,
                "identity": candidate if verdict == VERDICT_RECOGNIZED else None,
                "candidate": candidate,
                "similarity": round(float(best_score), 4) if best_score is not None else None,
                "yunet_confidence": round(float(face.confidence), 4),
                "bbox": {
                    "x": int(face.x),
                    "y": int(face.y),
                    "width": int(face.width),
                    "height": int(face.height),
                },
                "face_ratio": round(float(quality.face_ratio), 4),
                "sharpness": round(float(quality.sharpness), 2),
                "brightness": round(float(quality.brightness), 2),
                "quality_passed": bool(quality.passed),
                "quality_checks": quality_breakdown(ai.quality_gate, quality),
                "embedding_dim": int(embedding.shape[0]) if embedding is not None else 0,
                "embedding_norm": round(float(np.linalg.norm(embedding)), 4) if embedding is not None else None,
                "sface_ms": round(sface_ms, 2),
                "sface_error": sface_error,
            })

        self.faces = results
        self.yunet_ms = yunet_ms
        self.last_update = time.time()

        return display

    def status(self) -> dict:
        ai = runtime.ai
        return {
            "active": not self.finished,
            "faces": self.faces,
            "face_count": len(self.faces),
            "yunet_ms": round(self.yunet_ms, 2),
            "db_embedding_count": len(self.person_database),
            "db_member_count": self.member_count,
            "match_threshold": ai.matcher.match_threshold if ai else MATCH_THRESHOLD,
            "quality_thresholds": {
                "min_confidence": ai.quality_gate.min_confidence,
                "min_face_ratio": ai.quality_gate.min_face_ratio,
                "min_sharpness": ai.quality_gate.min_sharpness,
                "min_brightness": ai.quality_gate.min_brightness,
                "max_brightness": ai.quality_gate.max_brightness,
            } if ai else None,
            "error": self.error,
        }
