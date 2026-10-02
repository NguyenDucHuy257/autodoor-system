import numpy as np
from pathlib import Path
import cv2

from app.database.enrollment_repository import EnrollmentRepository
from app.recognition.detector import FaceDetector, FaceDetection
from app.recognition.embedder import FaceEmbedder
from app.database.models import Member


from app.recognition.face_quality import FaceQualityGate

from app.config import (
    YAW_THRESHOLD,
    DOWN_THRESHOLD,
    FAR_FACE_RATIO,
)

class EnrollmentService:

    REQUIRED_SLOTS = (
        "FRONT",
        "LEFT",
        "RIGHT",
        "DOWN",
        "FAR_FRONT"
    )

    
    def __init__(
        self,
        repository: EnrollmentRepository,
        detector: FaceDetector,
        embedder: FaceEmbedder,
        quality_gate: FaceQualityGate,
    ):
        self.repository = repository
        self.detector = detector
        self.embedder = embedder

        self.active = False
        self.full_name: str | None = None
        self.room_number: str | None = None

        self.collected_frames: dict[
            str,
            tuple[np.ndarray,FaceDetection]
        ] = {}

        self.quality_gate = quality_gate

    def _save_debug_frames(self, code: str) -> None:
        folder = Path("data/enrollment") / code

        folder.mkdir(
            parents=True,
            exist_ok=True
        )
        for pose, (frame, face) in self.collected_frames.items():

            file_path = folder / f"{pose}.png" #LIB PATH HỖ TRỢ /
            cv2.imwrite(
                str(file_path),
                frame
            )
    def _classify_pose(
        self,
        face: FaceDetection,
        frame_width: int,
    ) -> str | None:

        right_eye = face.landmarks[0]
        left_eye = face.landmarks[1]
        nose = face.landmarks[2]
        right_mouth = face.landmarks[3]
        left_mouth = face.landmarks[4]

        eye_center = (right_eye + left_eye)/2
        mouth_center = (right_mouth + left_mouth)/2

        face_ratio = face.width / frame_width

        eye_span = abs(left_eye[0]-right_eye[0])
        if eye_span < 1:
            return None

        yaw_score = (nose[0] - eye_center[0])/eye_span

        if yaw_score <= -YAW_THRESHOLD:
            return "LEFT"
        if yaw_score >= YAW_THRESHOLD:
            return "RIGHT"

        vertical_span = eye_center[1]-mouth_center[1]

        if abs(vertical_span) < 1:
            return None

        pitch = (nose[1]-mouth_center[1])/vertical_span

        if pitch < DOWN_THRESHOLD:
            return "DOWN"

        if face_ratio < FAR_FACE_RATIO:
            return "FAR_FRONT"

        return "FRONT"

    def _extract_embeddings(self) -> dict[str, np.ndarray]:

        embeddings = {}
        
        for pose,(frame,face) in self.collected_frames.items():
            embedding = self.embedder.extract(
                frame=frame,
                face=face
            )
            embeddings[pose] = embedding

        return embeddings

    def _generate_member_code(self) -> str:

        number = self.repository.next_member_code_number()
        return f"MS{number:06d}"
    
    def get_next_required_pose(self) -> str | None:
        for pose in self.REQUIRED_SLOTS:
            if pose not in self.collected_frames:
                return pose
            
        return None
    
    def start_enrollment(
        self,
        full_name: str,
        room_number: str | None
    ) -> None:

        if not full_name.strip():
            raise ValueError("Full name is required!") 
        
        self.active = True

        self.full_name = full_name.strip()
        self.room_number = room_number

        self.collected_frames = {}


    def process_frame(
        self,
        frame: np.ndarray | None,
    ) -> dict[str,bool | list[str]] | None:
        
        if not self.active:
            raise RuntimeError("Call 'start_enrollment' function before use this!")

        if frame is None or frame.size == 0:
            return None

        faces = self.detector.detect(frame)

        if len(faces) != 1:
            return None
        
        face = faces[0]
        # print(face.landmarks)
        # print(face.landmarks.shape)

        quality_gate_result = self.quality_gate.evaluate(
            frame=frame,
            face=face
        )
        if not quality_gate_result.passed:
            print("Ảnh không đạt chuẩn!")
            return None

        frame_height, frame_width = frame.shape[:2]
        pose = self._classify_pose(
            face=face,
            frame_width=frame_width
        )

        required_pose = self.get_next_required_pose()

        if pose is None:
            return None
        if pose not in self.REQUIRED_SLOTS:
            return None

        if pose == required_pose:
            self.collected_frames[pose] = (frame.copy(),face)

        if all(pose in self.collected_frames for pose in self.REQUIRED_SLOTS):
            return {
                "complete": True,
                "collected": list(self.collected_frames.keys()),
                "next_pose": None
            }
        return {
            "complete": False,
            "collected": list(self.collected_frames.keys()),
            "next_pose": self.get_next_required_pose(),
            "debug": {
                "confidence": face.confidence,
                "landmarks": face.landmarks,
                "bbox": (face.x, face.y, face.width, face.height),
                "face_ratio": face.width / frame_width,
            }
        }

    def finish_enrollment(self) -> Member:

        if not self.active:
            raise RuntimeError("start_enrollment first!")

        if not all(pose in self.collected_frames for pose in self.REQUIRED_SLOTS):

            missing = [
                pose
                for pose in self.REQUIRED_SLOTS
                if pose not in self.collected_frames
            ]
            raise ValueError(f"REQUIRED SLOTS: missing {missing}")

        embeddings = self._extract_embeddings()

        code = self._generate_member_code()

        member = self.repository.save_enrollment(
            code=code,
            full_name=self.full_name,
            room_number=self.room_number,
            embeddings=embeddings
        )

        self._save_debug_frames(code)
        
        self.active = False
        self.full_name = None
        self.room_number = None
        self.collected_frames = {}

        return member
