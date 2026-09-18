import cv2
import numpy as np
from app.recognition.detector import FaceDetection
from app.recognition.embedder import FaceEmbedder




class SFaceEmbedder(FaceEmbedder):
    def __init__(
        self,
        model_path: str,
    ):
        self.recognizer = cv2.FaceRecognizerSF.create(
            model_path,
            "",
        )
    def _to_sface_input(
        self,
        face: FaceDetection,
    ) -> np.ndarray:

        bbox = np.array(
            [
                face.x,
                face.y,
                face.width,
                face.height
            ],
            dtype=np.float32
        )

        landmarks = face.landmarks.astype(np.float32).reshape(-1)

        confidence = np.array(
            [face.confidence],
            dtype=np.float32
        )

        return np.concatenate(
            [
                bbox,
                landmarks,
                confidence
            ]
        )
    def extract(
        self,
        frame: np.ndarray,
        face: FaceDetection
    ) -> np.ndarray:
        if frame is None or frame.size == 0:
            raise ValueError("Frame is empty")

        face_data = self._to_sface_input(face)

        aligned_face = self.recognizer.alignCrop(
            frame,
            face_data
        )    

        feature = self.recognizer.feature(
            aligned_face
        ) #convert sang vector 
        embedding = feature.flatten().astype(np.float32) # trải phẳng (1,128) -> (128,)
    
        norm = np.linalg.norm(embedding) # tính độ dài vector 

        if norm == 0:
            raise ValueError("Invalid embedding")

        return embedding / norm # L2 normalization

