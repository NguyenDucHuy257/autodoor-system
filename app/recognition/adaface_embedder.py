import cv2
import numpy as np
import torch  # Load bundled CUDA/cuDNN DLLs before ONNX Runtime creates a session.
import onnxruntime as ort

from app.recognition.detector import FaceDetection
from app.recognition.embedder import FaceEmbedder


class AdaFaceEmbedder(FaceEmbedder):
    def __init__(self, model_path):
        self.model_path = model_path

        self.session = ort.InferenceSession(
            model_path,
            providers=[
                "CUDAExecutionProvider",
                "CPUExecutionProvider"
            ]
        )

    def _align(
        self,
        frame: np.ndarray,
        face: FaceDetection
    ) -> np.ndarray:

        if frame is None or frame.size == 0:
            raise ValueError("Frame is empty!")
        
        landmarks = face.landmarks.astype(np.float32)

        if landmarks.shape != (5,2) or not np.isfinite(landmarks).all():
            raise ValueError("Not Enought 5 landmarks!")

        source_landmarks = landmarks[[1,0,2,4,3]]

        destination_landmarks = np.array([
            [38.2946, 51.6963],  # Mắt trái trên ảnh
            [73.5318, 51.5014],  # Mắt phải trên ảnh
            [56.0252, 71.7366],  # Mũi
            [41.5493, 92.3655],  # Khóe miệng trái
            [70.7299, 92.2041]   # Khóe miệng phải
        ], dtype=np.float32)

        matrix, inliers = cv2.estimateAffinePartial2D(
            from_=source_landmarks,
            to=destination_landmarks,
            method=cv2.LMEDS 
        ) #Nhân ma trận để căn góc

        if matrix is None or not np.isfinite(matrix).all():
            raise ValueError("Matrix after estimateAffinePartial2D is None!")

        aligned_face = cv2.warpAffine(
            src=frame,
            M=matrix,
            dsize=(112,112)
        )

        return aligned_face

    def _preprocess(
        self,
        aligned_face: np.ndarray
    ) -> np.ndarray:
        if (aligned_face.shape != (112,112,3)):
            raise ValueError("Expected aligned image shape (112, 112, 3)")
        
        aligned_face = aligned_face.astype(np.float32)

        nomalized_face = (aligned_face - 127.5)/127.5

        CHW_face = np.transpose(nomalized_face, (2, 0, 1))

        add_batch_face = np.expand_dims(CHW_face, axis=0)

        return add_batch_face
    def extract(
        self,
        frame: np.ndarray,
        face: FaceDetection
    ) -> np.ndarray:

        aligned_face = self._align(
           frame=frame,
           face=face
        )

        formatted_face = self._preprocess(aligned_face=aligned_face)

        embedding = self.session.run(
            ["embedding"],
            {"input": formatted_face}
        )[0]

         # 4. (1, 512) -> (512,)
        embedding = embedding.flatten().astype(np.float32)

        if embedding.shape != (512,) or not np.isfinite(embedding).all():
            raise ValueError("Invalid AdaFace embedding")

        # 5. L2 normalization
        norm = np.linalg.norm(embedding)

        if not np.isfinite(norm) or norm <= 0:
            raise ValueError("Invalid embedding norm")

        embedding = embedding / norm

        return embedding
