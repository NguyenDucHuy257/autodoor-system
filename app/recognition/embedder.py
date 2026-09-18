from abc import ABC,abstractmethod

import numpy as np

from app.recognition.detector import FaceDetection

class FaceEmbedder(ABC):
    @abstractmethod
    def extract(
        self,
        frame: np.ndarray,
        face: FaceDetection,
    ) -> np.ndarray:
        pass