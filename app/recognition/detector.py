from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

@dataclass
class FaceDetection:
    x: int
    y: int
    width: int
    height: int 

    landmarks: np.ndarray
    confidence: float

class FaceDetector(ABC):
    @abstractmethod
    def detect(
        self,
        frame: np.ndarray,

    ) -> list[FaceDetection]:
        pass
