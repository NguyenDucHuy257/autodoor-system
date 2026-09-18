import cv2

import numpy as np

class FaceMatcher():
    def __init__(
        self,
        match_threshold: float = 0.4
    ):
        self.match_threshold = match_threshold

    def similarity(
        self,
        embedding_root: np.ndarray,
        embedding_checker: np.ndarray       
    ) -> float:

        score = np.dot(
            embedding_root,
            embedding_checker
        )        

        return float(score)

    def is_match(
        self,
        embedding_root: np.ndarray,
        embedding_checker: np.ndarray  
    ) -> bool:
        score = self.similarity(embedding_root, embedding_checker)
        return (score >= self.match_threshold,score)

        