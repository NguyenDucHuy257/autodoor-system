import cv2
import numpy as np 

from app.recognition.detector import (
    FaceDetection,
    FaceDetector
)

class YuNetDetector(FaceDetector):
    def __init__(
            self,
            model_path: str,
            score_threshold: float = 0.9,
            nms_threshold: float = 0.3, #iou - tỉ lệ phần giao > 0,3 ưu tiên box score cao hơn 
            top_k: int = 5000
    ):
        self.detector = cv2.FaceDetectorYN.create(
            model_path,
            "",
            (320,320),
            score_threshold,
            nms_threshold,  
            top_k
        ) #load trước model 1 lần 
    def detect(
            self,
            frame: np.ndarray,
    ) -> list[FaceDetection]:
        
        if frame is None or frame.size == 0:
            return []

        height, width = frame.shape[:2]

        self.detector.setInputSize(
            (width,height)
        )

        _, faces = self.detector.detect(frame)

        if faces is None or len(faces) == 0:
            return []

        results = []
        for face in faces:
            x,y,w,h = face[:4]
            landmarks = face[4:14].reshape(5,2) # 4 - 13: landmarks, convert sang 5 landmarks (x,y)
            confidence = float(face[14]) #score
            results.append(
                FaceDetection(
                    x=int(x),
                    y=int(y),
                    width=int(w),
                    height=int(h),
                    landmarks=landmarks,
                    confidence=confidence
                )
            )   

        return results



