import numpy as np
import cv2

from dataclasses import dataclass

from app.recognition.detector import FaceDetection

@dataclass
class FaceQualityResult:
    passed: bool
    score: float
    sharpness: float
    brightness: float
    face_ratio: float
    confidence: float

class FaceQualityGate:
    def __init__(
        self,
        min_confidence: float,
        min_face_ratio: float,
        min_sharpness: float,
        min_brightness: float,
        max_brightness: float      
    ):
        self.min_confidence = min_confidence
        self.min_face_ratio = min_face_ratio
        self.min_sharpness = min_sharpness

        self.min_brightness = min_brightness
        self.max_brightness = max_brightness

    def _crop_face_roi(
        self,
        frame: np.ndarray,
        face: FaceDetection,
    ) -> np.ndarray | None:
        
        frame_height, frame_width = frame.shape[:2]

        x1 = max(face.x,0)
        y1 = max(face.y,0)

        x2 = min(face.x + face.width,frame_width)
        y2 = min(face.y + face.height,frame_height)

        if x2 <= x1 or y2 <= y1:
            return None

        visible_width = x2-x1
        visible_height = y2-y1

        if visible_height/face.height < 0.7 or visible_width/face.width < 0.7:
            return None
        
        crop_face = frame[y1:y2,x1:x2]

        return crop_face
        
    def _calculate_sharpness(
        self,
        face_roi: np.ndarray
    ) -> float:

        gray_face = cv2.cvtColor(
            face_roi,
            cv2.COLOR_BGR2GRAY
        )

        laplacian = cv2.Laplacian(
            gray_face,
            cv2.CV_64F
        )

        return float(laplacian.var())

    def _calculate_brightness(
        self,
        face_roi: np.ndarray,
    ) -> float:

        gray_face = cv2.cvtColor(
            face_roi,
            cv2.COLOR_BGR2GRAY
        )

        brightness = np.mean(gray_face)

        return float(brightness)
    
    def evaluate(
        self,
        frame: np.ndarray,
        face: FaceDetection,
    ) -> FaceQualityResult:

        face_roi = self._crop_face_roi(
            frame=frame,
            face=face
        )

        if face_roi is None:
            return FaceQualityResult(
                passed=False,
                score=0,
                sharpness=0,
                brightness=0,
                face_ratio=0,
                confidence=0
            )

        sharpness = self._calculate_sharpness(face_roi)#25%
        brightness = self._calculate_brightness(face_roi)#25%

        frame_height, frame_width = frame.shape[:2]

        face_ratio = face.width/frame_width #50%
        print("face_ratio=",face_ratio)

        confidence_score = 0 #face_ratio*0.5 + sharpness*0.25 + brightness*0.25
        
        quality_result = FaceQualityResult(
            passed = True if (face.confidence > self.min_confidence) and (self.max_brightness > brightness) and (self.min_brightness < brightness)  and (sharpness > self.min_sharpness) and (face_ratio > self.min_face_ratio) else False,
            score=confidence_score,
            sharpness=sharpness,
            brightness=brightness,
            face_ratio=face_ratio,
            confidence=face.confidence
        )
        
        return quality_result