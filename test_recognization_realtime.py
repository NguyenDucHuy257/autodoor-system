import cv2
import numpy as np
from glob import glob

from app.recognition.sface_embedder import SFaceEmbedder
from app.recognition.yunet_detector import YuNetDetector
from app.recognition.face_matcher import FaceMatcher


embedder = SFaceEmbedder(
    model_path="models/face_recognition_sface_2021dec.onnx"
)

detector = YuNetDetector(
    model_path="models/face_detection_yunet_2023mar.onnx",
    score_threshold=0.7
)
matcher = FaceMatcher()

frame_path = "data/huy_cam.jpg"
frame = cv2.resize(cv2.imread(frame_path),(960, 540))
print(frame.shape)

root_faces = detector.detect(frame=frame)
if len(root_faces) == 0:
    raise ValueError(f"Cannot find any face in this image {frame_path}")

root_face = max(
    root_faces,
    key= lambda face: face.confidence
)

root_embedding = embedder.extract(frame=frame,face=root_face)

cap = cv2.VideoCapture(0)
while True:

    ret,frame = cap.read()

    frame = cv2.resize(
        frame,
        (960,540)
    )

    if not ret:
        break

    faces = detector.detect(frame=frame)

    print(frame_path, "faces =", len(faces))

    for i,face in enumerate(faces,start=0):
        embedding = embedder.extract(frame=frame,face=face)
        cv2.rectangle(
            frame,
            (face.x, face.y),
            (face.x+face.width, face.y+face.height),
            (0,255,0),
            4
        )

        cv2.putText(
            frame,
            f"{face.confidence}",
            (face.x, face.y + 30),
            cv2.FONT_HERSHEY_COMPLEX,
            1,
            (0,255,0),
            1
        )
        
        for point in face.landmarks:
            x, y = point.astype(int)
        
            cv2.circle(
                frame,
                (x, y),
                6,
                (0,0,255),
                -1
            )
        is_match,score = matcher.is_match(root_embedding,embedding)
        cv2.putText(
            frame,
            f"is_match: {is_match} {score}",
            (face.x, face.y-5),
            cv2.FONT_HERSHEY_COMPLEX,
            1,
            (0,0,255),
            1
        )
            
    cv2.imshow("REALTIME RECONIZATION", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()

