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
)
matcher = FaceMatcher()

frames_path = glob("data/huy*.jpg")
len_frames = len(frames_path)

embeddings = []

show_landmarks = []

for frame_path in frames_path:

    frame = cv2.resize(cv2.imread(frame_path),(960,1200))
    print(frame.shape)
    faces = detector.detect(frame=frame)

    print(frame_path, "faces =", len(faces))
    
    for i,face in enumerate(faces,start=0):
        embedding = embedder.extract(frame=frame,face=face)
        
        print(embedding.shape)
        print(np.linalg.norm(embedding))
        print(embedding[:5])

        embeddings.append(embedding)

        cv2.rectangle(
            frame,
            (face.x, face.y),
            (face.x+face.width, face.y+face.height),
            (0,255,0),
            8
        )

        cv2.putText(
            frame,
            f"{face.confidence}",
            (face.x, face.y - 5),
            cv2.FONT_HERSHEY_COMPLEX,
            5,
            (0,255,0),
            3
        )
        
        for point in face.landmarks:
            x, y = point.astype(int)
        
            cv2.circle(
                frame,
                (x, y),
                10,
                (0,0,255),
                -1
            )

        
    show_landmarks.append(frame)
print("\n\n")
if len(embeddings) >= 2:
    for i in range(len_frames-1):
        print(matcher.is_match(embeddings[0],embeddings[i+1]))

show_full_landmarks = cv2.hconcat(show_landmarks)
show_full_landmarks = cv2.resize(
    show_full_landmarks,
    (1200, 300)
)

cv2.imshow("SHOW LANDMARKS", show_full_landmarks)
cv2.waitKey(0)
cv2.destroyAllWindows()
