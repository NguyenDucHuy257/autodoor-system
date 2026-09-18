import cv2
import time
import numpy as np
from glob import glob

from app.recognition.sface_embedder import SFaceEmbedder
from app.recognition.yunet_detector import YuNetDetector
from app.recognition.face_matcher import FaceMatcher
from app.camera.rtsp_camera import RTSPCamera

from app.config import (
    CAMRERA_IP,
    CAMERA_USERNAME,
    CAMERA_PASSWORD,
    CAMERA_RTSP_PORT
)

rtsp_url = f"rtsp://{CAMERA_USERNAME}:{CAMERA_PASSWORD}@{CAMRERA_IP}:{CAMERA_RTSP_PORT}/cam/realmonitor?channel=1&subtype=0"

embedder = SFaceEmbedder(
    model_path="models/face_recognition_sface_2021dec.onnx"
)

detector = YuNetDetector(
    model_path="models/face_detection_yunet_2023mar.onnx",
)
matcher = FaceMatcher()

# Lấy mặt gốc
root_frame = cv2.imread("data/huy1.jpg")
root_faces = detector.detect(frame=root_frame)
root_face = root_faces[0]
root_embedding = embedder.extract(frame=root_frame, face=root_face)

rtsp_camera = RTSPCamera(url=rtsp_url)

rtsp_camera.start()
try:
    while True:

        frame = rtsp_camera.get_latest_frame()

        if frame is None:
            time.sleep(0.01)
            continue

        frame = cv2.resize(frame,(960,540))

        faces = detector.detect(frame=frame)

        for i,face in enumerate(faces,start=0):
            embedding = embedder.extract(frame=frame,face=face)

            cv2.rectangle(
                frame,
                (face.x, face.y),
                (face.x+face.width, face.y+face.height),
                (0,255,0),
                2
            )
            is_match,score = matcher.is_match(
                embedding_root=root_embedding,
                embedding_checker=embedding
            )
            cv2.putText(
                frame,
                f"{is_match} & Similarity: {score:.2f}",
                (face.x, face.y - 5),
                cv2.FONT_HERSHEY_COMPLEX,
                1,
                (0,0,250),
                1
            )
            
            for point in face.landmarks:
                x, y = point.astype(int)
            
                cv2.circle(
                    frame,
                    (x, y),
                    3,
                    (0,0,255),
                    -1
                )

        cv2.imshow("CAMERA", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
finally:
    rtsp_camera.stop()
        
    cv2.destroyAllWindows()