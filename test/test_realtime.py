import cv2
import time
import sys

from datetime import datetime
import numpy as np
from glob import glob

from app.recognition.sface_embedder import SFaceEmbedder
from app.recognition.yunet_detector import YuNetDetector
from app.recognition.face_matcher import FaceMatcher

from sqlalchemy import select
from app.database.database import SessionLocal
from app.database.models import Member, FaceEmbedding

from app.camera.rtsp_camera import RTSPCamera

from app.config import (
    CAMRERA_IP,
    CAMERA_USERNAME,
    CAMERA_PASSWORD,
    CAMERA_RTSP_PORT
)



channel = 16

if len(sys.argv) > 1:
    channel = int(sys.argv[1])

rtsp_url = (
    f"rtsp://{CAMERA_USERNAME}:{CAMERA_PASSWORD}"
    f"@{CAMRERA_IP}:{CAMERA_RTSP_PORT}"
    f"/cam/realmonitor?channel={channel}&subtype=0"
)

session = SessionLocal()

# KHỞI TẠO CÁC CHỨC NĂNG
embedder = SFaceEmbedder(
    model_path="models/face_recognition_sface_2021dec.onnx",

)

detector = YuNetDetector(
    model_path="models/face_detection_yunet_2023mar.onnx",
    score_threshold=0.7
)
matcher = FaceMatcher(
    match_threshold=0.4
)

# TRUY VẤN DATABASE
stmt = (
    select(
        Member,
        FaceEmbedding
    )
    .join(
        FaceEmbedding,
        FaceEmbedding.member_id == Member.id
    )
    .where(
        Member.active.is_(True)
    )
)

# LẤY DANH SÁCH ENROLLED
rows = session.execute(
    stmt
).all()

person_database = []
for member, face_embedding in rows:
    #ĐỔI BYTES SANG NP.NDARRAY
    embedding = np.frombuffer(
        face_embedding.embedding,
        dtype=np.float32
    )

    person_database.append(
    {
        "member_id": member.id,
        "code": member.code,
        "name": member.full_name,
        "room": member.room_number,
        "pose": face_embedding.pose,
        "embedding": embedding,
    }
)

session.close()

if len(person_database) == 0:
    print("Chưa đăng ký khuôn mặt nào")
    raise SystemExit

rtsp_camera = RTSPCamera(url=rtsp_url)

rtsp_camera.start()
# XỬ LÝ CHÍNH

try:
    while True:

        frame = rtsp_camera.get_latest_frame()
        
        if frame is None:
            time.sleep(0.01)
            continue

        #resized_frame = cv2.resize(frame,(960,540))

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

            cv2.putText(
                frame,
                f"{face.confidence*100:.2f}% là khuôn mặt",
                (face.x, face.y - 40),
                cv2.FONT_HERSHEY_COMPLEX,
                1,
                (0,255,0),
                1
            )
            
            # for point in face.landmarks:
            #     x, y = point.astype(int)
            
            #     cv2.circle(
            #         frame,
            #         (x, y),
            #         3,
            #         (0,0,255),
            #         -1
            #     )

            # person_database.append(
            #     {
            #         "member_id": member.id,
            #         "code": member.code,
            #         "name": member.full_name,
            #         "room": member.room_number,
            #         "pose": face_embedding.pose,
            #         "embedding": embedding,
            #     }

            best_person = None
            best_score = -1

            for person in person_database:

                is_match,score = matcher.is_match(person["embedding"],embedding)

                if is_match and score > best_score:
                    best_person = person
                    best_score = score

            cv2.putText(
                frame,
                f"Similarity:{best_score:.2f}" if best_person is not None else "Không rõ danh tính",
                (face.x, face.y-10),
                cv2.FONT_HERSHEY_COMPLEX,
                0.9,
                (0,0,255),
                1
            )

            if best_person is not None:

                name = best_person.get("name")
                room = best_person.get("room")
                pose = best_person.get("pose")

                cv2.putText(
                    frame,
                    f"{name},P:{room},Hướng: {pose}",
                    (face.x, face.y + face.height+40),
                    cv2.FONT_HERSHEY_COMPLEX,
                    0.9,
                    (0,255,0),
                    1
                )
        if len(faces) >= 1:
            file_name = datetime.now().strftime("%Y%m%d_%H%M%S_%f") + ".jpg"
            cv2.imwrite(f"data/reco_data/{channel} - {file_name}",frame)    
                    
        display_frame = cv2.resize(frame, (480, 270))

        cv2.imshow(f"REALTIME CHANNEL {channel}", display_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
finally:
    rtsp_camera.stop()
    cv2.destroyAllWindows()

