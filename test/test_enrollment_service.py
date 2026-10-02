import cv2
import time

from app.database.database import SessionLocal
from app.database.enrollment_repository import EnrollmentRepository

from app.camera.rtsp_camera import RTSPCamera
from app.recognition.yunet_detector import YuNetDetector
from app.recognition.sface_embedder import SFaceEmbedder
from app.recognition.face_quality import FaceQualityGate

from app.services.enrollment_service import EnrollmentService

from app.config import (
    CAMRERA_IP,
    CAMERA_USERNAME,
    CAMERA_PASSWORD,
    CAMERA_RTSP_PORT
)

from app.config import (
    FACE_MIN_CONFIDENCE,
    FACE_MIN_RATIO,
    FACE_MIN_SHARPNESS,
    FACE_MIN_BRIGHTNESS,
    FACE_MAX_BRIGHTNESS
)

detector = YuNetDetector(
    model_path="models/face_detection_yunet_2023mar.onnx"
)
embedder = SFaceEmbedder(
    model_path="models/face_recognition_sface_2021dec.onnx"
)
quality_gate = FaceQualityGate(
    min_confidence=FACE_MIN_CONFIDENCE,
    min_face_ratio=FACE_MIN_RATIO,
    min_sharpness=FACE_MIN_SHARPNESS,
    min_brightness=FACE_MIN_BRIGHTNESS,
    max_brightness=FACE_MAX_BRIGHTNESS  
)

session = SessionLocal()
repository = EnrollmentRepository(session=session)

enrollment = EnrollmentService(
    repository=repository,
    detector=detector,
    embedder=embedder,
    quality_gate=quality_gate
)

channel = 1
url = (
    f"rtsp://{CAMERA_USERNAME}:{CAMERA_PASSWORD}"
    f"@{CAMRERA_IP}:{CAMERA_RTSP_PORT}"
    f"/cam/realmonitor?channel={channel}&subtype=0"
)

# camera = RTSPCamera(
#     url=url
# )

# camera.start()
test_cam = cv2.VideoCapture(0)


while True:
    ret, frame = test_cam.read()

    if not ret:
        time.sleep(0.1)
        continue
    
    if frame is None:
        time.sleep(0.1)
        continue
    frame = cv2.flip(frame, 1)
    display_frame = frame.copy()
    faces = detector.detect(frame)

    if len(faces) == 1:
        face = faces[0]
        cv2.rectangle(
            display_frame,
            (face.x, face.y),
            (face.x + face.width, face.y + face.height),
            (0, 255, 0),
            2
        )
        cv2.putText(
            display_frame,
            f"{face.confidence * 100:.1f}%",
            (face.x, face.y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )
    cv2.imshow("DISPLAY CAMERA",cv2.resize(display_frame, (960, 540)))
    key = cv2.waitKey(1) & 0xFF
    if key == 13:
        break
enrollment.start_enrollment(

    full_name="Nguyễn Đức Huy",
    room_number="301"
)

next_pose = enrollment.get_next_required_pose()
while (True):
    #frame = camera.get_latest_frame()
    ret, frame = test_cam.read()
    

    if not ret:
        time.sleep(0.1)
        continue
    
    if frame is None:
        time.sleep(0.1)
        continue
    frame = cv2.flip(frame, 1)
    display_frame = cv2.resize(frame,(960,540))
    
    if next_pose == "RIGHT":
        cv2.arrowedLine(
            display_frame,
            (740,405),
            (930,405),
            (0, 255, 0),
            8
        )
    elif next_pose == "LEFT":
        cv2.arrowedLine(
            display_frame,
            (740,405),
            (510,405),
            (0, 255, 0),
            8
        )
    elif next_pose == "FRONT":
        cv2.circle(
            display_frame,
            (740,405),
            12,
            (0, 255, 0),
            -1
        )
    elif next_pose == "FAR_FRONT":
        center = (740, 405)
        radius = 40
        line_len = 20

        cv2.circle(
            display_frame,
            center,
            radius,
            (0, 255, 0),
            3
        )

        cx, cy = center

        # nét ngang của dấu +
        cv2.line(
            display_frame,
            (cx - line_len, cy),
            (cx + line_len, cy),
            (0, 255, 0),
            3
        )

        # nét dọc của dấu +
        cv2.line(
            display_frame,
            (cx, cy - line_len),
            (cx, cy + line_len),
            (0, 255, 0),
            3
        )
    elif next_pose == "DOWN":
        cv2.arrowedLine(
            display_frame,
            (740, 300),
            (740, 510),
            (0, 255, 0),
            8
        )
    
    result = enrollment.process_frame(frame)
    

    if result is not None:
        next_pose = result['next_pose']
        if result['complete']:

            member = enrollment.finish_enrollment()

            print("Enrollment success")
            print("ID:", member.id)
            print("Code:", member.code)
            print("Name:", member.full_name)

            break

        
    cv2.imshow(
        "DISPLAY CAMERA",
        display_frame
    )

    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

#camera.stop()
test_cam.release()
session.close()
cv2.destroyAllWindows()


        