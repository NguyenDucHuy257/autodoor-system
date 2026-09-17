import cv2
from glob import glob

from app.recognition.yunet_detector import YuNetDetector


detector = YuNetDetector(
    "models/face_detection_yunet_2023mar.onnx",
    score_threshold=0.6
)

cap = cv2.VideoCapture("data/WalkByShop1cor.mpg")

while True:
    ret, frame = cap.read()

    if not ret:
        break
    
    faces = detector.detect(frame=frame)

    len_faces = len(faces)
    print("Faces: ", len_faces)

    for i, face in enumerate(faces, start=0) :
        print(f"{i}th:")
        x = face.x
        y = face.y
        w = face.width
        h = face.height

        print(face.landmarks)
        print(f"Score={face.confidence}\n")

        cv2.rectangle(
            frame,
            (x, y),
            (x + w, y + h),
            (0,255,0),
            1
        )
        cv2.putText(
            frame,
            f"{face.confidence: .2f}",
            (x,y - 5),
            cv2.FONT_HERSHEY_COMPLEX,
            0.5, #font size
            (0,255,0),
            1 #thickness
        )
    cv2.imshow("YuNet Realtime",frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
cap.release()
cv2.destroyAllWindows()