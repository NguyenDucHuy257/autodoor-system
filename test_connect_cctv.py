import cv2

rtsp_url = (
    "rtsp://admin:PASSWORD@192.168.0.20:554/"
    "cam/realmonitor?channel=1&subtype=0"
)

cap = cv2.VideoCapture(
    rtsp_url,
    cv2.CAP_FFMPEG
)

if not cap.isOpened():
    raise RuntimeError("Cannot connect to RTSP camera")

while True:
    ret, frame = cap.read()

    if not ret:
        print("Cannot read frame")
        break

    cv2.imshow("RTSP Camera", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()