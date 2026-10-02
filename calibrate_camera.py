import cv2
import time
import numpy as np

from app.recognition.yunet_detector import YuNetDetector


STAGES = (
    "FRONT_NEAR",
    "LEFT",
    "RIGHT",
    "DOWN",
    "FAR_FRONT",
)

PREPARE_SECONDS = 5
SAMPLES_PER_STAGE = 30


detector = YuNetDetector(
    model_path="models/face_detection_yunet_2023mar.onnx"
)

camera = cv2.VideoCapture(0)


def extract_metrics(face, frame) -> dict | None:
    right_eye = face.landmarks[0]
    left_eye = face.landmarks[1]
    nose = face.landmarks[2]
    right_mouth = face.landmarks[3]
    left_mouth = face.landmarks[4]

    # -------------------------
    # YAW
    # -------------------------
    eye_center = (right_eye + left_eye) / 2

    eye_span = abs(
        left_eye[0] - right_eye[0]
    )

    if eye_span < 1:
        return None

    yaw = (
        nose[0] - eye_center[0]
    ) / eye_span

    # -------------------------
    # PITCH
    # -------------------------
    mouth_center = (
        right_mouth + left_mouth
    ) / 2

    vertical_span = (
        eye_center[1] - mouth_center[1]
    )

    if abs(vertical_span) < 1:
        return None

    pitch = (
        nose[1] - mouth_center[1]
    ) / vertical_span

    # -------------------------
    # FACE RATIO
    # -------------------------
    frame_width = frame.shape[1]

    face_ratio = (
        face.width / frame_width
    )

    return {
        "yaw": float(yaw),
        "pitch": float(pitch),
        "face_ratio": float(face_ratio),
        "confidence": float(face.confidence),
    }


# =========================
# STATE
# =========================

stage_index = 0

phase = "WAIT_ENTER"

phase_started_at = None

stage_samples = []

all_samples = {
    stage: []
    for stage in STAGES
}

calibration_complete = False


try:
    while True:

        ret, frame = camera.read()

        if not ret or frame is None:
            continue

        # Giữ cùng quy ước với Enrollment
        frame = cv2.flip(frame, 1)

        display_frame = frame.copy()

        current_stage = STAGES[stage_index]

        # Detect liên tục để preview
        faces = detector.detect(frame)

        # =========================
        # DRAW FACE DEBUG
        # =========================

        if len(faces) == 1:

            face = faces[0]

            cv2.rectangle(
                display_frame,
                (face.x, face.y),
                (
                    face.x + face.width,
                    face.y + face.height
                ),
                (0, 255, 0),
                2
            )

            cv2.putText(
                display_frame,
                f"Detect: {face.confidence * 100:.1f}%",
                (
                    face.x,
                    max(face.y - 10, 20)
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )

            # 5 landmarks
            for point in face.landmarks:

                x, y = point

                cv2.circle(
                    display_frame,
                    (
                        int(x),
                        int(y)
                    ),
                    3,
                    (0, 0, 255),
                    -1
                )

        # =========================
        # WAIT ENTER
        # =========================

        if phase == "WAIT_ENTER":

            cv2.putText(
                display_frame,
                f"{current_stage}",
                (30, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 255),
                2
            )

            cv2.putText(
                display_frame,
                "Press ENTER when ready",
                (30, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2
            )

        # =========================
        # PREPARE
        # =========================

        elif phase == "PREPARE":

            elapsed = (
                time.monotonic()
                - phase_started_at
            )

            remaining = max(
                0,
                PREPARE_SECONDS - elapsed
            )

            cv2.putText(
                display_frame,
                f"{current_stage}",
                (30, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 255),
                2
            )

            cv2.putText(
                display_frame,
                f"Ready in {remaining:.1f}s",
                (30, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 255),
                2
            )

            if remaining <= 0:
                phase = "CAPTURE"
                stage_samples = []

        # =========================
        # CAPTURE
        # =========================

        elif phase == "CAPTURE":

            cv2.putText(
                display_frame,
                f"CAPTURING {current_stage}",
                (30, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2
            )

            cv2.putText(
                display_frame,
                (
                    f"{len(stage_samples)}"
                    f"/{SAMPLES_PER_STAGE}"
                ),
                (30, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2
            )

            # Chỉ lấy khi đúng 1 mặt
            if len(faces) == 1:

                face = faces[0]

                metrics = extract_metrics(
                    face=face,
                    frame=frame
                )

                if metrics is not None:

                    stage_samples.append(
                        metrics
                    )

            # Đã đủ 30 mẫu
            if (
                len(stage_samples)
                >= SAMPLES_PER_STAGE
            ):

                all_samples[current_stage] = (
                    stage_samples.copy()
                )

                print(
                    f"{current_stage}: "
                    f"{len(stage_samples)} samples"
                )

                stage_index += 1

                # Hoàn thành tất cả stage
                if stage_index >= len(STAGES):
                    calibration_complete = True
                    break

                # Stage mới lại chờ Enter
                phase = "WAIT_ENTER"

                stage_samples = []

                phase_started_at = None

        # =========================
        # DISPLAY
        # =========================

        display_frame = cv2.resize(
            display_frame,
            (960, 540)
        )

        cv2.imshow(
            "CAMERA CALIBRATION",
            display_frame
        )

        key = cv2.waitKey(1) & 0xFF

        # Enter
        if (
            key == 13
            and phase == "WAIT_ENTER"
        ):
            phase = "PREPARE"

            phase_started_at = (
                time.monotonic()
            )

        # Quit
        if key == ord("q"):
            break

finally:
    camera.release()
    cv2.destroyAllWindows()


# =====================================
# CALCULATE MEDIAN
# =====================================

if calibration_complete:

    print("\n")
    print("=" * 50)
    print("CALIBRATION RESULT")
    print("=" * 50)

    for stage in STAGES:

        samples = all_samples[stage]

        yaw_values = [
            sample["yaw"]
            for sample in samples
        ]

        pitch_values = [
            sample["pitch"]
            for sample in samples
        ]

        face_ratio_values = [
            sample["face_ratio"]
            for sample in samples
        ]

        confidence_values = [
            sample["confidence"]
            for sample in samples
        ]

        yaw_median = float(
            np.median(yaw_values)
        )

        pitch_median = float(
            np.median(pitch_values)
        )

        face_ratio_median = float(
            np.median(face_ratio_values)
        )

        confidence_median = float(
            np.median(confidence_values)
        )

        print()
        print(f"[{stage}]")
        print(
            f"Yaw median        : "
            f"{yaw_median:.4f}"
        )
        print(
            f"Pitch median      : "
            f"{pitch_median:.4f}"
        )
        print(
            f"Face ratio median : "
            f"{face_ratio_median:.4f}"
        )
        print(
            f"Confidence median : "
            f"{confidence_median:.4f}"
        )

else:
    print(
        "Calibration cancelled before completion."
    )