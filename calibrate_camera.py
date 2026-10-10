import cv2
import time
import numpy as np

from pathlib import Path
from dotenv import set_key

from app.recognition.yunet_detector import YuNetDetector
from app.recognition.face_quality import FaceQualityGate


# =========================================================
# CONFIG
# =========================================================

ROOT_DIR = Path(__file__).resolve().parent

ENV_PATH = ROOT_DIR / ".env"

MODEL_PATH = (
    ROOT_DIR
    / "models"
    / "face_detection_yunet_2023mar.onnx"
)

STAGES = (
    "FRONT_NEAR",
    "LEFT",
    "RIGHT",
    "DOWN",
    "FAR_FRONT",
)

STAGE_GUIDES = {
    "FRONT_NEAR": "Stand near - Look straight",
    "LEFT": "Turn LEFT",
    "RIGHT": "Turn RIGHT",
    "DOWN": "Look DOWN",
    "FAR_FRONT": "Step back - Look straight",
}

PREPARE_SECONDS = 5

SAMPLES_PER_STAGE = 30

# Tránh lấy 30 frame gần như y hệt nhau quá nhanh
SAMPLE_INTERVAL_SECONDS = 0.05


# =========================================================
# DEPENDENCIES
# =========================================================

detector = YuNetDetector(
    model_path=str(MODEL_PATH)
)

# Calibration gate:
# threshold cực thoáng để tận dụng chính FaceQualityGate
# chỉ nhằm LẤY METRIC, không dùng passed để reject.
quality_gate = FaceQualityGate(
    min_confidence=0.0,
    min_face_ratio=0.0,
    min_sharpness=0.0,
    min_brightness=0.0,
    max_brightness=255.0,
)


# =========================================================
# POSE METRICS
# Hiện EnrollmentService chưa expose yaw/pitch thô,
# nên calibration vẫn phải lấy 2 metric này.
# =========================================================

def extract_pose_metrics(face) -> dict | None:

    right_eye = face.landmarks[0]
    left_eye = face.landmarks[1]

    nose = face.landmarks[2]

    right_mouth = face.landmarks[3]
    left_mouth = face.landmarks[4]

    eye_center = (
        right_eye + left_eye
    ) / 2

    mouth_center = (
        right_mouth + left_mouth
    ) / 2

    # -------------------------
    # YAW
    # -------------------------

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

    vertical_span = (
        eye_center[1]
        - mouth_center[1]
    )

    if abs(vertical_span) < 1:
        return None

    pitch = (
        nose[1] - mouth_center[1]
    ) / vertical_span

    return {
        "yaw": float(yaw),
        "pitch": float(pitch),
    }


# =========================================================
# STATISTICS
# =========================================================

def calculate_medians(
    all_samples: dict[str, list[dict]]
) -> dict[str, dict[str, float]]:

    medians = {}

    for stage, samples in all_samples.items():

        medians[stage] = {
            "yaw": float(
                np.median([
                    sample["yaw"]
                    for sample in samples
                ])
            ),
            "pitch": float(
                np.median([
                    sample["pitch"]
                    for sample in samples
                ])
            ),
            "face_ratio": float(
                np.median([
                    sample["face_ratio"]
                    for sample in samples
                ])
            ),
            "confidence": float(
                np.median([
                    sample["confidence"]
                    for sample in samples
                ])
            ),
            "sharpness": float(
                np.median([
                    sample["sharpness"]
                    for sample in samples
                ])
            ),
            "brightness": float(
                np.median([
                    sample["brightness"]
                    for sample in samples
                ])
            ),
        }

    return medians


# =========================================================
# THRESHOLD CALCULATOR
# =========================================================

def calculate_thresholds(
    medians: dict[str, dict[str, float]],
    all_samples: dict[str, list[dict]],
) -> dict[str, float]:

    front = medians["FRONT_NEAR"]
    left = medians["LEFT"]
    right = medians["RIGHT"]
    down = medians["DOWN"]
    far = medians["FAR_FRONT"]

    # -----------------------------------------------------
    # 1. YAW_THRESHOLD
    #
    # Runtime hiện tại dùng:
    #
    # yaw <= -threshold -> LEFT
    # yaw >= +threshold -> RIGHT
    #
    # Vì người thật có thể quay trái/phải không cân nhau,
    # lấy độ quay trung bình hai phía rồi lấy midpoint
    # giữa FRONT và TURN.
    # -----------------------------------------------------

    front_yaw_abs = abs(
        front["yaw"]
    )

    average_turn_yaw = (
        abs(left["yaw"])
        + abs(right["yaw"])
    ) / 2

    yaw_threshold = (
        front_yaw_abs
        + average_turn_yaw
    ) / 2

    # -----------------------------------------------------
    # 2. DOWN_THRESHOLD
    #
    # Runtime:
    #
    # pitch < threshold -> DOWN
    #
    # lấy midpoint FRONT <-> DOWN
    # -----------------------------------------------------

    # if down["pitch"] >= front["pitch"]:
    #     raise ValueError(
    #         "Calibration DOWN không hợp lệ: "
    #         "pitch DOWN phải nhỏ hơn pitch FRONT."
    #     )

    down_threshold = (
        front["pitch"]
        + down["pitch"]
    ) / 2

    # -----------------------------------------------------
    # 3. FAR_FACE_RATIO
    #
    # Runtime:
    #
    # face_ratio < threshold -> FAR_FRONT
    #
    # midpoint NEAR <-> FAR
    # -----------------------------------------------------

    if far["face_ratio"] >= front["face_ratio"]:
        raise ValueError(
            "Calibration FAR không hợp lệ: "
            "face_ratio FAR phải nhỏ hơn FRONT_NEAR."
        )

    far_face_ratio = (
        front["face_ratio"]
        + far["face_ratio"]
    ) / 2

    # =====================================================
    # QUALITY THRESHOLDS
    # =====================================================

    all_quality_samples = []

    for samples in all_samples.values():
        all_quality_samples.extend(samples)

    confidences = np.array([
        s["confidence"]
        for s in all_quality_samples
    ])

    sharpnesses = np.array([
        s["sharpness"]
        for s in all_quality_samples
    ])

    brightnesses = np.array([
        s["brightness"]
        for s in all_quality_samples
    ])

    far_face_ratios = np.array([
        s["face_ratio"]
        for s in all_samples["FAR_FRONT"]
    ])

    # -----------------------------------------------------
    # Confidence
    #
    # lấy vùng thấp của những frame calibration tốt
    # rồi chừa thêm margin nhỏ.
    # -----------------------------------------------------

    min_confidence = float(
        np.percentile(
            confidences,
            5
        )
        - 0.02
    )

    min_confidence = max(
        0.0,
        min(
            1.0,
            min_confidence
        )
    )

    # -----------------------------------------------------
    # Minimum face ratio
    #
    # FAR_FRONT vẫn là một ảnh hợp lệ.
    # Cho phép nhỏ hơn FAR calibration khoảng 20%.
    # -----------------------------------------------------

    min_face_ratio = float(
        np.percentile(
            far_face_ratios,
            5
        )
        * 0.8
    )

    # -----------------------------------------------------
    # Sharpness
    #
    # Lấy percentile thấp rồi nới thêm.
    # -----------------------------------------------------

    min_sharpness = float(
        np.percentile(
            sharpnesses,
            5
        )
        * 0.8
    )

    # -----------------------------------------------------
    # Brightness
    #
    # Vì calibration hiện mới dùng một điều kiện ánh sáng,
    # mở rộng range thêm 30 mức gray.
    # -----------------------------------------------------

    min_brightness = float(
        max(
            0,
            np.percentile(
                brightnesses,
                5
            ) - 30
        )
    )

    max_brightness = float(
        min(
            255,
            np.percentile(
                brightnesses,
                95
            ) + 30
        )
    )

    return {
        "YAW_THRESHOLD": yaw_threshold,
        "DOWN_THRESHOLD": down_threshold,
        "FAR_FACE_RATIO": far_face_ratio,

        "FACE_MIN_CONFIDENCE": min_confidence,
        "FACE_MIN_RATIO": min_face_ratio,
        "FACE_MIN_SHARPNESS": min_sharpness,
        "FACE_MIN_BRIGHTNESS": min_brightness,
        "FACE_MAX_BRIGHTNESS": max_brightness,
    }


# =========================================================
# SAVE .ENV
# =========================================================

def save_thresholds_to_env(
    thresholds: dict[str, float]
) -> None:

    if not ENV_PATH.exists():
        ENV_PATH.touch()

    for key, value in thresholds.items():

        set_key(
            str(ENV_PATH),
            key,
            f"{value:.4f}"
        )


# =========================================================
# CAMERA
# =========================================================

camera = cv2.VideoCapture(0)

if not camera.isOpened():
    raise RuntimeError(
        "Cannot open camera."
    )


# =========================================================
# STATE
# =========================================================

stage_index = 0

phase = "WAIT_ENTER"

phase_started_at = None

last_sample_at = 0.0

stage_samples = []

all_samples = {
    stage: []
    for stage in STAGES
}

calibration_complete = False


# =========================================================
# MAIN LOOP
# =========================================================

try:

    while True:

        ret, frame = camera.read()

        if not ret or frame is None:
            continue

        # Giữ cùng hướng với Enrollment hiện tại
        frame = cv2.flip(
            frame,
            1
        )

        display_frame = frame.copy()

        current_stage = STAGES[
            stage_index
        ]

        faces = detector.detect(
            frame
        )

        # =================================================
        # FACE PREVIEW
        # =================================================

        if len(faces) == 1:

            face = faces[0]

            cv2.rectangle(
                display_frame,
                (
                    face.x,
                    face.y
                ),
                (
                    face.x + face.width,
                    face.y + face.height
                ),
                (0, 255, 0),
                2
            )

            cv2.putText(
                display_frame,
                (
                    f"Detect: "
                    f"{face.confidence * 100:.1f}%"
                ),
                (
                    face.x,
                    max(
                        face.y - 10,
                        20
                    )
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )

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

        # =================================================
        # WAIT ENTER
        # =================================================

        if phase == "WAIT_ENTER":

            cv2.putText(
                display_frame,
                STAGE_GUIDES[
                    current_stage
                ],
                (30, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
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

        # =================================================
        # PREPARE
        # =================================================

        elif phase == "PREPARE":

            elapsed = (
                time.monotonic()
                - phase_started_at
            )

            remaining = max(
                0,
                PREPARE_SECONDS
                - elapsed
            )

            cv2.putText(
                display_frame,
                STAGE_GUIDES[
                    current_stage
                ],
                (30, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 255),
                2
            )

            cv2.putText(
                display_frame,
                (
                    f"Ready in "
                    f"{remaining:.1f}s"
                ),
                (30, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 255),
                2
            )

            if remaining <= 0:

                phase = "CAPTURE"

                stage_samples = []

                last_sample_at = 0.0

        # =================================================
        # CAPTURE
        # =================================================

        elif phase == "CAPTURE":

            cv2.putText(
                display_frame,
                (
                    f"CAPTURING "
                    f"{current_stage}"
                ),
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

            if len(faces) == 1:

                now = time.monotonic()

                if (
                    now - last_sample_at
                    >= SAMPLE_INTERVAL_SECONDS
                ):

                    face = faces[0]

                    pose_metrics = (
                        extract_pose_metrics(
                            face
                        )
                    )

                    quality_result = (
                        quality_gate.evaluate(
                            frame=frame,
                            face=face
                        )
                    )

                    if (
                        pose_metrics is not None
                        and quality_result.face_ratio > 0
                    ):

                        sample = {
                            "yaw": (
                                pose_metrics["yaw"]
                            ),
                            "pitch": (
                                pose_metrics["pitch"]
                            ),
                            "face_ratio": (
                                quality_result.face_ratio
                            ),
                            "confidence": (
                                quality_result.confidence
                            ),
                            "sharpness": (
                                quality_result.sharpness
                            ),
                            "brightness": (
                                quality_result.brightness
                            ),
                        }

                        stage_samples.append(
                            sample
                        )

                        last_sample_at = now

            # =============================================
            # STAGE COMPLETE
            # =============================================

            if (
                len(stage_samples)
                >= SAMPLES_PER_STAGE
            ):

                all_samples[
                    current_stage
                ] = stage_samples.copy()

                print(
                    f"{current_stage}: "
                    f"{len(stage_samples)} samples"
                )

                stage_index += 1

                if stage_index >= len(STAGES):

                    calibration_complete = True

                    break

                phase = "WAIT_ENTER"

                phase_started_at = None

                stage_samples = []

        # =================================================
        # DISPLAY
        # =================================================

        display_frame = cv2.resize(
            display_frame,
            (960, 540)
        )

        cv2.imshow(
            "CAMERA CALIBRATION",
            display_frame
        )

        key = (
            cv2.waitKey(1)
            & 0xFF
        )

        # ENTER
        if (
            key == 13
            and phase == "WAIT_ENTER"
        ):

            phase = "PREPARE"

            phase_started_at = (
                time.monotonic()
            )

        # Q
        if key == ord("q"):
            break


finally:

    camera.release()

    cv2.destroyAllWindows()


# =========================================================
# RESULT
# =========================================================

if not calibration_complete:

    print(
        "Calibration cancelled."
    )

    raise SystemExit


medians = calculate_medians(
    all_samples
)

thresholds = calculate_thresholds(
    medians=medians,
    all_samples=all_samples,
)


# =========================================================
# PRINT MEDIANS
# =========================================================

print()
print("=" * 60)
print("CALIBRATION MEDIANS")
print("=" * 60)

for stage in STAGES:

    data = medians[stage]

    print()
    print(f"[{stage}]")

    print(
        f"Yaw        : "
        f"{data['yaw']:.4f}"
    )

    print(
        f"Pitch      : "
        f"{data['pitch']:.4f}"
    )

    print(
        f"Face ratio : "
        f"{data['face_ratio']:.4f}"
    )

    print(
        f"Confidence : "
        f"{data['confidence']:.4f}"
    )

    print(
        f"Sharpness  : "
        f"{data['sharpness']:.2f}"
    )

    print(
        f"Brightness : "
        f"{data['brightness']:.2f}"
    )


# =========================================================
# PRINT THRESHOLDS
# =========================================================

print()
print("=" * 60)
print("GENERATED CAMERA SETTINGS")
print("=" * 60)

for key, value in thresholds.items():

    print(
        f"{key}={value:.4f}"
    )


# =========================================================
# WRITE .ENV
# =========================================================

save_thresholds_to_env(
    thresholds
)

print()
print(
    f"Updated .env: {ENV_PATH}"
)

print(
    "Restart Enrollment application "
    "to load the new values."
)