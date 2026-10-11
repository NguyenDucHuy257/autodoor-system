"""
Test-only adapter for camera calibration math.

calibrate_camera.py is a monolithic script (it opens the webcam and runs its
main loop at import time), so it cannot be imported by the web tool.
The pure functions below mirror calibrate_camera.py 1:1 so the web page
produces the same thresholds. They are NOT AI algorithms from app/ (SCRFD and
FaceQualityGate are still imported from app/ and reused).

Keep in sync with calibrate_camera.py:
  * extract_pose_metrics
  * calculate_medians
  * calculate_thresholds
  * save_thresholds_to_env
"""

from pathlib import Path

import numpy as np
from dotenv import set_key


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
SAMPLE_INTERVAL_SECONDS = 0.05

# Only the keys the system currently reads (app/config.py). No new keys.
ENV_KEYS = (
    "YAW_THRESHOLD",
    "DOWN_THRESHOLD",
    "FAR_FACE_RATIO",
    "FACE_MIN_CONFIDENCE",
    "FACE_MIN_RATIO",
    "FACE_MIN_SHARPNESS",
    "FACE_MIN_BRIGHTNESS",
    "FACE_MAX_BRIGHTNESS",
)

METRIC_KEYS = ("yaw", "pitch", "face_ratio", "confidence", "sharpness", "brightness")


def extract_pose_metrics(face) -> dict | None:
    right_eye = face.landmarks[0]
    left_eye = face.landmarks[1]
    nose = face.landmarks[2]
    right_mouth = face.landmarks[3]
    left_mouth = face.landmarks[4]

    eye_center = (right_eye + left_eye) / 2
    mouth_center = (right_mouth + left_mouth) / 2

    eye_span = abs(left_eye[0] - right_eye[0])
    if eye_span < 1:
        return None

    yaw = (nose[0] - eye_center[0]) / eye_span

    vertical_span = eye_center[1] - mouth_center[1]
    if abs(vertical_span) < 1:
        return None

    pitch = (nose[1] - mouth_center[1]) / vertical_span

    return {
        "yaw": float(yaw),
        "pitch": float(pitch),
    }


def calculate_medians(all_samples: dict[str, list[dict]]) -> dict[str, dict[str, float]]:
    medians = {}
    for stage, samples in all_samples.items():
        medians[stage] = {
            key: float(np.median([sample[key] for sample in samples]))
            for key in METRIC_KEYS
        }
    return medians


def calculate_thresholds(
    medians: dict[str, dict[str, float]],
    all_samples: dict[str, list[dict]],
) -> dict[str, float]:

    front = medians["FRONT_NEAR"]
    left = medians["LEFT"]
    right = medians["RIGHT"]
    down = medians["DOWN"]
    far = medians["FAR_FRONT"]

    # YAW_THRESHOLD: midpoint between |front yaw| and mean |turn yaw|
    front_yaw_abs = abs(front["yaw"])
    average_turn_yaw = (abs(left["yaw"]) + abs(right["yaw"])) / 2
    yaw_threshold = (front_yaw_abs + average_turn_yaw) / 2

    # DOWN_THRESHOLD: midpoint FRONT <-> DOWN pitch
    down_threshold = (front["pitch"] + down["pitch"]) / 2

    # FAR_FACE_RATIO: midpoint NEAR <-> FAR
    if far["face_ratio"] >= front["face_ratio"]:
        raise ValueError(
            "Calibration FAR không hợp lệ: face_ratio FAR phải nhỏ hơn FRONT_NEAR."
        )

    far_face_ratio = (front["face_ratio"] + far["face_ratio"]) / 2

    # quality thresholds
    all_quality_samples = []
    for samples in all_samples.values():
        all_quality_samples.extend(samples)

    confidences = np.array([s["confidence"] for s in all_quality_samples])
    sharpnesses = np.array([s["sharpness"] for s in all_quality_samples])
    brightnesses = np.array([s["brightness"] for s in all_quality_samples])
    far_face_ratios = np.array([s["face_ratio"] for s in all_samples["FAR_FRONT"]])

    min_confidence = float(np.percentile(confidences, 5) - 0.02)
    min_confidence = max(0.0, min(1.0, min_confidence))

    min_face_ratio = float(np.percentile(far_face_ratios, 5) * 0.8)

    min_sharpness = float(np.percentile(sharpnesses, 5) * 0.8)

    min_brightness = float(max(0, np.percentile(brightnesses, 5) - 30))
    max_brightness = float(min(255, np.percentile(brightnesses, 95) + 30))

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


# ----------------------------------------------------------------------
# RELAXED profile (web tool only, NOT in calibrate_camera.py)
#
# Outdoor cameras: more lag, smaller and noisier faces than during
# calibration. The strict formula is derived from calibration frames where
# the person stands still, so enrollment often fails on sharpness /
# confidence. The relaxed profile widens the quality gate only;
# pose thresholds (YAW / DOWN / FAR) are midpoints and stay unchanged.
# ----------------------------------------------------------------------

RELAX_CONFIDENCE_DELTA = 0.06      # min confidence lowered by this
RELAX_SHARPNESS_FACTOR = 0.5       # min sharpness multiplied by this
RELAX_FACE_RATIO_FACTOR = 0.6      # min face ratio multiplied by this
RELAX_BRIGHTNESS_MARGIN = 20.0     # brightness window widened both sides
RELAX_CONFIDENCE_FLOOR = 0.5


def relax_thresholds(strict: dict[str, float]) -> dict[str, float]:
    relaxed = dict(strict)
    relaxed["FACE_MIN_CONFIDENCE"] = max(
        RELAX_CONFIDENCE_FLOOR,
        min(1.0, strict["FACE_MIN_CONFIDENCE"] - RELAX_CONFIDENCE_DELTA),
    )
    relaxed["FACE_MIN_SHARPNESS"] = strict["FACE_MIN_SHARPNESS"] * RELAX_SHARPNESS_FACTOR
    relaxed["FACE_MIN_RATIO"] = strict["FACE_MIN_RATIO"] * RELAX_FACE_RATIO_FACTOR
    relaxed["FACE_MIN_BRIGHTNESS"] = max(0.0, strict["FACE_MIN_BRIGHTNESS"] - RELAX_BRIGHTNESS_MARGIN)
    relaxed["FACE_MAX_BRIGHTNESS"] = min(255.0, strict["FACE_MAX_BRIGHTNESS"] + RELAX_BRIGHTNESS_MARGIN)
    return relaxed


def validate_threshold_values(values: dict) -> dict[str, float]:
    """Accept only the known keys; every value must be a finite float."""
    cleaned = {}
    for key, raw in values.items():
        if key not in ENV_KEYS:
            raise ValueError("Unknown key: %s" % key)
        try:
            value = float(raw)
        except (TypeError, ValueError):
            raise ValueError("%s is not a number: %r" % (key, raw))
        if not np.isfinite(value):
            raise ValueError("%s is not finite" % key)
        cleaned[key] = value
    if cleaned.get("FACE_MIN_BRIGHTNESS", 0) >= cleaned.get("FACE_MAX_BRIGHTNESS", 255):
        raise ValueError("FACE_MIN_BRIGHTNESS must be < FACE_MAX_BRIGHTNESS")
    return cleaned


def save_thresholds_to_env(env_path: Path, thresholds: dict[str, float]) -> list[str]:
    """
    Update ONLY the calibration keys in .env via python-dotenv set_key.
    Every other key (DB_*, CAMERA_*) is left untouched.
    """
    if not env_path.exists():
        env_path.touch()

    written = []
    for key in ENV_KEYS:
        if key not in thresholds:
            continue
        set_key(str(env_path), key, "%.4f" % thresholds[key])
        written.append(key)

    return written
