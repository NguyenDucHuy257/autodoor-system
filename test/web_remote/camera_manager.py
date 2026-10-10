"""
Camera source manager for the LAN web test tool.

TEST ONLY - lives in test/web_remote, never in app/camera.

Supports:
  * laptop webcam  -> cv2.VideoCapture(LAPTOP_CAMERA_INDEX)
  * RTSP channel   -> app.camera.rtsp_camera.RTSPCamera (reused, not copied)

Rules:
  * exactly one source open at a time
  * old source is stopped BEFORE the new one is started
  * laptop frames are mirrored, same as the existing test scripts
"""

import threading

import cv2

from app.camera.rtsp_camera import RTSPCamera
from app.config import (
    CAMRERA_IP,
    CAMERA_USERNAME,
    CAMERA_PASSWORD,
    CAMERA_RTSP_PORT,
)


LAPTOP_CAMERA_INDEX = 0
LAPTOP_MIRROR = True

MIN_CHANNEL = 1
MAX_CHANNEL = 20

SOURCE_NONE = "NONE"
SOURCE_LAPTOP = "LAPTOP"
SOURCE_RTSP = "RTSP"


def build_rtsp_url(channel: int) -> str:
    # Same format used by test/test_realtime.py and test/test_enrollment_service.py
    return (
        f"rtsp://{CAMERA_USERNAME}:{CAMERA_PASSWORD}"
        f"@{CAMRERA_IP}:{CAMERA_RTSP_PORT}"
        f"/cam/realmonitor?channel={channel}&subtype=0"
    )


class CameraManager:

    def __init__(self):
        self._lock = threading.RLock()

        self.source = SOURCE_NONE
        self.channel: int | None = None
        self.last_error: str | None = None

        self._laptop: cv2.VideoCapture | None = None
        self._rtsp: RTSPCamera | None = None

    # ------------------------------------------------------------------
    # lifecycle
    # ------------------------------------------------------------------

    def stop(self) -> None:
        with self._lock:
            if self._laptop is not None:
                try:
                    self._laptop.release()
                finally:
                    self._laptop = None

            if self._rtsp is not None:
                try:
                    self._rtsp.stop()
                finally:
                    self._rtsp = None

            self.source = SOURCE_NONE
            self.channel = None

    def use_laptop(self) -> tuple[bool, str]:
        with self._lock:
            # stop old source first
            self.stop()

            cap = cv2.VideoCapture(LAPTOP_CAMERA_INDEX)

            if not cap.isOpened():
                cap.release()
                self.last_error = "Cannot open laptop camera"
                return False, self.last_error

            self._laptop = cap
            self.source = SOURCE_LAPTOP
            self.channel = None
            self.last_error = None

            return True, "Laptop camera started"

    def use_rtsp(self, channel: int) -> tuple[bool, str]:
        if not (MIN_CHANNEL <= channel <= MAX_CHANNEL):
            return False, f"Channel must be {MIN_CHANNEL}..{MAX_CHANNEL}"

        with self._lock:
            # stop old source first - never more than one RTSPCamera alive
            self.stop()

            camera = RTSPCamera(url=build_rtsp_url(channel))
            camera.start()

            self._rtsp = camera
            self.source = SOURCE_RTSP
            self.channel = channel

            opened = camera.cap is not None and camera.cap.isOpened()

            if opened:
                self.last_error = None
                return True, f"RTSP channel {channel} started"

            # keep the camera object: RTSPCamera's loop keeps trying cap.read()
            self.last_error = f"RTSP channel {channel}: stream not opened (check IP / credentials / channel)"
            return True, self.last_error

    # ------------------------------------------------------------------
    # frames
    # ------------------------------------------------------------------

    def get_frame(self):
        with self._lock:
            if self.source == SOURCE_LAPTOP:
                if self._laptop is None:
                    return None

                ret, frame = self._laptop.read()

                if not ret or frame is None:
                    return None

                if LAPTOP_MIRROR:
                    frame = cv2.flip(frame, 1)

                return frame

            if self.source == SOURCE_RTSP:
                if self._rtsp is None:
                    return None

                return self._rtsp.get_latest_frame()

            return None

    # ------------------------------------------------------------------
    # info
    # ------------------------------------------------------------------

    def describe(self) -> dict:
        with self._lock:
            return {
                "source": self.source,
                "channel": self.channel,
                "error": self.last_error,
            }
