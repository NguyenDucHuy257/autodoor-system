"""Test-only camera capture. One owner thread, bounded latest-frame mailbox."""
from dataclasses import dataclass
import threading
import time
import cv2
from app.performance import performance
from app.config import CAMRERA_IP, CAMERA_USERNAME, CAMERA_PASSWORD, CAMERA_RTSP_PORT

MIN_CHANNEL, MAX_CHANNEL = 1, 20
SOURCE_NONE, SOURCE_LAPTOP, SOURCE_RTSP = "NONE", "LAPTOP", "RTSP"


def build_rtsp_url(channel):
    return (f"rtsp://{CAMERA_USERNAME}:{CAMERA_PASSWORD}@{CAMRERA_IP}:{CAMERA_RTSP_PORT}"
            f"/cam/realmonitor?channel={channel}&subtype=0")


@dataclass(frozen=True)
class FramePacket:
    frame: object
    generation: int
    frame_id: int
    timestamp: float
    source: str
    channel: int | None


class CameraManager:
    def __init__(self):
        self._lock = threading.Lock()
        self._wake = threading.Event()
        self._stop = threading.Event()
        self._thread = None
        self._desired = (SOURCE_NONE, None)
        self.generation = 0
        self.source, self.channel = SOURCE_NONE, None
        self.last_error = None
        self.switching = False
        self._packet = None
        self.last_frame_timestamp = None
        self._frame_id = 0

    def start(self):
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._capture, name="ui-camera-owner", daemon=True)
        self._thread.start()

    def _request(self, source, channel):
        self.start()
        with self._lock:
            self.generation += 1
            self._desired = (source, channel)
            self._packet = None
            self.source, self.channel = source, channel
            self.switching = source != SOURCE_NONE
            self.last_error = None
        self._wake.set()
        performance.event("camera_switch_requested", source=source, channel=channel,
                          generation=self.generation)
        return True, "Camera switch queued"

    def use_laptop(self):
        return self._request(SOURCE_LAPTOP, None)

    def use_rtsp(self, channel):
        if not MIN_CHANNEL <= channel <= MAX_CHANNEL:
            return False, "Invalid channel"
        return self._request(SOURCE_RTSP, channel)

    def stop(self):
        return self._request(SOURCE_NONE, None)

    def shutdown(self):
        self._stop.set()
        self._wake.set()
        if self._thread:
            self._thread.join(timeout=6)
        if self._thread and self._thread.is_alive():
            performance.event("capture_shutdown_timeout")

    def current_generation(self):
        with self._lock:
            return self.generation

    def packet(self):
        with self._lock:
            packet = self._packet
            if packet is None or packet.generation != self.generation:
                return None
            if time.monotonic() - packet.timestamp > 1.0:
                performance.count("stale_frame_rejections")
                return None
            return packet

    def get_frame(self):
        packet = self.packet()
        self.last_frame_timestamp = packet.timestamp if packet else None
        return packet.frame.copy() if packet else None

    def describe(self):
        with self._lock:
            return dict(source=self.source, channel=self.channel, error=self.last_error,
                        switching=self.switching, generation=self.generation)

    def _capture(self):
        cap = None
        opened_generation = -1
        retry_at = 0.0
        failures = 0
        try:
            while not self._stop.is_set():
                with self._lock:
                    generation = self.generation
                    source, channel = self._desired
                if generation != opened_generation:
                    if cap is not None:
                        cap.release()
                    cap = None
                    opened_generation = generation
                    retry_at = 0.0
                    failures = 0
                if source == SOURCE_NONE:
                    self._wake.wait(0.1)
                    self._wake.clear()
                    continue
                if cap is None:
                    if time.monotonic() < retry_at:
                        self._stop.wait(0.1)
                        continue
                    performance.event("camera_switch_begin", source=source, channel=channel,
                                      generation=generation)
                    with performance.measure("capture_open_ms"):
                        if source == SOURCE_RTSP:
                            cap = cv2.VideoCapture(build_rtsp_url(channel), cv2.CAP_FFMPEG,
                                [cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 3000,
                                 cv2.CAP_PROP_READ_TIMEOUT_MSEC, 1000])
                        else:
                            cap = cv2.VideoCapture(0)
                    if not cap.isOpened():
                        cap.release()
                        cap = None
                        with self._lock:
                            if generation == self.generation:
                                self.last_error = "Cannot open camera; retrying"
                                self.switching = False
                        performance.event("camera_open_failed", generation=generation)
                        retry_at = time.monotonic() + 2
                        continue
                    performance.event("camera_switch_end", source=source, channel=channel,
                                      generation=generation, opened=True)
                performance.update(capture_read_started=time.monotonic())
                with performance.measure("capture_read_ms"):
                    ok, frame = cap.read()
                performance.update(capture_read_started=None)
                if not ok or frame is None:
                    failures += 1
                    performance.count("capture_failures")
                    with self._lock:
                        if generation == self.generation:
                            self._packet = None
                            self.last_error = "Camera read failed; reconnecting"
                    if failures >= 3:
                        cap.release()
                        cap = None
                        retry_at = time.monotonic() + 1
                    self._stop.wait(0.05)
                    continue
                failures = 0
                if source == SOURCE_LAPTOP:
                    frame = cv2.flip(frame, 1)
                stamp = time.monotonic()
                with self._lock:
                    if generation != self.generation or self._stop.is_set():
                        continue
                    self._frame_id += 1
                    self._packet = FramePacket(frame, generation, self._frame_id, stamp, source, channel)
                    self.switching = False
                    self.last_error = None
                performance.count("captured_frames")
                performance.update(last_capture_at=stamp, source=source, channel=channel)
        finally:
            if cap is not None:
                cap.release()
            with self._lock:
                self._packet = None
