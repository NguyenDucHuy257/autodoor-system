"""Bounded recording of completed recognition frames, including face thumbnails."""
from dataclasses import dataclass
from collections import deque
from datetime import datetime
import json
from pathlib import Path
import queue
import threading
import time
import cv2
from app.performance import performance

RESULTS_DIR = Path(__file__).resolve().parent / "results"


@dataclass(frozen=True)
class RecordingPacket:
    frame: object
    timestamp: float
    generation: int
    channel: int | None


class EventRecorder:
    def __init__(self, directory=RESULTS_DIR):
        self.directory = Path(directory)
        self.queue = queue.Queue(maxsize=60)
        self.stop_event = threading.Event()
        self.thread = None
        self.error = None
        self.path = None
        self.writer = None
        self.token = None
        self.buffer = deque(maxlen=15)
        self.started_at = None
        self.last_presence = None
        self.frames = 0
        self.written = 0
        self.fps = 15

    def start(self):
        self.stop_event.clear()
        self.thread = threading.Thread(target=self._loop, name="ui-video-recorder", daemon=True)
        self.thread.start()

    def submit(self, packet, token, presence, display=None, faces=None):
        if display is not None:
            packet = RecordingPacket(display, packet.timestamp, packet.generation, packet.channel)
        item = (packet, token, presence, faces or [])
        try:
            self.queue.put_nowait(item)
        except queue.Full:
            try:
                self.queue.get_nowait()
            except queue.Empty:
                pass
            performance.count("recording_frames_dropped")
            try:
                self.queue.put_nowait(item)
            except queue.Full:
                performance.count("recording_frames_dropped")

    def stop(self):
        self.stop_event.set()
        if self.thread:
            self.thread.join(timeout=5)

    def _close(self):
        writer, path = self.writer, self.path
        self.writer = None
        self.path = None
        try:
            if writer:
                writer.release()
                path.with_suffix(".json").write_text(json.dumps({
                    "file": path.name, "frames": self.written, "fps": self.fps,
                    "source_generation": self.token[0], "start_monotonic": self.started_at,
                    "encoding": "recognition frames with boxes and face thumbnails, mp4v, no audio",
                    "unique_processed_frames": self.frames,
                }, indent=2), encoding="utf-8")
                performance.event("recording_closed", file=path.name, frames=self.written)
        except Exception as error:
            self.error = type(error).__name__ + ": recording finalization failed"
            performance.event("recording_error", error=type(error).__name__)
        finally:
            self.started_at = None
            self.frames = self.written = 0

    def _write(self, packet, faces=None):
        # Timestamp-driven CFR: duplicate only to represent elapsed capture time.
        target = int((packet.timestamp-self.started_at)*self.fps)+1
        repeats = max(0, min(target-self.written, self.fps))
        if repeats:
            from test.web_remote.overlay import recording_frame
            display = recording_frame(packet.frame, faces or [])
        for _ in range(repeats):
            self.writer.write(display)
            self.written += 1
        self.frames += 1

    def _consume(self, packet, token, presence, faces=None):
        if token != self.token:
            self._close()
            self.buffer.clear()
            self.token = token
            self.last_presence = None
        if token is None:
            return
        self.buffer.append((packet, faces or []))
        if presence is not None:
            self.last_presence = presence
        if self.writer and (self.last_presence is None or packet.timestamp-self.last_presence>3):
            self._close()
        if not self.writer and presence is not None and packet.timestamp-presence <= 3:
            self.directory.mkdir(parents=True, exist_ok=True)
            self.path = self.directory / (datetime.now().strftime("%Y%m%d_%H%M%S_%f") + f"_cam{packet.channel or 'laptop'}.mp4")
            h, w = packet.frame.shape[:2]
            self.writer = cv2.VideoWriter(str(self.path), cv2.VideoWriter_fourcc(*"mp4v"), self.fps, (w, h))
            if not self.writer.isOpened():
                self.writer.release()
                self.writer = None
                self.error = "Video codec unavailable; recording could not start"
                return
            self.error = None
            self.started_at = self.buffer[0][0].timestamp
            for buffered, buffered_faces in self.buffer:
                self._write(buffered, buffered_faces)
            self.buffer.clear()
            performance.event("recording_started", file=self.path.name)
        elif self.writer:
            self._write(packet, faces)

    def _loop(self):
        try:
            while not self.stop_event.is_set():
                try:
                    item = self.queue.get(timeout=0.2)
                except queue.Empty:
                    if self.writer and time.monotonic()-self.last_presence > 3:
                        self._close()
                    continue
                try:
                    with performance.measure("recording_ms"):
                        self._consume(*item)
                except Exception as error:
                    self.error = type(error).__name__ + ": recording failed"
                    performance.event("recording_error", error=type(error).__name__)
                    self._close()
        finally:
            self._close()
