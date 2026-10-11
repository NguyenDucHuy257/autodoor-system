import cv2
import threading
import time
from app.performance import performance

class RTSPCamera:
    def __init__(
        self,
        url
    ):
        self.url = url
        self.cap = None
        self.lastest_frame = None
        self.thread = None
        self.lock = threading.Lock()

        self.frame_id = 0
        self.frame_timestamp = None
        self._last_consumed_id = None

    def start(self):
        
        with performance.measure("capture_open_ms"):
            self.cap = cv2.VideoCapture(self.url)
        performance.event("capture_open", opened=self.cap.isOpened())
        self.running = True

        self.thread = threading.Thread(
            target=self._capture_loop,
            daemon=True
        )

        self.thread.start()

    def _capture_loop(self):
        while self.running:

            performance.update(capture_read_started=time.monotonic())
            with performance.measure("capture_read_ms"):
                ret, frame = self.cap.read()
            performance.update(capture_read_started=None)
            if not ret or frame is None: #Tránh lấy frames cũ 
                with self.lock:
                    self.lastest_frame = None
                performance.count("capture_failures")
                continue

            with self.lock:
                self.lastest_frame = frame
                self.frame_id += 1
                self.frame_timestamp = time.monotonic()
            performance.count("captured_frames")
            performance.update(last_capture_at=self.frame_timestamp)

    def get_latest_frame(self, with_metadata=False):
        with self.lock:
            if self.lastest_frame is None:
                performance.count("capture_empty_reads")
                return None

            if self.frame_timestamp is None:
                return None

            age = time.monotonic() - self.frame_timestamp
            performance.observe("frame_age_ms", age * 1000)

            if age > 1.0:
                performance.count("stale_frame_rejections")
                return None

            if self._last_consumed_id is not None:
                if self.frame_id == self._last_consumed_id:
                    performance.count("duplicate_frame_reads")
                else:
                    performance.count("frames_dropped", max(0, self.frame_id - self._last_consumed_id - 1))
            self._last_consumed_id = self.frame_id
            frame = self.lastest_frame.copy()
            return (frame, self.frame_id, self.frame_timestamp) if with_metadata else frame

    def stop(self):
        started = time.perf_counter()
        self.running = False
        if self.thread is not None:
            self.thread.join(timeout=2)
        if self.cap is not None:
            self.cap.release()
        performance.observe("capture_stop_ms", (time.perf_counter() - started) * 1000)
        performance.event("capture_stop", thread_alive=self.thread.is_alive() if self.thread else False)



         
