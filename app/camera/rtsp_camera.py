import cv2
import threading


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

    def start(self):
        
        self.cap = cv2.VideoCapture(self.url)
        self.running = True

        self.thread = threading.Thread(
            target=self._capture_loop,
            daemon=True
        )

        self.thread.start()

    def _capture_loop(self):
        while self.running:

            ret, frame = self.cap.read()
            if not ret:
                continue

            with self.lock:
                self.lastest_frame = frame

    def get_latest_frame(self):

        with self.lock:

            if self.lastest_frame is None:
                return None
            return self.lastest_frame.copy()

    def stop(self):
        self.running = False
        if self.thread is not None:
            self.thread.join(timeout=2)
        if self.cap is not None:
            self.cap.release()



         