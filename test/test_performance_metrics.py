"""Offline diagnostics tests: no .env, camera, models or database access."""
import json
import ast
import logging
from pathlib import Path
import tempfile
import time
import threading
import unittest
from unittest.mock import patch

import numpy as np

from app.performance import PerformanceMetrics
from app.camera.rtsp_camera import RTSPCamera


class PerformanceTests(unittest.TestCase):
    def test_bounded_samples_and_exact_counts(self):
        metrics = PerformanceMetrics()
        metrics.enabled = True
        for value in range(3000):
            metrics.observe("detection_ms", value)
        metrics.count("captured_frames", 50)
        report = metrics.snapshot(5)
        timing = report["timings"]["detection_ms"]
        self.assertEqual(timing["samples"], 2048)
        self.assertEqual(timing["calls"], 3000)
        self.assertEqual(timing["max"], 2999)
        self.assertEqual(report["rates"]["capture_fps"], 10)
        self.assertEqual(metrics.snapshot(5)["timings"], {})

    def test_wrapper_preserves_return_and_exception(self):
        metrics = PerformanceMetrics()
        metrics.enabled = True
        self.assertEqual(metrics.wrap("work_ms", lambda: 42)(), 42)
        def failing():
            raise ValueError("expected")
        with self.assertRaises(ValueError):
            metrics.wrap("work_ms", failing)()
        self.assertEqual(metrics.snapshot(5)["timings"]["work_ms"]["calls"], 2)

    def test_each_run_has_its_own_log_and_removes_handlers(self):
        metrics = PerformanceMetrics()
        logger = logging.getLogger("uvicorn.error")
        before = list(logger.handlers)
        with tempfile.TemporaryDirectory() as directory:
            paths = []
            for _ in range(2):
                metrics.start(directory, embedder="test")
                paths.append(metrics.log_path)
                record = logging.LogRecord("uvicorn.error", logging.WARNING, "", 0,
                                           "Invalid HTTP request received.", (), None)
                metrics.http_handler.emit(record)
                metrics.count("published_frames", 3)
                metrics.stop()
                records = [json.loads(line) for line in Path(paths[-1]).read_text(encoding="utf-8").splitlines()]
                self.assertEqual(records[0]["event"], "startup")
                self.assertEqual(records[-1]["event"], "shutdown")
                window = next(r for r in records if r["event"] == "window")
                self.assertEqual(window["counts"]["invalid_http_requests"], 1)
                self.assertEqual(window["counts"]["published_frames"], 3)
            self.assertNotEqual(paths[0], paths[1])
        self.assertEqual(logger.handlers, before)

    def test_frame_metadata_duplicates_drops_and_stale(self):
        metrics = PerformanceMetrics()
        metrics.enabled = True
        camera = RTSPCamera("unused")
        camera.lastest_frame = np.zeros((2, 2, 3), np.uint8)
        camera.frame_timestamp = time.monotonic()
        camera.frame_id = 1
        with patch("app.camera.rtsp_camera.performance", metrics):
            frame, frame_id, stamp = camera.get_latest_frame(with_metadata=True)
            self.assertEqual(frame_id, 1)
            self.assertEqual(stamp, camera.frame_timestamp)
            self.assertIsNot(frame, camera.lastest_frame)
            self.assertIsInstance(camera.get_latest_frame(), np.ndarray)
            camera.frame_id = 4
            camera.get_latest_frame()
            camera.frame_timestamp = time.monotonic() - 2
            self.assertIsNone(camera.get_latest_frame())
        counts = metrics.snapshot(5)["counts"]
        self.assertEqual(counts["duplicate_frame_reads"], 1)
        self.assertEqual(counts["frames_dropped"], 2)
        self.assertEqual(counts["stale_frame_rejections"], 1)

    def test_real_worker_loop_with_fake_camera_and_session(self):
        # Compile only Runtime; importing its module would load .env and DB modules.
        path = Path(__file__).resolve().parent / "web_remote" / "runtime.py"
        tree = ast.parse(path.read_text(encoding="utf-8"))
        node = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "Runtime")
        metrics = PerformanceMetrics()
        metrics.enabled = True
        class FakeCamera:
            source = "RTSP"
            channel = 1
            last_frame_timestamp = None
            def get_frame(self):
                self.last_frame_timestamp = time.monotonic()
                return np.zeros((16, 16, 3), np.uint8)
        class FakeSession:
            finished = False
            mode = "RECOGNITION"
            def process(self, frame):
                time.sleep(0.005)
                return frame
        import cv2
        namespace = dict(threading=threading, time=time, np=np, cv2=cv2,
                         CameraManager=FakeCamera, BaseSession=FakeSession,
                         AIComponents=object, performance=metrics,
                         MODE_IDLE="IDLE", MAX_LOOP_FPS=30,
                         PREVIEW_MAX_WIDTH=960, JPEG_QUALITY=70,
                         log=logging.getLogger("offline-worker-test"))
        exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), namespace)
        runtime = namespace["Runtime"]()
        runtime._session = FakeSession()
        original_publish = runtime._publish
        published = []
        def publish(frame):
            original_publish(frame)
            published.append(runtime._frame_id)
            if len(published) == 3:
                runtime._stop_event.set()
        runtime._publish = publish
        runtime._loop()
        report = metrics.snapshot(1)
        self.assertEqual(published, [1, 2, 3])
        self.assertEqual(report["counts"]["published_frames"], 3)
        self.assertEqual(report["timings"]["processing_ms"]["calls"], 3)
        self.assertEqual(report["timings"]["frame_age_at_publish_ms"]["calls"], 3)
        self.assertIsNone(metrics.state["processing_started"])


if __name__ == "__main__":
    unittest.main()
