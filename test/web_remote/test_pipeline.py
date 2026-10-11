"""Offline concurrency/regression checks; never opens a physical camera or DB."""
import ast
from collections import Counter
from pathlib import Path
import tempfile
import threading
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import cv2
import numpy as np

UI = Path(__file__).resolve().parent

class Metrics:
    def __init__(self): self.counts = Counter()
    def count(self, name, amount=1): self.counts[name] += amount
    def observe(self, *args): pass
    def event(self, *args, **kwargs): pass
    def update(self, **kwargs): pass
    def measure(self, *args): return __import__("contextlib").nullcontext()


def runtime_types():
    tree = ast.parse((UI/"runtime.py").read_text(encoding="utf-8"))
    tree.body = [node for node in tree.body if isinstance(node, ast.ClassDef) and node.name in ("BaseSession", "BusyError", "Runtime")]
    from test.web_remote.recorder import EventRecorder
    metrics = Metrics()
    env = dict(threading=threading, time=time, cv2=cv2, np=np, CameraManager=lambda: SimpleNamespace(current_generation=lambda: 1),
               EventRecorder=EventRecorder, performance=metrics, log=__import__("logging").getLogger("test"),
               MODE_IDLE="IDLE", MODE_RECOGNITION="RECOGNITION", MAX_LOOP_FPS=30,
               PREVIEW_MAX_WIDTH=960, JPEG_QUALITY=70)
    exec(compile(tree, str(UI/"runtime.py"), "exec"), env)
    return env["Runtime"], env["BaseSession"], metrics


class PipelineTests(unittest.TestCase):
    def setUp(self):
        cls, base, self.metrics = runtime_types()
        self.runtime = cls()
        self.session = base()
        self.runtime._session = self.session
        self.packet = SimpleNamespace(frame=np.full((40, 60, 3), 75, np.uint8), generation=1,
                                      frame_id=1, timestamp=time.monotonic(), channel=1)

    def test_result_uses_exact_source_frame(self):
        self.runtime._publish(self.packet.frame, "ai", self.packet, self.session)
        jpeg, _ = self.runtime.wait_for_frame(-1, kind="ai")
        decoded = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
        self.assertLess(abs(float(decoded.mean())-75), 2)
        self.assertIsNone(self.runtime._jpeg)

    def test_old_camera_result_rejected(self):
        self.packet.generation = 0
        self.runtime._publish(self.packet.frame, "ai", self.packet, self.session)
        self.assertIsNone(self.runtime._ai_jpeg)

    def test_old_session_result_rejected(self):
        self.runtime._session = None
        self.runtime._publish(self.packet.frame, "ai", self.packet, self.session)
        self.assertIsNone(self.runtime._ai_jpeg)

    def test_stale_result_rejected(self):
        self.packet.timestamp -= 1.1
        self.runtime._publish(self.packet.frame, "ai", self.packet, self.session)
        self.assertIsNone(self.runtime._ai_jpeg)

    def test_clear_invalidates_pending_jobs(self):
        self.runtime._detect_job = self.runtime._reco_job = object()
        self.runtime._clear_ai()
        self.assertIsNone(self.runtime._detect_job)
        self.assertIsNone(self.runtime._reco_job)
        self.assertIsNone(self.runtime._ai_jpeg)

    def test_face_metadata_stays_with_published_frame(self):
        self.session.mode = "RECOGNITION"
        self.session.faces = [{"identity": {"name": "Person A"}}]
        self.runtime._publish(self.packet.frame, "ai", self.packet, self.session)
        self.session.faces = [{"identity": {"name": "Person B"}}]
        jpeg, frame_id, generation, faces = self.runtime.wait_for_delivery(-1, kind="ai")
        self.assertIsNotNone(jpeg)
        self.assertEqual(faces[0]["identity"]["name"], "Person A")
        self.runtime._clear_ai()
        self.assertEqual(self.runtime.wait_for_delivery(frame_id, kind="ai")[3], [])

    def test_crop_and_boxes_do_not_change_source(self):
        from test.web_remote.overlay import draw_faces, face_thumbnail
        import base64
        source = np.full((200, 200, 3), 75, np.uint8)
        box = dict(x=-5, y=20, width=165, height=170)
        thumbnail = face_thumbnail(source, box)
        crop = cv2.imdecode(np.frombuffer(base64.b64decode(thumbnail.split(",")[1]), np.uint8), cv2.IMREAD_COLOR)
        self.assertLessEqual(max(crop.shape[:2]), 128)
        faces = [{"bbox": dict(x=40,y=40,width=30,height=30), "verdict": "RECOGNIZED"},
                 {"bbox": dict(x=100,y=100,width=30,height=30), "verdict": "UNKNOWN"}]
        output = draw_faces(source, faces)
        self.assertTrue(np.array_equal(output[40,40], [40,220,40]))
        self.assertTrue(np.array_equal(output[100,100], [30,30,240]))
        self.assertTrue(np.all(source == 75))
        self.assertTrue(np.all(output[:20] == 75))

    def test_preview_runs_while_processing_blocked(self):
        self.session.mode = "ENROLLMENT"
        entered, release = threading.Event(), threading.Event()
        self.session.process = lambda frame: (entered.set(), release.wait(2), frame)[2]
        self.runtime.camera.packet = lambda: self.packet
        self.runtime._placeholder = lambda: self.packet.frame
        workers = [threading.Thread(target=self.runtime._loop), threading.Thread(target=self.runtime._detect_loop)]
        for worker in workers: worker.start()
        try:
            self.assertTrue(entered.wait(1))
            previous = self.runtime._frame_id
            self.packet = SimpleNamespace(**{**vars(self.packet), "frame_id": 2, "timestamp": time.monotonic()})
            deadline = time.monotonic()+1
            while self.runtime._frame_id == previous and time.monotonic()<deadline: time.sleep(.01)
            self.assertGreater(self.runtime._frame_id, previous)
        finally:
            self.runtime._stop_event.set()
            release.set()
            with self.runtime._jobs: self.runtime._jobs.notify_all()
            for worker in workers: worker.join(2)

    def test_gallery_rejects_different_model(self):
        from test.web_remote.adaface_engine import DemoGallery
        with tempfile.TemporaryDirectory() as folder:
            vectors = {str(i): np.ones(512, np.float32) for i in range(5)}
            gallery = DemoGallery("model-a", folder)
            gallery.save("Nguyen", "101", vectors)
            people, matrix = gallery.load()
            self.assertEqual(len(people), 5)
            self.assertTrue(np.allclose(np.linalg.norm(matrix, axis=1), 1))
            self.assertEqual(len(DemoGallery("model-b", folder).load()[0]), 0)
            vectors["0"][:] = 0
            with self.assertRaises(ValueError): gallery.save("Bad", None, vectors)

    def test_recorder_closes_after_three_seconds_and_switch(self):
        from test.web_remote.recorder import EventRecorder
        class Writer:
            closed = False
            def isOpened(self): return True
            def write(self, frame): pass
            def release(self): self.closed = True
        with tempfile.TemporaryDirectory() as folder, patch("test.web_remote.recorder.cv2.VideoWriter", side_effect=lambda *a: Writer()):
            recorder = EventRecorder(folder)
            token = (1, 123)
            recorder._consume(self.packet, token, self.packet.timestamp)
            writer = recorder.writer
            self.assertIsNotNone(writer)
            late = SimpleNamespace(**{**vars(self.packet), "timestamp": self.packet.timestamp+3.1})
            recorder._consume(late, token, self.packet.timestamp)
            self.assertTrue(writer.closed)
            self.assertIsNone(recorder.writer)
            self.assertEqual(len(list(Path(folder).glob("*.json"))), 1)
            recorder._consume(late, (2, 123), late.timestamp)
            writer = recorder.writer
            recorder._consume(late, None, None)
            self.assertTrue(writer.closed)

    def test_recorder_writes_processed_frame_and_unknown_crop(self):
        from test.web_remote.recorder import EventRecorder
        from test.web_remote.overlay import face_thumbnail
        raw = np.zeros((216,384,3), np.uint8)
        display = np.full_like(raw, 75)
        packet = SimpleNamespace(frame=raw, timestamp=time.monotonic(), generation=1, channel=1)
        face = dict(verdict="UNKNOWN", crop=face_thumbnail(raw, dict(x=20,y=20,width=50,height=50)))
        class Writer:
            def __init__(self): self.frames=[]
            def isOpened(self): return True
            def write(self, frame): self.frames.append(frame.copy())
            def release(self): pass
        with tempfile.TemporaryDirectory() as folder, patch("test.web_remote.recorder.cv2.VideoWriter", side_effect=lambda *a: Writer()):
            recorder = EventRecorder(folder)
            recorder.submit(packet, (1,123), packet.timestamp, display, [face])
            recorder._consume(*recorder.queue.get_nowait())
            saved = recorder.writer.frames[0]
            self.assertTrue(np.all(saved[:40] == 75))
            self.assertTrue(np.all(saved[100,320] == 0))
            self.assertTrue(np.all(raw == 0))
            recorder._close()
            self.assertIsNone(recorder.writer)
            self.assertEqual(len(list(Path(folder).glob("*.json"))), 1)

    def test_camera_switch_returns_without_open_and_drops_old_frame(self):
        tree = ast.parse((UI/"camera_manager.py").read_text(encoding="utf-8"))
        tree.body = [n for n in tree.body if not isinstance(n, (ast.Import, ast.ImportFrom))]
        from dataclasses import dataclass
        env = dict(dataclass=dataclass, threading=threading, time=time, cv2=cv2, performance=self.metrics)
        exec(compile(tree, "camera_manager", "exec"), env)
        camera = env["CameraManager"]()
        camera.start = lambda: None
        with patch.object(cv2, "VideoCapture") as capture:
            self.assertTrue(camera.use_rtsp(3)[0])
            self.assertTrue(camera.use_rtsp(7)[0])
            capture.assert_not_called()
        self.assertEqual(camera.describe()["channel"], 7)
        camera._packet = env["FramePacket"](self.packet.frame, camera.generation-1, 1, time.monotonic(), "RTSP", 3)
        self.assertIsNone(camera.packet())

if __name__ == "__main__": unittest.main()
