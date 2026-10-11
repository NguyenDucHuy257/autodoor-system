"""Bounded, opt-in diagnostics. Never records images, identities or URLs."""

from collections import Counter, deque
from contextlib import contextmanager
from datetime import datetime
from functools import wraps
import json
import logging
import os
from pathlib import Path
import threading
import time


class _HttpWarningCounter(logging.Handler):
    def __init__(self, metrics):
        super().__init__(level=logging.WARNING)
        self.metrics = metrics

    def emit(self, record):
        if record.getMessage() == "Invalid HTTP request received.":
            self.metrics.count("invalid_http_requests")


class PerformanceMetrics:
    def __init__(self):
        self.enabled = False
        self.lock = threading.Lock()
        self.samples = {}
        self.counts = Counter()
        self.state = {}
        self.stop_event = threading.Event()
        self.thread = None
        self.handler = None
        self.logger = logging.getLogger("autodoor.performance")
        self.log_path = None
        self.http_handler = _HttpWarningCounter(self)

    def observe(self, name, value):
        if self.enabled:
            with self.lock:
                self.samples.setdefault(name, deque(maxlen=2048)).append(float(value))
                self.counts[name + "_calls"] += 1

    def count(self, name, value=1):
        if self.enabled:
            with self.lock:
                self.counts[name] += value

    def update(self, **values):
        if self.enabled:
            with self.lock:
                self.state.update(values)

    @contextmanager
    def measure(self, name):
        started = time.perf_counter()
        try:
            yield
        finally:
            self.observe(name, (time.perf_counter() - started) * 1000)

    def wrap(self, name, function):
        @wraps(function)
        def measured(*args, **kwargs):
            with self.measure(name):
                return function(*args, **kwargs)
        return measured

    def snapshot(self, elapsed):
        with self.lock:
            samples, counts, state = self.samples, self.counts, dict(self.state)
            self.samples, self.counts = {}, Counter()
        timings = {}
        for name, values in samples.items():
            ordered = sorted(values)
            position = (len(ordered) - 1) * 0.95
            lower = int(position)
            upper = min(lower + 1, len(ordered) - 1)
            p95 = ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)
            timings[name] = {
                "samples": len(ordered), "calls": counts.pop(name + "_calls", 0),
                "mean": round(sum(ordered) / len(ordered), 3),
                "p95": round(p95, 3), "max": round(ordered[-1], 3),
            }
        now = time.monotonic()
        for key in ("capture_read_started", "last_capture_at", "processing_started"):
            stamp = state.pop(key, None)
            state[key + "_age_ms"] = round((now - stamp) * 1000, 3) if stamp is not None else None
        rates = {name: round(counts.get(counter, 0) / max(elapsed, 0.001), 3)
                 for name, counter in (("capture_fps", "captured_frames"),
                                       ("publish_fps", "published_frames"),
                                       ("mjpeg_yield_fps_all_clients", "mjpeg_frames"))}
        return {"window_s": round(elapsed, 3), "state": state,
                "rates": rates, "counts": dict(counts), "timings": timings}

    def start(self, directory, **metadata):
        if self.thread is not None:
            return
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        self.log_path = directory / (datetime.now().strftime("performance_%Y%m%d_%H%M%S_%f")
                                     + f"_pid{os.getpid()}.jsonl")
        self.handler = logging.FileHandler(self.log_path, encoding="utf-8")
        self.handler.setFormatter(logging.Formatter("%(message)s"))
        self.logger.addHandler(self.handler)
        self.logger.setLevel(logging.INFO)
        self.samples, self.counts, self.state = {}, Counter(), {}
        self.enabled = True
        logging.getLogger("uvicorn.error").addHandler(self.http_handler)
        self.stop_event.clear()
        self.event("startup", **metadata, pid=os.getpid(), log_path=str(self.log_path))
        self.thread = threading.Thread(target=self._report, name="performance-reporter", daemon=True)
        self.thread.start()

    def event(self, name, **values):
        if self.enabled:
            self.logger.info(json.dumps({"time": datetime.now().astimezone().isoformat(),
                                         "event": name, **values}, ensure_ascii=False))

    def _report(self):
        started = time.perf_counter()
        while True:
            stopping = self.stop_event.wait(5.0)
            now = time.perf_counter()
            self.event("window", **self.snapshot(now - started))
            started = now
            if stopping:
                return

    def stop(self):
        if self.thread is None:
            return
        self.stop_event.set()
        self.thread.join(timeout=2)
        self.thread = None
        self.event("shutdown")
        self.enabled = False
        logging.getLogger("uvicorn.error").removeHandler(self.http_handler)
        self.logger.removeHandler(self.handler)
        self.handler.close()
        self.handler = None


performance = PerformanceMetrics()
