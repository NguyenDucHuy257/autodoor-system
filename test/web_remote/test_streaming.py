"""Offline tests of bounded streaming with delayed or missing browser feedback."""
import asyncio
import json
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from fastapi import WebSocketDisconnect
from test.web_remote.routes import camera

class Socket:
    def __init__(self, runtime, acknowledge):
        self.runtime = runtime
        self.acknowledge = acknowledge
        self.replies = asyncio.Queue()
        self.metadata = []
        self.sent = 0
        self.closed = []
    async def accept(self): pass
    async def receive_text(self):
        reply = await self.replies.get()
        if isinstance(reply, Exception): raise reply
        return reply
    async def send_json(self, metadata): self.metadata.append(metadata)
    async def send_bytes(self, data):
        self.sent += 1
        self.acknowledge(self)
    async def close(self, code=1000): self.closed.append(code)
    def ack(self, sequence):
        self.replies.put_nowait(json.dumps(dict(sequence=sequence, rendered=True, decode_ms=4, render_ms=.1)))

class StreamingTests(unittest.IsolatedAsyncioTestCase):
    def runtime(self):
        runtime = SimpleNamespace(shutting_down=False)
        runtime.frame_id = 0
        def latest(*args):
            runtime.frame_id += 1
            return b"jpeg", runtime.frame_id, 7, [{"index":runtime.frame_id}]
        runtime.wait_for_delivery = latest
        return runtime

    async def test_sends_multiple_frames_before_first_feedback(self):
        runtime = self.runtime()
        def acknowledge(socket):
            if socket.sent == 3:
                for sequence in (1,2,3): socket.ack(sequence)
            elif socket.sent > 3: socket.ack(socket.sent)
            if socket.sent == 6: runtime.shutting_down=True
        socket = Socket(runtime, acknowledge)
        with patch.object(camera, "runtime", runtime):
            await asyncio.wait_for(camera.video_live(socket, "ai"), 1)
        self.assertEqual(socket.sent, 6)
        self.assertEqual(socket.metadata[2]["faces"], [{"index":3}])
        self.assertEqual(socket.metadata[2]["sequence"], 3)

    async def test_missing_feedback_bounds_frames_and_closes_transport(self):
        runtime = self.runtime()
        socket = Socket(runtime, lambda socket: None)
        with patch.object(camera, "runtime", runtime), patch.object(camera, "MAX_DELIVERY_AGE", .05):
            await asyncio.wait_for(camera.video_live(socket), 1)
        self.assertEqual(socket.sent, camera.MAX_IN_FLIGHT)
        self.assertIn(1013, socket.closed)

    async def test_idle_disconnect_terminates_receiver(self):
        runtime = self.runtime()
        runtime.wait_for_delivery = lambda *args: (None, -1, 7, [])
        socket = Socket(runtime, lambda socket: None)
        socket.replies.put_nowait(WebSocketDisconnect())
        with patch.object(camera, "runtime", runtime):
            await asyncio.wait_for(camera.video_live(socket), 1)
        self.assertEqual(socket.sent, 0)

if __name__ == "__main__": unittest.main()
