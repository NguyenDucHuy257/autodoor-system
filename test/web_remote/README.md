# Web Remote Demo ? SCRFD + AdaFace

Run from the project root using the existing project environment:

```powershell
python -m uvicorn test.web_remote.app:app --host 127.0.0.1 --port 8000
```

Requires `models/scrfd/det_10g.onnx` and `models/adaface_r100.onnx`, the existing ONNX Runtime installation, and Pillow. Model initialization reports the actual ONNX session providers; CUDA availability alone does not prove every operator executes on GPU.

The camera owner captures full source frames. The small direct preview uses a 320-pixel JPEG width during recognition; the recognition view retains its 960-pixel display width; the camera resolution and AI input source are unchanged. Preview, SCRFD, AdaFace, JPEG delivery, and recording have separate workers. Latest-only mailboxes discard pending older work. SCRFD has one owner; AdaFace inference is serialized. WebSocket delivery allows up to four frames in flight with an independent feedback receiver. It does not wait for each frame acknowledgement. The browser keeps one latest pending frame and serializes decoding. A transport stalled for more than one second is closed and reconnected to discard queued old data.

Recognition shows a smooth raw preview and a separate result image containing only green (recognized) or red (other) face boxes. Face crops and names appear in the bottom-right corner, with contrasting caption backgrounds. Detailed recognition results appear below the image. JPEG, crops, and details are delivered together from the exact source frame processed by SCRFD and AdaFace. Jobs/results older than one second are discarded; stale result images and details are cleared. Camera switches are queued immediately, capture releases/opens the source on its owner thread, and generations invalidate old jobs/results.

Enrollment writes five normalized AdaFace embeddings to `test/web_remote/gallery`. The gallery fingerprint includes the model file hash and the existing core alignment/preprocessing convention. It does not read/write the production database or mix SFace embeddings. The demo cosine threshold is **0.50** and requires calibration against your own positive/negative samples before relying on identity decisions. AdaFace provides embeddings, not age/gender estimates; displayed details include identity/unknown, cosine, detector confidence, quality, embedding dimension/norm, and active providers.

During recognition, completed face results start a recognition-view MP4 clip in `test/web_remote/results`, including face boxes and a bottom-right crop/name strip. Recording ends after three seconds without a detection, camera/session change, or shutdown. The bounded recording queue drops old frames under load; MP4 uses 15 FPS timestamp-driven encoding and has a JSON sidecar. Clips contain completed recognition frames and no audio; the 15 FPS container may repeat frames when AI produces fewer results. The sidecar also reports the number of unique processed frames. Storage failures appear in status and performance events.

Existing performance logs are created for each server run under project `logs`. New counters include `ai_jobs_dropped`, `ai_results_dropped`, `ai_published_frames`, `client_rendered_frames`, `client_ai_rendered_frames`, and `recording_frames_dropped`. Client counters measure decode/draw acknowledgements, not physical display refresh. Timing windows remain managed by the existing performance collector.

Terminal logging is redirected to a separate per-run `logs/web_remote_*.log`, including Uvicorn access/error messages and native AI stdout/stderr. This applies only to the test server process; errors remain available in log files.

Offline regression checks (no camera or database connection):

```powershell
python -m unittest test.web_remote.test_pipeline test.web_remote.test_streaming test.web_remote.test_scrfd_adapter -v
```

Static JS/CSS URLs include file modification versions to invalidate old browser caches. WebSocket logs now separate send duration, acknowledgement wait, browser decode, and browser render timings. These distinguish network/transport delay from browser work without lowering recognition FPS.

Streaming telemetry uses `client_*_delivery_ack_ms` (send-to-feedback latency), `client_*_frames_dropped`, `client_*_server_frames_skipped`, and `client_*_slow_client_reconnects`. Decode/render timings and actually rendered counters continue separately. Acknowledgement latency includes network and browser/server scheduling; it is not pure network RTT.

The web demo uses only SCRFD-10G (`models/scrfd/det_10g.onnx`) for calibration, enrollment, and recognition. Its default detection input is 640x640 with aspect-preserving resize and padding, RGB normalization, threshold 0.50, and NMS IoU 0.40. Boxes and landmarks are mapped back to the original frame; landmarks are reordered to the core contract so AdaFace alignment and pose classification remain compatible. AdaFace remains the only recognition embedder. The unfinished core SCRFD adapter is extended in `test/web_remote/scrfd_adapter.py`; no core source is changed. Decoder reference: https://github.com/deepinsight/insightface/blob/master/python-package/insightface/model_zoo/scrfd.py
