# AutoDoor Test Remote (LAN web test tool)

TEST ONLY. Nothing here is imported by production code in `app/`.
All AI (YuNet, SFace, FaceMatcher, FaceQualityGate, EnrollmentService) is
imported from `app/` and reused as-is.

## Install

```powershell
.venv\Scripts\activate
pip install -r test\web_remote\requirements_web_test.txt
```

## Run (from the project root)

```powershell
python -m uvicorn test.web_remote.app:app --host 0.0.0.0 --port 8000
```

* PC:    http://127.0.0.1:8000
* Phone: http://<PC-LAN-IP>:8000  (same Wi-Fi, the LAN IP is printed at startup)

Windows Firewall: allow inbound TCP 8000 for python.exe (or run once as admin):

```powershell
netsh advfirewall firewall add rule name="AutoDoor Test Remote 8000" dir=in action=allow protocol=TCP localport=8000
```

## Pages

| Page            | Route          | Mirrors                         |
|-----------------|----------------|---------------------------------|
| Home            | `/`            | -                               |
| Calibration     | `/calibration` | `calibrate_camera.py`           |
| Enrollment      | `/enrollment`  | `test/test_enrollment_service.py` |
| Realtime debug  | `/recognition` | `test/test_realtime.py`         |

## API

```
GET  /api/status
POST /api/camera/laptop
POST /api/camera/channel/{1..20}
POST /api/camera/stop
GET  /api/video/stream          (MJPEG)
GET  /api/video/snapshot        (single JPEG)

POST /api/calibration/start | ready | cancel | save
GET  /api/calibration/status

POST /api/enrollment/start      {"full_name": "...", "room_number": "..."}
POST /api/enrollment/cancel
GET  /api/enrollment/status

POST /api/recognition/start | stop
GET  /api/recognition/status
```

```
GET  /api/thresholds            thresholds currently loaded in the running server
POST /api/thresholds/reload     force re-read of .env
```

Only one mode runs at a time (IDLE / CALIBRATION / ENROLLMENT / RECOGNITION).
Camera switching is locked while CALIBRATION or ENROLLMENT is running.

## Thresholds: hot reload, strict vs relaxed

* `.env` is watched (mtime). Any change, from SAVE TO .ENV, an editor, or
  `calibrate_camera.py`, is applied to the running server within one status
  poll: the shared `FaceQualityGate` is updated in place and the YAW / DOWN /
  FAR globals used by `EnrollmentService._classify_pose` are patched in the
  process. No restart needed. Files in `app/` are not modified.
* The calibration page shows two profiles: **strict** (exact
  `calibrate_camera.py` formula) and **relaxed** (web-only: confidence -0.06,
  sharpness x0.5, face ratio x0.6, brightness window +/-20). The `save` column
  is editable; SAVE writes those values.
* Enrollment / recognition pages show every gate condition with the loaded
  threshold, so a FAIL names the metric.

## Files

```
app.py                  FastAPI app, lifespan (loads models, releases cameras on shutdown)
camera_manager.py       LAPTOP / RTSP source switch (uses app.camera.rtsp_camera.RTSPCamera)
runtime.py              single worker thread, mode manager, MJPEG frame buffer
calibration_adapter.py  pure math mirrored from calibrate_camera.py (not importable: runs at import)
calibration_session.py  WAIT -> PREPARE(5s) -> CAPTURE(30 samples) per stage, SAVE TO .ENV on demand
enrollment_session.py   EnrollmentService wrapper with 1.5 s cooldown between poses
recognition_session.py  multi-face debug: quality, SFace, best match, verdict
routes/                 camera.py, calibration.py, enrollment.py, recognition.py
templates/, static/     Jinja2 + vanilla JS, dark mobile UI
```
