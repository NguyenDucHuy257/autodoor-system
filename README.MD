## Chạy Web Remote

Khởi động FastAPI bằng Uvicorn:

```bash
python -m uvicorn test.web_remote.app:app --host 127.0.0.1 --port 8000
```

Truy cập: http://127.0.0.1:8000

API Documentation: http://127.0.0.1:8000/docs

## Performance logs

Mỗi lần khởi động Web Remote tạo một tệp riêng trong `logs/`:
`performance_YYYYMMDD_HHMMSS_microseconds_pidNNN.jsonl`.
Đường dẫn được in khi startup. Restart web để áp dụng instrumentation mới.

Mỗi 5 giây có một bản ghi `window`, kể cả khi worker đang bị chặn:

- `rates`: capture FPS, publish FPS, tổng MJPEG yield FPS của mọi client.
- `timings`: mean/p95/max (ms), số mẫu và tổng số lần gọi cho capture read,
  get frame (gồm chờ lock), detection, quality, embedding, matching,
  processing, resize, JPEG encode, publish và finish enrollment.
- `counts`: capture failures, frame bị từ chối do stale, đọc trùng frame,
  frame capture bị consumer bỏ qua (`frames_dropped`), JPEG/MJPEG bytes,
  client mở/đóng và `invalid_http_requests` do Uvicorn báo.
- `state`: mode, camera source/channel, frame size và tuổi của read/processing
  đang chạy. `frame_age_at_publish_ms` đo từ timestamp decode đến publish.

`frames_dropped` không phải packet loss RTSP. Capture FPS hiện đo cho RTSP;
laptop không có capture thread riêng. MJPEG yield FPS là phía server, không
phải browser receive/render FPS. Timestamp decode không đo được tuổi frame
trong buffer camera/FFmpeg. Khi không có mẫu, metric không xuất hiện.
Timings giữ tối đa 2048 mẫu cuối mỗi cửa sổ; counts vẫn đếm toàn bộ lời gọi.

Log không lưu ảnh, embeddings, tên thành viên, RTSP URL hay secrets.
Để chẩn đoán lag, chạy IDLE, recognition và enrollment rồi so sánh các cửa sổ
trong cùng tệp. Tệp không tự xóa hoặc xoay vòng; có thể xóa log cũ khi không cần.

Kiểm tra offline (không truy cập camera/database):

```powershell
.\.venv\Scripts\python.exe -B -m unittest test.test_performance_metrics
```
