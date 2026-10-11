import cv2
import numpy as np
import onnxruntime as ort

from app.recognition.detector import FaceDetector, FaceDetection


class SCRFDDetector(FaceDetector):
    def __init__(
        self,
        model_path: str,
        input_size: tuple[int, int] = (640, 640),
        score_threshold: float = 0.5,
    ):
        self.input_size = input_size
        self.score_threshold = score_threshold

        self.session = ort.InferenceSession(
            model_path,
            providers=[
                "CUDAExecutionProvider",
                "CPUExecutionProvider",
            ],
        )

        self.input_name = self.session.get_inputs()[0].name

    def _preprocess(self, frame: np.ndarray):
        if frame is None or frame.size == 0:
            raise ValueError("Frame is empty")

        height, width = frame.shape[:2]

        target_width, target_height = self.input_size

        scale = min(
            target_width / width,
            target_height / height
        )

        new_width = max(1, min(target_width, int(width * scale)))
        new_height = max(1, min(target_height, int(height * scale)))

        resized = cv2.resize(
            frame,
            (new_width, new_height)
        )

        canvas = np.zeros(
            (target_height, target_width, 3),
            dtype=np.uint8
        )

        canvas[:new_height, :new_width] = resized

        normalized = (canvas.astype(np.float32) - 127.5) / 128.0

        chw = np.transpose(normalized, (2, 0, 1))
        tensor = np.expand_dims(chw, axis=0)

        tensor = np.ascontiguousarray(tensor, dtype=np.float32)
        return tensor, scale