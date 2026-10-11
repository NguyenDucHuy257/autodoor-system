import torch
import onnxruntime as ort

session = ort.InferenceSession(
    "models/scrfd/det_10g.onnx",
    providers=[
        "CUDAExecutionProvider",
        "CPUExecutionProvider"
    ]
)

print("Active providers:", session.get_providers())