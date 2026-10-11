import torch
import onnxruntime as ort

session = ort.InferenceSession(
    "models/adaface_r100.onnx",
    providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
)

print("Active providers:", session.get_providers())