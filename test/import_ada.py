import torch
from models.AdaFace.net import build_model


class AdaFaceONNXWrapper(torch.nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone

    def forward(self, x):
        embedding, norm = self.backbone(x)
        return embedding

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

checkpoint = torch.load(
    r"models\AdaFace\adaface_ir101_webface12m.ckpt",
    map_location="cpu",
    weights_only=True
)

model = build_model("ir_101")

state_dict = checkpoint["state_dict"]

clean_state_dict = {
    key.removeprefix("model."): value
    for key, value in state_dict.items()
    if key.startswith("model.")
}

model.load_state_dict(clean_state_dict, strict=True)

model = model.to(device)
model.eval()

print("AdaFace R100 loaded successfully!")
print("Device:", device)

dummy_input = torch.randn(
    1, 3, 112, 112,
    device=device
)
with torch.inference_mode():
    output = model(dummy_input)

print("Output type:", type(output))
print("Number of outputs:", len(output))

for i, item in enumerate(output):
    print(f"Output {i}:")
    print("  Type:", type(item))

    if isinstance(item, torch.Tensor):
        print("  Shape:", item.shape)
        print("  Device:", item.device)

#============
export_model = AdaFaceONNXWrapper(model)
export_model.eval()

with torch.inference_mode():
    output = export_model(dummy_input)

print("Wrapper output:", output.shape)
print("Device:", output.device)

#==============
from pathlib import Path

output_path = Path("models/adaface_r100.onnx")
output_path.parent.mkdir(parents=True, exist_ok=True)

torch.onnx.export(
    export_model,
    (dummy_input,),
    str(output_path),
    input_names=["input"],
    output_names=["embedding"],
    opset_version=17,
    dynamo=False
)

print("ONNX exported:", output_path)
#==================================
import onnx
import onnxruntime as ort

onnx_path = "models/adaface_r100.onnx"

# Kiểm tra cấu trúc file ONNX
onnx_model = onnx.load(onnx_path)
onnx.checker.check_model(onnx_model)
print("ONNX model valid!")

# Tạo phiên inference
session = ort.InferenceSession(
    onnx_path,
    providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
)

print("Active providers:", session.get_providers())
print("Input:", session.get_inputs()[0].shape)
print("Output:", session.get_outputs()[0].shape)

import numpy as np

with torch.inference_mode():
    torch_output = export_model(dummy_input).cpu().numpy()

onnx_input = dummy_input.cpu().numpy()

onnx_output = session.run(
    ["embedding"],
    {"input": onnx_input}
)[0]

max_diff = np.max(np.abs(torch_output - onnx_output))

print("PyTorch shape:", torch_output.shape)
print("ONNX shape:", onnx_output.shape)
print("Max difference:", max_diff)