from pathlib import Path
import torch

model_path = Path("models/adaface/adaface_ir101_webface12m.ckpt")

checkpoint = torch.load(
    model_path,
    map_location="cpu",
    weights_only=True
)

print("Checkpoint type:", type(checkpoint))
print("Keys:", list(checkpoint.keys())[:20])
state_dict = checkpoint["state_dict"]

print("Type:", type(state_dict))
print("Total keys:", len(state_dict))

for name, value in list(state_dict.items())[:10]:
    print(name, value.shape)