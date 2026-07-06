"""
Export trained PongMLP weights to JSON for in-browser inference.
"""
import json
import torch
import sys

sys.path.insert(0, ".")
from play_vs_bc import PongMLP, FEATURES

def export(model_path, out_path):
    model = PongMLP()
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()

    layers = []
    for name, param in model.net.named_children():
        if isinstance(param, torch.nn.Linear):
            layers.append({
                "weight": param.weight.detach().tolist(),
                "bias":   param.bias.detach().tolist(),
            })

    out = {
        "features": FEATURES,
        "layers": layers,
    }

    with open(out_path, "w") as f:
        json.dump(out, f)

    size_kb = len(json.dumps(out)) / 1024
    print(f"Exported {model_path} → {out_path}  ({size_kb:.0f} KB)")

if __name__ == "__main__":
    export("models/bc_model.pt",  "site/bc_model.json")
    export("models/rl_model.pt",  "site/rl_model.json")
