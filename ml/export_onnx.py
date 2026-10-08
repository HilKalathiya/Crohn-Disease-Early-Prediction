"""Export a backbone (optionally with trained 7-class head) to ONNX with two outputs: logits/pooled features and the last conv feature map (for CAM)."""
import sys, os, torch, torch.nn as nn
sys.path.insert(0, os.path.dirname(__file__))
from backbones import create

class Wrap(nn.Module):
    def __init__(self, m, with_head):
        super().__init__(); self.m = m; self.with_head = with_head
    def forward(self, x):
        fmap = self.m.forward_features(x)                 # B x 1280 x 7 x 7
        pooled = fmap.mean((2, 3))
        out = self.m.classifier(pooled) if self.with_head else pooled
        return out, fmap

def export(key, out_path, state=None, num_classes=0):
    m = create(key, num_classes=num_classes, pretrained=state is None)
    if state is not None: m.load_state_dict(torch.load(state, map_location="cpu"))
    w = Wrap(m.eval(), with_head=num_classes > 0)
    torch.onnx.export(w, torch.randn(1, 3, 224, 224), out_path, input_names=["image"], output_names=["out", "fmap"],
                      dynamic_axes={"image": {0: "b"}, "out": {0: "b"}, "fmap": {0: "b"}}, opset_version=17, dynamo=False)
    print("exported", out_path, os.path.getsize(out_path) / 1e6, "MB")

if __name__ == "__main__":
    key, out = sys.argv[1], sys.argv[2]
    export(key, out, state=sys.argv[3] if len(sys.argv) > 3 else None, num_classes=int(sys.argv[4]) if len(sys.argv) > 4 else 0)
