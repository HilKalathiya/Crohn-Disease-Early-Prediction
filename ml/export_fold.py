"""Export a fine-tuned CrohnIPI fold model to ONNX (+ classifier weights for CAM) and check parity with PyTorch."""
import sys, os, numpy as np, torch, onnxruntime as ort
sys.path.insert(0, os.path.dirname(__file__))
from export_onnx import export
from backbones import create
k = sys.argv[1]; out = f"export/crohnipi_effb0_fold{k}.onnx"; st = f"runs/effb0_ra/fold{k}.pt"
export("effb0_ra", out, state=st, num_classes=7)
sd = torch.load(st, map_location="cpu"); np.savez(f"export/crohnipi_effb0_fold{k}_head.npz", W=sd["classifier.weight"].numpy(), b=sd["classifier.bias"].numpy())
m = create("effb0_ra", num_classes=7, pretrained=False); m.load_state_dict(sd); m.eval()
x = np.random.rand(2, 3, 224, 224).astype(np.float32)
with torch.no_grad(): ref = m(torch.from_numpy(x)).numpy()
o = ort.InferenceSession(out).run(None, {"image": x})[0]; print("fold", k, "parity max abs diff", float(np.abs(ref - o).max()))
