"""Export the 5 fold models of a GPU run to ONNX and switch the app to them.

    python export_to_app.py --runs runs_gpu/effb2 --backbone effb2

Writes into ../app/models:
    crohnipi_<backbone>_fold<k>.onnx       (inputs: "image" B x 3 x 224 x 224; outputs: "out" logits, "fmap" last conv map)
    crohnipi_<backbone>_fold<k>_head.npz   (classifier W, b - used for the heat-maps)
and updates crohnipi_meta.json -> "model_prefix" so the app loads the new files.
The previous model files are left in place: to go back, set "model_prefix" to "crohnipi_effb0" in crohnipi_meta.json.
"""
import argparse, json, os, sys
import numpy as np
import torch
import torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from train_gpu import ARCH, make_model  # noqa: E402

APP_MODELS = os.path.normpath(os.path.join(HERE, "..", "app", "models"))


class Wrap(nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, x):
        fmap = self.m.forward_features(x)
        logits = self.m.get_classifier()(fmap.mean((2, 3)))
        return logits, fmap


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", required=True)
    ap.add_argument("--backbone", required=True, choices=sorted(ARCH))
    ap.add_argument("--img", type=int, default=224)
    ap.add_argument("--models", default=APP_MODELS)
    ap.add_argument("--no-switch", action="store_true", help="export only; keep the app on its current model")
    ap.add_argument("--prefix", default=None, help="model file prefix (default crohnipi_<backbone>; use a new one, e.g. crohnipi_effb0v2, so the current files are kept)")
    a = ap.parse_args()
    runs = a.runs if os.path.isabs(a.runs) else os.path.join(HERE, a.runs)
    prefix = a.prefix or f"crohnipi_{a.backbone}"
    if prefix == "crohnipi_effb0" and os.path.exists(os.path.join(a.models, "crohnipi_effb0_fold1.onnx")):
        prefix = "crohnipi_effb0v2"  # never overwrite the original app model
        print("Using prefix", prefix, "so the original crohnipi_effb0 files stay untouched")
    try:
        import onnxruntime as ort
    except ImportError:
        ort = None
        print("onnxruntime not installed: skipping the PyTorch-vs-ONNX parity check")
    n = 0
    for k in range(1, 6):
        st = os.path.join(runs, f"fold{k}.pt")
        if not os.path.exists(st):
            print(f"fold {k}: no {st}, skipped")
            continue
        m = make_model(a.backbone, pretrained=False)
        m.load_state_dict(torch.load(st, map_location="cpu", weights_only=True))
        m.eval()
        w = Wrap(m).eval()
        out = os.path.join(a.models, f"{prefix}_fold{k}.onnx")
        torch.onnx.export(w, torch.randn(1, 3, a.img, a.img), out, input_names=["image"], output_names=["out", "fmap"],
                          dynamic_axes={"image": {0: "b"}, "out": {0: "b"}, "fmap": {0: "b"}}, opset_version=17, dynamo=False)
        clf = m.get_classifier()
        np.savez(os.path.join(a.models, f"{prefix}_fold{k}_head.npz"), W=clf.weight.detach().numpy(), b=clf.bias.detach().numpy())
        msg = f"fold {k}: {os.path.getsize(out) / 1e6:.1f} MB"
        if ort is not None:
            x = np.random.rand(2, 3, a.img, a.img).astype(np.float32)
            with torch.no_grad():
                ref = w(torch.from_numpy(x))[0].numpy()
            got = ort.InferenceSession(out, providers=["CPUExecutionProvider"]).run(None, {"image": x})[0]
            msg += f", parity max |diff| {float(np.abs(ref - got).max()):.2e}"
        print(msg)
        n += 1
    if n < 5:
        print(f"WARNING: only {n}/5 folds exported")
    if n and not a.no_switch:
        mp = os.path.join(a.models, "crohnipi_meta.json")
        meta = json.load(open(mp)) if os.path.exists(mp) else {}
        meta.update(model_prefix=prefix, backbone=ARCH[a.backbone], input_size=a.img, n_folds=n)
        json.dump(meta, open(mp, "w"), indent=1)
        print(f"App switched to {prefix} ({n} folds). Restart the app:  python run.py")


if __name__ == "__main__":
    main()
