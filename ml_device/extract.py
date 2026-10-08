"""Resumable, time-boxed ONNX feature extraction over a packed zip. Saves chunk files feat_<name>_<i>.npy (float16, pooled 1280-d, orig+hflip mean)."""
import os, sys, io, time, zipfile, csv, numpy as np, onnxruntime as ort
from PIL import Image
B = os.path.expanduser("~/mnt/MP_Cohn Disease/interim_build"); P = os.path.join(B, "_packed"); OUT = os.path.join(B, "_features")
MEAN = np.array([0.485, 0.456, 0.406], np.float32); STD = np.array([0.229, 0.224, 0.225], np.float32)
CH = 512
def main(name, model="backbone_effb0_ra.onnx", tag="imnet", budget=150):
    t0 = time.time(); so = ort.SessionOptions(); so.intra_op_num_threads = 2
    s = ort.InferenceSession(os.path.join(B, "app", "models", model), so, providers=["CPUExecutionProvider"])
    keys = [r["key"] for r in csv.DictReader(open(os.path.join(P, f"{name}_index.csv")))]
    z = zipfile.ZipFile(os.path.join(P, f"{name}.zip"))
    nch = (len(keys) + CH - 1) // CH
    for c in range(nch):
        fp = os.path.join(OUT, f"feat_{tag}_{name}_{c:03d}.npy")
        if os.path.exists(fp): continue
        if time.time() - t0 > budget: print("budget stop at chunk", c, "/", nch); return
        ks = keys[c * CH:(c + 1) * CH]; feats = []
        for i in range(0, len(ks), 32):
            x = np.stack([np.asarray(Image.open(io.BytesIO(z.read(k))).convert("RGB"), np.float32) / 255.0 for k in ks[i:i + 32]])
            x = ((x - MEAN) / STD).transpose(0, 3, 1, 2).copy()
            o1 = s.run(["out"], {"image": x})[0]; o2 = s.run(["out"], {"image": x[:, :, :, ::-1].copy()})[0]
            feats.append(((o1 + o2) / 2).astype(np.float16))
        np.save(fp, np.concatenate(feats)); print(name, "chunk", c, "/", nch, round(time.time() - t0, 1), "s", flush=True)
    print(name, "ALL DONE", nch)
if __name__ == "__main__":
    main(sys.argv[1], *(sys.argv[2:4]))
