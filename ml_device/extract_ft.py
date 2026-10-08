"""Features from the CrohnIPI fine-tuned fold-1 backbone (global-average-pooled last feature map), orig + h-flip.
Used to test whether endoscopy-adapted features help the UC grader."""
import os, sys, io, time, zipfile, csv, numpy as np, onnxruntime as ort
from PIL import Image
B = os.path.expanduser("~/mnt/MP_Cohn Disease/interim_build"); P = os.path.join(B, "_packed"); OUT = os.path.join(B, "_features")
MEAN = np.array([0.485, 0.456, 0.406], np.float32); STD = np.array([0.229, 0.224, 0.225], np.float32); CH = 512
def main(name, budget=160):
    t0 = time.time(); so = ort.SessionOptions(); so.intra_op_num_threads = 2
    s = ort.InferenceSession(os.path.join(B, "app", "models", "crohnipi_effb0_fold1.onnx"), so, providers=["CPUExecutionProvider"])
    keys = [r["key"] for r in csv.DictReader(open(os.path.join(P, f"{name}_index.csv")))]; z = zipfile.ZipFile(os.path.join(P, f"{name}.zip"))
    n = (len(keys) + CH - 1) // CH
    for c in range(n):
        fp = os.path.join(OUT, f"feat_ft1_{name}_{c:03d}.npy")
        if os.path.exists(fp): continue
        if time.time() - t0 > budget: print("budget stop", c, "/", n); return
        ks = keys[c * CH:(c + 1) * CH]; F = []
        for i in range(0, len(ks), 32):
            x = np.stack([np.asarray(Image.open(io.BytesIO(z.read(k))).convert("RGB"), np.float32) / 255.0 for k in ks[i:i + 32]])
            x = ((x - MEAN) / STD).transpose(0, 3, 1, 2).copy()
            f1 = s.run(["fmap"], {"image": x})[0].mean((2, 3)); f2 = s.run(["fmap"], {"image": x[:, :, :, ::-1].copy()})[0].mean((2, 3))
            F.append(((f1 + f2) / 2).astype(np.float16))
        np.save(fp, np.concatenate(F)); print(name, c, "/", n, round(time.time() - t0), flush=True)
    print(name, "ALL DONE")
if __name__ == "__main__": main(sys.argv[1])
