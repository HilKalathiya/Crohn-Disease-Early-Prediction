"""Re-run the external tests (no retraining) with GPU-trained fold models, e.g. the EfficientNet-B2 320 px model.

    python external_gpu.py                      # both seeds of runs_gpu/effb2_320*, ~40 min on an RTX 3050

Writes, in the same formats the CPU analysis scripts already read:
  results/bulk/kvasir_frames_<prefix>/shard_merged_000000_all.npz    every labelled Kvasir-Capsule frame (47,238):
                                                                        ensemble probs + spread across models
  _external_<prefix>/{kvasir,seeai}_f<m>_000.npz                     the Kvasir subset and SEE-AI frames used by
                                                                        ml_device/external_eval.py: probs + class maps per model
Each image is read from the ORIGINAL dataset file and resized once to the model input (as the app does), one view,
mixed precision. Resumable: finished chunks are kept in <out>/parts and skipped on a rerun.
Then (on the CPU side):  MODEL_PREFIX=<prefix> python ml_device/analyze_kvasir_full.py
                         MODEL_PREFIX=<prefix> FOLDS=1,...,10 python ml_device/external_eval.py report
"""
import argparse, csv, glob, os, sys, time
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import torch
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from train_gpu import make_model  # noqa: E402

BASE = os.path.normpath(os.path.join(HERE, ".."))                        # interim_build
SRC = os.path.normpath(os.path.join(BASE, "..", "temorary solution"))
KC = os.path.join(SRC, "Kvasir-Capsule")
SEE = os.path.join(SRC, "The SEE-AI Project Dataset")
SEE_COLS = ["angiodysplasia", "erosion", "stenosis", "lymphangiectasia", "lymph follicle", "SMT", "polyp-like", "bleeding",
            "diverticulum", "erythema", "foreign body", "vein"]


def kvasir_all():
    """(key, path, cls_dir, video_id, finding_class) for every labelled Kvasir-Capsule image"""
    meta = {}
    with open(os.path.join(KC, "metadata.csv"), encoding="utf-8", errors="replace") as f:
        for r in csv.DictReader(f, delimiter=";"):
            meta[r["filename"]] = (r["video_id"], r["finding_class"])
    rows = []
    for p in sorted(glob.glob(os.path.join(KC, "labelled_images", "*", "*", "*.jpg"))):
        cls_dir = os.path.basename(os.path.dirname(os.path.dirname(p))); fn = os.path.basename(p)
        vid, fc = meta.get(fn, ("", ""))
        rows.append((f"{cls_dir}/{os.path.basename(os.path.dirname(p))}/{fn}", p, cls_dir, vid, fc))
    return rows


def kvasir_subset(paths_by_name):
    """the 8,266 Kvasir frames of ml_device/external_eval.py (keys 'class/filename')"""
    keys = [r["key"] for r in csv.DictReader(open(os.path.join(BASE, "_packed", "kvasir_capsule_index.csv")))]
    return [(k, paths_by_name[os.path.basename(k)]) for k in keys]


def seeai_subset():
    """the SEE-AI selection of ml_device/external_eval.py, reproduced exactly (pandas, seed 0)"""
    import pandas as pd
    a = pd.read_csv(os.path.join(SEE, "all_annotation.csv"), skiprows=[1]); a = a[a.image_number.notna()].copy()
    a["key"] = a.image_number.astype(int).map(lambda n: f"image{n:05d}.jpg"); rng = np.random.default_rng(0); keys = []
    none = a[a[SEE_COLS].sum(axis=1) == 0].key.values; keys += list(rng.choice(none, 2000, replace=False))
    for c in ["erosion", "stenosis", "erythema"]:
        k = a[a[c] > 0].key.values; keys += list(rng.choice(k, min(1200, len(k)), replace=False))
    img = os.path.join(SEE, "SEE_AI_project_all_images", "SEE_AI_project_all_images")
    return [(k, os.path.join(img, k)) for k in sorted(set(keys))]


class Ensemble:
    def __init__(self, runs, backbone, img, dev):
        self.models, self.W, self.dev, self.img = [], [], dev, img
        for r in runs:
            for k in range(1, 6):
                m = make_model(backbone, pretrained=False)
                m.load_state_dict(torch.load(os.path.join(r, f"fold{k}.pt"), map_location="cpu", weights_only=True))
                m = m.to(dev).eval(); self.models.append(m); self.W.append(m.get_classifier().weight.detach())
        cfg = self.models[0].pretrained_cfg
        self.mean = torch.tensor(cfg["mean"], device=dev).view(1, 3, 1, 1); self.std = torch.tensor(cfg["std"], device=dev).view(1, 3, 1, 1)
        self.pool = ThreadPoolExecutor(6)

    def load(self, paths):
        return np.stack(list(self.pool.map(lambda p: np.asarray(Image.open(p).convert("RGB").resize((self.img, self.img), Image.BICUBIC)), paths)))

    @torch.no_grad()
    def run(self, x_uint8, with_cam):
        x = torch.from_numpy(x_uint8).to(self.dev).permute(0, 3, 1, 2).float().div_(255.0)
        x = (x - self.mean) / self.std
        P, C = [], []
        for m, W in zip(self.models, self.W):
            with torch.autocast(device_type=self.dev.type, dtype=torch.float16, enabled=self.dev.type == "cuda"):
                fm = m.forward_features(x)
            fm = fm.float(); logits = m.get_classifier()(fm.mean((2, 3)))
            P.append(torch.softmax(logits, 1).cpu().numpy())
            if with_cam:
                C.append(torch.einsum("ck,bkhw->bchw", W.float(), fm).half().cpu().numpy())
        return np.stack(P), (np.stack(C) if with_cam else None)   # (models,B,7), (models,B,7,h,w)


def run_set(ens, name, items, out_dir, with_cam, chunk=1024, bs=32, limit=0):
    """score items [(key, path)] in resumable chunks -> concatenated probs (models,N,7) and cams"""
    if limit:
        items = items[:limit]
    pdir = os.path.join(out_dir, "parts"); os.makedirs(pdir, exist_ok=True)
    t0, n = time.time(), len(items)
    for c in range(0, n, chunk):
        fp = os.path.join(pdir, f"{name}_{c // chunk:04d}.npz")
        if os.path.exists(fp):
            continue
        P, C = [], []
        for i in range(c, min(n, c + chunk), bs):
            p, cm = ens.run(ens.load([it[1] for it in items[i:i + bs]]), with_cam)
            P.append(p); C.append(cm)
        np.savez(fp + ".tmp.npz", probs=np.concatenate(P, 1), **({"cam": np.concatenate(C, 1)} if with_cam else {}))
        os.replace(fp + ".tmp.npz", fp)
        done = min(n, c + chunk); el = time.time() - t0
        print(f"  {name}: {done}/{n} images, {el / 60:.1f} min", flush=True)
    parts = [np.load(os.path.join(pdir, f"{name}_{c // chunk:04d}.npz")) for c in range(0, n, chunk)]
    P = np.concatenate([z["probs"] for z in parts], 1)
    C = np.concatenate([z["cam"] for z in parts], 1) if with_cam else None
    return items, P, C


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", nargs="+", default=[os.path.join(HERE, "runs_gpu", "effb2_320"), os.path.join(HERE, "runs_gpu", "effb2_320_s1")])
    ap.add_argument("--backbone", default="effb2"); ap.add_argument("--img", type=int, default=320)
    ap.add_argument("--prefix", default="crohnipi_effb2")
    ap.add_argument("--only", choices=["kvasir_all", "external"], default=None)
    ap.add_argument("--limit", type=int, default=0, help="smoke test: first N images of each set")
    ap.add_argument("--cpu", action="store_true")
    a = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() and not a.cpu else "cpu")
    if dev.type == "cuda":
        torch.backends.cudnn.benchmark = True
        print("GPU:", torch.cuda.get_device_name(0), flush=True)
    ens = Ensemble(a.runs, a.backbone, a.img, dev)
    print(f"{len(ens.models)} models loaded ({a.backbone}, {a.img} px)", flush=True)
    rows = kvasir_all()
    print(f"Kvasir-Capsule labelled images found: {len(rows)}", flush=True)

    if a.only in (None, "kvasir_all"):
        out = os.path.join(BASE, "results", "bulk", f"kvasir_frames_{a.prefix}"); os.makedirs(out, exist_ok=True)
        items, P, _ = run_set(ens, "kvasir_all", [(r[0], r[1]) for r in rows], out, False, chunk=2048, limit=a.limit)
        meta = rows[:len(items)]
        np.savez_compressed(os.path.join(out, "shard_merged_000000_all.npz"), cls_dir=np.array([m[2] for m in meta]),
                            video_id=np.array([m[3] for m in meta]), finding_class=np.array([m[4] for m in meta]),
                            path=np.array([m[0] for m in meta]), probs=P.mean(0).astype(np.float32), probs_sd=P.std(0).astype(np.float32))
        print("wrote", out, flush=True)

    if a.only in (None, "external"):
        out = os.path.join(BASE, f"_external_{a.prefix}"); os.makedirs(out, exist_ok=True)
        by_name = {os.path.basename(r[1]): r[1] for r in rows}
        for name, items in (("kvasir", kvasir_subset(by_name)), ("seeai", seeai_subset())):
            items, P, C = run_set(ens, name, items, out, True, limit=a.limit)
            keys = np.array([it[0] for it in items])
            for m in range(P.shape[0]):  # one file per model, the layout external_eval.py's report reads (FOLDS=1..10)
                np.savez(os.path.join(out, f"{name}_f{m + 1}_000.npz"), keys=keys, probs=P[m].astype(np.float32), cam=C[m])
            print(f"wrote {name}: {len(keys)} images x {P.shape[0]} models", flush=True)
    print("\nDone. Tell Claude; the analysis runs on the CPU side.")


if __name__ == "__main__":
    main()
