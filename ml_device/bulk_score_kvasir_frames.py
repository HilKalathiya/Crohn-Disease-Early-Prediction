"""Bulk-score every labelled Kvasir-Capsule frame with the 5 fold ONNX models.

Resumable:
- enumerates frames once into a work-list CSV (video_id, class, path) sorted
- processes `--batch N` frames per invocation, writes an npz shard with
  per-frame probs + lesion_score + top_class per model
- a progress file records the next index to process
"""
import os, sys, time, csv, json, argparse, numpy as np
from PIL import Image
import onnxruntime as ort
PREFIX = os.environ.get("MODEL_PREFIX", "crohnipi_effb0")  # crohnipi_effb0v2 = improved model

BASE = os.path.expanduser("~/mnt/MP_Cohn Disease")
IMG_ROOT = os.path.join(BASE, "temorary solution/Kvasir-Capsule/labelled_images")
MODELS = os.path.join(BASE, "interim_build/app/models")
FOLDS = [int(f) for f in os.environ.get("FOLDS", "1,2,3,4,5").split(",")]  # score one fold at a time if needed
OUT = os.path.join(BASE, "interim_build/results/bulk/kvasir_frames" + ("" if PREFIX == "crohnipi_effb0" else "_" + PREFIX)
                   + ("" if FOLDS == [1, 2, 3, 4, 5] else "_f" + "".join(map(str, FOLDS))))
META_CSV = os.path.join(BASE, "temorary solution/Kvasir-Capsule/metadata.csv")
WORKLIST = os.path.join(OUT, "worklist.csv")
PROGRESS = os.path.join(OUT, "progress.json")

CLASSES = ["N", "E", "O", "AU", "U3-10", "U>10", "S"]
MEAN = np.array([0.485, 0.456, 0.406], np.float32).reshape(1, 3, 1, 1)
STD = np.array([0.229, 0.224, 0.225], np.float32).reshape(1, 3, 1, 1)

def build_worklist():
    os.makedirs(OUT, exist_ok=True)
    """Enumerate all labelled images with their video_id and finding_class."""
    if os.path.exists(WORKLIST):
        return
    # read metadata.csv to map filename -> (video_id, finding_class)
    print(f"[build] reading {META_CSV}", flush=True)
    meta = {}
    with open(META_CSV, encoding="utf-8", errors="replace") as f:
        r = csv.DictReader(f, delimiter=";")
        for row in r:
            meta[row["filename"]] = (row["video_id"], row["finding_class"])
    print(f"[build] metadata rows: {len(meta)}", flush=True)
    rows = []
    # walk the labelled_images folders
    for cls_dir in sorted(os.listdir(IMG_ROOT)):
        p = os.path.join(IMG_ROOT, cls_dir)
        if not os.path.isdir(p):
            continue
        # inside each class dir there's one inner folder
        for inner in os.listdir(p):
            q = os.path.join(p, inner)
            if not os.path.isdir(q):
                continue
            for fn in sorted(os.listdir(q)):
                if not fn.lower().endswith((".jpg", ".png")):
                    continue
                m = meta.get(fn, ("?", "?"))
                rows.append((cls_dir, m[0], m[1], os.path.join(q, fn)))
    with open(WORKLIST, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["cls_dir", "video_id", "finding_class", "path"])
        w.writerows(rows)
    print(f"[build] wrote {WORKLIST} with {len(rows)} rows", flush=True)

def load_sessions():
    so = ort.SessionOptions()
    so.intra_op_num_threads = 2
    so.inter_op_num_threads = 1
    so.enable_cpu_mem_arena = False
    sess = []
    for i in FOLDS:
        p = os.path.join(MODELS, f"{PREFIX}_fold{i}.onnx")
        sess.append(ort.InferenceSession(p, so, providers=["CPUExecutionProvider"]))
    return sess

def preprocess_batch(paths):
    arr = np.empty((len(paths), 3, 224, 224), np.float32)
    for i, p in enumerate(paths):
        try:
            im = Image.open(p).convert("RGB").resize((224, 224), Image.BICUBIC)
            arr[i] = (np.asarray(im, np.float32).transpose(2, 0, 1) / 255.0)
        except Exception as e:
            print("ERR", p, e, flush=True)
            arr[i] = 0
    arr = (arr - MEAN) / STD
    return np.ascontiguousarray(arr, np.float32)

def softmax(x, axis=-1):
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)

def scored_paths():
    """Union of frame paths already stored in valid shard files (corrupt shards are renamed *.bad)."""
    import glob
    done = set()
    for f in sorted(glob.glob(os.path.join(OUT, "shard_*.npz"))):
        try:
            with np.load(f) as z:
                done.update(z["path"].tolist())
        except Exception as e:
            print("BAD shard", f, e, flush=True)
            os.replace(f, f + ".bad")
    return done

def save_shard(first, cls, vid, fc, paths, probs, sds):
    name = os.path.join(OUT, f"shard_{first:06d}_n{len(paths):04d}.npz")
    tmp = name + ".tmp.npz"
    np.savez_compressed(tmp, cls_dir=np.array(cls), video_id=np.array(vid), finding_class=np.array(fc),
                        path=np.array(paths), probs=np.stack(probs).astype(np.float32), probs_sd=np.stack(sds).astype(np.float32))
    os.replace(tmp, name)  # atomic: a killed call never leaves a half-written shard

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=int, default=140, help="max wall seconds per call")
    ap.add_argument("--batch", type=int, default=8, help="frames per batch")
    ap.add_argument("--chunk-size", type=int, default=128, help="frames per saved shard")
    args = ap.parse_args()
    build_worklist()
    rows = list(csv.reader(open(WORKLIST)))[1:]
    done = scored_paths()
    todo = [i for i, r in enumerate(rows) if r[3] not in done]
    print(f"[run] worklist {len(rows)} | scored {len(rows) - len(todo)} | to do {len(todo)}", flush=True)
    if not todo:
        print("[run] ALL DONE", flush=True); return
    sess = load_sessions()
    t0 = time.time(); k = 0
    buf = ([], [], [], [], [], []); first = None
    while k < len(todo) and (time.time() - t0) < args.seconds:
        idx = todo[k:k + args.batch]; k += len(idx)
        batch = [rows[i] for i in idx]
        x = preprocess_batch([r[3] for r in batch])
        P = np.stack([softmax(s.run(None, {s.get_inputs()[0].name: x})[0], 1) for s in sess], 0)
        pm, ps = P.mean(0), P.std(0)
        if first is None: first = idx[0]
        for j, r in enumerate(batch):
            for lst, v in zip(buf, (r[0], r[1] or "?", r[2], r[3], pm[j], ps[j])): lst.append(v)
        if len(buf[3]) >= args.chunk_size:
            save_shard(first, *buf); buf = ([], [], [], [], [], []); first = None
    if buf[3]:
        save_shard(first, *buf)
    left = len(todo) - k
    print(f"[run] scored {k} this call in {time.time() - t0:.0f}s | remaining {left} | total done {len(rows) - left}/{len(rows)} ({100 * (len(rows) - left) / len(rows):.1f}%)", flush=True)

if __name__ == "__main__":
    main()
