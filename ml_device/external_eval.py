"""External validation of the CrohnIPI 5-fold EfficientNet-B0 ensemble on Kvasir-Capsule and SEE-AI (no re-training),
plus the 'pointing game' (does the hottest heat-map point fall inside the expert lesion box?).
Resumable: run `python external_eval.py score <set>` repeatedly (time-boxed), then `python external_eval.py report`."""
import os, sys, io, json, time, glob, zipfile, numpy as np, pandas as pd, onnxruntime as ort
from PIL import Image
B = os.path.expanduser("~/mnt/MP_Cohn Disease/interim_build"); P = os.path.join(B, "_packed"); M = os.path.join(B, "app", "models")
PREFIX = os.environ.get("MODEL_PREFIX", "crohnipi_effb0")  # e.g. crohnipi_effb0v2 for the improved model
OUT = os.path.join(B, "_external" if PREFIX == "crohnipi_effb0" else f"_external_{PREFIX}"); RES = os.path.join(B, "results"); os.makedirs(OUT, exist_ok=True)
SRC = os.path.expanduser("~/mnt/temorary solution")
MEAN = np.array([0.485, 0.456, 0.406], np.float32); STD = np.array([0.229, 0.224, 0.225], np.float32)
SEE_COLS = ["angiodysplasia", "erosion", "stenosis", "lymphangiectasia", "lymph follicle", "SMT", "polyp-like", "bleeding", "diverticulum", "erythema", "foreign body", "vein"]

def eval_list(name):
    if name == "kvasir":
        idx = pd.read_csv(os.path.join(P, "kvasir_capsule_index.csv")); return idx.key.tolist(), "kvasir_capsule"
    a = pd.read_csv(os.path.join(SRC, "The SEE-AI Project Dataset/all_annotation.csv"), skiprows=[1]); a = a[a.image_number.notna()].copy()
    a["key"] = a.image_number.astype(int).map(lambda n: f"image{n:05d}.jpg"); rng = np.random.default_rng(0); keys = []
    none = a[a[SEE_COLS].sum(1) == 0].key.values; keys += list(rng.choice(none, 2000, replace=False))
    for c in ["erosion", "stenosis", "erythema"]:
        k = a[a[c] > 0].key.values; keys += list(rng.choice(k, min(1200, len(k)), replace=False))
    return sorted(set(keys)), "seeai"

def score(name, fold, budget=160):
    """Score one fold model on one external set (resumable chunks). Stores class probs and class CAMs (7x7) per image."""
    t0 = time.time(); keys, zname = eval_list(name); z = zipfile.ZipFile(os.path.join(P, f"{zname}.zip"))
    so = ort.SessionOptions(); so.intra_op_num_threads = 2
    s = ort.InferenceSession(os.path.join(M, f"{PREFIX}_fold{fold}.onnx"), so, providers=["CPUExecutionProvider"])
    W = np.load(os.path.join(M, f"{PREFIX}_fold{fold}_head.npz"))["W"]
    CH = 512; n = (len(keys) + CH - 1) // CH
    for c in range(n):
        fp = os.path.join(OUT, f"{name}_f{fold}_{c:03d}.npz")
        if os.path.exists(fp): continue
        if time.time() - t0 > budget: print("budget stop", name, fold, c, "/", n); return
        ks = keys[c * CH:(c + 1) * CH]; P_, C_ = [], []
        for i in range(0, len(ks), 32):
            x = np.stack([np.asarray(Image.open(io.BytesIO(z.read(k))).convert("RGB"), np.float32) / 255.0 for k in ks[i:i + 32]])
            x = ((x - MEAN) / STD).transpose(0, 3, 1, 2).copy()
            lg, fm = s.run(None, {"image": x}); e = np.exp(lg - lg.max(1, keepdims=True)); P_.append(e / e.sum(1, keepdims=True))
            C_.append(np.einsum("ck,bkhw->bchw", W, fm).astype(np.float16))
        np.savez(fp, keys=np.array(ks), probs=np.concatenate(P_), cam=np.concatenate(C_)); print(name, "fold", fold, c, "/", n, round(time.time() - t0), flush=True)
    print(name, "fold", fold, "ALL DONE")

def load(name):
    """Average the 5 fold models: probs and class CAMs; lesion CAM = sum over lesion classes of p_c * normalised CAM_c."""
    Ps, Cs, keys = [], [], None
    folds = [int(x) for x in os.environ.get("FOLDS", "1,2,3,4,5").split(",")]
    for f in folds:
        fs = sorted(glob.glob(os.path.join(OUT, f"{name}_f{f}_*.npz"))); d = [np.load(x) for x in fs]
        k = np.concatenate([x["keys"] for x in d]); keys = k if keys is None else keys; assert (k == keys).all()
        Ps.append(np.concatenate([x["probs"] for x in d])); Cs.append(np.concatenate([x["cam"] for x in d]).astype(np.float32))
    p = np.mean(Ps, 0); cam = np.maximum(np.mean(Cs, 0), 0)
    cam = cam / np.maximum(cam.reshape(*cam.shape[:2], -1).max(-1)[..., None, None], 1e-8)
    les = (cam[:, 1:] * p[:, 1:, None, None]).sum(1)
    return keys, p, les

def up(cam, S=224):
    from PIL import Image as I
    return np.asarray(I.fromarray(cam.astype(np.float32)).resize((S, S), I.BILINEAR))

def pointing(cam, boxes):
    """cam 7x7; boxes list of (x0,y0,x1,y1) in [0,1]. hit if argmax of upsampled map inside any box."""
    u = up(cam); yy, xx = np.unravel_index(u.argmax(), u.shape); px, py = (xx + 0.5) / 224, (yy + 0.5) / 224
    hit = any(x0 <= px <= x1 and y0 <= py <= y1 for x0, y0, x1, y1 in boxes)
    area = min(1.0, sum((x1 - x0) * (y1 - y0) for x0, y0, x1, y1 in boxes)); return hit, area

def report():
    from sklearn.metrics import roc_auc_score
    thr = json.load(open(os.path.join(M, "crohnipi_meta.json")))["lesion_threshold"]; out = {}
    # Kvasir-Capsule
    k, p, cam = load("kvasir"); idx = pd.read_csv(os.path.join(P, "kvasir_capsule_index.csv")).set_index("key").loc[k]
    lab = idx.label.values; s = 1 - p[:, 0]; neg = lab == "normal_clean_mucosa"; kv = {}
    infl = np.isin(lab, ["ulcer", "erosion", "erythema"])
    m = infl | neg; kv["inflammatory_vs_normal"] = dict(auroc=roc_auc_score(infl[m], s[m]), n_pos=int(infl.sum()), n_neg=int(neg.sum()),
        sensitivity=float((s[infl] >= thr).mean()), specificity=float((s[neg] < thr).mean()))
    for c in ["ulcer", "erosion", "erythema", "angiectasia", "blood_fresh", "lymphangiectasia", "foreign_body", "polyp"]:
        pos = lab == c
        if pos.sum() >= 10: kv[c] = dict(n=int(pos.sum()), auroc_vs_normal=roc_auc_score(np.r_[np.ones(pos.sum()), np.zeros(neg.sum())], np.r_[s[pos], s[neg]]), mean_lesion_score=float(s[pos].mean()))
    kv["normal_mean_lesion_score"] = float(s[neg].mean())
    meta = pd.read_csv(os.path.join(SRC, "Kvasir-Capsule/metadata.csv"), sep=";").dropna(subset=["x1"])
    meta["key"] = meta.finding_class.map({"Ulcer": "ulcer", "Erosion": "erosion", "Erythema": "erythema"}) + "/" + meta.filename
    bx = {}
    for r in meta.dropna(subset=["key"]).itertuples():
        xs, ys = [r.x1, r.x2, r.x3, r.x4], [r.y1, r.y2, r.y3, r.y4]; bx.setdefault(r.key, []).append((min(xs) / 336, min(ys) / 336, max(xs) / 336, max(ys) / 336))
    pos = {kk: i for i, kk in enumerate(k)}; hits, areas = [], []
    for kk, b in bx.items():
        if kk in pos: h, a = pointing(cam[pos[kk]], b); hits.append(h); areas.append(a)
    kv["pointing"] = dict(n=len(hits), hit_rate=float(np.mean(hits)) if hits else None, baseline=float(np.mean(areas)) if areas else None)
    out["kvasir"] = kv
    # SEE-AI
    k, p, cam = load("seeai"); a = pd.read_csv(os.path.join(SRC, "The SEE-AI Project Dataset/all_annotation.csv"), skiprows=[1]); a = a[a.image_number.notna()].copy()
    a["key"] = a.image_number.astype(int).map(lambda n: f"image{n:05d}.jpg"); a = a.set_index("key").loc[k]; s = 1 - p[:, 0]
    none = (a[SEE_COLS].sum(1) == 0).values; se = {}
    for c in ["erosion", "stenosis", "erythema"]:
        pos_ = (a[c] > 0).values; se[c] = dict(n=int(pos_.sum()), auroc_vs_no_finding=roc_auc_score(np.r_[np.ones(pos_.sum()), np.zeros(none.sum())], np.r_[s[pos_], s[none]]),
                                               sensitivity=float((s[pos_] >= thr).mean()))
    se["no_finding"] = dict(n=int(none.sum()), specificity=float((s[none] < thr).mean()))
    # pointing game on SEE-AI erosion boxes (YOLO txt: cls cx cy w h, normalised; class 1 = erosion, 2 = stenosis)
    tdir = os.path.join(SRC, "The SEE-AI Project Dataset/SEE_AI_project_all_txt/SEE_AI_project_all_txt"); hits = {1: [], 2: []}; areas = {1: [], 2: []}; t0 = time.time()
    for i, kk in enumerate(k):
        row = a.iloc[i]
        if not (row["erosion"] > 0 or row["stenosis"] > 0): continue
        if time.time() - t0 > 60: break
        try: lines = open(os.path.join(tdir, kk.replace(".jpg", ".txt"))).read().split("\n")
        except FileNotFoundError: continue
        bb = {1: [], 2: []}
        for ln in lines:
            v = ln.split()
            if len(v) == 5 and int(v[0]) in bb:
                cx, cy, w, h = map(float, v[1:]); bb[int(v[0])].append((cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2))
        for cl in (1, 2):
            if bb[cl]: h, ar = pointing(cam[i], bb[cl]); hits[cl].append(h); areas[cl].append(ar)
    se["pointing"] = {nm: dict(n=len(hits[cl]), hit_rate=float(np.mean(hits[cl])) if hits[cl] else None, baseline=float(np.mean(areas[cl])) if areas[cl] else None)
                      for cl, nm in [(1, "erosion"), (2, "stenosis")]}
    out["seeai"] = se
    json.dump(out, open(os.path.join(RES, "crohnipi_external.json" if PREFIX == "crohnipi_effb0" else f"crohnipi_external_{PREFIX}.json"), "w"), indent=1, default=float); print(json.dumps(out, default=float, indent=1)[:3000])

if __name__ == "__main__":
    if sys.argv[1] == "score": score(sys.argv[2], int(sys.argv[3]))
    else: report()
