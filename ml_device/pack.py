"""Resize selected images to 224x224 JPEG and pack per dataset into a zip + index.csv (for transfer to the training box)."""
import os, sys, io, zipfile, random, glob, csv
from multiprocessing import Pool
from PIL import Image
SRC = os.path.expanduser("~/mnt/temorary solution")
OUT = os.path.expanduser("~/mnt/MP_Cohn Disease/interim_build/_packed")
S = 224

def load(args):
    path, key = args
    try:
        im = Image.open(path).convert("RGB"); w, h = im.size
        b = io.BytesIO(); im.resize((S, S), Image.BICUBIC).save(b, "JPEG", quality=92)
        return key, b.getvalue(), w, h
    except Exception as e:
        return key, None, str(e), 0

def pack(name, items, extra_cols=None):
    """items: list of (path, key, label, extra dict)"""
    zp = os.path.join(OUT, f"{name}.zip"); rows = []
    with zipfile.ZipFile(zp + ".part", "w", zipfile.ZIP_STORED) as z, Pool(2) as pool:
        meta = {k: (lab, ex) for _, k, lab, ex in items}
        for i, (k, data, w, h) in enumerate(pool.imap_unordered(load, [(p, k) for p, k, _, _ in items], chunksize=16)):
            if data is None: print("FAIL", k, w, flush=True); continue
            z.writestr(k, data); lab, ex = meta[k]; rows.append(dict(key=k, label=lab, orig_w=w, orig_h=h, **ex))
            if i % 1000 == 0: print(name, i, "/", len(items), flush=True)
    os.replace(zp + ".part", zp)
    with open(os.path.join(OUT, f"{name}_index.csv"), "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0].keys())); wr.writeheader(); wr.writerows(rows)
    print(name, "DONE", len(rows), flush=True)

def hyperkvasir():
    base = os.path.join(SRC, "hyper-kvasir/labeled-images/lower-gi-tract/pathological-findings"); items = []
    for g in sorted(os.listdir(base)):
        if not g.startswith("ulcerative-colitis"): continue
        for f in sorted(os.listdir(os.path.join(base, g))):
            items.append((os.path.join(base, g, f), f"{g}/{f}", g.replace("ulcerative-colitis-", ""), {}))
    pack("hyperkvasir_uc", items)

def limuc():
    base = os.path.join(SRC, "Labeled Images for Ulcerative Colitis (LIMUC)"); items = []
    for p in sorted(os.listdir(base)):
        pd_ = os.path.join(base, p)
        if not os.path.isdir(pd_): continue
        for m in sorted(os.listdir(pd_)):
            for f in sorted(os.listdir(os.path.join(pd_, m))):
                items.append((os.path.join(pd_, m, f), f"{p}/{m.replace(' ', '')}/{f.rsplit('.',1)[0]}.jpg", m, {"patient": p}))
    pack("limuc", items)

def kvasir_capsule():
    base = os.path.join(SRC, "Kvasir-Capsule/labelled_images"); items = []; random.seed(0)
    keep = ["ulcer", "erosion", "erythema", "angiectasia", "blood_fresh", "blood_hematin", "lymphangiectasia", "foreign_body", "polyp", "normal_clean_mucosa"]
    for c in keep:
        d = os.path.join(base, c); sub = [s for s in os.listdir(d)][0]; files = sorted(os.listdir(os.path.join(d, sub)))
        if c == "normal_clean_mucosa": files = random.sample(files, 4000)
        for f in files: items.append((os.path.join(d, sub, f), f"{c}/{f}", c, {"video": f.split("_")[0]}))
    pack("kvasir_capsule", items)

def seeai():
    base = os.path.join(SRC, "The SEE-AI Project Dataset/SEE_AI_project_all_images/SEE_AI_project_all_images"); items = []
    for f in sorted(os.listdir(base)):
        if f.endswith(".jpg"): items.append((os.path.join(base, f), f, "", {}))
    pack("seeai", items)

if __name__ == "__main__":
    for n in sys.argv[1:]: globals()[n]()
