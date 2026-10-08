"""Resumable, time-boxed packer: resize to 224 JPEG into <name>.zip (append), index rows to <name>_index.csv."""
import os, sys, io, zipfile, random, csv, time
from multiprocessing import Pool
from PIL import Image
sys.path.insert(0, os.path.dirname(__file__))
from pack import SRC, OUT, load
BUDGET = float(os.environ.get("BUDGET", 150))

def items_for(name):
    random.seed(0); items = []
    if name == "limuc":
        base = os.path.join(SRC, "Labeled Images for Ulcerative Colitis (LIMUC)")
        for p in sorted(os.listdir(base), key=lambda s: (len(s), s)):
            pd_ = os.path.join(base, p)
            if not os.path.isdir(pd_): continue
            for m in sorted(os.listdir(pd_)):
                for f in sorted(os.listdir(os.path.join(pd_, m))):
                    items.append((os.path.join(pd_, m, f), f"{p}/{m.replace(' ', '')}/{f.rsplit('.',1)[0]}.jpg", m, {"patient": p}))
    elif name == "kvasir_capsule":
        base = os.path.join(SRC, "Kvasir-Capsule/labelled_images")
        for c in ["ulcer", "erosion", "erythema", "angiectasia", "blood_fresh", "blood_hematin", "lymphangiectasia", "foreign_body", "polyp", "normal_clean_mucosa"]:
            d = os.path.join(base, c); sub = os.listdir(d)[0]; files = sorted(os.listdir(os.path.join(d, sub)))
            if c == "normal_clean_mucosa": files = random.sample(files, 4000)
            for f in files: items.append((os.path.join(d, sub, f), f"{c}/{f}", c, {"video": f.split("_")[0]}))
    elif name == "seeai":
        base = os.path.join(SRC, "The SEE-AI Project Dataset/SEE_AI_project_all_images/SEE_AI_project_all_images")
        for f in sorted(os.listdir(base)):
            if f.endswith(".jpg"): items.append((os.path.join(base, f), f, "", {}))
    return items

def run(name):
    t0 = time.time(); zp = os.path.join(OUT, f"{name}.zip.part"); ip = os.path.join(OUT, f"{name}_index.csv.part")
    cache = os.path.join(os.path.dirname(__file__), f"{name}_items.txt")
    if os.path.exists(cache):
        import json; items = json.load(open(cache))
    else:
        import json; items = items_for(name); json.dump(items, open(cache, "w"))
    done = set()
    if os.path.exists(ip):
        with open(ip) as f: done = {r["key"] for r in csv.DictReader(f)}
    todo = [it for it in items if it[1] not in done]
    print(name, "total", len(items), "done", len(done), "todo", len(todo), "listing s", round(time.time() - t0, 1), flush=True)
    if not todo:
        os.replace(zp, os.path.join(OUT, f"{name}.zip")); os.replace(ip, os.path.join(OUT, f"{name}_index.csv")); print(name, "FINALISED"); return
    meta = {it[1]: (it[2], it[3]) for it in todo}; new_file = not os.path.exists(ip)
    fields = ["key", "label", "orig_w", "orig_h"] + list(todo[0][3].keys())
    with zipfile.ZipFile(zp, "a", zipfile.ZIP_STORED) as z, open(ip, "a", newline="") as fi, Pool(2) as pool:
        wr = csv.DictWriter(fi, fieldnames=fields)
        if new_file: wr.writeheader()
        n = 0
        for k, data, w, h in pool.imap(load, [(it[0], it[1]) for it in todo], chunksize=8):
            if data is not None:
                z.writestr(k, data); lab, ex = meta[k]; wr.writerow(dict(key=k, label=lab, orig_w=w, orig_h=h, **ex)); n += 1
            if time.time() - t0 > BUDGET: break
        pool.terminate()
    print(name, "wrote", n, "in", round(time.time() - t0, 1), "s", flush=True)

if __name__ == "__main__":
    run(sys.argv[1])
