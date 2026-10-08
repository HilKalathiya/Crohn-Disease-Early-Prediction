"""Evaluate the hard-negative models on whole Kvasir-Capsule videos, against the original model, with videos held out.

    python eval_video_frames.py --pt-A runs_gpu/hn_A/all.pt --pt-B runs_gpu/hn_B/all.pt --backbone effb0

Model A (negatives from half A) scores the videos of half B and model B scores half A, so no video is scored by a model
that saw its frames. The original CrohnIPI-only model is the baseline: its fold-1 scores for the same frames are already
in index.csv (column v1_lesion_score), and its 5-model ensemble scores for the expert-labelled frames are in
results/bulk/kvasir_frames. Writes results/hard_negative_eval.json.
"""
import argparse, csv, glob, json, os, sys
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from train_gpu import make_model  # noqa: E402
from kvasir_groups import groups  # noqa: E402

BASE = os.path.normpath(os.path.join(HERE, ".."))
DEF_INDEX = os.path.join(BASE, "data_kvasir_video_frames", "index.csv")
KC = os.path.normpath(os.path.join(BASE, "..", "temorary solution", "Kvasir-Capsule"))
INFL = ("Ulcer", "Erosion", "Erythema")


def auroc(s, y):
    s, y = np.asarray(s, float), np.asarray(y, bool)
    if y.all() or (~y).all():
        return None
    r = np.argsort(np.argsort(s, kind="mergesort"), kind="mergesort") + 1.0
    for v in np.unique(s):
        t = s == v
        if t.sum() > 1:
            r[t] = r[t].mean()
    return float((r[y].sum() - y.sum() * (y.sum() + 1) / 2) / (y.sum() * (~y).sum()))


@torch.no_grad()
def score(model, paths, dev, mean, std, bs=64):
    model.eval(); out = []
    for i in range(0, len(paths), bs):
        x = torch.from_numpy(np.stack([np.asarray(Image.open(p).convert("RGB").resize((224, 224), Image.BICUBIC)) for p in paths[i:i + bs]]))
        x = (x.to(dev).permute(0, 3, 1, 2).float() / 255.0 - mean) / std
        with torch.autocast(device_type=dev.type, dtype=torch.float16, enabled=dev.type == "cuda"):
            p = F.softmax(model(x).float(), 1)
        out.append(1 - p[:, 0].cpu().numpy())
        if (i // bs) % 50 == 0:
            print(f"  scored {i + len(paths[i:i + bs])}/{len(paths)}", flush=True)
    return np.concatenate(out)


def video_metrics(rows, sc):
    by_v = {}
    for r, s in zip(rows, sc):
        by_v.setdefault(r["video_id"], []).append((int(r["frame"]), s, r["expert_label_within_25"], int(r["video_has_inflammatory"])))
    vids, rec5, rec10 = [], [], []
    for v, items in by_v.items():
        fr = np.array([i[0] for i in items]); s = np.array([i[1] for i in items]); has = items[0][3]
        vids.append(dict(video=v, has=has, p99=float(np.percentile(s, 99)), top1=float(np.sort(s)[-max(1, len(s) // 100):].mean()),
                         flagged=float((s >= 0.5).mean())))
        lesion_frames = fr[[i[2] in INFL for i in items]]
        if has and len(lesion_frames):
            order = np.argsort(-s)
            for q, store in ((5, rec5), (10, rec10)):
                top = fr[order[:max(1, int(len(s) * q / 100))]]
                store.append(float(np.mean([np.abs(top - n).min() <= 25 for n in lesion_frames])))
    y = [x["has"] for x in vids]
    return dict(n_videos=len(vids), auroc_p99=auroc([x["p99"] for x in vids], y), auroc_top1pct=auroc([x["top1"] for x in vids], y),
                auroc_flagged=auroc([x["flagged"] for x in vids], y),
                median_flagged_lesion_free=float(np.median([x["flagged"] for x in vids if not x["has"]])),
                median_lesion_recall_top5pct=float(np.median(rec5)) if rec5 else None,
                median_lesion_recall_top10pct=float(np.median(rec10)) if rec10 else None)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pt-A", required=True); ap.add_argument("--pt-B", required=True)
    ap.add_argument("--backbone", default="effb0"); ap.add_argument("--index", default=DEF_INDEX)
    ap.add_argument("--out", default=os.path.join(BASE, "results", "hard_negative_eval.json"))
    ap.add_argument("--quick", type=int, default=0, help="smoke test: only N frames per video and N labelled frames")
    a = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    root = os.path.dirname(a.index); g = groups(a.index)
    rows = list(csv.DictReader(open(a.index)))
    if a.quick:
        seen = {}; keep = []
        for r in rows:
            seen[r["video_id"]] = seen.get(r["video_id"], 0) + 1
            if seen[r["video_id"]] <= a.quick or r["expert_label_within_25"]: keep.append(r)
        rows = [r for r in keep if os.path.exists(os.path.join(root, r["file"]))]
    models = {}
    for half, pt in (("A", a.pt_A), ("B", a.pt_B)):
        m = make_model(a.backbone, pretrained=False); m.load_state_dict(torch.load(pt, map_location="cpu", weights_only=True))
        models[half] = m.to(dev)
    cfg = models["A"].pretrained_cfg
    mean = torch.tensor(cfg["mean"], device=dev).view(1, 3, 1, 1); std = torch.tensor(cfg["std"], device=dev).view(1, 3, 1, 1)
    new = np.zeros(len(rows))
    for scorer, half in (("A", "B"), ("B", "A")):  # model A scores half B, and vice versa
        idx = [i for i, r in enumerate(rows) if g[r["video_id"]] == half]
        print(f"Model {scorer} scores {len(idx)} frames of half {half}", flush=True)
        new[idx] = score(models[scorer], [os.path.join(root, rows[i]["file"]) for i in idx], dev, mean, std)
    base = np.array([float(r["v1_lesion_score"]) if r["v1_lesion_score"] not in ("", "nan") else np.nan for r in rows])
    ok = ~np.isnan(base)
    res = dict(note="Every 50th frame of the 43 labelled Kvasir-Capsule videos; each video scored by the model trained WITHOUT its half; "
                    "baseline = original CrohnIPI fold-1 model on the same frames",
               baseline=video_metrics([r for r, k in zip(rows, ok) if k], base[ok]), hard_negative=video_metrics(rows, new))
    # expert-labelled frames of each half, scored by the other half's model, vs the original 5-model ensemble
    if not os.path.exists(os.path.join(KC, "metadata.csv")):
        print("Kvasir-Capsule metadata not found; skipping the labelled-frame comparison")
        json.dump(res, open(a.out, "w"), indent=1); print(json.dumps(res, indent=1)); return
    meta = {r["filename"]: (r["video_id"], r["finding_class"]) for r in csv.DictReader(open(os.path.join(KC, "metadata.csv"), encoding="utf-8", errors="replace"), delimiter=";")}
    ens = {}
    for f in glob.glob(os.path.join(BASE, "results", "bulk", "kvasir_frames", "shard_*.npz")):
        z = np.load(f)
        for p, pr in zip(z["path"], z["probs"]):
            ens[os.path.basename(str(p))] = float(1 - pr[0])
    lab = [(fn, v, c) for fn, (v, c) in meta.items() if v in g and c in INFL + ("Normal clean mucosa",)]
    paths = {fn: p for p in glob.glob(os.path.join(KC, "labelled_images", "*", "*", "*.jpg")) for fn in [os.path.basename(p)]}
    lab = [t for t in lab if t[0] in paths]
    if a.quick:
        lab = lab[::max(1, len(lab) // a.quick)]
    sc_new = np.zeros(len(lab))
    for scorer, half in (("A", "B"), ("B", "A")):
        idx = [i for i, t in enumerate(lab) if g[t[1]] == half]
        print(f"Model {scorer} scores {len(idx)} expert-labelled frames of half {half}", flush=True)
        sc_new[idx] = score(models[scorer], [paths[lab[i][0]] for i in idx], dev, mean, std)
    y = np.array([t[2] in INFL for t in lab]); sc_ens = np.array([ens.get(t[0], np.nan) for t in lab]); k = ~np.isnan(sc_ens)
    res["labelled_frames"] = dict(n=int(len(lab)), inflammatory=int(y.sum()),
                                  auroc_hard_negative=auroc(sc_new, y), auroc_original_ensemble=auroc(sc_ens[k], y[k]),
                                  specificity_at_0_5_hard_negative=float((sc_new[~y] < 0.5).mean()),
                                  specificity_at_0_5_original=float((sc_ens[k][~y[k]] < 0.5).mean()),
                                  sensitivity_at_0_5_hard_negative=float((sc_new[y] >= 0.5).mean()),
                                  sensitivity_at_0_5_original=float((sc_ens[k][y[k]] >= 0.5).mean()))
    np.save(os.path.join(os.path.dirname(a.out), "hard_negative_video_scores.npy"), new)
    json.dump(res, open(a.out, "w"), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
