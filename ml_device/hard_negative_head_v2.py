"""Hard negatives, round 2: does a domain-balanced head fix study ranking WITHOUT losing Kvasir lesions?

Why: the full GPU retrain with raw-video negatives (gpu/eval_video_frames.py) raised study AUROC but dropped
lesion sensitivity on held-out Kvasir frames from 87% to 28%. Every negative came from Kvasir (Olympus capsule, Norway)
and every lesion from CrohnIPI (PillCam, France), so "looks like Kvasir" became a shortcut for "normal".
Fix tested here: also give the model Kvasir LESION frames (and Kvasir normal frames from lesion videos) from the
training half, so the camera no longer predicts the label.

Same frozen v2 fold-1 backbone and stored features as hard_negative_head.py; only the final layer is retrained.
Variants (all reported):
  base      the model's own classifier (app)
  head_ipi  CrohnIPI folds 2-5 only                                   (control)
  head_hn   + 400 frames per lesion-free video of the training half    (round 1)
  head_bal  head_hn + expert-labelled frames of the training half's lesion videos:
            Ulcer -> U3-10, Erosion -> AU, Erythema -> E, Normal clean mucosa -> N (up to 150 per video)
Every video is scored by the head trained on the OTHER half. New check: frame-level AUROC on the held-out half's
expert-labelled frames (inflammatory vs normal clean mucosa), i.e. whether a head throws away Kvasir lesions.
Settings fixed before this run, same as round 1 (C = 0.1, sqrt class weights, seed 0).

    python hard_negative_head_v2.py        # writes results/hard_negative_head_v2.json
"""
import os, sys, csv, glob, json
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from hard_negative_head import (IB, KC, PREFIX, VID, WORK, INDEX, INFL, NEG_PER_VIDEO, SEED, auroc, softmax,
                                fit_head, class_weights)  # noqa: E402

OUT = os.path.join(IB, "results", "hard_negative_head_v2.json")
MAP = {"Ulcer": 4, "Erosion": 3, "Erythema": 1, "Normal clean mucosa": 0}
NORMAL_PER_VIDEO = 150


def labelled_sampled(F, labs):
    """sampled frame index -> expert class, for sampled frames within +/-4 of exactly one expert class"""
    by_f = {}
    for n, c in labs:
        by_f.setdefault(n, set()).add(c)
    out = {}
    for i, f in enumerate(F):
        cs = set()
        for n in range(f - 4, f + 5):
            cs |= by_f.get(n, set())
        if len(cs) == 1:
            out[i] = next(iter(cs))
    return out


def main():
    from kvasir_groups import groups
    g = groups(INDEX)
    z = np.load(os.path.join(WORK, f"crohnipi_feat_{PREFIX}_f1.npz"))
    Xi, yi, fi = z["feat"].astype(np.float32), z["y"], z["fold"]
    h = np.load(os.path.join(IB, "app", "models", f"{PREFIX}_fold1_head.npz")); W0, b0 = h["W"], h["b"]
    tr, te = fi != 1, fi == 1
    meta = list(csv.DictReader(open(os.path.join(KC, "metadata.csv"), encoding="utf-8", errors="replace"), delimiter=";"))
    lab = {}
    for r in meta:
        lab.setdefault(r["video_id"], []).append((int(r["frame_number"]), r["finding_class"]))
    vids = sorted(os.path.basename(f)[:-4] for f in glob.glob(os.path.join(VID, "*.npz")) if "_seg" not in os.path.basename(f))
    rng = np.random.default_rng(SEED)
    neg = {"A": [], "B": []}; posX = {"A": [], "B": []}; posY = {"A": [], "B": []}; cnt = {"A": {}, "B": {}}
    for v in vids:
        zz = np.load(os.path.join(VID, v + ".npz")); half = g[v]
        if not int(zz["expert_inflammatory"]):  # lesion-free video: raw-video negatives, exactly as in round 1
            sc = 1 - zz["probs"][:, 0]; order = np.argsort(-sc)
            n_hard = min(len(sc), int(NEG_PER_VIDEO * 0.6)); rest = order[n_hard:]
            pick = np.concatenate([order[:n_hard], rng.choice(rest, size=min(len(rest), NEG_PER_VIDEO - n_hard), replace=False)])
            neg[half].append(zz["feat"][pick].astype(np.float32))
        else:  # lesion video: its expert-labelled sampled frames (used only by head_bal, only for the OTHER half's head)
            ls = labelled_sampled(zz["frame"], lab.get(v, []))
            infl = [i for i, c in ls.items() if c in INFL]
            norm = [i for i, c in ls.items() if c == "Normal clean mucosa"]
            norm = list(rng.choice(norm, size=min(len(norm), NORMAL_PER_VIDEO), replace=False)) if norm else []
            idx = infl + norm
            if idx:
                posX[half].append(zz["feat"][idx].astype(np.float32)); posY[half].append(np.array([MAP[ls[i]] for i in idx]))
                cnt[half][v] = dict(inflammatory=len(infl), normal=len(norm))
    yi_tr = yi[tr]
    heads = {"head_ipi": fit_head(Xi[tr], yi_tr, class_weights(yi_tr))}
    for half in ("A", "B"):
        Xn = np.concatenate(neg[half]); X1 = np.concatenate([Xi[tr], Xn]); y1 = np.concatenate([yi_tr, np.zeros(len(Xn), int)])
        heads["hn_" + half] = fit_head(X1, y1, class_weights(y1))
        Xp, yp = np.concatenate(posX[half]), np.concatenate(posY[half])
        X2, y2 = np.concatenate([X1, Xp]), np.concatenate([y1, yp])
        heads["bal_" + half] = fit_head(X2, y2, class_weights(y2))
        print(f"half {half}: {len(Xn)} raw-video negatives; {int((yp != 0).sum())} Kvasir lesion + {int((yp == 0).sum())} Kvasir normal frames", flush=True)

    def ipi(P):
        yb = yi[te] != 0; s = 1 - P[:, 0]
        return dict(auroc=auroc(s, yb), acc7=float((P.argmax(1) == yi[te]).mean()), acc_at_0_5=float(((s >= 0.5) == yb).mean()))
    res = dict(crohnipi_fold1=dict(base=ipi(softmax(Xi[te] @ W0.T + b0)), head_ipi=ipi(heads["head_ipi"](Xi[te])),
                                   head_hn=ipi(heads["hn_A"](Xi[te])), head_bal=ipi(heads["bal_A"](Xi[te]))))
    keys = ("base", "head_ipi", "head_hn", "head_bal")
    per_video, frame_s, frame_y = [], {k: [] for k in keys}, []
    for v in vids:
        zz = np.load(os.path.join(VID, v + ".npz")); Fr = zz["frame"]; Xv = zz["feat"].astype(np.float32)
        other = "B" if g[v] == "A" else "A"
        S = dict(base=1 - zz["probs"][:, 0], head_ipi=1 - heads["head_ipi"](Xv)[:, 0],
                 head_hn=1 - heads["hn_" + other](Xv)[:, 0], head_bal=1 - heads["bal_" + other](Xv)[:, 0])
        infl = np.array(sorted(n for n, c in lab.get(v, []) if c in INFL), int)
        ls = labelled_sampled(Fr, lab.get(v, []))
        li = [i for i, c in ls.items() if c in INFL or c == "Normal clean mucosa"]
        frame_y += [ls[i] in INFL for i in li]
        rec = dict(video=v, half=g[v], has=int(bool(zz["expert_inflammatory"])))
        for k, s in S.items():
            frame_s[k] += list(s[li])
            rec[k] = dict(p99=float(np.percentile(s, 99)), top1=float(np.sort(s)[-max(1, len(s) // 100):].mean()))
            if len(infl):
                order = np.argsort(-s)
                for q in (5, 10):
                    sel = Fr[order[:max(1, int(len(s) * q / 100))]]
                    rec[k][f"recall_top{q}"] = float(np.mean([np.abs(sel - n).min() <= 10 for n in infl]))
        per_video.append(rec)
    y = np.array([r["has"] for r in per_video]); fy = np.array(frame_y, bool)
    rb = np.random.default_rng(1); boot = [rb.integers(0, len(per_video), len(per_video)) for _ in range(2000)]
    summ = {}
    for k in keys:
        p = np.array([r[k]["p99"] for r in per_video]); fs = np.array(frame_s[k])
        bs = [a for a in (auroc(p[i], y[i]) for i in boot) if a is not None]
        lv = [r[k] for r in per_video if r["has"]]
        summ[k] = dict(study_auroc_p99=auroc(p, y), study_auroc_p99_ci95=[float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
                       study_auroc_top1pct=auroc([r[k]["top1"] for r in per_video], y),
                       median_recall_top5=float(np.median([r["recall_top5"] for r in lv])),
                       median_recall_top10=float(np.median([r["recall_top10"] for r in lv])),
                       heldout_frames_auroc=auroc(fs, fy), heldout_frames_sensitivity_0_5=float((fs[fy] >= 0.5).mean()),
                       heldout_frames_specificity_0_5=float((fs[~fy] < 0.5).mean()))
    res.update(note="Frozen v2 fold-1 backbone, final layer retrained; every 10th frame of the 43 labelled Kvasir-Capsule videos; "
                    "each video scored by the head trained on the OTHER half. heldout_frames = expert-labelled sampled frames "
                    "(inflammatory vs normal clean mucosa) of held-out videos.",
               heldout_frames=dict(n=int(len(fy)), inflammatory=int(fy.sum())), training_extra=cnt, summary=summ, per_video=per_video)
    json.dump(res, open(OUT, "w"), indent=1)
    print(json.dumps(dict(crohnipi_fold1=res["crohnipi_fold1"], heldout_frames=res["heldout_frames"], summary=summ), indent=1))


if __name__ == "__main__":
    main()
