"""Hard negatives, head only (laptop CPU, minutes): do raw-video negatives fix whole-study ranking?

The backbone stays frozen (v2 fold-1 model, the same model that ran stage 1 of the whole-video scan). Only its final
linear layer is retrained, on pooled features that are already stored for every 10th frame of the 43 Kvasir-Capsule videos.
  base      the model's own classifier, as in the app                               (no retraining)
  head_ipi  new logistic head, CrohnIPI folds 2-5 only                               (control: same retraining, no negatives)
  head_hn   new logistic head, CrohnIPI folds 2-5 + frames of lesion-free videos of ONE half of the videos
The 43 videos are split into halves A and B (kvasir_groups.py, as in the GPU kit). The head trained with half A's
negatives scores half B's videos and vice versa, so no video is scored by a head that saw any of its frames.
CrohnIPI fold 1 (never seen by the backbone or any head) checks that the lesion classifier itself is not damaged.

Settings fixed before the first run: 400 negatives per lesion-free training video (60% highest base score, 40% random,
seed 0), features standardised on the training set, multinomial logistic regression C = 0.1, square-root
inverse-frequency class weights, max_iter 3000.

    python hard_negative_head.py crohnipi   # pooled features of the 3,484 CrohnIPI frames (resumable, ~2 min)
    python hard_negative_head.py run        # train + evaluate -> results/hard_negative_head.json
"""
import os, sys, csv, glob, json, time
import numpy as np
from PIL import Image

BASE = os.path.expanduser("~/mnt/MP_Cohn Disease")
IB = os.path.join(BASE, "interim_build")
KC = os.path.join(BASE, "temorary solution", "Kvasir-Capsule")
IPI = os.path.join(BASE, "temorary solution", "DataCrohnIPI")
PREFIX = os.environ.get("MODEL_PREFIX", "crohnipi_effb0v2")
VID = os.path.join(IB, "results", "bulk", "kvasir_videos_" + PREFIX)
WORK = os.path.join(IB, "results", "bulk", "hard_negative_head")
OUT = os.path.join(IB, "results", "hard_negative_head.json")
INDEX = os.path.join(IB, "data_kvasir_video_frames", "index.csv")
CLASSES = ["N", "E", "O", "AU", "U3-10", "U>10", "S"]
INFL = ("Ulcer", "Erosion", "Erythema")
MEAN = np.array([0.485, 0.456, 0.406], np.float32).reshape(1, 3, 1, 1)
STD = np.array([0.229, 0.224, 0.225], np.float32).reshape(1, 3, 1, 1)
NEG_PER_VIDEO, C_REG, SEED = 400, 0.1, 0
sys.path.insert(0, os.path.join(IB, "gpu"))


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


def softmax(z):
    z = z - z.max(1, keepdims=True); e = np.exp(z); return e / e.sum(1, keepdims=True)


def crohnipi_features(seconds=150):
    import onnxruntime as ort
    os.makedirs(WORK, exist_ok=True)
    rows = list(csv.DictReader(open(os.path.join(IPI, "CrohnIPI_description.csv"))))
    keys = list(rows[0].keys())
    files = [r[keys[0]] for r in rows]; labels = [CLASSES.index(r[keys[1]]) for r in rows]; folds = [int(r[keys[2]]) for r in rows]
    cache = os.path.join(WORK, f"crohnipi_feat_{PREFIX}_f1.npz")
    F = np.load(cache)["feat"] if os.path.exists(cache) else np.zeros((0, 1280), np.float16)
    so = ort.SessionOptions(); so.intra_op_num_threads = 2
    s = ort.InferenceSession(os.path.join(IB, "app", "models", f"{PREFIX}_fold1.onnx"), so, providers=["CPUExecutionProvider"])
    t0 = time.time(); done = len(F)
    while done < len(files) and time.time() - t0 < seconds:
        chunk = files[done:done + 64]
        x = np.stack([np.asarray(Image.open(os.path.join(IPI, "imgs", f)).convert("RGB").resize((224, 224), Image.BICUBIC)) for f in chunk])
        x = np.ascontiguousarray((x.astype(np.float32).transpose(0, 3, 1, 2) / 255.0 - MEAN) / STD, np.float32)
        fmap = s.run(["fmap"], {"image": x})[0]
        F = np.concatenate([F, fmap.mean((2, 3)).astype(np.float16)]); done = len(F)
        np.savez(cache + ".tmp.npz", feat=F, y=np.array(labels[:done]), fold=np.array(folds[:done]))
        os.replace(cache + ".tmp.npz", cache)
    print(f"CrohnIPI features: {done}/{len(files)}")


def fit_head(X, y, w_class):
    from sklearn.linear_model import LogisticRegression
    mu, sd = X.mean(0), X.std(0) + 1e-6
    clf = LogisticRegression(C=C_REG, max_iter=3000)
    clf.fit((X - mu) / sd, y, sample_weight=w_class[y])
    return lambda Z: clf.predict_proba((np.asarray(Z, np.float32) - mu) / sd)


def class_weights(y):
    cnt = np.bincount(y, minlength=7).astype(float)
    w = (cnt.max() / np.maximum(cnt, 1)) ** 0.5
    return w / w.mean()


def run():
    from kvasir_groups import groups
    g = groups(INDEX)
    z = np.load(os.path.join(WORK, f"crohnipi_feat_{PREFIX}_f1.npz"))
    Xi, yi, fi = z["feat"].astype(np.float32), z["y"], z["fold"]
    assert len(Xi) == 3484, "run the crohnipi step first"
    h = np.load(os.path.join(IB, "app", "models", f"{PREFIX}_fold1_head.npz")); W0, b0 = h["W"], h["b"]
    tr, te = fi != 1, fi == 1
    meta = list(csv.DictReader(open(os.path.join(KC, "metadata.csv"), encoding="utf-8", errors="replace"), delimiter=";"))
    lab = {}
    for r in meta:
        lab.setdefault(r["video_id"], []).append((int(r["frame_number"]), r["finding_class"]))
    vids = sorted(os.path.basename(f)[:-4] for f in glob.glob(os.path.join(VID, "*.npz")) if "_seg" not in os.path.basename(f))
    assert len(vids) == 43, len(vids)
    rng = np.random.default_rng(SEED)
    neg = {"A": [], "B": []}; n_neg_v = {"A": 0, "B": 0}
    for v in vids:
        zz = np.load(os.path.join(VID, v + ".npz"))
        if int(zz["expert_inflammatory"]):
            continue
        sc = 1 - zz["probs"][:, 0]; order = np.argsort(-sc)
        n_hard = min(len(sc), int(NEG_PER_VIDEO * 0.6)); rest = order[n_hard:]
        pick = np.concatenate([order[:n_hard], rng.choice(rest, size=min(len(rest), NEG_PER_VIDEO - n_hard), replace=False)])
        neg[g[v]].append(zz["feat"][pick].astype(np.float32)); n_neg_v[g[v]] += 1
    heads = {}
    yi_tr = yi[tr]
    heads["head_ipi"] = fit_head(Xi[tr], yi_tr, class_weights(yi_tr))
    for half in ("A", "B"):
        Xn = np.concatenate(neg[half]); X = np.concatenate([Xi[tr], Xn]); y = np.concatenate([yi_tr, np.zeros(len(Xn), int)])
        heads["hn_" + half] = fit_head(X, y, class_weights(y))
        print(f"head trained with {len(Xn)} negatives from {n_neg_v[half]} lesion-free videos of half {half}", flush=True)

    def ipi_metrics(P):
        yb = yi[te] != 0; s = 1 - P[:, 0]
        return dict(auroc=auroc(s, yb), acc_at_0_5=float(((s >= 0.5) == yb).mean()), acc7=float((P.argmax(1) == yi[te]).mean()),
                    sensitivity=float((s[yb] >= 0.5).mean()), specificity=float((s[~yb] < 0.5).mean()))
    res = dict(crohnipi_fold1=dict(base=ipi_metrics(softmax(Xi[te] @ W0.T + b0)), head_ipi=ipi_metrics(heads["head_ipi"](Xi[te])),
                                   head_hn_A=ipi_metrics(heads["hn_A"](Xi[te])), head_hn_B=ipi_metrics(heads["hn_B"](Xi[te]))))
    per_video = []
    for v in vids:
        zz = np.load(os.path.join(VID, v + ".npz")); Fr = zz["frame"]; Xv = zz["feat"].astype(np.float32)
        other = "B" if g[v] == "A" else "A"  # head trained on the OTHER half's negatives
        scores = dict(base=1 - zz["probs"][:, 0], head_ipi=1 - heads["head_ipi"](Xv)[:, 0], head_hn=1 - heads["hn_" + other](Xv)[:, 0])
        infl = np.array(sorted(n for n, c in lab.get(v, []) if c in INFL), int)
        rec = dict(video=v, half=g[v], has=int(zz["expert_inflammatory"]), n=int(len(Fr)))
        for k, s in scores.items():
            rec[k] = dict(p99=float(np.percentile(s, 99)), top1=float(np.sort(s)[-max(1, len(s) // 100):].mean()), flagged=float((s >= 0.5).mean()))
            if len(infl):
                order = np.argsort(-s)
                for q in (1, 5, 10):
                    sel = Fr[order[:max(1, int(len(s) * q / 100))]]
                    rec[k][f"recall_top{q}"] = float(np.mean([np.abs(sel - n).min() <= 10 for n in infl]))
        per_video.append(rec)
    y = [r["has"] for r in per_video]
    summ = {}
    for k in ("base", "head_ipi", "head_hn"):
        lv = [r for r in per_video if r["has"] and "recall_top5" in r[k]]
        summ[k] = dict(study_auroc_p99=auroc([r[k]["p99"] for r in per_video], y), study_auroc_top1pct=auroc([r[k]["top1"] for r in per_video], y),
                       study_auroc_flagged=auroc([r[k]["flagged"] for r in per_video], y),
                       median_flagged_pct_lesion_free=float(100 * np.median([r[k]["flagged"] for r in per_video if not r["has"]])),
                       median_flagged_pct_lesion=float(100 * np.median([r[k]["flagged"] for r in per_video if r["has"]])),
                       median_recall_top1=float(np.median([r[k]["recall_top1"] for r in lv])), median_recall_top5=float(np.median([r[k]["recall_top5"] for r in lv])),
                       median_recall_top10=float(np.median([r[k]["recall_top10"] for r in lv])))
    # bootstrap over videos for the study AUROC difference (hard-negative head minus base)
    rb = np.random.default_rng(1); diffs = []
    yy = np.array(y)
    for _ in range(2000):
        i = rb.integers(0, len(per_video), len(per_video))
        a = auroc([per_video[j]["head_hn"]["p99"] for j in i], yy[i]); b = auroc([per_video[j]["base"]["p99"] for j in i], yy[i])
        if a is not None and b is not None:
            diffs.append(a - b)
    summ["auroc_p99_gain_ci95"] = [float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))]
    res.update(note="Frozen v2 fold-1 backbone; only the final layer retrained. Every 10th frame of the 43 labelled Kvasir-Capsule videos; "
                    "each video scored by the head trained with the OTHER half's negatives. Settings fixed before the first run.",
               settings=dict(neg_per_video=NEG_PER_VIDEO, C=C_REG, seed=SEED, negatives_A=int(sum(len(x) for x in neg["A"])),
                             negatives_B=int(sum(len(x) for x in neg["B"]))),
               summary=summ, per_video=per_video)
    json.dump(res, open(OUT, "w"), indent=1)
    print(json.dumps(dict(crohnipi_fold1=res["crohnipi_fold1"], summary=summ), indent=1))


if __name__ == "__main__":
    {"crohnipi": crohnipi_features, "run": run}[sys.argv[1]]()
