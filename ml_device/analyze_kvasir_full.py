"""A + B: full Kvasir-Capsule external test (all 47,238 labelled frames) and per-video study triage (43 videos).
Reads the shards written by bulk_score_kvasir_frames.py. NumPy only."""
import os, glob, json, numpy as np
BASE = os.path.expanduser("~/mnt/MP_Cohn Disease/interim_build")
PREFIX = os.environ.get("MODEL_PREFIX", "crohnipi_effb0"); SUF = "" if PREFIX == "crohnipi_effb0" else "_" + PREFIX
SH = os.path.join(BASE, "results/bulk/kvasir_frames" + SUF)
OUT = os.path.join(BASE, f"results/kvasir_full_external{SUF}.json")
TIER = np.array([0, 1, 1, 2, 2, 3, 3])
INFL = ("Ulcer", "Erosion", "Erythema")
rng = np.random.default_rng(0)

def auroc(s, y):
    y = np.asarray(y, bool); s = np.asarray(s, float)
    o = np.argsort(s, kind="mergesort"); r = np.empty(len(s)); ss = s[o]
    i = 0
    while i < len(ss):
        j = i
        while j + 1 < len(ss) and ss[j + 1] == ss[i]: j += 1
        r[o[i:j + 1]] = (i + j) / 2 + 1; i = j + 1
    p, n = y.sum(), (~y).sum()
    return float((r[y].sum() - p * (p + 1) / 2) / (p * n)) if p and n else float("nan")

def spearman(a, b):
    ra = np.argsort(np.argsort(a)); rb = np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])

paths, vids, fcs, probs, sds = [], [], [], [], []
for f in sorted(glob.glob(os.path.join(SH, "shard_*.npz"))):
    with np.load(f) as z:
        paths += z["path"].tolist(); vids += z["video_id"].tolist(); fcs += z["finding_class"].tolist()
        probs.append(z["probs"]); sds.append(z["probs_sd"])
P = np.concatenate(probs); SD = np.concatenate(sds)
paths = np.array(paths); vids = np.array(vids); fcs = np.array(fcs)
_, keep = np.unique(paths, return_index=True); keep = np.sort(keep)
P, SD, paths, vids, fcs = P[keep], SD[keep], paths[keep], vids[keep], fcs[keep]
lesion = 1 - P[:, 0]
tierP = np.stack([P[:, TIER == t].sum(1) for t in range(4)], 1)
tier = tierP.argmax(1); exp_tier = tierP @ np.arange(4)
normal = fcs == "Normal clean mucosa"; infl = np.isin(fcs, INFL)
print("frames", len(P), "videos", len(np.unique(vids)), "normal", normal.sum(), "inflammatory", infl.sum())

# ---------- A: full external test ----------
A = dict(frames=int(len(P)), videos=int(len(np.unique(vids))), normal_frames=int(normal.sum()), inflammatory_frames=int(infl.sum()))
m = infl | normal
A["inflammatory_vs_normal"] = dict(auroc=auroc(lesion[m], infl[m]), sensitivity_at_0_5=float((lesion[infl] >= 0.5).mean()),
                                   specificity_at_0_5=float((lesion[normal] < 0.5).mean()))
# video-cluster bootstrap CI (frames of one video are correlated, so resample whole videos)
uv = np.unique(vids); idx_by_v = {v: np.where(vids == v)[0] for v in uv}; bs = []
for _ in range(500):
    pick = np.concatenate([idx_by_v[v] for v in rng.choice(uv, len(uv))])
    mm = pick[m[pick]]
    if infl[mm].any() and normal[mm].any(): bs.append(auroc(lesion[mm], infl[mm]))
A["inflammatory_vs_normal"]["auroc_ci95_video_bootstrap"] = [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]
per = {}
for c in sorted(set(fcs)):
    sel = fcs == c
    d = dict(n=int(sel.sum()), mean_lesion_score=float(lesion[sel].mean()), flagged_at_0_5=float((lesion[sel] >= 0.5).mean()),
             tier_share=[float((tier[sel] == t).mean()) for t in range(4)])
    if c != "Normal clean mucosa":
        mm = sel | normal; d["auroc_vs_normal"] = auroc(lesion[mm], sel[mm])
    per[c] = d
A["per_class"] = per
# ranking all frames of the dataset (pooled triage)
order = np.argsort(-lesion); hit = infl[order]
A["pooled_triage"] = {f"top_{q}pct": float(hit[:int(len(order) * q / 100)].sum() / infl.sum()) for q in (5, 10, 20, 30)}
A["uncertainty"] = dict(mean_sd_lesion_frames=float(SD[infl].max(1).mean()), mean_sd_normal_frames=float(SD[normal].max(1).mean()))

# ---------- B: per-video study triage ----------
videos = []
for v in uv:
    s = idx_by_v[v]; li = infl[s]; ls = lesion[s]
    o = np.argsort(-ls); hits = li[o]; n = len(s)
    rec = dict(video_id=str(v), labelled_frames=int(n), inflammatory_frames=int(li.sum()), prevalence=float(li.mean()),
               flagged_pct=float((ls >= 0.5).mean() * 100), burden_index=float(100 * exp_tier[s].mean() / 3),
               mean_lesion_score=float(ls.mean()), worst_tier=int(tier[s].max()),
               top_classes={c: int((fcs[s] == c).sum()) for c in set(fcs[s].tolist())})
    if li.any():
        rec["recall_top10pct"] = float(hits[:max(1, int(n * 0.10))].sum() / li.sum())
        rec["recall_top20pct"] = float(hits[:max(1, int(n * 0.20))].sum() / li.sum())
        rec["recall_top30pct"] = float(hits[:max(1, int(n * 0.30))].sum() / li.sum())
        rec["rank_first_lesion"] = int(np.argmax(hits) + 1)
        rec["auroc_within_video"] = auroc(ls, li) if (~li).any() else None
    videos.append(rec)
videos.sort(key=lambda r: -r["burden_index"])
has = np.array([r["inflammatory_frames"] > 0 for r in videos]); bi = np.array([r["burden_index"] for r in videos])
fl = np.array([r["flagged_pct"] for r in videos]); prev = np.array([r["prevalence"] for r in videos])
les_v = [r for r in videos if r["inflammatory_frames"] > 0]
B = dict(n_videos=len(videos), videos_with_inflammatory_lesions=int(has.sum()),
         video_auroc_burden_index=auroc(bi, has), video_auroc_flagged_pct=auroc(fl, has),
         spearman_burden_vs_prevalence=spearman(bi, prev),
         median_recall_top10pct=float(np.median([r["recall_top10pct"] for r in les_v])),
         median_recall_top20pct=float(np.median([r["recall_top20pct"] for r in les_v])),
         median_recall_top30pct=float(np.median([r["recall_top30pct"] for r in les_v])),
         median_rank_first_lesion=float(np.median([r["rank_first_lesion"] for r in les_v])),
         videos=videos)
json.dump(dict(A=A, B=B, note="5-fold EfficientNet-B0 ensemble trained on CrohnIPI only; single view, no TTA (same as the app); lesion score = 1 - P(normal)"),
          open(OUT, "w"), indent=1)
print(json.dumps(A["inflammatory_vs_normal"], indent=1))
print("pooled triage", A["pooled_triage"])
for c, d in per.items(): print(f"{c:22s} n={d['n']:6d} auroc={d.get('auroc_vs_normal', float('nan')):.3f} flagged={d['flagged_at_0_5']:.3f}")
print({k: v for k, v in B.items() if k != "videos"})
for r in les_v: print(r["video_id"], r["labelled_frames"], r["inflammatory_frames"], round(r["burden_index"], 1), round(r["recall_top20pct"], 2), r["rank_first_lesion"], r.get("auroc_within_video"))
