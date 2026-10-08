"""B2 analysis: whole-video (study-level) results from scan_kvasir_videos.py.

Per video we have every 10th frame scored by the fold-1 CrohnIPI model, plus the 5-model ensemble on its top-150 frames.
Expert labels exist only for the annotated subset of frames (metadata.csv), so:
  * study level: does the video's lesion signal separate videos WITH expert inflammatory labels from videos without?
  * localisation: are expert inflammatory frames near (within +/-10 frames) the frames the scan ranks highest?
Also writes an app-ready timeline file (lesion score along the video, expert lesion positions, top thumbnails).
NumPy only.
"""
import os, csv, json, glob, numpy as np

BASE = os.path.expanduser("~/mnt/MP_Cohn Disease")
KC = os.path.join(BASE, "temorary solution/Kvasir-Capsule")
PREFIX = os.environ.get("MODEL_PREFIX", "crohnipi_effb0"); SUF = "" if PREFIX == "crohnipi_effb0" else "_" + PREFIX
VID = os.path.join(BASE, "interim_build/results/bulk/kvasir_videos" + SUF)
OUT = os.path.join(BASE, f"interim_build/results/kvasir_video_study{SUF}.json")
APP_OUT = os.path.join(BASE, f"interim_build/app/models/video_timelines{SUF}.json")
INFL = ("Ulcer", "Erosion", "Erythema")
TIER = np.array([0, 1, 1, 2, 2, 3, 3])
FILTER = os.path.join(BASE, "interim_build/results/bulk/quality_filter_video_scores.npz")  # optional, written by the filter step


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


def runs(mask, min_len=3):
    """number of runs of >= min_len consecutive True values"""
    n, c = 0, 0
    for m in mask:
        c = c + 1 if m else 0
        if c == min_len:
            n += 1
    return n


def main():
    meta = list(csv.DictReader(open(os.path.join(KC, "metadata.csv"), encoding="utf-8", errors="replace"), delimiter=";"))
    by_v = {}
    for r in meta:
        by_v.setdefault(r["video_id"], []).append((int(r["frame_number"]), r["finding_class"]))
    keep_mask = None
    if os.path.exists(FILTER):
        keep_mask = dict(np.load(FILTER, allow_pickle=True))
    videos, timelines = [], {}
    for f in sorted(x for x in glob.glob(os.path.join(VID, "*.npz")) if "_seg" not in os.path.basename(x)):
        v = os.path.basename(f)[:-4]
        z = np.load(f)
        F, P = z["frame"], z["probs"]
        lesion = 1 - P[:, 0]
        good = keep_mask[v].astype(bool) if keep_mask is not None and v in keep_mask else np.ones(len(F), bool)
        lab = by_v.get(v, [])
        infl_frames = np.array(sorted(n for n, c in lab if c in INFL), int)
        # small-bowel window from the expert landmark labels: after the last pylorus frame, before the first ileocecal-valve frame
        pyl = [n for n, c in lab if c == "Pylorus"]; icv = [n for n, c in lab if c == "Ileocecal valve"]
        sb_lo = max(pyl) if pyl else 0
        sb_hi = min([n for n in icv if n > sb_lo], default=int(z["n_frames"])) if icv else int(z["n_frames"])
        in_sb = (F > sb_lo) & (F < sb_hi)
        top_ens = 1 - z["top_probs_ens"][:, 0]
        rec = dict(video_id=v, total_frames=int(z["n_frames"]), sampled=int(len(F)), expert_inflammatory_frames=int(len(infl_frames)),
                   has_inflammatory=bool(len(infl_frames) > 0),
                   other_findings=sorted({c for _, c in lab if c not in INFL + ("Normal clean mucosa",)}),
                   sb_window=[int(sb_lo), int(sb_hi)], sb_window_from=("pylorus" if pyl else "start") + "-" + ("ileocecal valve" if icv else "end"),
                   inflammatory_inside_sb_window_pct=float(100 * ((infl_frames > sb_lo) & (infl_frames < sb_hi)).mean()) if len(infl_frames) else None)
        for tag, m in (("all", np.ones(len(F), bool)), ("filtered", good), ("sb", in_sb), ("sb_filtered", in_sb & good)):
            ls = lesion[m]
            if len(ls) == 0:
                continue
            rec[tag] = dict(frames_kept=int(m.sum()), flagged_pct=float((ls >= 0.5).mean() * 100), p99=float(np.percentile(ls, 99)),
                            top1pct_mean=float(np.sort(ls)[-max(1, len(ls) // 100):].mean()),
                            runs_ge_0_9=int(runs(lesion * m >= 0.9)), runs_per_1000=float(1000 * runs(lesion * m >= 0.9) / max(1, m.sum())))
        rec["ensemble_top150_mean"] = float(top_ens.mean())
        # localisation: expert inflammatory frames within +/-10 frames of a sampled frame in the video's top-k%
        if len(infl_frames):
            for tag, m in (("all", np.ones(len(F), bool)), ("sb_filtered", in_sb & good)):
                order = np.argsort(-np.where(m, lesion, -1)); nm = max(1, int(m.sum()))
                for q in (1, 5, 10):
                    sel = F[order[:max(1, int(nm * q / 100))]]
                    near = np.array([np.abs(sel - n).min() <= 10 for n in infl_frames])
                    rec[f"lesion_recall_top{q}pct" + ("" if tag == "all" else "_sb_filtered")] = float(near.mean())
        videos.append(rec)
        # timeline for the app: max-pool to <= 600 bins
        nb = min(600, len(F))
        edges = np.linspace(0, len(F), nb + 1).astype(int)
        timelines[v] = dict(total_frames=int(z["n_frames"]), sampled_every=10,
                            bins=[[int(F[a]), round(float(lesion[a:b].max()), 3), round(float(lesion[a:b].mean()), 3)] for a, b in zip(edges[:-1], edges[1:]) if b > a],
                            expert=[[int(n), c] for n, c in sorted(lab) if c in INFL],
                            top=[dict(frame=int(n), lesion=round(float(1 - p[0]), 3), tier=int(np.argmax([p[TIER == t].sum() for t in range(4)])),
                                      top_class=int(np.argmax(p)))
                                 for n, p in sorted(zip(z["top_frames"], z["top_probs_ens"]), key=lambda t: -(1 - t[1][0]))[:12]],
                            has_inflammatory=rec["has_inflammatory"], sb_window=rec["sb_window"])
    y = np.array([r["has_inflammatory"] for r in videos])
    study = dict(n_videos=len(videos), with_inflammatory=int(y.sum()), without=int((~y).sum()))
    for tag in ("all", "filtered", "sb", "sb_filtered"):
        for key in ("flagged_pct", "p99", "top1pct_mean", "runs_per_1000"):
            vals = [r[tag][key] if tag in r else np.nan for r in videos]
            if not np.isnan(vals).any():
                study[f"auroc_{tag}_{key}"] = auroc(vals, y)
    study["auroc_ensemble_top150_mean"] = auroc([r["ensemble_top150_mean"] for r in videos], y)
    lv = [r for r in videos if r["has_inflammatory"]]
    for q in (1, 5, 10):
        if lv:
            study[f"median_lesion_recall_top{q}pct"] = float(np.median([r[f"lesion_recall_top{q}pct"] for r in lv]))
            study[f"median_lesion_recall_top{q}pct_sb_filtered"] = float(np.median([r[f"lesion_recall_top{q}pct_sb_filtered"] for r in lv]))
    study["median_inflammatory_inside_sb_window_pct"] = float(np.median([r["inflammatory_inside_sb_window_pct"] for r in lv])) if lv else None
    json.dump(dict(study=study, videos=videos, filter_applied=keep_mask is not None,
                   note="Stage 1: fold-1 CrohnIPI model on every 10th frame; stage 2: 5-model ensemble on the top 150 frames. Expert labels cover only the annotated subset of frames."),
              open(OUT, "w"), indent=1)
    json.dump(timelines, open(APP_OUT, "w"))
    print(json.dumps(study, indent=1))
    for r in videos:
        a, b = r["all"], r.get("sb_filtered", r["all"])
        print(f"{r['video_id']} infl={r['expert_inflammatory_frames']:4d} sb={r['sb_window_from'][:14]:14s} flagged all={a['flagged_pct']:5.1f}% sb+f={b['flagged_pct']:5.1f}% "
              f"p99 {a['p99']:.3f}/{b['p99']:.3f} rec@5% {r.get('lesion_recall_top5pct', float('nan')):.2f}/{r.get('lesion_recall_top5pct_sb_filtered', float('nan')):.2f} "
              f"in_sb={r['inflammatory_inside_sb_window_pct']} other={','.join(r['other_findings'])[:40]}")


if __name__ == "__main__":
    main()
