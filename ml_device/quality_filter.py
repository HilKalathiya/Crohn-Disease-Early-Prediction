"""Frame-quality / landmark filter for capsule video (fixes the weak spot found on Kvasir-Capsule:
anatomical landmarks and poor-view frames are often flagged as lesions by the CrohnIPI model).

Training data: Kvasir-Capsule expert labels on the frames that the video scan sampled (every 10th frame), with the
pooled fold-1 CrohnIPI features the scan stored.  Target: DROP = ileocecal valve, pylorus, ampulla of Vater, reduced
mucosal view;  KEEP = normal clean mucosa + every pathology class (lesions must never be filtered away).
Validation: 5-fold GroupKFold by VIDEO (a video's frames never train the filter that scores it).
The drop threshold for each fold is chosen on the training videos so that >= 98% of their inflammatory frames are kept.
Outputs:
  results/quality_filter.json                         - held-out metrics
  results/bulk/quality_filter_video_scores.npz        - per video: keep-mask for every sampled frame (cross-fitted)
  app/models/quality_filter_head.npz (+ _meta.json)   - final filter trained on all videos, for the app
"""
import os, csv, glob, json, numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

BASE = os.path.expanduser("~/mnt/MP_Cohn Disease")
KC = os.path.join(BASE, "temorary solution/Kvasir-Capsule")
VID = os.path.join(BASE, "interim_build/results/bulk/kvasir_videos")
RES = os.path.join(BASE, "interim_build/results")
APPM = os.path.join(BASE, "interim_build/app/models")
DROP = ("Ileocecal valve", "Pylorus", "Ampulla of Vater", "Reduced Mucosal View")
INFL = ("Ulcer", "Erosion", "Erythema")


def main():
    meta = list(csv.DictReader(open(os.path.join(KC, "metadata.csv"), encoding="utf-8", errors="replace"), delimiter=";"))
    lab = {}
    for r in meta:
        lab[(r["video_id"], int(r["frame_number"]))] = r["finding_class"]
    X, y, g, cls, lesion, vids_all = [], [], [], [], [], {}
    for f in sorted(x for x in glob.glob(os.path.join(VID, "*.npz")) if "_seg" not in os.path.basename(x)):
        v = os.path.basename(f)[:-4]
        z = np.load(f)
        if "feat" not in z.files:
            print("no features (rescan needed):", v)
            continue
        F, FE, P = z["frame"], z["feat"].astype(np.float32), z["probs"]
        vids_all[v] = (F, FE)
        for i, n in enumerate(F):
            c = lab.get((v, int(n)))
            if c is None:
                continue
            X.append(FE[i]); y.append(int(c in DROP)); g.append(v); cls.append(c); lesion.append(float(1 - P[i, 0]))
    X, y, g, cls, lesion = np.array(X), np.array(y), np.array(g), np.array(cls), np.array(lesion)
    print(f"labelled sampled frames: {len(y)} from {len(set(g))} videos | drop-class {y.sum()} | keep-class {(1 - y).sum()}")
    pdrop = np.zeros(len(y)); thr_by_video = {}
    folds = GroupKFold(n_splits=min(5, len(set(g))))
    models = []
    for k, (tr, te) in enumerate(folds.split(X, y, g)):
        sc = StandardScaler().fit(X[tr])
        m = LogisticRegression(C=0.05, class_weight="balanced", max_iter=2000).fit(sc.transform(X[tr]), y[tr])
        p_tr = m.predict_proba(sc.transform(X[tr]))[:, 1]
        inf_tr = np.isin(cls[tr], INFL)
        t = float(np.quantile(p_tr[inf_tr], 0.98)) if inf_tr.any() else 0.5  # keep >= 98% of training inflammatory frames
        t = max(t, 0.5)
        pdrop[te] = m.predict_proba(sc.transform(X[te]))[:, 1]
        for v in set(g[te]):
            thr_by_video[v] = (sc, m, t)
        models.append(dict(fold=k, threshold=t, test_videos=sorted(set(g[te]))))
    dropped = np.array([pdrop[i] >= thr_by_video[g[i]][2] for i in range(len(y))])
    per = {c: dict(n=int((cls == c).sum()), dropped_pct=float(100 * dropped[cls == c].mean())) for c in sorted(set(cls))}
    fp = (lesion >= 0.5) & ~np.isin(cls, INFL) & np.isin(cls, DROP + ("Normal clean mucosa",))
    res = dict(n_labelled_sampled_frames=int(len(y)), videos=len(set(g)), per_class=per,
               inflammatory_kept_pct=float(100 * (~dropped[np.isin(cls, INFL)]).mean()) if np.isin(cls, INFL).any() else None,
               landmark_and_poor_view_dropped_pct=float(100 * dropped[y == 1].mean()),
               normal_dropped_pct=float(100 * dropped[cls == "Normal clean mucosa"].mean()),
               lesion_model_false_positives_before=int(fp.sum()), lesion_model_false_positives_removed=int((fp & dropped).sum()),
               folds=models, cv="5-fold GroupKFold by video; threshold per fold keeps >=98% of training inflammatory frames")
    masks = {}
    for v, (F, FE) in vids_all.items():
        if v in thr_by_video:
            sc, m, t = thr_by_video[v]
        else:
            sc, m, t = thr_by_video[next(iter(thr_by_video))]
        masks[v] = m.predict_proba(sc.transform(FE))[:, 1] < t
    np.savez(os.path.join(RES, "bulk/quality_filter_video_scores.npz"), **masks)
    res["sampled_frames_kept_pct_all_videos"] = float(100 * np.mean(np.concatenate(list(masks.values()))))
    json.dump(res, open(os.path.join(RES, "quality_filter.json"), "w"), indent=1)
    sc = StandardScaler().fit(X)
    m = LogisticRegression(C=0.05, class_weight="balanced", max_iter=2000).fit(sc.transform(X), y)
    p = m.predict_proba(sc.transform(X))[:, 1]
    t = max(0.5, float(np.quantile(p[np.isin(cls, INFL)], 0.98))) if np.isin(cls, INFL).any() else 0.5
    np.savez(os.path.join(APPM, "quality_filter_head.npz"), mu=sc.mean_.astype(np.float32), sd=sc.scale_.astype(np.float32),
             coef=m.coef_[0].astype(np.float32), b=np.float32(m.intercept_[0]))
    json.dump(dict(threshold=t, features="pooled last-conv features of crohnipi fold-1 model", drop_classes=list(DROP),
                   trained_on="Kvasir-Capsule expert-labelled frames sampled by the video scan", heldout=res),
              open(os.path.join(APPM, "quality_filter_meta.json"), "w"), indent=1)
    print(json.dumps({k: v for k, v in res.items() if k != "folds"}, indent=1))


if __name__ == "__main__":
    main()
