"""General small-bowel findings screen: one-vs-rest logistic heads trained on SEE-AI (Japan) and tested externally on Kvasir-Capsule (Norway).
Also reports the frame-split leakage effect on Kvasir-Capsule (random split vs capsule-video split)."""
import os, json, time, numpy as np, pandas as pd
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold
from sklearn.metrics import roc_auc_score, balanced_accuracy_score, f1_score, average_precision_score
exec(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "train_heads.py")).read().split("if __name__")[0])
SEE_COLS = ["angiodysplasia", "erosion", "stenosis", "lymphangiectasia", "lymph follicle", "SMT", "polyp-like", "bleeding", "diverticulum", "erythema", "foreign body", "vein"]
TARGETS = {"ulcer_erosion": ["erosion"], "erythema": ["erythema"], "angiectasia": ["angiodysplasia"], "bleeding": ["bleeding"],
           "lymphangiectasia": ["lymphangiectasia"], "polyp": ["polyp-like"], "foreign_body": ["foreign body"], "stenosis": ["stenosis"]}
KV_TO = {"ulcer": "ulcer_erosion", "erosion": "ulcer_erosion", "erythema": "erythema", "angiectasia": "angiectasia", "blood_fresh": "bleeding",
         "blood_hematin": "bleeding", "lymphangiectasia": "lymphangiectasia", "polyp": "polyp", "foreign_body": "foreign_body"}

def see_labels():
    a = pd.read_csv(os.path.join(os.path.expanduser("~/mnt/temorary solution/The SEE-AI Project Dataset/all_annotation.csv")), skiprows=[1])
    a = a[a.image_number.notna()].copy(); a["key"] = a.image_number.astype(int).map(lambda n: f"image{n:05d}.jpg"); idx = pd.read_csv(os.path.join(P, "seeai_index.csv"))
    a = idx[["key"]].merge(a, on="key", how="left"); assert len(a) == len(idx)
    Y = np.stack([(a[cols].fillna(0).sum(1) > 0).astype(int).values for cols in TARGETS.values()], 1)
    return a, Y

def main():
    t0 = time.time(); a, Y = see_labels(); X = feats("seeai"); none = (a[SEE_COLS].fillna(0).sum(1) == 0).values
    heads = {}; internal = {}
    cache = os.path.join(RES, "_screen_stage1.json")
    if os.path.exists(cache): return stage2(json.load(open(cache)), t0)
    sc = StandardScaler().fit(X); Xs = sc.transform(X)
    for j, t in enumerate(TARGETS):
        y = Y[:, j]; oof = np.zeros(len(y))
        for tr, te in StratifiedKFold(5, shuffle=True, random_state=0).split(Xs, y):
            lr = LogisticRegression(C=C_REG, max_iter=1000, class_weight="balanced").fit(Xs[tr], y[tr]); oof[te] = lr.predict_proba(Xs[te])[:, 1]
        internal[t] = dict(positives=int(y.sum()), auroc=roc_auc_score(y, oof), auprc=average_precision_score(y, oof))
        heads[t] = LogisticRegression(C=C_REG, max_iter=1000, class_weight="balanced").fit(Xs, y); print(t, round(internal[t]["auroc"], 3), round(time.time() - t0), flush=True)
    coef = np.stack([heads[t].coef_[0] for t in TARGETS]); b = np.array([heads[t].intercept_[0] for t in TARGETS])
    np.savez(os.path.join(MOD, "sb_findings_head.npz"), mu=sc.mean_.astype(np.float32), sd=sc.scale_.astype(np.float32), coef=coef.astype(np.float32),
             b=b.astype(np.float32), classes=np.array(list(TARGETS)), multilabel=np.array(True))
    json.dump(dict(internal=internal), open(cache, "w"), default=float); return stage2(dict(internal=internal), t0)

def stage2(st, t0):
    internal = st["internal"]; z = np.load(os.path.join(MOD, "sb_findings_head.npz")); coef, b = z["coef"], z["b"]
    sc = StandardScaler(); sc.mean_, sc.scale_ = z["mu"].astype(np.float64), z["sd"].astype(np.float64)
    kv = pd.read_csv(os.path.join(P, "kvasir_capsule_index.csv")); Xk = sc.transform(feats("kvasir_capsule")); Pk = 1 / (1 + np.exp(-(Xk @ coef.T + b)))
    external = {}
    for j, t in enumerate(TARGETS):
        pos = kv.label.map(KV_TO).eq(t).values; neg = kv.label.eq("normal_clean_mucosa").values; m = pos | neg
        if pos.sum() >= 10:
            external[t] = dict(positives=int(pos.sum()), negatives=int(neg.sum()), auroc_vs_normal=roc_auc_score(pos[m], Pk[m, j]))
    anyab = ~kv.label.eq("normal_clean_mucosa").values; score = Pk.max(1)
    external["any_finding_vs_normal"] = dict(auroc=roc_auc_score(anyab, score))
    # leakage demonstration on Kvasir-Capsule (multi-class, frozen features + same LR)
    kidx = kv; yk = kidx.label.map(KV_MAP).map(KV_CLASSES.index).values; gk = kidx.video.values; Xk2 = feats("kvasir_capsule"); leak = {}
    for name, splitter, groups in [("random_frame_split", StratifiedKFold(5, shuffle=True, random_state=0), None),
                                   ("video_grouped_split", StratifiedGroupKFold(5, shuffle=True, random_state=0), gk)]:
        pred = np.zeros(len(yk), int)
        for tr, te in splitter.split(Xk2, yk, groups):
            s2, l2 = fit(Xk2[tr], yk[tr]); pred[te] = l2.predict(s2.transform(Xk2[te]))
        if name == "random_frame_split" and "leak_random" in st: leak[name] = st["leak_random"]; continue
        leak[name] = dict(balanced_accuracy=balanced_accuracy_score(yk, pred), macro_f1=f1_score(yk, pred, average="macro"))
        print(name, leak[name], round(time.time() - t0), flush=True)
    vids = pd.Series(gk).groupby(yk).nunique(); leak["videos_per_class"] = {KV_CLASSES[i]: int(v) for i, v in vids.items()}
    out = dict(train="SEE-AI (18,481 images, image-level labels from expert boxes)", internal_random_cv=internal,
               external_kvasir=external, leakage_kvasir=leak,
               note="SEE-AI has no patient IDs, so internal CV is frame-level (optimistic); the Kvasir-Capsule external test is the honest number.")
    json.dump(out, open(os.path.join(RES, "sb_screen_metrics.json"), "w"), indent=1, default=float)
    json.dump(dict(trained_on="SEE-AI", multilabel=True, classes=list(TARGETS)), open(os.path.join(MOD, "sb_findings_head_meta.json"), "w"))
    print(json.dumps(external, default=float))
main()
