"""Linear heads on frozen ImageNet EfficientNet-B0 features (run on the local machine with onnxruntime-extracted features).

  python train_heads.py limuc      -> UC Mayo grader, 5-fold patient-grouped CV + final head + HyperKvasir external test
  python train_heads.py kvasir     -> small-bowel multi-finding screen, 5-fold video-grouped CV + final head
"""
import os, sys, json, glob, time, numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score, cohen_kappa_score, roc_auc_score,
                             confusion_matrix, recall_score)
from scipy.stats import spearmanr
B = os.path.expanduser("~/mnt/MP_Cohn Disease/interim_build"); FE = os.path.join(B, "_features"); P = os.path.join(B, "_packed")
MOD = os.path.join(B, "app", "models"); RES = os.path.join(B, "results"); os.makedirs(MOD, exist_ok=True)
C_REG = 0.05  # fixed a priori (same value as the CrohnIPI linear probe) - no tuning on test folds

def feats(name, tag="imnet"):
    fs = sorted(glob.glob(os.path.join(FE, f"feat_{tag}_{name}_*.npy"))); return np.concatenate([np.load(f) for f in fs]).astype(np.float32)

def fit(X, y):
    sc = StandardScaler().fit(X); lr = LogisticRegression(C=C_REG, max_iter=2000, class_weight="balanced").fit(sc.transform(X), y)
    return sc, lr

def save_head(name, sc, lr, classes, meta):
    np.savez(os.path.join(MOD, f"{name}.npz"), mu=sc.mean_.astype(np.float32), sd=sc.scale_.astype(np.float32),
             coef=lr.coef_.astype(np.float32), b=lr.intercept_.astype(np.float32), classes=np.array(classes))
    json.dump(meta, open(os.path.join(MOD, f"{name}_meta.json"), "w"), indent=1)

def boot_ci(fn, y, p, groups, n=500, seed=0):
    rng = np.random.default_rng(seed); ug = np.unique(groups); idx = {g: np.where(groups == g)[0] for g in ug}; v = []
    for _ in range(n):
        s = np.concatenate([idx[g] for g in rng.choice(ug, len(ug))])
        try: v.append(fn(y[s], p[s]))
        except Exception: pass
    return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]

def limuc():
    t0 = time.time(); idx = pd.read_csv(os.path.join(P, "limuc_index.csv")); X = feats("limuc"); assert len(X) == len(idx)
    y = idx.label.str.extract(r"(\d)")[0].astype(int).values; g = idx.patient.astype(str).values
    Pr = np.zeros((len(y), 4)); cv = StratifiedGroupKFold(5, shuffle=True, random_state=0)
    for k, (tr, te) in enumerate(cv.split(X, y, g)):
        sc, lr = fit(X[tr], y[tr]); Pr[te] = lr.predict_proba(sc.transform(X[te])); print("fold", k, round(time.time() - t0), flush=True)
    pred = Pr.argmax(1); act = (y >= 2).astype(int); pa = Pr[:, 2:].sum(1)
    m = dict(dataset="LIMUC", images=int(len(y)), patients=int(len(np.unique(g))), class_counts=np.bincount(y).tolist(),
             cv="5-fold StratifiedGroupKFold by patient (no patient in both train and test)",
             accuracy=accuracy_score(y, pred), macro_f1=f1_score(y, pred, average="macro"), qwk=cohen_kappa_score(y, pred, weights="quadratic"),
             qwk_ci95=boot_ci(lambda a, b: cohen_kappa_score(a, b.argmax(1), weights="quadratic"), y, Pr, g),
             per_class_recall=recall_score(y, pred, average=None).tolist(), confusion=confusion_matrix(y, pred).tolist(),
             remission_vs_active=dict(auroc=roc_auc_score(act, pa), auroc_ci95=boot_ci(roc_auc_score, act, pa, g),
                                      accuracy=accuracy_score(act, pa >= 0.5), sensitivity=recall_score(act, pa >= 0.5), specificity=recall_score(1 - act, pa < 0.5)))
    np.save(os.path.join(RES, "oof_limuc.npy"), Pr)
    sc, lr = fit(X, y); save_head("uc_mayo_head", sc, lr, [0, 1, 2, 3], dict(trained_on="LIMUC (all 11,276 images)", cv=m))
    # external test: HyperKvasir UC grades (different hospital, Norway)
    hk = pd.read_csv(os.path.join(P, "hyperkvasir_uc_index.csv")); Xh = feats("hyperkvasir_uc"); Ph = lr.predict_proba(sc.transform(Xh))
    gmap = {"grade-0-1": 0.5, "grade-1": 1, "grade-1-2": 1.5, "grade-2": 2, "grade-2-3": 2.5, "grade-3": 3}
    gnum = hk.label.map(gmap).values; expm = Ph @ np.arange(4); keep = gnum != 1.5
    act_h = (gnum[keep] >= 2).astype(int); pa_h = Ph[keep][:, 2:].sum(1); single = np.isin(gnum, [1, 2, 3])
    ext = dict(dataset="HyperKvasir UC subset", images=int(len(hk)), grade_counts=hk.label.value_counts().to_dict(),
               spearman_expected_mayo_vs_grade=float(spearmanr(expm, gnum).correlation),
               remission_vs_active=dict(n=int(keep.sum()), auroc=roc_auc_score(act_h, pa_h), accuracy=accuracy_score(act_h, pa_h >= 0.5),
                                        sensitivity=recall_score(act_h, pa_h >= 0.5), specificity=recall_score(1 - act_h, pa_h < 0.5)),
               exact_mayo_accuracy_on_single_grade_images=float((Ph[single].argmax(1) == gnum[single]).mean()),
               mean_expected_mayo_by_grade={k: float(expm[hk.label.values == k].mean()) for k in gmap if (hk.label.values == k).any()},
               note="HyperKvasir has no Mayo-0 class; images labelled 1-2 are excluded from the binary test")
    out = dict(internal=m, external=ext); json.dump(out, open(os.path.join(RES, "uc_metrics.json"), "w"), indent=1, default=float)
    print(json.dumps(dict(acc=m["accuracy"], mf1=m["macro_f1"], qwk=m["qwk"], rem_auc=m["remission_vs_active"]["auroc"],
                          ext_auc=ext["remission_vs_active"]["auroc"], ext_rho=ext["spearman_expected_mayo_vs_grade"]), default=float))

KV_MAP = {"normal_clean_mucosa": "normal", "ulcer": "ulcer", "erosion": "erosion", "erythema": "erythema", "angiectasia": "angiectasia",
          "blood_fresh": "blood", "blood_hematin": "blood", "lymphangiectasia": "lymphangiectasia", "foreign_body": "foreign_body", "polyp": "polyp"}
KV_CLASSES = ["normal", "ulcer", "erosion", "erythema", "angiectasia", "blood", "lymphangiectasia", "foreign_body", "polyp"]

def kvasir():
    t0 = time.time(); idx = pd.read_csv(os.path.join(P, "kvasir_capsule_index.csv")); X = feats("kvasir_capsule"); assert len(X) == len(idx)
    lab = idx.label.map(KV_MAP); y = lab.map(KV_CLASSES.index).values; g = idx.video.astype(str).values
    Pr = np.zeros((len(y), len(KV_CLASSES))); cv = StratifiedGroupKFold(5, shuffle=True, random_state=0)
    for k, (tr, te) in enumerate(cv.split(X, y, g)):
        sc, lr = fit(X[tr], y[tr]); Pr[te][:, :] = 0
        Pr[np.ix_(te, lr.classes_)] = lr.predict_proba(sc.transform(X[te])); print("fold", k, round(time.time() - t0), flush=True)
    pred = Pr.argmax(1); ab = (y != 0).astype(int); pab = 1 - Pr[:, 0]
    m = dict(dataset="Kvasir-Capsule (labelled images; normal class randomly sub-sampled to 4,000 of 34,338)", images=int(len(y)),
             videos=int(len(np.unique(g))), class_counts={c: int((y == i).sum()) for i, c in enumerate(KV_CLASSES)},
             cv="5-fold StratifiedGroupKFold by capsule video", accuracy=accuracy_score(y, pred), balanced_accuracy=balanced_accuracy_score(y, pred),
             macro_f1=f1_score(y, pred, average="macro"), per_class_recall=dict(zip(KV_CLASSES, recall_score(y, pred, average=None).tolist())),
             confusion=confusion_matrix(y, pred).tolist(),
             normal_vs_abnormal=dict(auroc=roc_auc_score(ab, pab), auroc_ci95=boot_ci(roc_auc_score, ab, pab, g),
                                     sensitivity=recall_score(ab, pab >= 0.5), specificity=recall_score(1 - ab, pab < 0.5)))
    np.save(os.path.join(RES, "oof_kvasir.npy"), Pr)
    sc, lr = fit(X, y); save_head("sb_findings_head", sc, lr, KV_CLASSES, dict(trained_on="Kvasir-Capsule", cv=m))
    json.dump(m, open(os.path.join(RES, "kvasir_findings_metrics.json"), "w"), indent=1, default=float)
    print(json.dumps(dict(acc=m["accuracy"], bal=m["balanced_accuracy"], mf1=m["macro_f1"], auc=m["normal_vs_abnormal"]["auroc"],
                          recall=m["per_class_recall"]), default=float))

if __name__ == "__main__":
    globals()[sys.argv[1]]()
