"""Export the HMP2 random forest + imputer to plain numpy arrays so the app needs no scikit-learn at runtime."""
import json, joblib, numpy as np, os, shutil
b = joblib.load("artifacts/flare/flare_model.joblib"); pipe = b["model"]; imp = pipe[0]; rf = pipe[-1]
trees = rf.estimators_; n = max(t.tree_.node_count for t in trees); T = len(trees)
L = np.full((T, n), -1, np.int32); R = np.full((T, n), -1, np.int32); F = np.zeros((T, n), np.int32); TH = np.zeros((T, n), np.float64); V = np.zeros((T, n), np.float64)
for i, t in enumerate(trees):
    tr = t.tree_; c = tr.node_count; val = tr.value[:, 0, :]; frac = val[:, 1] / val.sum(1)
    L[i, :c], R[i, :c], F[i, :c], TH[i, :c], V[i, :c] = tr.children_left, tr.children_right, tr.feature, tr.threshold, frac
ind = imp.indicator_.features_ if imp.indicator_ is not None else np.array([], int)
np.savez_compressed("app/models/flare_rf.npz", left=L, right=R, feature=F, threshold=TH, value=V, medians=imp.statistics_, indicator_features=ind)
json.dump(dict(features=b["features"], nice=b["nice"], best=b["best"]), open("app/models/flare_meta.json", "w"), indent=1)
for f in ["flare_metrics.json", "flare_timelines.json"]: shutil.copy(f"artifacts/flare/{f}", f"app/models/{f}")
# parity check
import pandas as pd
X = b["X_background"]; ref = pipe.predict_proba(X)[:, 1]
Xi = X.values.astype(float).copy(); med = imp.statistics_
for j in range(Xi.shape[1]): Xi[np.isnan(Xi[:, j]), j] = med[j]
Xi = np.hstack([Xi, np.isnan(X.values[:, ind]).astype(float)])
def tree_pred(x):
    out = 0
    for i in range(T):
        nd = 0
        while L[i, nd] != -1: nd = L[i, nd] if x[F[i, nd]] <= TH[i, nd] else R[i, nd]
        out += V[i, nd]
    return out / T
mine = np.array([tree_pred(x) for x in Xi]); print("parity max abs diff", np.abs(mine - ref).max(), "trees", T, "nodes", n)
