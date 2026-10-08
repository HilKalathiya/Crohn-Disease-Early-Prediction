"""Metrics for a new CrohnIPI run (same format as results_summary['crohnipi']['finetuned']).
Usage: python ml/assemble_v2.py <run_dir> <out_json> [label]"""
import os, sys, json, numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from data import table
from metrics import evaluate
from sklearn.metrics import roc_auc_score
run, out = sys.argv[1], sys.argv[2]; label = sys.argv[3] if len(sys.argv) > 3 else "EfficientNet-B0 v2 (ours)"
d = table(); y = d.y.values; P = np.zeros((len(y), 7)); seen = np.zeros(len(y), bool)
for k in range(1, 6):
    idx = np.load(f"{run}/idx_fold{k}.npy"); P[idx] = np.load(f"{run}/oof_fold{k}.npy"); seen[idx] = True
assert seen.all(), "missing folds"
ft = evaluate(y, P)
rng = np.random.default_rng(0); yb = (y != 0).astype(int); s = 1 - P[:, 0]; bs = []
for _ in range(1000):
    i = rng.integers(0, len(y), len(y)); bs.append(roc_auc_score(yb[i], s[i]))
ft["binary"]["auroc_ci95"] = [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]
ft["binary"]["note"] = "image-level bootstrap (CrohnIPI release has no patient IDs)"
grid = np.linspace(0.05, 0.95, 181); pred_cf = np.zeros(len(y), bool); thr = {}
for k in sorted(set(d.fold.values)):
    m = d.fold.values == k; o = ~m
    t = float(grid[int(np.argmax([((s[o] >= g) == yb[o]).mean() for g in grid]))]); thr[int(k)] = t; pred_cf[m] = s[m] >= t
ft["binary"]["crossfit"] = dict(accuracy=float((pred_cf == yb).mean()), sensitivity=float(pred_cf[yb == 1].mean()),
                                specificity=float((~pred_cf[yb == 0]).mean()), thresholds=thr,
                                note="threshold for each fold chosen on the other folds' out-of-fold scores")
ft["per_fold"] = []
for k in range(1, 6):
    m = d.fold.values == k; e = evaluate(y[m], P[m])
    ft["per_fold"].append(dict(fold=k, acc7=e["multiclass"]["accuracy"], bin_acc=e["binary"]["accuracy"], auroc=e["binary"]["auroc"], qwk=e["severity"]["qwk"]))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "gpu"))
import pandas as pd
from evaluate_oof import evaluate as ev2
dd = d.rename(columns={"label": "label"}).copy()
m2 = ev2(P, dd.assign(y=y))
ft["severity_crossfit"] = dict(qwk=m2["severity"]["crossfit_cut_qwk"], accuracy=m2["severity"]["crossfit_cut_accuracy"], cuts=m2["severity"]["crossfit_cuts"],
                               note="tier = expected severity cut at points chosen on the other four folds")
ft["label"] = label
np.save(os.path.join(run, "oof_all.npy"), P)
json.dump(ft, open(out, "w"), indent=1, default=float)
print(json.dumps(dict(acc7=ft["multiclass"]["accuracy"], bal=ft["multiclass"]["balanced_accuracy"], auroc=ft["binary"]["auroc"],
                      crossfit=ft["binary"]["crossfit"]["accuracy"], at05=ft["binary"]["accuracy"], qwk=ft["severity"]["qwk"],
                      qwk_xf=ft["severity_crossfit"]["qwk"]), indent=1, default=float))
