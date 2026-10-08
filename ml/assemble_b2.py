"""Metrics for the B2-320 two-seed ensemble (and each seed alone), same format as crohnipi_v2_metrics.json.
Usage: python ml/assemble_b2.py <out_json> <run_dir> [<run_dir> ...]"""
import os, sys, json, numpy as np
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "gpu"))
from data import table
from metrics import evaluate
from sklearn.metrics import roc_auc_score
from evaluate_oof import evaluate as ev2
out, runs = sys.argv[1], sys.argv[2:]
d = table(); y = d.y.values

def load(run):
    P = np.zeros((len(y), 7)); seen = np.zeros(len(y), bool)
    for k in range(1, 6):
        idx = np.load(f"{run}/idx_fold{k}.npy"); P[idx] = np.load(f"{run}/oof_fold{k}.npy"); seen[idx] = True
    assert seen.all(), run
    # check: every row's held-out fold matches the official fold
    for k in range(1, 6):
        idx = np.load(f"{run}/idx_fold{k}.npy"); assert (d.fold.values[idx] == k).all(), (run, k)
    return P

def full(P, label):
    ft = evaluate(y, P)
    rng = np.random.default_rng(0); yb = (y != 0).astype(int); s = 1 - P[:, 0]; bs = []
    for _ in range(1000):
        i = rng.integers(0, len(y), len(y)); bs.append(roc_auc_score(yb[i], s[i]))
    ft["binary"]["auroc_ci95"] = [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]
    ft["binary"]["note"] = "image-level bootstrap (CrohnIPI release has no patient IDs)"
    grid = np.linspace(0.05, 0.95, 181); pred = np.zeros(len(y), bool); thr = {}
    for k in sorted(set(d.fold.values)):
        m = d.fold.values == k; o = ~m
        t = float(grid[int(np.argmax([((s[o] >= g) == yb[o]).mean() for g in grid]))]); thr[int(k)] = t; pred[m] = s[m] >= t
    ft["binary"]["crossfit"] = dict(accuracy=float((pred == yb).mean()), sensitivity=float(pred[yb == 1].mean()),
                                    specificity=float((~pred[yb == 0]).mean()), thresholds=thr,
                                    note="threshold for each fold chosen on the other folds' out-of-fold scores")
    ft["per_fold"] = []
    for k in range(1, 6):
        m = d.fold.values == k; e = evaluate(y[m], P[m])
        ft["per_fold"].append(dict(fold=k, acc7=e["multiclass"]["accuracy"], bin_acc=e["binary"]["accuracy"], auroc=e["binary"]["auroc"], qwk=e["severity"]["qwk"]))
    m2 = ev2(P, d.assign(y=y))
    ft["severity_crossfit"] = dict(qwk=m2["severity"]["crossfit_cut_qwk"], accuracy=m2["severity"]["crossfit_cut_accuracy"], cuts=m2["severity"]["crossfit_cuts"],
                                   note="tier = expected severity cut at points chosen on the other four folds")
    cm = np.array(ft["severity"]["confusion"]); ft["severity"]["within_one"] = float(sum(cm[i, j] for i in range(4) for j in range(4) if abs(i - j) <= 1) / cm.sum())
    ft["label"] = label
    return ft

Ps = [load(r) for r in runs]
res = full(np.mean(Ps, 0), "EfficientNet-B2 320 px, 2 seeds (ours)")
res["members"] = [os.path.basename(r.rstrip("/")) for r in runs]
res["recipe"] = dict(backbone="efficientnet_b2.ra_in1k", input_size=320, epochs=40, batch="8 x 4 accumulation", ema=0.998, tta="8 views", seeds=[0, 1],
                     decided_before_scoring="two-seed average as final model; switch the app only if it beats v2")
res["single_seed"] = {os.path.basename(r.rstrip("/")): {k: v for k, v in full(P, r).items() if k in ("binary", "multiclass", "severity", "severity_crossfit")} for r, P in zip(runs, Ps)}
np.save(os.path.join(os.path.dirname(out), "oof_b2_ensemble.npy"), np.mean(Ps, 0))
json.dump(res, open(out, "w"), indent=1, default=float)
def show(t, r):
    print(t, "AUROC %.4f CI %s | crossfit %.4f (sens %.3f spec %.3f) | @0.5 %.4f (sens %.3f spec %.3f) | acc7 %.4f bal %.4f F1 %.4f | sev acc %.4f within1 %s qwk %.4f xf %.4f" % (
        r["binary"]["auroc"], np.round(r["binary"].get("auroc_ci95", [0, 0]), 3), r["binary"]["crossfit"]["accuracy"], r["binary"]["crossfit"]["sensitivity"], r["binary"]["crossfit"]["specificity"],
        r["binary"]["accuracy"], r["binary"]["sensitivity"], r["binary"]["specificity"], r["multiclass"]["accuracy"], r["multiclass"]["balanced_accuracy"], r["multiclass"]["macro_f1"],
        r["severity"]["accuracy"], round(r["severity"].get("within_one", -1), 4), r["severity"]["qwk"], r["severity_crossfit"]["qwk"]))
show("ENSEMBLE", res)
for k, r in res["single_seed"].items(): show(k, r)
print("per-class recall", {k: round(v, 3) for k, v in res["multiclass"]["per_class_recall"].items()})
print("per-class precision", {k: round(v, 3) for k, v in res["multiclass"]["per_class_precision"].items()})
print("per fold", [{k: round(v, 4) for k, v in f.items()} for f in res["per_fold"]])
print("severity confusion", res["severity"]["confusion"])
