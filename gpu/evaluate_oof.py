"""Score out-of-fold (OOF) predictions from one or more training runs, alone or as an ensemble.

    python evaluate_oof.py --runs runs_gpu/effb0
    python evaluate_oof.py --runs runs_gpu/effb2 runs_gpu/effb0          # ensemble of two backbones
    python evaluate_oof.py --runs runs_gpu/effb2 --with-current           # + the model now in the app

A run folder holds oof_fold<k>.npy (probabilities) and idx_fold<k>.npy (row numbers in CrohnIPI_description.csv).
Everything is computed out-of-fold. The two tuned decision rules (lesion threshold, severity cut-points) are
CROSS-FITTED: for each test fold they are chosen on the other four folds only, so the numbers stay honest.
NumPy + pandas only (no scikit-learn needed).
"""
import argparse, json, os, sys
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DATA = os.path.normpath(os.path.join(HERE, "..", "..", "temorary solution", "DataCrohnIPI"))
CURRENT_OOF = os.path.normpath(os.path.join(HERE, "..", "results", "oof_finetuned.npy"))
CURRENT_SUMMARY = os.path.normpath(os.path.join(HERE, "..", "app", "models", "results_summary.json"))
CLASSES = ["N", "E", "O", "AU", "U3-10", "U>10", "S"]
TIER = np.array([0, 1, 1, 2, 2, 3, 3])


def auroc(score, label):
    label = np.asarray(label).astype(bool)
    order = np.argsort(score, kind="mergesort")
    s = np.asarray(score)[order]
    ranks = np.empty(len(s))
    i = 0
    while i < len(s):  # average ranks for ties
        j = i
        while j + 1 < len(s) and s[j + 1] == s[i]:
            j += 1
        ranks[i:j + 1] = (i + j) / 2 + 1
        i = j + 1
    r = np.empty(len(s))
    r[order] = ranks
    npos, nneg = label.sum(), (~label).sum()
    return float((r[label].sum() - npos * (npos + 1) / 2) / (npos * nneg))


def qwk(a, b, k=4):
    O = np.bincount(np.asarray(a) * k + np.asarray(b), minlength=k * k).reshape(k, k).astype(float)
    W = np.array([[(i - j) ** 2 for j in range(k)] for i in range(k)]) / (k - 1) ** 2
    E = np.outer(O.sum(1), O.sum(0)) / O.sum()
    return float(1 - (W * O).sum() / (W * E).sum())


def macro_f1(y, p, k=7):
    f = []
    for c in range(k):
        tp = ((p == c) & (y == c)).sum()
        fp = ((p == c) & (y != c)).sum()
        fn = ((p != c) & (y == c)).sum()
        f.append(0.0 if tp == 0 else 2 * tp / (2 * tp + fp + fn))
    return float(np.mean(f))


def load_run(path, n):
    P = np.full((n, 7), np.nan)
    found = 0
    for k in range(1, 6):
        po, pi = os.path.join(path, f"oof_fold{k}.npy"), os.path.join(path, f"idx_fold{k}.npy")
        if os.path.exists(po) and os.path.exists(pi):
            P[np.load(pi)] = np.load(po)
            found += 1
    if found == 0:
        sys.exit(f"No oof_fold*.npy / idx_fold*.npy in {path}")
    return P, found


def best_threshold(score, truth):
    grid = np.arange(0.05, 0.951, 0.005)
    acc = [((score >= t) == truth).mean() for t in grid]
    return float(grid[int(np.argmax(acc))])


def best_cuts(expected, tier):
    """Cut-points c1<c2<c3 on the expected tier (0..3) that maximise QWK; coarse grid then refine."""
    best, cuts = -1, (0.5, 1.5, 2.5)
    g = np.arange(0.2, 2.81, 0.1)
    for c1 in g:
        for c2 in g[g > c1]:
            for c3 in g[g > c2]:
                q = qwk(tier, np.digitize(expected, [c1, c2, c3]))
                if q > best:
                    best, cuts = q, (c1, c2, c3)
    return cuts


def evaluate(P, d):
    y, fold = d.y.values, d.fold.values
    ok = ~np.isnan(P).any(1)
    if not ok.all():
        print(f"NOTE: {int((~ok).sum())} frames have no OOF prediction (folds not trained yet); scoring the rest.")
    P, y, fold = P[ok], y[ok], fold[ok]
    pred = P.argmax(1)
    lesion = 1 - P[:, 0]
    truth = y != 0
    # binary: fixed 0.5 (what the app uses) and cross-fitted threshold
    xf_pred = (lesion >= 0.5).copy()
    thr = {}
    for k in (np.unique(fold) if len(np.unique(fold)) > 1 else []):  # cross-fitting needs at least 2 folds
        t = best_threshold(lesion[fold != k], truth[fold != k])
        thr[int(k)] = t
        xf_pred[fold == k] = lesion[fold == k] >= t
    # severity tiers
    tier_true = TIER[y]
    tierP = np.stack([P[:, TIER == t].sum(1) for t in range(4)], 1)
    tier_arg = tierP.argmax(1)
    expected = tierP @ np.arange(4)
    tier_xf = tier_arg.copy()
    cuts = {}
    for k in (np.unique(fold) if len(np.unique(fold)) > 1 else []):
        c = best_cuts(expected[fold != k], tier_true[fold != k])
        cuts[int(k)] = [round(v, 2) for v in c]
        tier_xf[fold == k] = np.digitize(expected[fold == k], c)
    rec = {c: float((pred[y == i] == i).mean()) for i, c in enumerate(CLASSES)}
    return dict(
        n=int(len(y)),
        multiclass=dict(accuracy=float((pred == y).mean()), balanced_accuracy=float(np.mean(list(rec.values()))),
                        macro_f1=macro_f1(y, pred), per_class_recall=rec),
        binary=dict(auroc=auroc(lesion, truth),
                    acc_at_0_5=float(((lesion >= 0.5) == truth).mean()),
                    sens_at_0_5=float((lesion[truth] >= 0.5).mean()), spec_at_0_5=float((lesion[~truth] < 0.5).mean()),
                    crossfit_accuracy=float((xf_pred == truth).mean()),
                    crossfit_sensitivity=float(xf_pred[truth].mean()), crossfit_specificity=float((~xf_pred[~truth]).mean()),
                    crossfit_thresholds=thr),
        severity=dict(argmax_accuracy=float((tier_arg == tier_true).mean()), argmax_qwk=qwk(tier_true, tier_arg),
                      within_one=float((np.abs(tier_arg - tier_true) <= 1).mean()),
                      crossfit_cut_accuracy=float((tier_xf == tier_true).mean()), crossfit_cut_qwk=qwk(tier_true, tier_xf),
                      crossfit_cuts=cuts),
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", nargs="*", default=[], help="run folders to average")
    ap.add_argument("--with-current", action="store_true", help="also average in the model currently in the app")
    ap.add_argument("--data", default=DEFAULT_DATA)
    ap.add_argument("--save", default=None, help="write metrics JSON here")
    a = ap.parse_args()
    d = pd.read_csv(os.path.join(a.data, "CrohnIPI_description.csv"))
    d.columns = ["frame", "label", "fold"]
    d["y"] = d.label.map(CLASSES.index)
    members, names = [], []
    for r in a.runs:
        P, found = load_run(os.path.join(HERE, r) if not os.path.isabs(r) else r, len(d))
        members.append(P)
        names.append(f"{r} ({found}/5 folds)")
    if a.with_current:
        if not os.path.exists(CURRENT_OOF):
            sys.exit(f"Missing {CURRENT_OOF}")
        members.append(np.load(CURRENT_OOF))
        names.append("current app model (EfficientNet-B0 x5)")
    if not members:
        sys.exit("Give --runs and/or --with-current")
    P = np.mean(members, 0)
    m = evaluate(P, d)
    m["members"] = names
    print("Members:", *names, sep="\n  ")
    b, mc, s = m["binary"], m["multiclass"], m["severity"]
    rows = [("Lesion vs normal AUROC", b["auroc"], 0.985, None),
            ("Lesion vs normal accuracy (cross-fitted threshold)", b["crossfit_accuracy"], 0.943, 0.96),
            ("Lesion vs normal accuracy at 0.5 (app)", b["acc_at_0_5"], 0.927, None),
            ("7-class accuracy", mc["accuracy"], 0.868, 0.90),
            ("7-class balanced accuracy", mc["balanced_accuracy"], 0.776, None),
            ("Severity QWK (argmax tier)", s["argmax_qwk"], 0.893, 0.90),
            ("Severity QWK (cross-fitted cut-points)", s["crossfit_cut_qwk"], None, 0.90)]
    print(f"\n{'Metric':52s} {'this':>7s} {'current':>8s} {'target':>7s}")
    for name, v, cur, tgt in rows:
        print(f"{name:52s} {v:7.3f} {('' if cur is None else f'{cur:.3f}'):>8s} {('' if tgt is None else f'{tgt:.2f}'):>7s}")
    print("\nPer-class recall:", {k: round(v, 3) for k, v in mc["per_class_recall"].items()})
    print("Cross-fitted lesion thresholds:", b["crossfit_thresholds"])
    if a.save:
        json.dump(m, open(a.save, "w"), indent=1)
        print("Saved", a.save)


if __name__ == "__main__":
    main()
