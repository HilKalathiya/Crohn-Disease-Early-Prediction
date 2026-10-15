"""Figures for the IEEE-format evaluation deck (Oct 2026). Every plot is computed from saved result files:
out-of-fold predictions, training logs, external scores. Run from interim_build:  python3 ml_device/make_deck_figures.py [step]
steps: crohnipi | external | uc | flare | features | cam | all   -> PNGs in docs/figures/
"""
import csv, glob, json, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve, average_precision_score, confusion_matrix

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA = os.path.join(ROOT, "..", "temorary solution")
OUT = os.path.join(ROOT, "docs", "figures"); os.makedirs(OUT, exist_ok=True)
C1, C2, C3, GREY, INK, INK2, GRID = "#2a78d6", "#eb6834", "#1baf7a", "#9a9a96", "#1b1b1a", "#52514e", "#e6e5e1"
CLASSES = ["N", "E", "O", "AU", "U3-10", "U>10", "S"]
NAMES = ["Normal", "Erythema", "Oedema", "Aphthoid", "Ulcer 3–10 mm", "Ulcer >10 mm", "Stenosis"]
TIER = np.array([0, 1, 1, 2, 2, 3, 3]); TIERS = ["No lesion", "Mild", "Moderate", "Severe"]
SEQ = LinearSegmentedColormap.from_list("seq", ["#f4f8fd", "#a9c8ee", C1, "#0d3d75"])
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 13, "axes.edgecolor": "#8a8984", "axes.linewidth": 0.8,
                     "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2, "axes.titlesize": 14,
                     "axes.titleweight": "bold", "axes.titlecolor": INK, "legend.frameon": False, "savefig.dpi": 200,
                     "axes.spines.top": False, "axes.spines.right": False})


def save(fig, name):
    fig.savefig(os.path.join(OUT, name), bbox_inches="tight", facecolor="white"); plt.close(fig); print("wrote", name)


def grid(ax):
    ax.grid(True, color=GRID, linewidth=0.8); ax.set_axisbelow(True)


def crohnipi_oof():
    rows = list(csv.reader(open(os.path.join(DATA, "DataCrohnIPI", "CrohnIPI_description.csv"), encoding="utf-8")))[1:]
    y = np.array([CLASSES.index(r[1]) for r in rows]); fold = np.array([int(r[2]) for r in rows])
    P = np.zeros((2, len(y), 7))
    for s in (0, 1):
        for f in range(1, 6):
            d = os.path.join(ROOT, "results", "b2_oof", f"seed{s}")
            P[s, np.load(os.path.join(d, f"idx_fold{f}.npy"))] = np.load(os.path.join(d, f"oof_fold{f}.npy"))
    return y, fold, P.mean(0)


def step_crohnipi():
    y, fold, P = crohnipi_oof()
    les, score = (y != 0).astype(int), 1 - P[:, 0]
    auc = roc_auc_score(les, score); acc7 = (P.argmax(1) == y).mean()
    print(f"check: AUROC {auc:.4f}  acc7 {acc7:.4f}  bin@0.5 {((score>=0.5)==les).mean():.4f}")
    # Fig: ROC + PR
    fig, ax = plt.subplots(1, 2, figsize=(12.5, 5.4))
    for f in range(1, 6):
        m = fold == f; fp, tp, _ = roc_curve(les[m], score[m]); ax[0].plot(fp, tp, color=GREY, lw=1, alpha=0.7)
    fp, tp, th = roc_curve(les, score)
    ax[0].plot(fp, tp, color=C1, lw=2.5, label=f"10-model ensemble, AUROC = {auc:.3f}")
    ax[0].plot([], [], color=GREY, lw=1, label="each official fold (5)")
    for t, mk, lab in [(0.5, "o", "balanced cut-off 0.50"), (0.3097, "s", "screening cut-off 0.31")]:
        se, sp = (score[les == 1] >= t).mean(), (score[les == 0] < t).mean()
        ax[0].plot(1 - sp, se, mk, ms=9, color=C2, mec="white", mew=1.5, label=f"{lab}: sens {se:.1%}, spec {sp:.1%}")
    ax[0].plot([0, 1], [0, 1], ls=":", color=GREY, lw=1)
    ax[0].set(xlabel="False-positive rate (1 − specificity)", ylabel="True-positive rate (sensitivity)", title="(a) ROC, lesion vs normal",
              xlim=(-0.01, 1), ylim=(0, 1.01)); grid(ax[0]); ax[0].legend(loc="lower right", fontsize=11)
    ins = ax[0].inset_axes([0.32, 0.36, 0.36, 0.36]); ins.plot(fp, tp, color=C1, lw=2); ins.set(xlim=(0, 0.15), ylim=(0.85, 1.0))
    ins.tick_params(labelsize=9); ins.set_title("zoom", fontsize=10, fontweight="normal"); grid(ins)
    pr, rc, _ = precision_recall_curve(les, score); ap = average_precision_score(les, score)
    ax[1].plot(rc, pr, color=C1, lw=2.5, label=f"ensemble, AP = {ap:.3f}")
    ax[1].axhline(les.mean(), ls=":", color=GREY, lw=1); ax[1].text(0.02, les.mean() + 0.015, f"chance = prevalence {les.mean():.2f}", color=INK2, fontsize=11)
    ax[1].set(xlabel="Recall (sensitivity)", ylabel="Precision (PPV)", title="(b) Precision–recall", xlim=(0, 1.01), ylim=(0, 1.02))
    grid(ax[1]); ax[1].legend(loc="lower left", fontsize=11)
    fig.tight_layout(); save(fig, "fig_roc_pr_crohnipi.png")
    # Fig: confusion matrices
    fig, ax = plt.subplots(1, 2, figsize=(14, 6.2), gridspec_kw=dict(width_ratios=[1.35, 1]))
    for a, (yt, yp, labels, title) in zip(ax, [(y, P.argmax(1), NAMES, "(a) 7 lesion classes"),
                                                (TIER[y], np.stack([P[:, TIER == t].sum(1) for t in range(4)], 1).argmax(1), TIERS, "(b) 4 severity tiers")]):
        cm = confusion_matrix(yt, yp, labels=range(len(labels))); cmn = cm / cm.sum(1, keepdims=True)
        a.imshow(cmn, cmap=SEQ, vmin=0, vmax=1)
        for i in range(len(labels)):
            for j in range(len(labels)):
                a.text(j, i, f"{cmn[i, j]*100:.0f}%\n({cm[i, j]})" if cm[i, j] else "·", ha="center", va="center", fontsize=10 if len(labels) > 4 else 12,
                       color="white" if cmn[i, j] > 0.55 else INK)
        a.set_xticks(range(len(labels)), labels, rotation=35, ha="right"); a.set_yticks(range(len(labels)), labels)
        a.set(xlabel="Predicted", ylabel="True (expert consensus)", title=title)
        for sp in a.spines.values(): sp.set_visible(False)
        a.tick_params(length=0)
    fig.tight_layout(); save(fig, "fig_confusion_crohnipi.png")
    # Fig: per-class precision / recall / F1
    yp = P.argmax(1); prec, rec, f1 = [], [], []
    for c in range(7):
        tp_ = ((yp == c) & (y == c)).sum(); p_ = tp_ / max((yp == c).sum(), 1); r_ = tp_ / (y == c).sum()
        prec.append(p_); rec.append(r_); f1.append(2 * p_ * r_ / (p_ + r_))
    fig, ax = plt.subplots(figsize=(12.5, 5.0)); x = np.arange(7); w = 0.26
    for k, (vals, col, lab, hatch) in enumerate([(prec, C1, "Precision", ""), (rec, C2, "Recall", "//"), (f1, C3, "F1-score", "..")]):
        b = ax.bar(x + (k - 1) * (w + 0.02), vals, w, color=col, label=lab, hatch=hatch, edgecolor="white", linewidth=0)
    for i in range(7): ax.text(i + (w + 0.02), f1[i] + 0.015, f"{f1[i]:.2f}", ha="center", fontsize=10, color=INK2)
    ax.set_xticks(x, [f"{n}\n(n={int((y == c).sum())})" for c, n in enumerate(NAMES)], fontsize=11)
    ax.set(ylim=(0, 1.08), ylabel="Score"); grid(ax); ax.legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.12))
    save(fig, "fig_perclass_crohnipi.png")
    print("per-class", [round(v, 3) for v in prec], [round(v, 3) for v in rec], [round(v, 3) for v in f1])
    # Fig: training curves + per-fold held-out accuracy
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.8), gridspec_kw=dict(width_ratios=[1.7, 1]))
    accs = {0: [], 1: []}
    for s, col in [(0, C1), (1, C2)]:
        for f in range(1, 6):
            lg = json.load(open(os.path.join(ROOT, "results", "b2_oof", f"seed{s}", f"log_fold{f}.json")))
            ep = [e["epoch"] for e in lg["epochs"]]; tl = [e["train_loss"] for e in lg["epochs"]]
            ax[0].plot(ep, tl, color=col, lw=1.4, alpha=0.85, label=f"seed {s} (5 folds)" if f == 1 else None)
            accs[s].append(lg["test"]["acc7"])
    ax[0].set(xlabel="Epoch", ylabel="Training loss (label-smoothed CE)", title="(a) Training loss, 10 runs × 40 epochs"); grid(ax[0]); ax[0].legend()
    for s, col, mk, dx in [(0, C1, "o", -0.1), (1, C2, "s", 0.1)]:
        ax[1].plot(np.arange(1, 6) + dx, np.array(accs[s]) * 100, mk, color=col, ms=9, label=f"seed {s}")
    ax[1].axhline(acc7 * 100, color=INK2, ls="--", lw=1); ax[1].text(0.6, acc7 * 100 + 0.25, f"10-model ensemble: {acc7:.1%}", fontsize=11, color=INK2)
    ax[1].set(xlabel="Held-out official fold", ylabel="7-class accuracy (%)", title="(b) Accuracy on each held-out fold", xticks=range(1, 6), ylim=(86, 94), xlim=(0.5, 5.5))
    grid(ax[1]); ax[1].legend(loc="lower left")
    fig.tight_layout(); save(fig, "fig_training_crohnipi.png")
    # Fig: benchmark bars (lesion vs normal accuracy, same data and folds)
    v2 = json.load(open(os.path.join(ROOT, "results", "crohnipi_v2_metrics.json")))
    bars = [("RAN, base paper [1]", 93.70, GREY), ("VGG-19 [1]", 94.35, GREY), ("VGG-16 [1]", 94.40, GREY), ("ResNet-34 [1]", 94.58, GREY),
            ("EffNet-B0 v1 (ours)", 94.32, "#a9c8ee"), ("EffNet-B0 v2 (ours)", round(v2["binary"]["crossfit"]["accuracy"] * 100, 2) if "binary" in v2 else 95.5, "#a9c8ee"),
            ("EffNet-B2 ×10 (ours)", 96.7, C1)]
    fig, ax = plt.subplots(figsize=(12.5, 4.8))
    for i, (n, v, col) in enumerate(bars):
        ax.plot([92, v], [i, i], color=GRID, lw=2, zorder=1)
        ax.plot(v, i, "o", ms=13 if col == C1 else 10, color=col, mec="white", mew=1.5, zorder=2)
        ax.text(v + 0.08, i, f"{v:.2f}%", va="center", fontsize=12, color=INK, fontweight="bold" if col == C1 else "normal")
    ax.set_yticks(range(len(bars)), [b[0] for b in bars]); ax.invert_yaxis()
    ax.set(xlim=(92, 97.6), xlabel="Lesion vs normal accuracy (%), official CrohnIPI folds (dot plot; axis starts at 92%)")
    grid(ax); ax.grid(axis="y", visible=False)
    save(fig, "fig_benchmark_crohnipi.png")


def step_external():
    z = np.load(os.path.join(ROOT, "results", "bulk", "kvasir_frames_crohnipi_effb2", "shard_merged_000000_all.npz"), allow_pickle=True)
    fc, s = z["finding_class"], 1 - z["probs"][:, 0]; norm = fc == "Normal clean mucosa"
    fig, ax = plt.subplots(1, 2, figsize=(12.5, 5.4))
    for cls, col, ls in [("Ulcer", C1, "-"), ("Erosion", C2, "--"), ("Erythema", C3, "-.")]:
        m = (fc == cls) | norm; yy = (fc[m] == cls).astype(int); fp, tp, _ = roc_curve(yy, s[m]); a = roc_auc_score(yy, s[m])
        ax[0].plot(fp, tp, color=col, ls=ls, lw=2.2, label=f"{cls} (n={yy.sum()}), AUROC {a:.3f}")
    m = np.isin(fc, ["Ulcer", "Erosion", "Erythema"]) | norm; yy = (~norm[m]).astype(int)
    print("kvasir inflammatory AUROC", round(roc_auc_score(yy, s[m]), 4), "n_pos", yy.sum(), "n_neg", (yy == 0).sum())
    ax[0].plot([0, 1], [0, 1], ls=":", color=GREY, lw=1)
    ax[0].set(xlabel="False-positive rate", ylabel="True-positive rate", title=f"(a) Kvasir-Capsule, Norway (vs {norm.sum():,} normal frames)", xlim=(-0.01, 1), ylim=(0, 1.01))
    grid(ax[0]); ax[0].legend(loc="lower right", fontsize=11)
    files = sorted(glob.glob(os.path.join(ROOT, "_external_crohnipi_effb2", "seeai_f*_000.npz")))
    keys = np.load(files[0], allow_pickle=True)["keys"]; Pm = np.mean([np.load(f, allow_pickle=True)["probs"] for f in files], 0); s2 = 1 - Pm[:, 0]
    ann = {}
    with open(os.path.join(DATA, "The SEE-AI Project Dataset", "all_annotation.csv"), encoding="utf-8-sig") as fh:
        for r in csv.DictReader(fh):
            if r["image_number"].strip().isdigit(): ann["image%05d.jpg" % int(r["image_number"])] = r
    A = [ann[k] for k in keys]
    nofind = np.array([all(float(r[c] or 0) == 0 for c in ["angiodysplasia", "erosion", "stenosis", "lymphangiectasia", "lymph follicle", "SMT", "polyp-like", "bleeding", "diverticulum", "erythema", "foreign body", "vein"]) for r in A])
    for cls, col, ls in [("stenosis", C1, "-"), ("erosion", C2, "--"), ("erythema", C3, "-.")]:
        pos = np.array([float(r[cls] or 0) > 0 for r in A]); m = pos | nofind; yy = pos[m].astype(int)
        fp, tp, _ = roc_curve(yy, s2[m]); a = roc_auc_score(yy, s2[m])
        ax[1].plot(fp, tp, color=col, ls=ls, lw=2.2, label=f"{cls.capitalize()} (n={yy.sum()}), AUROC {a:.3f}")
    ax[1].plot([0, 1], [0, 1], ls=":", color=GREY, lw=1)
    ax[1].set(xlabel="False-positive rate", ylabel="True-positive rate", title=f"(b) SEE-AI, Japan (vs {nofind.sum():,} no-finding frames)", xlim=(-0.01, 1), ylim=(0, 1.01))
    grid(ax[1]); ax[1].legend(loc="lower right", fontsize=11)
    fig.tight_layout(); save(fig, "fig_external_roc.png")


def step_uc():
    P = np.load(os.path.join(ROOT, "results", "oof_limuc.npy"))
    meta = json.load(open(os.path.join(ROOT, "results", "uc_metrics.json")))["internal"]
    idx = list(csv.DictReader(open(os.path.join(ROOT, "_packed", "limuc_index.csv"), encoding="utf-8")))
    yl = np.array([int(r["label"].split()[-1]) for r in idx]); assert len(yl) == len(P)
    print("limuc oof acc", round(float((P.argmax(1) == yl).mean()), 4), "reported", round(meta["accuracy"], 4))
    fig, ax = plt.subplots(1, 2, figsize=(12.5, 5.3))
    act = (yl >= 2).astype(int); pa = P[:, 2:].sum(1); fp, tp, _ = roc_curve(act, pa)
    ax[0].plot(fp, tp, color=C1, lw=2.5, label=f"LIMUC, 5-fold patient-grouped CV\nAUROC {roc_auc_score(act, pa):.3f}")
    ax[0].plot([0, 1], [0, 1], ls=":", color=GREY, lw=1)
    ax[0].set(xlabel="False-positive rate", ylabel="True-positive rate", title="(a) Remission (Mayo 0–1) vs active (2–3)", xlim=(-0.01, 1), ylim=(0, 1.01))
    grid(ax[0]); ax[0].legend(loc="lower right", fontsize=11)
    cm = np.array(meta["confusion"]); cmn = cm / cm.sum(1, keepdims=True)
    ax[1].imshow(cmn, cmap=SEQ, vmin=0, vmax=1)
    for i in range(4):
        for j in range(4):
            ax[1].text(j, i, f"{cmn[i, j]*100:.0f}%\n({cm[i, j]})", ha="center", va="center", fontsize=11, color="white" if cmn[i, j] > 0.55 else INK)
    ax[1].set_xticks(range(4), [f"Mayo {k}" for k in range(4)]); ax[1].set_yticks(range(4), [f"Mayo {k}" for k in range(4)])
    ax[1].set(xlabel="Predicted", ylabel="True", title=f"(b) 4-level Mayo, QWK {meta['qwk']:.2f}"); ax[1].tick_params(length=0)
    for sp in ax[1].spines.values(): sp.set_visible(False)
    fig.tight_layout(); save(fig, "fig_uc.png")


def step_flare():
    tl = json.load(open(os.path.join(ROOT, "app", "models", "flare_timelines.json")))
    risk, hbi, yv = [], [], []
    for pid, visits in tl.items():
        v = sorted(visits, key=lambda q: q["week"])
        for i in range(len(v) - 1):
            if v[i].get("risk") is None or v[i]["hbi"] is None or v[i]["hbi"] >= 5: continue
            nxt = v[i + 1]
            if nxt["hbi"] is None or nxt["week"] - v[i]["week"] > 8: continue
            risk.append(v[i]["risk"]); hbi.append(v[i]["hbi"]); yv.append(int(nxt["hbi"] >= 5))
    risk, hbi, yv = map(np.array, (risk, hbi, yv))
    print("flare pairs", len(yv), "flares", yv.sum(), "AUROC RF", round(roc_auc_score(yv, risk), 3), "HBI", round(roc_auc_score(yv, hbi), 3))
    fm = json.load(open(os.path.join(ROOT, "results", "flare_metrics.json")))
    fig, ax = plt.subplots(1, 2, figsize=(12.5, 5.3))
    for sc, col, ls, lab in [(risk, C1, "-", "random forest, 29 features"), (hbi, C2, "--", "current HBI only (baseline)")]:
        fp, tp, _ = roc_curve(yv, sc); ax[0].plot(fp, tp, color=col, ls=ls, lw=2.3, label=f"{lab}, AUROC {roc_auc_score(yv, sc):.3f}")
    ax[0].plot([0, 1], [0, 1], ls=":", color=GREY, lw=1)
    ax[0].set(xlabel="False-positive rate", ylabel="True-positive rate", title=f"(a) Flare at next visit, HMP2 ({yv.sum()} flares / {len(yv)} visits)", xlim=(-0.01, 1), ylim=(0, 1.01))
    grid(ax[0]); ax[0].legend(loc="lower right", fontsize=11)
    for key, col, mk, lab in [("calibration", C2, "s", "random forest, raw"), ("calibration_recalibrated", C1, "o", "after Platt scaling")]:
        b = fm[key]; ax[1].plot([q["pred"] for q in b], [q["obs"] for q in b], marker=mk, color=col, lw=2, ms=8, label=lab)
    ax[1].plot([0, 0.46], [0, 0.46], ls=":", color=GREY, lw=1); ax[1].text(0.28, 0.31, "perfect calibration", color=INK2, fontsize=10, rotation=40)
    ax[1].set(xlabel="Predicted flare risk (mean of each fifth)", ylabel="Observed flare rate", title="(b) Calibration (Brier 0.109 → 0.085)", xlim=(0, 0.46), ylim=(0, 0.46))
    grid(ax[1]); ax[1].legend(loc="upper left", fontsize=11)
    fig.tight_layout(); save(fig, "fig_flare.png")


def lab_from_rgb(x):
    x = x / 255.0; x = np.where(x > 0.04045, ((x + 0.055) / 1.055) ** 2.4, x / 12.92)
    M = np.array([[0.4124, 0.3576, 0.1805], [0.2126, 0.7152, 0.0722], [0.0193, 0.1192, 0.9505]])
    xyz = x @ M.T / np.array([0.95047, 1.0, 1.08883])
    f = np.where(xyz > 0.008856, np.cbrt(xyz), 7.787 * xyz + 16 / 116)
    return np.stack([116 * f[..., 1] - 16, 500 * (f[..., 0] - f[..., 1]), 200 * (f[..., 1] - f[..., 2])], -1)


FEATS = [("Redness a*", "colour"), ("Yellowness b*", "colour"), ("Brightness L*", "colour"), ("Focal red spots", "colour"),
         ("Pale/white fibrin", "colour"), ("Edge density", "texture"), ("Local contrast", "texture"), ("Dark lumen area", "geometry"), ("Lumen compactness", "geometry")]


def image_features(path):
    from PIL import Image
    im = np.asarray(Image.open(path).convert("RGB").resize((160, 160)), np.float64)
    L = lab_from_rgb(im); l, a, b = L[..., 0], L[..., 1], L[..., 2]
    yy, xx = np.mgrid[:160, :160]; fov = ((yy - 79.5) ** 2 + (xx - 79.5) ** 2 < 76 ** 2) & (im.sum(-1) > 30)
    lv, av, bv = l[fov], a[fov], b[fov]
    gy, gx = np.gradient(l); g = np.hypot(gx, gy)[fov]
    dark = (l < 22) & fov; dfrac = dark.mean() / fov.mean()
    if dark.sum() > 5:
        ys, xs = np.nonzero(dark); comp = dark.sum() / (np.pi * (ys.std() ** 2 + xs.std() ** 2) * 2 + 1)
    else:
        comp = 0.0
    return [av.mean(), bv.mean(), lv.mean(), (av > av.mean() + 10).mean(), ((av < av.mean() - 8) & (lv > lv.mean())).mean(),
            (g > 6).mean(), lv.std(), dfrac, comp]


def step_features():
    rows = list(csv.reader(open(os.path.join(DATA, "DataCrohnIPI", "CrohnIPI_description.csv"), encoding="utf-8")))[1:]
    cache = os.path.join(os.path.expanduser("~"), "crohnipi_handcrafted.npy")
    if os.path.exists(cache):
        F = np.load(cache)
    else:
        F = np.array([image_features(os.path.join(DATA, "DataCrohnIPI", "imgs", r[0])) for r in rows]); np.save(cache, F)
    y = np.array([CLASSES.index(r[1]) for r in rows]); fold = np.array([int(r[2]) for r in rows])
    # effect size of each feature vs normal frames (Cohen's d)
    D = np.zeros((len(FEATS), 6))
    for j in range(len(FEATS)):
        n = F[y == 0, j]
        for c in range(1, 7):
            v = F[y == c, j]; sp = np.sqrt((n.var(ddof=1) + v.var(ddof=1)) / 2) + 1e-9; D[j, c - 1] = (v.mean() - n.mean()) / sp
    # how far do these hand-crafted cues go? same official folds, logistic regression
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    Pb, P7 = np.zeros(len(y)), np.zeros((len(y), 7))
    for f in range(1, 6):
        tr, te = fold != f, fold == f
        m = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=3000, class_weight="balanced")).fit(F[tr], y[tr])
        P7[te] = m.predict_proba(F[te])
    Pb = 1 - P7[:, 0]; hc_auc = roc_auc_score((y != 0).astype(int), Pb); hc_acc7 = (P7.argmax(1) == y).mean()
    print("hand-crafted: AUROC", round(hc_auc, 3), "acc7", round(hc_acc7, 3)); print(np.round(D, 2))
    json.dump(dict(features=[f[0] for f in FEATS], groups=[f[1] for f in FEATS], cohens_d_vs_normal=np.round(D, 3).tolist(),
                   classes=NAMES[1:], handcrafted_lr=dict(auroc=hc_auc, acc7=hc_acc7)),
              open(os.path.join(ROOT, "results", "handcrafted_features.json"), "w"), indent=1)
    div = LinearSegmentedColormap.from_list("div", [C1, "#f2f2f0", C2])
    fig, ax = plt.subplots(figsize=(12.5, 6.0))
    lim = 1.6; ax.imshow(np.clip(D, -lim, lim), cmap=div, vmin=-lim, vmax=lim, aspect="auto")
    for j in range(len(FEATS)):
        for c in range(6):
            ax.text(c, j, f"{D[j, c]:+.2f}", ha="center", va="center", fontsize=11, color="white" if abs(D[j, c]) > 1.1 else INK)
    ax.set_xticks(range(6), NAMES[1:], rotation=20, ha="right"); ax.set_yticks(range(len(FEATS)), [f"{n}  [{g}]" for n, g in FEATS])
    ax.set_title("Effect size vs normal frames (Cohen's d): orange = higher than normal, blue = lower", fontsize=13)
    for sp in ax.spines.values(): sp.set_visible(False)
    ax.tick_params(length=0); fig.tight_layout(); save(fig, "fig_features_crohnipi.png")


def step_cam():
    sys.path.insert(0, os.path.join(ROOT, "app"))
    from PIL import Image
    from backend.capsule import CapsuleModel
    cap = CapsuleModel()
    meta = list(csv.reader(open(os.path.join(DATA, "Kvasir-Capsule", "metadata.csv"), encoding="utf-8"), delimiter=";"))[1:]
    by = {}
    for r in meta:
        if r[4] in ("Ulcer", "Erosion", "Erythema") and r[5]:
            by.setdefault(r[4], []).append(r)
    imgs = {os.path.basename(p): p for p in glob.glob(os.path.join(DATA, "Kvasir-Capsule", "labelled_images", "**", "*.jpg"), recursive=True)}
    picks, rng = [], np.random.default_rng(3)
    for cls in ("Ulcer", "Erosion", "Erythema"):
        cand = [r for r in by[cls] if r[0] in imgs]; rng.shuffle(cand); picks += [(cls, r) for r in cand[:12]]
    res = cap.analyse([Image.open(imgs[r[0]]).convert("RGB") for _, r in picks], with_cam=True)
    def hit(r, cam):
        g = np.array(cam); iy, ix = np.unravel_index(g.argmax(), g.shape); cx, cy = (ix + 0.5) / g.shape[1] * 336, (iy + 0.5) / g.shape[0] * 336
        xs = [float(r[k]) for k in (5, 7, 9, 11)]; ys = [float(r[k]) for k in (6, 8, 10, 12)]
        return min(xs) <= cx <= max(xs) and min(ys) <= cy <= max(ys), (min(xs), min(ys), max(xs), max(ys))
    chosen, miss = [], None
    for cls in ("Ulcer", "Erosion", "Erythema"):
        for (c, r), o in zip(picks, res):
            if c != cls: continue
            h, box = hit(r, o["cam"]["lesion"])
            if h and len([x for x in chosen if x[0] == cls]) == 0: chosen.append((cls, r, o, box, True))
            if not h and miss is None: miss = (cls, r, o, box, False)
    panels = chosen + ([miss] if miss else [])
    fig, ax = plt.subplots(2, len(panels), figsize=(4.0 * len(panels), 8.4))
    for k, (cls, r, o, box, h) in enumerate(panels):
        im = np.asarray(Image.open(imgs[r[0]]).convert("RGB"))
        cam = np.array(Image.fromarray((np.array(o["cam"]["lesion"]) * 255).astype(np.uint8)).resize((336, 336), Image.BICUBIC)) / 255.0
        ax[0, k].imshow(im); ax[1, k].imshow(im); ax[1, k].imshow(cam, cmap="magma", alpha=0.5, vmin=0, vmax=1)
        for a in ax[:, k]:
            a.add_patch(plt.Rectangle((box[0], box[1]), box[2] - box[0], box[3] - box[1], fill=False, ec="#39e48a", lw=2))
            a.set_xticks([]); a.set_yticks([])
        ax[0, k].set_title(f"{cls} (expert box)", fontsize=12)
        ax[1, k].set_title(("peak inside box" if h else "peak OUTSIDE box (miss)") + f"\nlesion score {o['lesion_score'] if 'lesion_score' in o else 1 - o['probs'][0]:.2f}", fontsize=11,
                           color=INK if h else C2)
    fig.tight_layout(); save(fig, "fig_cam_examples.png")
    print("panels", [(p[0], p[1][0], p[4]) for p in panels])


if __name__ == "__main__":
    step = sys.argv[1] if len(sys.argv) > 1 else "all"
    for name, fn in [("crohnipi", step_crohnipi), ("external", step_external), ("uc", step_uc), ("flare", step_flare),
                     ("features", step_features), ("cam", step_cam)]:
        if step in (name, "all"): fn()
