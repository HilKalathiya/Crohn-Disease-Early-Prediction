"""Collect every metric into app/models/results_summary.json (the single source the website and report read from)."""
import os, sys, json, glob, numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from data import table
from metrics import evaluate
from sklearn.metrics import roc_auc_score
DEV = "devres/results"
d = table(); y = d.y.values
P = np.zeros((len(y), 7)); seen = np.zeros(len(y), bool)
for k in range(1, 6):
    if not os.path.exists(f"runs/effb0_ra/idx_fold{k}.npy"): continue
    idx = np.load(f"runs/effb0_ra/idx_fold{k}.npy"); P[idx] = np.load(f"runs/effb0_ra/oof_fold{k}.npy"); seen[idx] = True
PARTIAL = "--partial" in sys.argv
if not PARTIAL: assert seen.all(), "missing folds"
else: print("PARTIAL: using", seen.sum(), "frames"); d = d[seen].reset_index(drop=True); y = y[seen]; P = P[seen]
ft = evaluate(y, P)
rng = np.random.default_rng(0); yb = (y != 0).astype(int); s = 1 - P[:, 0]; bs = []
for _ in range(1000):
    i = rng.integers(0, len(y), len(y)); bs.append(roc_auc_score(yb[i], s[i]))
ft["binary"]["auroc_ci95"] = [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]
ft["binary"]["note"] = "image-level bootstrap (CrohnIPI release has no patient IDs)"
# operating threshold: for each test fold, the threshold that maximises accuracy on the OTHER folds' out-of-fold scores (cross-fitting)
grid = np.linspace(0.05, 0.95, 181); pred_cf = np.zeros(len(y), bool); thr_used = {}
for k in sorted(set(d.fold.values)):
    m = d.fold.values == k; o = ~m
    accs = [((s[o] >= t) == yb[o]).mean() for t in grid]; t_k = float(grid[int(np.argmax(accs))]); thr_used[int(k)] = t_k
    pred_cf[m] = s[m] >= t_k
ft["binary"]["crossfit"] = dict(accuracy=float((pred_cf == yb).mean()), sensitivity=float(pred_cf[yb == 1].mean()), specificity=float((~pred_cf[yb == 0]).mean()),
                                thresholds=thr_used, note="threshold for each fold chosen on the other folds' out-of-fold scores")
app_thr = float(grid[int(np.argmax([((s >= t) == yb).mean() for t in grid]))])
json.dump(dict(lesion_threshold=0.5, threshold_rule="0.5 favours sensitivity, as a screening/triage tool should",
               oof_sensitivity_at_0_5=ft["binary"]["sensitivity"], oof_specificity_at_0_5=ft["binary"]["specificity"], accuracy_optimal_threshold=app_thr,
               classes=["N", "E", "O", "AU", "U3-10", "U>10", "S"]), open("app/models/crohnipi_meta.json", "w"), indent=1)
per_fold = []
for k in range(1, 6):
    m = d.fold.values == k
    if not m.any(): continue; e = evaluate(y[m], P[m]); per_fold.append(dict(fold=k, acc7=e["multiclass"]["accuracy"], bin_acc=e["binary"]["accuracy"], auroc=e["binary"]["auroc"], qwk=e["severity"]["qwk"]))
ft["per_fold"] = per_fold
np.save("artifacts/crohnipi/oof_finetuned.npy", P)
probe = json.load(open("artifacts/crohnipi/probe_metrics.json"))
NICE = {"mnv3_21k": "MobileNetV3 (IN-21k) frozen + LR", "effb0_ns": "EfficientNet-B0 (Noisy Student) frozen + LR", "effb0_ra": "EfficientNet-B0 (ImageNet) frozen + LR"}
ablation = [dict(label=NICE[k], balanced_accuracy=v["multiclass"]["balanced_accuracy"], auroc=v["binary"]["auroc"], accuracy7=v["multiclass"]["accuracy"], bin_acc=v["binary"]["accuracy"]) for k, v in probe.items()]
ablation.append(dict(label="EfficientNet-B0 fine-tuned (ours)", balanced_accuracy=ft["multiclass"]["balanced_accuracy"], auroc=ft["binary"]["auroc"],
                     accuracy7=ft["multiclass"]["accuracy"], bin_acc=ft["binary"]["accuracy"], ours=True))
bench = [dict(label="Recurrent attention net (de Maissin 2021)", accuracy=0.937, sensitivity=0.9209, specificity=0.9476),
         dict(label="ResNet-34 (de Maissin 2021)", accuracy=0.9458), dict(label="VGG-16 (de Maissin 2021)", accuracy=0.9440),
         dict(label="VGG-19 (de Maissin 2021)", accuracy=0.9435),
         dict(label="EfficientNet-B0 (ours)", accuracy=ft["binary"]["crossfit"]["accuracy"], sensitivity=ft["binary"]["crossfit"]["sensitivity"], specificity=ft["binary"]["crossfit"]["specificity"], ours=True)]
ext = json.load(open(f"{DEV}/crohnipi_external.json")) if os.path.exists(f"{DEV}/crohnipi_external.json") else None
ext_rows, pointing = [], None
if ext:
    kv, se = ext["kvasir"], ext["seeai"]
    ext_rows = [dict(label="Kvasir: ulcer + erosion + erythema", value=kv["inflammatory_vs_normal"]["auroc"], ours=True),
                dict(label="Kvasir: ulcer", value=kv["ulcer"]["auroc_vs_normal"]), dict(label="Kvasir: erosion", value=kv["erosion"]["auroc_vs_normal"]),
                dict(label="Kvasir: erythema", value=kv["erythema"]["auroc_vs_normal"]),
                dict(label="SEE-AI: erosion", value=se["erosion"]["auroc_vs_no_finding"]), dict(label="SEE-AI: stenosis", value=se["stenosis"]["auroc_vs_no_finding"]),
                dict(label="SEE-AI: erythema", value=se["erythema"]["auroc_vs_no_finding"])]
    pointing = dict(n=kv["pointing"]["n"], hit_rate=kv["pointing"]["hit_rate"], baseline=kv["pointing"]["baseline"], seeai=se.get("pointing"))
uc = json.load(open(f"{DEV}/uc_metrics.json"))
if os.path.exists(f"{DEV}/uc_ablation_crohnipi_features.json"): uc["ablation_crohnipi_features"] = json.load(open(f"{DEV}/uc_ablation_crohnipi_features.json"))
sc = json.load(open(f"{DEV}/sb_screen_metrics.json"))
SN = {"ulcer_erosion": "Ulcer / erosion", "erythema": "Erythema", "angiectasia": "Angiectasia", "bleeding": "Bleeding", "lymphangiectasia": "Lymphangiectasia", "polyp": "Polyp", "foreign_body": "Foreign body"}
screen_rows = [dict(label=SN[k], value=v["auroc_vs_normal"], note=f"{v['positives']} Kvasir frames; internal SEE-AI AUROC {sc['internal_random_cv'][k]['auroc']:.3f}") for k, v in sc["external_kvasir"].items() if k in SN]
flare = json.load(open("artifacts/flare/flare_metrics.json"))
totals = dict(images=3484 + 8266 + 18481 + 11276 + 851, datasets=6, hmp2_visits=flare["summary"]["hbi_visits"])
out = dict(totals=totals, crohnipi=dict(finetuned=ft, probe=probe, ablation=ablation, benchmarks=bench, external=ext, external_rows=ext_rows, pointing=pointing,
           recipe=dict(model="EfficientNet-B0 (timm efficientnet_b0.ra_in1k)", epochs=10, batch=32, lr=6e-4, optimizer="AdamW wd 0.01, cosine, 1 warm-up epoch",
                       loss="cross-entropy, label smoothing 0.1, sqrt inverse-frequency class weights", input="224x224 full frame", tta="4 views")),
           uc=uc, screen=sc, screen_rows=screen_rows, leakage=sc["leakage_kvasir"], flare=flare)
json.dump(out, open("app/models/results_summary.json", "w"), indent=1, default=float)
print("crossfit", ft["binary"]["crossfit"], "app thr", app_thr)
print(json.dumps(dict(ft7=ft["multiclass"]["accuracy"], bal=ft["multiclass"]["balanced_accuracy"], mf1=ft["multiclass"]["macro_f1"], bin=ft["binary"], tier=ft["severity"]["accuracy"], qwk=ft["severity"]["qwk"],
                      recall=ft["multiclass"]["per_class_recall"]), default=float, indent=1))
