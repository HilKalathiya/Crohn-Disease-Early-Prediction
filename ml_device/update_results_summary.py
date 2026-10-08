"""Merge the new full-dataset results into app/models/results_summary.json (the file the website's Results page reads)
and copy the whole-video files the app needs. Keeps a one-time backup of the original summary.
Run from interim_build:  python ml_device/update_results_summary.py"""
import os, sys, json, shutil, glob
import numpy as np
import pandas as pd

BASE = os.path.expanduser("~/mnt/MP_Cohn Disease/interim_build")
RES = os.path.join(BASE, "results")
APPM = os.path.join(BASE, "app/models")
SUMMARY = os.path.join(APPM, "results_summary.json")
BACKUP = os.path.join(APPM, "results_summary.before_full_data.json")
sys.path.insert(0, os.path.join(BASE, "gpu"))
from evaluate_oof import evaluate, CLASSES  # noqa: E402


def load(p):
    p = os.path.join(RES, p)
    return json.load(open(p)) if os.path.exists(p) else None


def main():
    if not os.path.exists(BACKUP):
        shutil.copy2(SUMMARY, BACKUP)
    r = json.load(open(BACKUP))  # always rebuild from the original, so re-running is safe
    # 1. honest severity cut-points (cross-fitted) for the current app model
    d = pd.read_csv(os.path.join(BASE, "..", "temorary solution", "DataCrohnIPI", "CrohnIPI_description.csv"))
    d.columns = ["frame", "label", "fold"]
    d["y"] = d.label.map(CLASSES.index)
    m = evaluate(np.load(os.path.join(RES, "oof_finetuned.npy")), d)
    r["crohnipi"]["finetuned"]["severity_crossfit"] = dict(qwk=m["severity"]["crossfit_cut_qwk"], accuracy=m["severity"]["crossfit_cut_accuracy"],
                                                          cuts=m["severity"]["crossfit_cuts"],
                                                          note="tier = expected severity cut at points chosen on the other four folds")
    # 2. new external results
    k = load("kvasir_full_external.json")
    if k:
        r["kvasir_full"] = k
    v = load("kvasir_video_study.json")
    if v:
        r["video_study"] = v["study"]
        shutil.copy2(os.path.join(RES, "kvasir_video_study.json"), os.path.join(APPM, "video_study.json"))
    q = load("quality_filter.json")
    if q:
        r["quality_filter"] = {kk: vv for kk, vv in q.items() if kk != "folds"}
    qr = load("quality_filter_REJECTED.json")
    if qr:
        r["quality_filter_rejected"] = dict(landmark_and_poor_view_dropped_pct=qr["landmark_and_poor_view_dropped_pct"],
                                            inflammatory_kept_pct=qr["inflammatory_kept_pct"], normal_dropped_pct=qr["normal_dropped_pct"],
                                            ulcer_dropped_pct=qr["per_class"].get("Ulcer", {}).get("dropped_pct"),
                                            videos=qr["videos"], frames=qr["n_labelled_sampled_frames"], cv=qr["cv"],
                                            decision="rejected: removes too many real lesion frames on held-out videos")
    h = load("hyperkvasir_extra.json")
    if h:
        r["hyperkvasir_extra"] = h
    extra_imgs = (k["A"]["frames"] - 8266) if k else 0  # Kvasir frames beyond the earlier subset
    r["totals"]["images"] = r["totals"]["images"] + extra_imgs + (1000 if h and "polyp_localisation" in h else 0)
    if v:
        r["totals"]["videos_scanned"] = v["study"]["n_videos"]
    if h and "uc_video_test" in h:
        r["totals"]["colonoscopy_videos"] = h["uc_video_test"]["n_colitis"] + h["uc_video_test"]["n_clean"]
    # optional: switch every CrohnIPI-model number to the improved v2 model (USE_V2=1), keeping v1 for comparison
    if os.environ.get("USE_V2") == "1":
        v2 = load("crohnipi_v2_metrics.json")
        assert v2, "results/crohnipi_v2_metrics.json missing"
        c = r["crohnipi"]; c["finetuned_v1"] = c["finetuned"]; c["finetuned"] = v2
        for b in c["benchmarks"]:
            if b.get("ours"):
                b.update(label="EfficientNet-B0 v2 (ours)", accuracy=v2["binary"]["crossfit"]["accuracy"],
                         sensitivity=v2["binary"]["crossfit"]["sensitivity"], specificity=v2["binary"]["crossfit"]["specificity"])
        for a in c["ablation"]:
            if a.get("ours"):
                a["label"] = "EfficientNet-B0 fine-tuned v1 (10 epochs)"; a.pop("ours")
        c["ablation"].append(dict(label="EfficientNet-B0 fine-tuned v2 (ours)", balanced_accuracy=v2["multiclass"]["balanced_accuracy"],
                                  auroc=v2["binary"]["auroc"], accuracy7=v2["multiclass"]["accuracy"], bin_acc=v2["binary"]["accuracy"], ours=True))
        c["recipe"].update(epochs=15, tta="8 views (4 rotations x flip)", ema="exponential moving average of weights, decay 0.998",
                           version="v2")
        ext2 = load("crohnipi_external_crohnipi_effb0v2.json")
        if ext2:
            kv, se = ext2["kvasir"], ext2["seeai"]; c["external_v1"] = c["external"]; c["external"] = ext2
            c["external_rows"] = [dict(label="Kvasir: ulcer + erosion + erythema", value=kv["inflammatory_vs_normal"]["auroc"], ours=True),
                                  dict(label="Kvasir: ulcer", value=kv["ulcer"]["auroc_vs_normal"]), dict(label="Kvasir: erosion", value=kv["erosion"]["auroc_vs_normal"]),
                                  dict(label="Kvasir: erythema", value=kv["erythema"]["auroc_vs_normal"]),
                                  dict(label="SEE-AI: erosion", value=se["erosion"]["auroc_vs_no_finding"]), dict(label="SEE-AI: stenosis", value=se["stenosis"]["auroc_vs_no_finding"]),
                                  dict(label="SEE-AI: erythema", value=se["erythema"]["auroc_vs_no_finding"])]
            c["pointing"] = dict(n=kv["pointing"]["n"], hit_rate=kv["pointing"]["hit_rate"], baseline=kv["pointing"]["baseline"], seeai=se.get("pointing"))
        k2 = load("kvasir_full_external_crohnipi_effb0v2.json")
        if k2:
            r["kvasir_full_v1"] = r.get("kvasir_full"); r["kvasir_full"] = k2
        v2v = load("kvasir_video_study_crohnipi_effb0v2.json")
        if v2v:
            r["video_study_v1"] = r.get("video_study"); r["video_study"] = v2v["study"]
            shutil.copy2(os.path.join(RES, "kvasir_video_study_crohnipi_effb0v2.json"), os.path.join(APPM, "video_study.json"))
            if os.path.exists(os.path.join(APPM, "video_timelines_crohnipi_effb0v2.json")):
                shutil.copy2(os.path.join(APPM, "video_timelines_crohnipi_effb0v2.json"), os.path.join(APPM, "video_timelines.json"))
            for vd in glob.glob(os.path.join(RES, "bulk/kvasir_videos_crohnipi_effb0v2/thumbs/*")):  # v2 top-frame thumbnails
                od = os.path.join(BASE, "app/samples/videos", os.path.basename(vd)); os.makedirs(od, exist_ok=True)
                for f in glob.glob(os.path.join(vd, "*.jpg")):
                    if not os.path.exists(os.path.join(od, os.path.basename(f))): shutil.copy2(f, od)
        print("switched results to v2:", {k: bool(x) for k, x in dict(external=ext2, kvasir_full=k2, video=v2v).items()})
    # optional: switch the CrohnIPI numbers to the GPU-trained EfficientNet-B2 (320 px, 2 seeds) model (USE_B2=1, after USE_V2=1)
    if os.environ.get("USE_B2") == "1":
        b2 = load("crohnipi_b2_metrics.json")
        assert b2, "results/crohnipi_b2_metrics.json missing"
        c = r["crohnipi"]; c["finetuned_v2"] = c["finetuned"]; c["finetuned"] = b2
        for b in c["benchmarks"]:
            if b.get("ours"):
                b.update(label="EfficientNet-B2 x2 seeds (ours)", accuracy=b2["binary"]["crossfit"]["accuracy"],
                         sensitivity=b2["binary"]["crossfit"]["sensitivity"], specificity=b2["binary"]["crossfit"]["specificity"])
        for a in c["ablation"]:
            if a.get("ours"):
                a["label"] = "EfficientNet-B0 fine-tuned v2 (15 epochs, 224 px)"; a.pop("ours")
        c["ablation"].append(dict(label="EfficientNet-B2 fine-tuned, 320 px, 2 seeds (ours)", balanced_accuracy=b2["multiclass"]["balanced_accuracy"],
                                  auroc=b2["binary"]["auroc"], accuracy7=b2["multiclass"]["accuracy"], bin_acc=b2["binary"]["accuracy"], ours=True))
        c["recipe"].update(model="EfficientNet-B2 (timm efficientnet_b2.ra_in1k), 2 seeds x 5 folds", epochs=40, batch="8 x 4 accumulation",
                           input="320x320 full frame (native size)", hardware="RTX 3050 laptop GPU, mixed precision", version="B2")
        c["external_model"] = "v2 (EfficientNet-B0, 224 px)"  # until the B2 external files below exist
        ext3 = load("crohnipi_external_crohnipi_effb2.json")  # gpu/external_gpu.py + ml_device/external_eval.py report
        if ext3:
            kv, se = ext3["kvasir"], ext3["seeai"]; c["external_v2"] = c["external"]; c["external"] = ext3; c["external_model"] = "B2"
            c["external_rows"] = [dict(label="Kvasir: ulcer + erosion + erythema", value=kv["inflammatory_vs_normal"]["auroc"], ours=True),
                                  dict(label="Kvasir: ulcer", value=kv["ulcer"]["auroc_vs_normal"]), dict(label="Kvasir: erosion", value=kv["erosion"]["auroc_vs_normal"]),
                                  dict(label="Kvasir: erythema", value=kv["erythema"]["auroc_vs_normal"]),
                                  dict(label="SEE-AI: erosion", value=se["erosion"]["auroc_vs_no_finding"]), dict(label="SEE-AI: stenosis", value=se["stenosis"]["auroc_vs_no_finding"]),
                                  dict(label="SEE-AI: erythema", value=se["erythema"]["auroc_vs_no_finding"])]
            c["pointing"] = dict(n=kv["pointing"]["n"], hit_rate=kv["pointing"]["hit_rate"], baseline=kv["pointing"]["baseline"], seeai=se.get("pointing"))
        k3 = load("kvasir_full_external_crohnipi_effb2.json")
        if k3:
            r["kvasir_full_v2"] = r.get("kvasir_full"); r["kvasir_full"] = k3
        print("switched CrohnIPI results to B2", {"external": bool(ext3), "kvasir_full": bool(k3)})
    hn = load("hard_negative_head.json")  # head-only hard-negative experiment (frozen v2 fold-1 backbone)
    if hn:
        r["hard_negative_head"] = dict(summary=hn["summary"], crohnipi_fold1=hn["crohnipi_fold1"], settings=hn["settings"], note=hn["note"])
    g = load("hard_negative_eval.json")  # full GPU retrain with raw-video negatives (gpu/eval_video_frames.py)
    if g:
        r["hard_negative_gpu"] = g
    h2 = load("hard_negative_head_v2.json")  # head-only round 2: adds the held-out lesion-frame check and a domain-balanced head
    if h2:
        r["hard_negative_head_v2"] = h2["summary"]
    json.dump(r, open(SUMMARY, "w"), indent=1)
    # 3. thumbnails of the top frames for the Video Scan page
    src = os.path.join(RES, "bulk/kvasir_videos/thumbs")
    dst = os.path.join(BASE, "app/samples/videos")
    n = 0
    for vd in glob.glob(os.path.join(src, "*")):
        od = os.path.join(dst, os.path.basename(vd))
        os.makedirs(od, exist_ok=True)
        for f in glob.glob(os.path.join(vd, "*.jpg")):
            if not os.path.exists(os.path.join(od, os.path.basename(f))):
                shutil.copy2(f, od)
                n += 1
    print("results_summary.json updated; totals:", r["totals"], "| thumbnails copied:", n,
          "| severity cross-fitted QWK", round(m["severity"]["crossfit_cut_qwk"], 3))


if __name__ == "__main__":
    main()
