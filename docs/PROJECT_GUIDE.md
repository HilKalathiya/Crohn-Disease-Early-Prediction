# CrohnSight: the complete project guide

**Project:** AI-Based Early Prediction of Crohn's Disease Severity Using Machine Learning (Minor Project 2026-27, PDEU, topic T-6)
**Team:** Hil Kalathiya (23BIT092), Sahaj Saliya (23BIT128) · **Guide:** Prof. Dr. Babalu Kumar Singh
**App name:** CrohnSight · **Guide written:** 8 Oct 2026, the night before the 9 Oct evaluation

This guide explains every corner of the project: what we built, why we chose each piece, how the ideas developed, what went wrong and how we fixed it, and how to run or rebuild anything. It is long on purpose. For the short version, read sections 1 and 2.

> Research prototype. Not a medical device. Never use it to diagnose a patient.

---

## Contents

1. [The project on one page](#1-the-project-on-one-page)
2. [How the project developed (the thought process)](#2-how-the-project-developed-the-thought-process)
3. [Medical background you need](#3-medical-background-you-need)
4. [Questions a first reader asks, with answers](#4-questions-a-first-reader-asks-with-answers)
5. [The data](#5-the-data)
6. [System architecture](#6-system-architecture)
7. [Methods, module by module](#7-methods-module-by-module)
8. [Results](#8-results)
9. [Things that did not work (and what they taught us)](#9-things-that-did-not-work-and-what-they-taught-us)
10. [Problems we faced and how we solved them](#10-problems-we-faced-and-how-we-solved-them)
11. [Code map](#11-code-map)
12. [How to run, re-check and rebuild](#12-how-to-run-re-check-and-rebuild)
13. [Our evaluation rules (why the numbers can be trusted)](#13-our-evaluation-rules-why-the-numbers-can-be-trusted)
14. [Limitations](#14-limitations)
15. [Future work](#15-future-work)
16. [Viva questions and answers](#16-viva-questions-and-answers)
17. [Glossary](#17-glossary)
18. [References and data sources](#18-references-and-data-sources)

---

## 1. The project on one page

**Problem.** Crohn's disease (CD) inflames the gut, often the small bowel. A capsule camera swallowed by the patient records 50,000 or more frames of the small bowel. A doctor must read them all to find the few lesions (ulcers, erosions, narrowing) that show how active and how severe the disease is. That often takes 30 to 120 minutes per study, and small, early lesions are easy to miss.

**What CrohnSight does.**

| Module | Input | Output |
| --- | --- | --- |
| 1. Lesion Analyzer (main) | One capsule frame | 7 lesion classes, 4-tier severity, lesion score, heat-map showing where the lesion is, model agreement |
| 1b. Study Triage | Many frames or a .zip | Frames ranked most-suspicious first, study summary, how much reading time could be saved |
| 1c. Video Scan | A whole capsule video (upload or put it in a folder) | Timeline of lesion score, top suspicious frames, heat-map for any of them |
| 2. UC Grader (add-on) | A colonoscopy image | Ulcerative colitis Mayo score 0 to 3 |
| 3. Flare Risk (add-on) | Patient visit history (HMP2) | Risk of a flare at the next visit, with the reasons |

**The final Crohn's model.** Ten EfficientNet-B2 networks (5 cross-validation folds × 2 random seeds), trained at the native 320 × 320 px frame size for 40 epochs on an RTX 3050 laptop GPU, averaged. It runs on any CPU through ONNX Runtime (about 1 second per frame on the laptop).

**Headline results** (CrohnIPI official 5-fold, out-of-fold, so every frame is scored by a model that never saw it):

| Measure | Ours | Target / best published |
| --- | --- | --- |
| Lesion vs normal AUROC | **0.992** (95% CI 0.989–0.995) | — |
| Lesion vs normal accuracy | **96.7%** | best published 94.6% (ResNet-34) |
| 7-class accuracy | **91.1%** (balanced 84.0%) | target 90% |
| 4-tier severity, quadratic weighted kappa | **0.94** (0.95 with cross-fitted cut-points) | target 0.90 |
| Severity within one tier | **97.7%** | — |

**Does it work on other hospitals?** Tested unchanged on two datasets it never saw (different countries and capsule brands):

* Kvasir-Capsule (Norway, Olympus capsule), all 47,238 labelled frames: inflammatory vs normal AUROC **0.944**. Ulcer 0.989, erosion 0.920, erythema 0.785.
* SEE-AI (Japan): erosion 0.897, stenosis 0.916, erythema 0.897.
* Heat-maps land inside the expert's lesion box in 69% of Kvasir frames (chance 4%) and 93% of SEE-AI stenosis frames.

**Two operating modes.** "Balanced" (cut-off 0.5) raises few false alarms: 2.6% of normal Kvasir frames. "Screening" (cut-off 0.31, fixed on CrohnIPI before any external test) catches 93.4% of Kvasir lesion frames and about 88–90% of SEE-AI lesions.

**Honest limits.** Telling a whole lesion video from a lesion-free one is near chance (study AUROC 0.57). The UC grader drops from 0.943 to 0.746 on a new hospital. The flare model only ties the simple HBI baseline (AUROC 0.711). All three are reported in the app, slides and report.

---

## 2. How the project developed (the thought process)

This section is the story in order. It matters because examiners ask "why did you do it this way?", and the honest answer is usually "because the step before showed us a problem".

### 2.1 The registered topic and the first plan (late September)

The registered title is about *early prediction* and *severity* of Crohn's disease. The first plan (still saved in `04_architecture/` and `05_plan/`) used **clinical records**:

* **MIMIC-IV** (US hospital records): find Crohn's patients by ICD codes, predict onset and disease course from labs and history.
* **HMP2 / IBDMDB**: predict flares from visit data.
* Three models (onset, course, flare) behind a Streamlit demo.

We wrote a disease primer (`01_understanding/`), a 37-paper literature table and a 25-dataset catalogue (`02_literature/`, `03_datasets/`).

### 2.2 The blocker: no MIMIC-IV access

MIMIC-IV needs PhysioNet credentialing (CITI training plus approval). The approval did not arrive in time. We checked a Kaggle "MIMIC-IV cleaned transcripts" dataset as a shortcut and **rejected it**: 232,158 rows, of which 209,951 were exact duplicates from a bad table join, only 7 patients and 13 admissions, no Crohn's patients at all, and no ICD codes, labs or dates.

**Decision (with the professor):** keep the MIMIC-IV plan *on hold, not dropped*, and build a fully working interim system on open data that answers the same question (how severe is the disease?) from a different signal: **images of the gut**.

### 2.3 Why capsule endoscopy

* Severity in Crohn's is judged by doctors mainly from what the bowel looks like (ulcers, their size, narrowing). Scores such as Lewis and CECDAI are built from exactly these findings.
* The small bowel is where Crohn's most often starts, and capsule endoscopy is the standard way to see it.
* An open, expert-labelled Crohn's dataset exists: **CrohnIPI** (Nantes, France), with 7 lesion classes and official 5-fold splits, so we can compare with published results fairly.
* Reading a capsule video is slow and tiring for doctors, so a model that ranks frames saves real time. That is a clear, useful product.

We added open datasets to test on other hospitals (Kvasir-Capsule, SEE-AI) and two add-ons that cover the rest of IBD: ulcerative colitis grading (LIMUC, HyperKvasir) and flare risk (HMP2).

### 2.4 Interim build, version 1 (6 Oct)

* Frozen-backbone baselines first (linear probes on MobileNetV3 and EfficientNet-B0 features). Good AUROC (0.94–0.95) but only ~75% 7-class accuracy. Lesson: the network must be fine-tuned.
* **v1:** EfficientNet-B0 fine-tuned 10 epochs at 224 px on the cloud CPU, official folds. AUROC 0.985, accuracy 94.3%, 7-class 86.8%, kappa 0.89.
* External tests, heat-map pointing game and a leakage experiment (section 7.8).
* Web app: FastAPI backend + Next.js website, runs offline on CPU with ONNX models.

### 2.5 "Use every dataset, no weak spots hidden" (7–8 Oct, overnight)

Hil asked to use everything downloaded, including the videos. So we:

* scored **all 47,238** labelled Kvasir-Capsule frames (not only a subset);
* scanned **all 43** labelled Kvasir videos (2 million frames);
* ran the UC grader on 46 HyperKvasir colonoscopy videos;
* tested a landmark/poor-view filter (rejected: it removed real lesions).

This exposed the biggest weakness: the model finds lesions *inside* a video well, but cannot tell a lesion video from a lesion-free one (AUROC 0.55–0.57). We report this openly.

### 2.6 v2 (8 Oct, early morning)

Better recipe, same network: 15 epochs, exponential moving average (EMA) of weights, 8-view test-time augmentation. 7-class 87.9%, kappa 0.91. The recipe was fixed before scoring all folds. We also tested averaging v1 + v2 and did **not** adopt it, because picking a mix after seeing the test results is selection on test data.

### 2.7 The GPU phase (8 Oct, day)

Hil has an RTX 3050 laptop GPU and offered to run any number of epochs. We wrote a GPU training kit (`gpu/`). Two lines of work:

1. **Hard negatives** to fix the whole-video weakness: add thousands of raw Kvasir video frames from lesion-free videos as "normal". It raised video-level AUROC a little but **destroyed** detection of real Kvasir lesions (sensitivity 87% → 28%). Cause: every "normal" came from the Kvasir camera and every lesion from CrohnIPI, so the network learned "Kvasir camera = normal". Not adopted, and an earlier over-optimistic claim was corrected everywhere.
2. **Bigger model, native resolution:** EfficientNet-B2 at 320 px (the real frame size), 40 epochs, 2 seeds. Training was first 10× too slow; a benchmark showed the channels-last memory layout was 7× slower on this GPU, so we turned it off.

### 2.8 B2 becomes the app model (8 Oct, afternoon and evening)

Rule fixed before scoring: final model = average of both seeds; switch the app only if it beats v2. It did, on every measure, and met all three targets (7-class 91.1%, binary 96.7%, kappa 0.94). External tests were re-run on the GPU with B2: Kvasir AUROC 0.915 → 0.944, false alarms on normal frames 26% → 2.6%.

### 2.9 The trade-off question and screening mode (8 Oct, night)

At the fixed 0.5 cut-off, B2 caught 79% of Kvasir lesion frames, v2 caught 88%. Hil asked whether we could overcome this. Analysis:

* B2 is not worse, it is *more cautious*. At the same false-alarm rate as v2 (specificity 73.8%), B2 catches **92.1%** vs v2's 88.3%. At B2's own false-alarm rate (specificity 97.4%), B2 catches 79.4% vs v2's 65.5%.
* So the fix is to offer a second operating point. We fixed a **screening cut-off of 0.31** on CrohnIPI training data only (the score at which 99% of CrohnIPI lesions are caught), before looking at any external data. On Kvasir it catches **93.4%** of lesion frames (specificity 68.4%); on SEE-AI 88–90% of erosions, stenoses and erythema.
* The app now has a Balanced / Screening switch on the Lesion Analyzer, Study Triage and Video pages.

### 2.10 Scan your own video (8 Oct, night)

Upload a capsule video in the browser, or drop it in `app/user_videos/`, and the app scans it in two stages (fast pass on every Nth frame, then the full 10-model ensemble on the top 24 frames). Tested on a real 16,151-frame Kvasir video: 38 seconds on the laptop's 2-core Linux VM.

---

## 3. Medical background you need

### 3.1 Crohn's disease

* A chronic inflammatory bowel disease (IBD). Can affect any part of the gut, most often the end of the small bowel (terminal ileum) and the colon.
* Inflammation comes and goes: **flares** (active disease) and **remission**.
* Inflammation is *transmural* (through the whole bowel wall) and *patchy* ("skip lesions").
* Complications: strictures (narrowing, "stenosis"), fistulas, abscesses.

### 3.2 Capsule endoscopy

* The patient swallows a pill-sized camera. It records 2 to 6 frames per second for about 8 to 12 hours: often 50,000+ frames.
* Frames are round (circular field of view) with no fixed "up". This is why rotating and flipping training images is valid.
* Brands differ in colour, resolution and lighting: PillCam SB3 (Medtronic; CrohnIPI), Olympus (Kvasir-Capsule), and others in SEE-AI. This "domain shift" is why we test on other datasets.

### 3.3 The 7 CrohnIPI lesion classes and our 4 severity tiers

| Code | Finding | Our tier | Why this tier |
| --- | --- | --- | --- |
| N | Normal mucosa | 0 No lesion | — |
| E | Erythema (redness) | 1 Mild | Inflammation without tissue loss; low weight in capsule scores |
| O | Oedema (villous swelling) | 1 Mild | Same |
| AU | Aphthoid ulceration | 2 Moderate | Small superficial ulcer; earliest visible Crohn's lesion; CECDAI groups aphthae with erosions and small ulcers |
| U3-10 | Ulcer 3–10 mm | 2 Moderate | True ulcer, medium size |
| U>10 | Ulcer > 10 mm | 3 Severe | Large ulcers weigh most in Lewis / CECDAI |
| S | Stenosis (narrowing) | 3 Severe | Structural damage; can block the capsule |

The tier is computed by adding the model's class probabilities inside each tier and taking the largest. This makes severity an *ordinal* (ordered) grade, so we measure it with weighted kappa.

### 3.4 Clinical scores (context only)

* **Lewis score** and **CECDAI** grade small-bowel Crohn's from capsule findings (villous oedema, ulcers by number/size, stenosis). Our tiers follow the same logic but work per frame, not per study.
* **Mayo endoscopic subscore** (ulcerative colitis): 0 normal, 1 mild, 2 moderate, 3 severe. Used by the UC grader.
* **Harvey-Bradshaw Index (HBI)**: a symptom score for Crohn's (wellbeing, pain, stools, complications). HBI < 5 means remission. Used to define flares in HMP2.

---

## 4. Questions a first reader asks, with answers

**"The title says early prediction. Where is prediction?"**
Two places. (1) Early *detection*: aphthoid ulcers are the earliest visible Crohn's lesion, and the model finds them and grades severity per frame. (2) The flare module predicts a flare at the *next* visit from the current one. The original onset-prediction plan on MIMIC-IV is on hold until access arrives.

**"Why images and not patient records?"**
MIMIC-IV access did not arrive; the only shortcut dataset was unusable (section 2.2). Severity in Crohn's is judged mainly from endoscopy, so images answer the severity question directly.

**"96.7% sounds too good. Is there leakage?"**
We used the dataset's official 5 folds, so our numbers compare fairly with published ones (best 94.6%). Every number is out-of-fold. The two tuned decision rules (lesion threshold and severity cut-points) are cross-fitted: chosen on four folds, applied to the fifth. And we test on two other countries' datasets without retraining. One honest caveat: CrohnIPI has no patient IDs, so frames from one patient may sit in different folds. That is why the external tests matter more.

**"Why is AUROC 0.944 on Kvasir but sensitivity only 79%?"**
AUROC measures ranking at every cut-off. 79% is at one fixed cut-off (0.5) chosen to keep false alarms low (2.6%). At the screening cut-off the same model catches 93.4%.

**"Why not just train on Kvasir too?"**
Then Kvasir is no longer an independent test. Kvasir also has very few lesion videos (ulcer frames come from only 3 videos), so a fair Kvasir-trained model is not possible; our leakage experiment shows random frame splits inflate accuracy from 18% to 96%.

**"Can it read a whole capsule study?"**
It finds lesions inside a study well (10 of 11 lesion videos have a real lesion as their top-ranked labelled frame), and you can upload a full video. But deciding "this patient has lesions / has none" from the whole video is near chance (0.57). We say this clearly.

**"Is it a medical device?"**
No. It is a research prototype trained on public data.

---

## 5. The data

### 5.1 Summary

| Dataset | Country / camera | Size used | Labels | Licence | Role |
| --- | --- | --- | --- | --- | --- |
| **CrohnIPI** | France, PillCam SB3 | 3,484 labelled frames | 7 lesion classes, official 5 folds | CC BY-NC-ND, research use only | Train + internal test (main) |
| **Kvasir-Capsule** | Norway, Olympus | 47,238 labelled frames, 43 labelled videos | 14 findings/landmarks, video IDs, boxes | CC BY 4.0 | External test, video tests, leakage experiment |
| **SEE-AI** | Japan | 18,481 frames (4,455 used for the Crohn's test) | 12 finding flags + boxes | Kaggle dataset licence | External test; trains the findings screen |
| **LIMUC** | Turkey | 11,276 images, 564 patients | Mayo 0–3 | CC BY 4.0 | Trains the UC grader |
| **HyperKvasir** | Norway | 851 UC images, 46 colonoscopy videos, 1,000 polyp masks | UC grade 0–3, polyp masks | CC BY 4.0 | External test for the UC grader and heat-maps |
| **HMP2 / IBDMDB** | USA | 66 CD patients, 873 visits | HBI, labs, meds | Open (IBDMDB) | Flare forecaster |

Total: 82,330 images across 5 image datasets (3,484 + 47,238 + 18,481 + 11,276 + 851 + 1,000 polyp images), plus 43 capsule videos (about 2 million frames) and 46 colonoscopy videos.

### 5.2 CrohnIPI (main)

* 3,498 image files but only **3,484** have a label in `CrohnIPI_description.csv` (14 unlabelled; we use the 3,484).
* Class counts: Normal 2,124 · Ulcer 3–10 mm 408 · Ulcer > 10 mm 297 · Aphthoid 251 · Oedema 149 · Stenosis 130 · Erythema 125. Strong imbalance: 17 normal frames for every erythema frame.
* The CSV gives an **official fold number** (1–5) for each frame. We always use it, never our own random split, so results compare with the papers.
* Frames are 320 × 320 px. v1/v2 resized to 224; B2 uses the full 320.
* No patient IDs in the release, so confidence intervals are image-level bootstrap (stated wherever shown).

### 5.3 Kvasir-Capsule

* 47,238 labelled frames in 14 classes: Normal 34,338 · Ileocecal valve 4,189 · Reduced mucosal view 2,906 · Pylorus 1,529 · Angiectasia 866 · Ulcer 854 · Foreign body 776 · Lymphangiectasia 592 · Erosion 506 · Blood (fresh) 446 · Erythema 159 · Polyp 55 · Blood (hematin) 12 · Ampulla of Vater 10.
* "Inflammatory" (what our model should flag) = ulcer + erosion + erythema: 1,519 frames in the 8,266-frame test subset (854 + 506 + 159), 1,527 in the all-frames analysis.
* 43 labelled videos (32 GB) and 71 unlabelled videos (55 GB).
* **Key fact:** lesion frames come from very few videos (ulcer: 3 videos, erythema: 3, erosion: 9). This explains the leakage result and why video-level work is hard.

### 5.4 SEE-AI

* 18,481 frames, each with 12 finding flags from expert boxes. Erosion 4,390 · polyp-like 2,157 · lymph follicle 1,788 · erythema 915 · angiodysplasia 825 · bleeding 801 · vein 723 · foreign body 583 · lymphangiectasia 572 · SMT 528 · stenosis 442 · diverticulum 40 · no finding 6,161.
* For the Crohn's external test we draw (seed 0): 2,000 no-finding frames, 1,200 erosion frames, all 915 erythema and all 442 stenosis frames. The sets overlap, giving 4,455 unique frames; a frame counts for every finding it carries (so 1,427 erosion frames are tested).

### 5.5 LIMUC, HyperKvasir, HMP2

* **LIMUC:** 11,276 colonoscopy images from 564 UC patients, Mayo 0: 6,105 · 1: 3,052 · 2: 1,254 · 3: 865. Patient folders allow patient-grouped CV.
* **HyperKvasir:** UC subset of 851 images (grades 1, 2, 3 and mixed 0-1, 1-2, 2-3; no clean Mayo 0), 46 videos (15 colitis, 31 clean bowel), 1,000 polyp images with masks.
* **HMP2:** 66 Crohn's patients, 873 visits with HBI, 629 "remission now → next visit within 8 weeks" pairs, 61 flares (9.7%) in 27 patients.

### 5.6 What we did not use, on purpose

71 unlabelled Kvasir videos and 99,417 unlabelled HyperKvasir images (nothing to test against), HyperKvasir classes unrelated to IBD (polyps for grading, oesophagitis), box coordinates (we use image-level labels; heat-maps are checked against boxes).

### 5.7 What is in this repository

No dataset is in the repository (licences and size). The repo holds code, trained models, metrics and small out-of-fold prediction files. Sample images included in the app are only from CC BY 4.0 datasets (Kvasir-Capsule, HyperKvasir, LIMUC). See the README for download links and the expected folder layout.

---

## 6. System architecture

```
                 ┌────────────────────────── Training (offline) ──────────────────────────┐
 CrohnIPI ──►  gpu/train_gpu.py (PyTorch, RTX 3050)  ──►  fold*.pt  ──►  gpu/export_to_app.py ──► ONNX
 LIMUC / SEE-AI ──► ml_device/extract.py (ONNX backbone features) ──► train_heads.py ──► linear heads (.npz)
 HMP2 ──► ml/flare.py (scikit-learn RF) ──► ml/flare_export.py ──► flare_rf.npz (NumPy trees)
                 └──────────────────────────────────────────────────────────────────────────┘

                 ┌────────────────────────── App (runs on any CPU) ────────────────────────┐
 Browser  ◄──►  Next.js static site (frontend_dist)  ◄──►  FastAPI (backend/app.py)
                                                              │
                     capsule.py  — 10 × EfficientNet-B2 ONNX (+ CAM weights) — lesion, tier, heat-map
                     video_scan.py — two-stage video scan, background jobs
                     heads.py    — shared EfficientNet-B0 ONNX backbone + UC head + findings head
                     flare.py    — random forest in pure NumPy + Platt calibration
                     core.py     — ONNX Runtime sessions (CPU, or GPU if onnxruntime-gpu), image preprocessing
                 └──────────────────────────────────────────────────────────────────────────┘
```

**Why this design**

* **ONNX Runtime at run time, PyTorch only for training.** The demo laptop needs no GPU, no PyTorch and no internet. `pip install` is small and quick.
* **FastAPI + static Next.js export.** One process serves both the API and the website (`python run.py`). No Node.js needed at run time.
* **Class activation maps (CAM) without extra cost.** Each ONNX model outputs the logits *and* the last feature map; the heat-map is the classifier weights times the feature map. No second pass, no gradients.
* **Everything the website shows comes from one file** (`app/models/results_summary.json`), generated by `ml_device/update_results_summary.py` from the raw results. So the website, slides and report cannot drift apart silently.

---

## 7. Methods, module by module

### 7.1 Crohn's lesion model: how it evolved

| Version | Network | Input | Training | Where | 7-class | Binary (cross-fit) | Kappa |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Probe | MobileNetV3 / EffNet-B0 frozen + logistic regression | 224 | features only | laptop CPU | 75–76% | 84–86% (at 0.5) | — |
| v1 | EfficientNet-B0, fine-tuned | 224 | 10 epochs | cloud CPU | 86.8% | 94.3% | 0.89 |
| v2 | EfficientNet-B0 | 224 | 15 epochs, EMA, 8-view TTA | cloud CPU | 87.9% | 95.5% | 0.91 |
| **B2 (final)** | **EfficientNet-B2 × 2 seeds** | **320** | **40 epochs, EMA, 8-view TTA, AMP** | **RTX 3050** | **91.1%** | **96.7%** | **0.94** |

**Final recipe (B2)** — `gpu/train_gpu.py --backbone effb2 --img 320 --epochs 40 --batch 8 --accum 4`

| Choice | Value | Why |
| --- | --- | --- |
| Network | `efficientnet_b2.ra_in1k` (timm, ImageNet weights) | Best accuracy per compute in its size class; B0 was the CPU-friendly start, B2 is the next step that fits a 4 GB GPU at 320 px |
| Input | 320 × 320, the native CrohnIPI size | Resizing to 224 throws away detail of small lesions (aphthae, erythema) |
| Loss | Cross-entropy, label smoothing 0.1, class weights = sqrt(max count / class count) | Weights counter the 17:1 imbalance without over-boosting tiny classes; smoothing helps because neighbouring classes (e.g. aphthoid vs small ulcer) are genuinely ambiguous |
| Optimiser | AdamW, lr 6e-4, weight decay 0.01, 1 warm-up epoch, cosine decay | Standard, stable fine-tuning schedule |
| Batch | 8 images × 4 accumulation steps = effective 32 | 320 px B2 does not fit 32 at once in 4 GB |
| Augmentation | Random rotation 0–360°, flip, zoom 0.9–1.18, shift ±5%, brightness/contrast ±15%, saturation ±10% (on the GPU) | Capsule frames have no "up"; colour jitter mimics other cameras |
| EMA | Exponential moving average of weights, decay 0.998; the EMA model is evaluated | Smoother, more general weights; decided in advance so no epoch picking on the test fold |
| Test-time augmentation | 8 views (4 rotations × flip), probabilities averaged | Valid because of rotation symmetry; reduces variance |
| Mixed precision (AMP) | float16 compute | ~2× faster, half the memory |
| Seeds | 2 (0 and 1), averaged | Decided before scoring: average both seeds; each seed alone gives 96.5–96.7% and 91.1–91.3%, so the result is not luck |
| Final app ensemble | 10 ONNX models (5 folds × 2 seeds), probabilities averaged | Each fold model saw a different 80% of the data; agreement count is shown as an uncertainty signal |

Note: the app runs one view per model (not 8) to stay fast on a CPU; the parity check showed the same class 100% of the time vs the out-of-fold predictions.

### 7.2 Decision rules (threshold and severity cut-points)

* **Lesion score** = 1 − P(normal). A frame is flagged if the score ≥ the cut-off.
* **Balanced cut-off 0.5** (default). CrohnIPI out-of-fold: sensitivity 97.1%, specificity 95.7%.
* **Cross-fitted threshold** for the reported accuracy: for each fold, the best threshold is chosen on the other four folds' out-of-fold scores, then applied to this fold (0.66–0.72). This gives 96.7% without touching the test fold.
* **Screening cut-off 0.3097**: the score at which 99% of CrohnIPI lesion frames are caught (out-of-fold). Fixed on CrohnIPI only, before any external test. Cross-fitted on CrohnIPI: sensitivity 98.8%, specificity 85.6%.
* **Severity**: argmax of the summed tier probabilities (QWK 0.942), or cross-fitted cut-points on the expected tier (QWK 0.951).

### 7.3 Heat-maps (class activation maps)

EfficientNet ends with global average pooling and one linear layer. For class *c*, the CAM is `Σ_k W[c,k] · F[k,x,y]` over the last feature map F (10 × 10 for B2 at 320 px). The app averages the CAM over the 10 models, normalises it and draws it over the frame. The "lesion" map weights each lesion class map by its probability.

**Pointing game** (does the hottest pixel fall inside the expert's box?): Kvasir 68.8% (chance 4.1% = average box area), SEE-AI erosion 77.8% (chance 17.6%), stenosis 92.8% (chance 54.5%). So the model looks at the lesion, not at the background or the camera border.

### 7.4 Study triage

Upload many frames or a .zip. The app scores each frame, sorts them by lesion score, and reports: frames flagged (both modes), worst tier, tier counts, a burden index (mean expected tier × 100 / 3) and reading time saved (frames a doctor can skip). The bundled demo study is 160 frames from one Kvasir video with expert labels shown next to the model's ranking. Because it is fixed input, its result is computed once per model set and cached in `app/models/demo_study_cache.json` (the page then says "cached result · first run took N s"); delete that file to recompute.

### 7.5 Video scanning

**Offline scan of the 43 Kvasir videos** (`ml_device/scan_kvasir_videos.py`, v2 models, CPU): 1,979,285 frames; every 10th frame scored by one model; the top 150 per study re-scored by all fold models. Results drive the Video Scan page timelines.

**Scan your own video** (`app/backend/video_scan.py`), two stages:

1. **Stage 1 (fast):** decode every Nth frame (N chosen so about 500 / 2,000 / 5,000 frames, or every frame, are scored), run one network (seed 0, fold 1), keep the 240 highest-scoring frames as small JPEGs in memory.
2. **Stage 2 (accurate):** pick the top 24 frames that are at least `max(5N, total/400)` frames apart (so you see 24 *different* places, not 24 copies of one ulcer), re-score them with all 10 models. Click any of them for a heat-map.

Decoding uses OpenCV if installed, otherwise ffmpeg (bundled by `pip install imageio-ffmpeg`). Uploads stream to disk in 4 MB chunks; file names are sanitised (no path traversal); one scan runs at a time; the website polls the job every 1.5 s. Tested: a 16,151-frame Kvasir video (9 min of footage), 150 frames scored, top 24 re-scored, 38 s on the laptop's 2-core Linux VM.

### 7.6 External validation design

* The CrohnIPI models are applied **unchanged** to Kvasir-Capsule and SEE-AI. No retraining, no threshold tuning on them.
* Kvasir classes map to "inflammatory" (ulcer, erosion, erythema) vs "normal clean mucosa". Other classes (angiectasia, blood, landmarks) are reported separately, because the model was never taught them.
* AUROC confidence intervals on Kvasir use a **video bootstrap** (resample videos, not frames), because frames inside one video are strongly correlated.
* Triage curve: share of all lesion frames found in the top 5/10/20/30% of frames ranked by score.

### 7.7 The leakage experiment (why we never quote random-split numbers)

We trained a 9-class Kvasir finding classifier twice: once with frames split at random, once with whole videos held out. Balanced accuracy: **95.7%** (random) vs **17.9%** (video-grouped). Neighbouring frames of one video are near-identical, so a random split lets the model "recognise the video" instead of the lesion. Many papers report random-split numbers on Kvasir; we show why they mislead.

### 7.8 UC grader (add-on 2)

* Shared EfficientNet-B0 ONNX backbone (ImageNet weights) → 1,280 features per image → multinomial logistic regression head trained on LIMUC.
* 5-fold StratifiedGroupKFold by **patient** (no patient in train and test).
* Temperature scaling (T = 2.24) for honest probabilities: expected calibration error 0.144 → 0.014.
* Ablation: features from our CrohnIPI fine-tuned network were *worse* for UC (QWK 0.70 vs 0.74), so the app uses ImageNet features.

### 7.9 Findings screen (experimental)

Same shared backbone, one-vs-rest logistic heads for 8 SEE-AI findings (ulcer/erosion, erythema, angiectasia, bleeding, lymphangiectasia, polyp, foreign body, stenosis). Shown as a secondary panel only.

### 7.10 Flare forecaster (add-on 3)

* HMP2 Crohn's patients. A *pair* = a visit in remission (HBI < 5) and the next visit within 8 weeks. Flare = next HBI ≥ 5. 629 pairs, 61 flares.
* 29 features: HBI now / previous / change / mean and max of last 3 / highest ever, prior flares, weeks since flare, symptom items, antibiotics, immunosuppressants, calprotectin, CRP, ESR (with "weeks since"), age, disease years, sex, Montreal location, gap to next visit, visit count.
* Logistic regression, random forest and XGBoost compared with 5-fold StratifiedGroupKFold by patient, repeated 5 times; 95% CI from a patient-level bootstrap. Random forest chosen; Platt recalibration (Brier 0.109 → 0.085). Exported as pure NumPy trees, so the app needs no scikit-learn.
* Baseline: current HBI alone.

---

## 8. Results

### 8.1 CrohnIPI (official 5-fold, out-of-fold, final B2 model)

| Measure | Value |
| --- | --- |
| Lesion vs normal AUROC | 0.992 (95% CI 0.989–0.995, image-level bootstrap) |
| Accuracy, cross-fitted threshold | 96.7% (sensitivity 95.1%, specificity 97.6%) |
| Accuracy at 0.5 | 96.3% (sensitivity 97.1%, specificity 95.7%) |
| 7-class accuracy | 91.1% (balanced 84.0%, macro-F1 0.84) |
| Severity (4 tiers) | 93.1% exact, 97.7% within one tier, QWK 0.942 (argmax) / 0.951 (cross-fitted cut-points) |
| Per fold, 7-class | 89.9% – 92.3% |
| Recall per class | Normal 97.6%, erythema 79.2%, oedema 78.5%, aphthoid 81.7%, ulcer 3–10 mm 75.7%, ulcer > 10 mm 84.8%, stenosis 90.8% |

Severity confusion matrix (rows = truth, columns = prediction; No lesion / Mild / Moderate / Severe):

```
              No lesion  Mild  Moderate  Severe
No lesion        2064     26      30        4
Mild               21    226      20        7
Moderate           33     28     560       38
Severe              1      7      26      393
```

Only 1 severe frame of 427 was called "no lesion".

**Comparison with published CrohnIPI results** (same official folds): recurrent attention net 93.7%, VGG-19 94.35%, VGG-16 94.4%, ResNet-34 94.58%, **ours 96.7%**.

**Ablation** (what each step added, 7-class / binary at 0.5): frozen MobileNetV3 76.1% / 84.4% → frozen EfficientNet-B0 75.3% / 84.0% → v1 fine-tuned 86.8% / 92.7% → v2 87.9% / 93.9% → B2 320 px 2 seeds 91.1% / 96.3%. Fine-tuning gave the biggest jump; native resolution plus the bigger network gave the next.

### 8.2 External: Kvasir-Capsule (Norway), all 47,238 labelled frames, no retraining

| Measure | v2 (B0) | **B2 (final)** |
| --- | --- | --- |
| Inflammatory vs normal AUROC | 0.915 | **0.944** (CI 0.86–0.98, video bootstrap) |
| Ulcer / erosion / erythema AUROC | 0.961 / 0.873 / 0.803 | **0.989 / 0.920 / 0.785** |
| At 0.5: lesion frames caught | 88.3% | 79.4% |
| At 0.5: normal frames flagged | 26.2% | **2.6%** |
| Lesions caught at v2's false-alarm rate (spec 73.8%) | 88.3% | **92.1%** |
| Lesions caught at B2's false-alarm rate (spec 97.4%) | 65.5% | **79.4%** |
| **Screening cut-off 0.31**: caught / specificity | — | **93.4% / 68.4%** |
| Lesion frames in top 10 / 20 / 30% of ranked frames | 71 / 81 / 87% | **81 / 88 / 91%** |
| Lesion videos whose top-ranked labelled frame is a lesion | 9 of 11 | **10 of 11** |
| Pylorus / ileocecal valve frames flagged | 64% / 45% | **25% / 17%** |

### 8.3 External: SEE-AI (Japan), no retraining (B2)

| Finding | AUROC vs no finding | Caught at 0.5 | Caught at screening 0.31 |
| --- | --- | --- | --- |
| Erosion (1,427) | 0.897 | 70.9% | 88.2% |
| Stenosis (442) | 0.916 | 78.1% | 90.0% |
| Erythema (915) | 0.897 | 65.8% | 89.8% |
| No finding (2,000): specificity | — | 91.6% | 68.8% |

### 8.4 Heat-maps (pointing game, B2)

Kvasir 68.8% (chance 4.1%) · SEE-AI erosion 77.8% (17.6%) · SEE-AI stenosis 92.8% (54.5%).

### 8.5 Whole videos (43 Kvasir studies, v2 scan)

* Inside a lesion study: a median 67% of the expert's lesion frames are in that study's top 5% of frames.
* Lesion study vs lesion-free study: AUROC 0.57 (CI 0.35–0.76), near chance. Lesion-free studies also contain frames that look suspicious (bubbles, debris, landmarks), and lesions are rare and small.

### 8.6 UC grader

| Test | Result |
| --- | --- |
| LIMUC (patient-grouped CV), 4-class | 67.2% accuracy, QWK 0.742 (CI 0.720–0.760) |
| LIMUC remission (Mayo 0–1) vs active (2–3) | AUROC 0.943 (CI 0.934–0.950) |
| HyperKvasir (new hospital), remission vs active | AUROC 0.746 |
| HyperKvasir, 46 whole colonoscopy videos (colitis vs clean) | AUROC 0.981 |

### 8.7 Findings screen (experimental)

SEE-AI internal (random CV) AUROC 0.78–0.94; on Kvasir (external) 0.60–0.88 (bleeding 0.876, angiectasia 0.790, lymphangiectasia 0.766, erythema 0.701, ulcer/erosion 0.611, polyp 0.601). Polyp heat-map on HyperKvasir colonoscopy: 33% hits vs 16% chance. Kept as "experimental" in the app.

### 8.8 Flare forecaster (HMP2)

| Model | AUROC (95% CI) |
| --- | --- |
| Current HBI only (baseline) | 0.711 (0.633–0.781) |
| Random forest (chosen) | 0.711 (0.649–0.781); Brier 0.109 → 0.085 after recalibration |
| Logistic regression | 0.633 |
| XGBoost | 0.629 |

The model ties the baseline. With 61 flares in 27 patients there is not enough signal to beat it; we say so.

---

## 9. Things that did not work (and what they taught us)

Negative results are part of the project. Each one below was tested properly, reported, and left out of the app.

| Idea | What we expected | What happened | Lesson |
| --- | --- | --- | --- |
| **Frozen backbone + linear head** for the Crohn's model | Fast and good enough | 75% 7-class (vs 87–91% fine-tuned) | Medical frames are far from ImageNet; the network itself must adapt |
| **Averaging v1 + v2** | Small free gain | Tiny gain | Not adopted: choosing the mix after seeing test results is selection on test data |
| **Landmark / poor-view filter** before scoring videos | Fewer false alarms from pylorus, valve, bubbles | Removed 42% of real lesion frames on held-out videos (only 58% kept) | Rejected. A filter must be tested on held-out videos, not on the frames it was tuned on |
| **Small-bowel window** (score only the part of the video between pylorus and ileocecal valve) | Better study-level AUROC | No reliable gain: study AUROC 0.49–0.61 depending on the summary used, vs 0.56–0.59 without the window, all within chance-level uncertainty | The weakness is not "wrong organ" frames |
| **Hard negatives, full GPU retrain** (≈6,000 raw Kvasir video frames from lesion-free videos as "normal") | Whole-study AUROC up | Study AUROC 0.51 → 0.61, but held-out Kvasir lesion frames: AUROC 0.91 → 0.77, caught at 0.5 87% → 28% | **Camera shortcut:** all negatives were Kvasir camera, all lesions CrohnIPI camera, so the network learned the camera |
| **Hard negatives, head only** (laptop CPU) | Same, cheaper | Study 0.57 → 0.74, but lesion frames 0.75 → 0.68, caught 65% → 2% | Same shortcut. We had first called this "a fix that works"; that claim was corrected everywhere the same day |
| **Domain-balanced head** (add Kvasir lesion + normal frames from the training half) | Remove the shortcut | Study 0.49, frames 0.65: worse | With lesions from only 13 Kvasir videos, there is not enough in-domain lesion data |
| **CrohnIPI features for the UC grader** | Gut-specific features help UC | QWK 0.70 vs 0.74 with ImageNet features | Fine-tuning on 7 Crohn's classes narrows the features |
| **ML flare model vs HBI** | Beat the symptom score | Ties it (0.711 vs 0.711) | 61 flares in 27 patients are too few; more data or better signals (calprotectin trends) needed |

**The main lesson:** whole-study ranking is limited by in-domain lesion data, not by compute or model size. We proved this three ways before saying it.

---

## 10. Problems we faced and how we solved them

| # | Problem | How we found it | Solution |
| --- | --- | --- | --- |
| 1 | No MIMIC-IV access in time | Credentialing still pending at build time | Pivot to open imaging data for the interim build; MIMIC-IV plan kept on hold |
| 2 | The Kaggle "MIMIC-IV transcripts" shortcut | Data audit: 209,951 of 232,158 rows duplicated, 7 patients, 0 Crohn's | Rejected the dataset and documented why |
| 3 | CrohnIPI has 3,498 files but 3,484 labels | Count mismatch between folder and CSV | Use the 3,484 labelled frames; state both numbers |
| 4 | No patient IDs in CrohnIPI | Release format | Official folds (as the papers use), image-level bootstrap CIs (stated), plus external tests |
| 5 | Strong class imbalance (17:1) | Class counts | sqrt inverse-frequency class weights, label smoothing, report balanced accuracy and macro-F1 next to accuracy |
| 6 | Tuning a threshold on the test fold inflates accuracy | Design review | Cross-fitting: thresholds and severity cut-points chosen on the other four folds only |
| 7 | Random frame splits on Kvasir look great but lie | Leakage experiment: 95.7% vs 17.9% | Never quote random-split numbers; Kvasir used only as an external test with video-bootstrap CIs |
| 8 | CPU-only training limited v1/v2 (224 px, 10–15 epochs) | Targets 90% / 96% not met by v2 | GPU kit for Hil's RTX 3050 (`gpu/`): AMP, EMA, TTA, resumable |
| 9 | "unrecognized arguments --kvasir-index" on the laptop | Laptop had an old `train_gpu.py` | Re-synced; every transfer now checked with an md5 hash |
| 10 | A harmless PyTorch warning made it look like training had failed | User stopped the run | Explained it; filtered the warning; added progress lines in the first epoch |
| 11 | CUDA out-of-memory at epoch 20 (19 epochs lost), another program was using the GPU | Crash log | Resume checkpoint every epoch (model, optimiser, scheduler, scaler, EMA, random state) and automatic retry after an OOM; tested that crash + resume gives bit-identical weights |
| 12 | GPU training 10× slower than expected (35 img/s) | `bench_gpu.py` | Channels-last memory layout was 7× slower on this GPU/cuDNN; turned off by default: B0 35 → 261 img/s, B2-320 13 → 77 img/s |
| 13 | B2 at 320 px does not fit batch 32 in 4 GB | OOM | Batch 8 × 4 gradient-accumulation steps (effective 32) + mixed precision |
| 14 | Hard negatives taught the camera, not the disease | Held-out labelled Kvasir lesion check | Not adopted; root cause documented; earlier claim corrected |
| 15 | Whole-study ranking near chance | 43-video scan | Reported as the main limit; the app shows per-frame ranking, not a per-patient verdict |
| 16 | B2 at 0.5 caught fewer Kvasir lesions (79% vs v2's 88%) | External re-run | Matched-specificity comparison (B2 better at every false-alarm rate) + a Screening mode with a cut-off fixed on CrohnIPI (93.4% on Kvasir) |
| 17 | Landmarks flagged as lesions (v2: pylorus 64%, valve 45%) | Per-class flag rates on 47k frames | B2 lowered them to 25% / 17%; remaining rate is documented |
| 18 | Moving 31 MB model files laptop ↔ cloud (30 MB limit per file) | Transfer tool refused | Split into 16 MB parts, joined on the laptop, md5 checked for all 10 models |
| 19 | Large transfers timing out | Tool timeouts | Staged one file at a time; packed images as 224 px zips for the early CPU work |
| 20 | PyTorch could not be installed in the laptop's sandbox | Blocked download | ONNX export done in the cloud from the GPU checkpoints |
| 21 | ONNX parity looked wrong on a random-noise input (0.435) | Parity check | Re-checked on real CrohnIPI frames: max difference ~0.001, same class 100% |
| 22 | New API routes not reachable | Browser test | FastAPI matches routes in order; API routes moved before the static-site mount |
| 23 | Videos need a decoder; OpenCV may be missing on Windows | Design | OpenCV first, else ffmpeg from `imageio-ffmpeg` (pip), else a system ffmpeg |
| 24 | Big video uploads and unsafe file names | Design review | Stream to disk in 4 MB chunks, sanitise names, block path traversal, one scan at a time |
| 25 | UC grader weaker on a new hospital (0.943 → 0.746) | HyperKvasir test | Reported; temperature-scaled probabilities; whole-video UC test (0.981) shows video-level use works |
| 26 | Flare model only ties HBI | Baseline comparison | Reported honestly; probabilities recalibrated so risks are realistic |
| 27 | A slide fact was wrong (erythema count) | Cross-check against the CSV | Fixed to 125; every headline number is now generated from one results file |
| 28 | Dataset licences (CrohnIPI: research use only) | Licence review before publishing | No datasets in the repository; only CC BY 4.0 sample images; trained weights marked research/non-commercial |
| 29 | The demo study took 1.5–3 minutes with 10 B2 networks on a laptop CPU | Fresh-clone test before the evaluation | Result of the fixed demo study cached once per model set (now loads in under a second); uploaded studies are still scored live |

---

## 11. Code map

Repository layout (this is the `interim_build` folder on the laptop):

```
.
├── README.md                 start here: what it is, how to run, data download
├── app/                      the web app (runs on any CPU)
│   ├── run.py                one-command launcher: python run.py
│   ├── requirements.txt      fastapi, uvicorn, python-multipart, onnxruntime, numpy, pillow, imageio-ffmpeg
│   ├── backend/              FastAPI service
│   ├── frontend/             Next.js + TypeScript + Tailwind source
│   ├── frontend_dist/        pre-built website (static export) served by the backend
│   ├── models/               ONNX models, heads, metas, results_summary.json
│   ├── samples/              CC BY sample frames, 160-frame demo study, video thumbnails
│   ├── user_videos/          drop capsule videos here to scan them
│   ├── Dockerfile, docker-compose.yml, DOCKER.md
├── gpu/                      GPU training + evaluation kit (PyTorch, RTX 3050)
├── ml/                       first training code (cloud CPU): probes, v1/v2 fine-tuning, flare, ONNX export
├── ml_device/                laptop scripts (onnxruntime): features, heads, external tests, video scans, summaries
├── results/                  every metric as JSON + out-of-fold predictions (results/b2_oof)
└── docs/                     this guide + the development log
```

### 11.1 Backend (`app/backend/`)

| File | What it does |
| --- | --- |
| `app.py` | FastAPI app: all routes, study upload (.zip or many files, up to 400), demo study (cached in `models/demo_study_cache.json`), sample gallery, UC, flare, video scan jobs; mounts `/samples` and the website |
| `core.py` | ONNX Runtime session factory (GPU if `onnxruntime-gpu` is installed and `CROHNSIGHT_DEVICE` is not `cpu`), image reading, resize/normalise, softmax, CAM normalisation, thumbnails |
| `capsule.py` | Loads the Crohn's ensemble listed in `crohnipi_meta.json`; `analyse()` returns probabilities, spread, tier, lesion flags (balanced and screening), agreement, heat-maps; `study_summary()` |
| `video_scan.py` | Video folder listing, safe upload paths, decoder choice, two-stage scan as background jobs, heat-map for any stored top frame |
| `heads.py` | Shared EfficientNet-B0 ONNX backbone + linear heads: UC Mayo grader (temperature-scaled) and SEE-AI findings screen |
| `flare.py` | Random forest stored as NumPy arrays (no scikit-learn), Platt calibration, per-feature contributions for "what-if" |

### 11.2 API

| Method | Path | Purpose |
| --- | --- | --- |
| GET | `/api/health` | Which models are loaded |
| GET | `/api/meta` | Classes, tiers, thresholds (balanced and screening), operating points, model info |
| GET | `/api/results` | Every metric on the Results page (`results_summary.json`) |
| POST | `/api/capsule/analyze` | One frame → classes, tier, lesion score, heat-maps, agreement |
| POST | `/api/capsule/study` | Many frames or a .zip → ranked queue + summary |
| GET | `/api/capsule/demo-study` | The bundled 160-frame Kvasir study with expert labels |
| GET | `/api/samples` | Sample gallery (skips images that are not on disk) |
| POST | `/api/uc/grade` | Colonoscopy image → Mayo 0–3 with probabilities |
| GET | `/api/flare/patients`, `/api/flare/patient/{id}`, `/api/flare/schema` | HMP2 timelines with out-of-fold risk; feature schema |
| POST | `/api/flare/predict` | What-if flare risk with feature contributions |
| GET | `/api/videos`, `/api/videos/{vid}` | The 43 offline-scanned Kvasir videos: timelines, top frames, expert labels |
| GET | `/api/video/local` | Videos in the scan folder + which decoder is available |
| POST | `/api/video/upload` | Stream a video file into the scan folder |
| POST | `/api/video/scan` | Start a scan job `{name, target}` |
| GET | `/api/video/job/{id}` | Job progress and result |
| GET | `/api/video/job/{id}/frame/{n}` | Full 10-model analysis + heat-map for one top frame |

### 11.3 Website (`app/frontend/`)

| Page | Route | What it shows |
| --- | --- | --- |
| Overview | `/` | What CrohnSight does, headline numbers, links |
| Lesion Analyzer | `/capsule` | Upload/sample frame → class bars, severity tier, heat-map overlay (per class), agreement, balanced and screening cut-offs |
| Study Triage | `/study` | Many frames → ranked queue, summary, Balanced/Screening switch, demo study with expert labels |
| Video Scan | `/video` | **Scan your own capsule video** (upload or folder) + the 43 offline-scanned Kvasir studies |
| UC Grader | `/uc` | Colonoscopy image → Mayo grade |
| Flare Risk | `/flare` | HMP2 patient timelines, what-if calculator |
| Results | `/results` | Every result, including the negative ones |
| Data & Methods | `/data` | Datasets, licences, methods, model versions |

Components: `CamOverlay.tsx` (heat-map on canvas), `Mode.tsx` (Balanced/Screening switch + `useMeta`), `OwnVideo.tsx` (video upload/scan card), `Shell.tsx` (layout, navigation), `charts.tsx` (bars, timelines, calibration plots), `ui.tsx` (cards, image drop zone, stats). `lib/api.ts` holds the API types and fetch helpers.

### 11.4 Training and analysis scripts

**`gpu/` (PyTorch, NVIDIA GPU)**

| Script | Purpose |
| --- | --- |
| `train_gpu.py` | Train on CrohnIPI official folds (fixed recipe, EMA, AMP, 8-view TTA out-of-fold predictions, resume, OOM retry, optional hard negatives) |
| `evaluate_oof.py` | Metrics from out-of-fold files of one or more runs (ensembles), with cross-fitted rules |
| `export_to_app.py` | Export fold models to ONNX (+ CAM weights) into `app/models` (`--prefix`, `--no-switch`) |
| `external_gpu.py` | Re-run all external tests with GPU models (47k Kvasir frames, Kvasir subset, SEE-AI) |
| `eval_video_frames.py` | Hard-negative evaluation on held-out videos |
| `kvasir_groups.py` | Fixed split of the 43 Kvasir videos into halves A/B |
| `bench_gpu.py` | One-minute speed test (found the channels-last problem) |
| `README_GPU.md` | Step-by-step commands for the GPU work |

**`ml/` (cloud CPU, first phase)**: `data.py` (CrohnIPI loader), `backbones.py`, `extract_features.py` + `linear_probe.py` (frozen baselines), `finetune.py` (v1/v2), `metrics.py`, `export_onnx.py` / `export_fold.py` (ONNX with CAM output + parity check), `flare.py` / `flare_export.py`, `assemble_results.py` / `assemble_v2.py` / `assemble_b2.py` (metrics JSON), `screenshots.py`.

**`ml_device/` (laptop, onnxruntime)**: `pack.py` / `pack2.py` (resize + zip for transfer), `extract.py` / `extract_ft.py` (features), `train_heads.py` (UC + findings heads), `train_screen.py` (findings screen), `external_eval.py` (Kvasir subset + SEE-AI + pointing game), `bulk_score_kvasir_frames.py` + `merge_fold_scores.py` + `analyze_kvasir_full.py` (all 47,238 frames), `extract_video_frames.py`, `scan_kvasir_videos.py` + `analyze_kvasir_videos.py` (43 whole videos), `quality_filter.py` (rejected filter), `hyperkvasir_eval.py` (UC videos, polyp heat-maps), `hard_negative_head.py` / `_v2.py`, `update_results_summary.py` (builds `results_summary.json`).

### 11.5 Model files (`app/models/`)

| File | Content |
| --- | --- |
| `crohnipi_effb2s{0,1}_fold{1-5}.onnx` | The 10 Crohn's networks (31 MB each). Input `image` (N×3×320×320), outputs `out` (logits) and `fmap` (last feature map, for CAM) |
| `crohnipi_effb2s*_fold*_head.npz` | Classifier weights W, b for the heat-maps |
| `crohnipi_meta.json` | Which models to load (`model_prefixes`), input size, thresholds, operating points |
| `backbone_effb0_ra.onnx` | Shared ImageNet EfficientNet-B0 for the UC and findings heads |
| `uc_mayo_head.npz` + meta | UC logistic head + temperature |
| `sb_findings_head.npz` + meta | Findings-screen heads |
| `flare_rf.npz`, `flare_meta.json`, `flare_metrics.json`, `flare_timelines.json` | Flare forest, features, metrics, HMP2 timelines |
| `results_summary.json` | Every number the website shows |
| `video_study.json`, `video_timelines.json` | Offline scan of the 43 Kvasir videos |

To switch models, edit `model_prefixes` and `input_size` in `crohnipi_meta.json` and restart.

---

## 12. How to run, re-check and rebuild

### 12.1 Run the app

```bash
cd app
pip install -r requirements.txt
python run.py                 # opens http://127.0.0.1:8000
python run.py --no-browser    # server only;  set PORT=8080 (Windows) / PORT=8080 python run.py (Linux/macOS)
```

Docker: `cd app && docker compose up -d --build`, then http://localhost:8000 (details in `app/DOCKER.md`).

### 12.2 Scan your own capsule video

1. Open **Video Scan**. Either drag a video onto the upload box, or copy it into `app/user_videos/` and press **Refresh**.
2. Choose how many frames to score: 500 (fast), 2,000 (default), 5,000, or every frame.
3. Press **Scan**. Watch the progress; the result shows the score timeline (with the cut-off line), the top 24 frames ranked by the full ensemble, and a heat-map when you click a frame.

Formats: mp4, avi, mov, mkv, mpg, mpeg, wmv, m4v, webm. If the page says no decoder was found: `pip install imageio-ffmpeg`. A different folder can be used by setting the environment variable `CROHNSIGHT_VIDEO_DIR`.

### 12.3 Re-check the headline numbers without retraining

The out-of-fold predictions of the final model are in `results/b2_oof/seed0` and `seed1`. With the CrohnIPI CSV downloaded:

```bash
cd gpu
python evaluate_oof.py --runs ../results/b2_oof/seed0 ../results/b2_oof/seed1
# add  --data "<path to DataCrohnIPI>"  if the dataset is not in ../../temorary solution/DataCrohnIPI
```

This prints AUROC 0.992, cross-fitted accuracy 96.7%, accuracy at 0.5 96.3%, 7-class 91.1% (balanced 84.0%), QWK 0.942 / 0.951 (checked on 8 Oct from the repository files). The "current" column compares with the v1 model.

### 12.4 Retrain on a GPU

```bash
cd gpu
pip install torch timm pandas pillow onnx onnxruntime     # torch with CUDA from pytorch.org
python train_gpu.py --backbone effb2 --img 320 --epochs 40 --batch 8 --accum 4 --out runs_gpu/effb2_320
python train_gpu.py --backbone effb2 --img 320 --epochs 40 --batch 8 --accum 4 --seed 1 --out runs_gpu/effb2_320_s1
python evaluate_oof.py --runs runs_gpu/effb2_320 runs_gpu/effb2_320_s1
python export_to_app.py --runs runs_gpu/effb2_320 --backbone effb2 --img 320 --prefix crohnipi_effb2s0 --no-switch
python export_to_app.py --runs runs_gpu/effb2_320_s1 --backbone effb2 --img 320 --prefix crohnipi_effb2s1 --no-switch
```

Then set `model_prefixes` in `app/models/crohnipi_meta.json`. On an RTX 3050 one B2 run (5 folds × 40 epochs) takes about 2.5 hours. If training crashes, run the same command again: it resumes after the last finished epoch.

### 12.5 Re-run the external tests

```bash
python gpu/external_gpu.py                                   # GPU, ~20 min: writes per-frame scores
MODEL_PREFIX=crohnipi_effb2 python ml_device/analyze_kvasir_full.py
MODEL_PREFIX=crohnipi_effb2 FOLDS=1,2,3,4,5,6,7,8,9,10 python ml_device/external_eval.py report
USE_V2=1 USE_B2=1 python ml_device/update_results_summary.py # refresh the website's numbers
```

### 12.6 Rebuild the website

```bash
cd app/frontend
npm install
npm run build          # writes frontend/out
```

Replace the contents of `app/frontend_dist` with `frontend/out`.

### 12.7 Where the data must be

Scripts expect the datasets in a folder called `temorary solution` (spelling kept from the original laptop layout) **next to** the repository folder:

```
<parent>/
├── <this repository>/
└── temorary solution/
    ├── DataCrohnIPI/                 CrohnIPI_description.csv, imgs/
    ├── Kvasir-Capsule/               labelled_images/, labelled_videos/, metadata.csv
    ├── The SEE-AI Project Dataset/   SEE_AI_project_all_images/, all_annotation.csv
    ├── Labeled Images for Ulcerative Colitis (LIMUC)/
    ├── hyper-kvasir/
    └── hmp2/                         hmp2_metadata_2018-08-20.csv
```

---

## 13. Our evaluation rules (why the numbers can be trusted)

1. **Official CrohnIPI folds only.** Same splits as the published papers.
2. **Out-of-fold everything.** Every reported CrohnIPI number comes from a model that did not see that frame.
3. **Fixed recipes, no epoch picking.** The number of epochs and the EMA model were decided before training; we never pick the best epoch on the test fold.
4. **Cross-fitting** for the two tuned rules (lesion threshold, severity cut-points).
5. **Decisions written down before scoring.** "Average both seeds", "switch the app only if B2 beats v2", "screening cut-off = 99% CrohnIPI sensitivity" were all fixed before we looked at the result they decide.
6. **Never tune on external data.** Kvasir-Capsule and SEE-AI are test-only. No threshold, filter or model choice was made on them.
7. **Grouped splits wherever groups exist:** patients (LIMUC, HMP2), videos (Kvasir).
8. **Confidence intervals** on every headline number, and the type of bootstrap stated.
9. **Negative results reported**, and wrong claims corrected publicly (section 9).
10. **One source of numbers.** The website reads `results_summary.json`, which is generated from the raw result files.

---

## 14. Limitations

* **Frame level, not patient level.** Whole-study "lesion / no lesion" is near chance (0.57). The tool ranks frames for a reader; it does not diagnose a patient.
* **No patient IDs in CrohnIPI.** Internal numbers may be slightly optimistic; the external tests are the real check.
* **One training hospital and camera** (Nantes, PillCam). Sensitivity drops on other cameras at the balanced cut-off (79% Kvasir), which is why Screening mode exists.
* **Erythema is the hardest class** (Kvasir AUROC 0.785): redness depends strongly on camera colour.
* **Landmarks still raise false alarms** (pylorus 25%, ileocecal valve 17% flagged at 0.5).
* **UC grader** drops on a new hospital (0.746 vs 0.943).
* **Flare model** ties the HBI baseline.
* **Not clinically validated.** No prospective study, no regulatory approval.

---

## 15. Future work

1. **MIMIC-IV modules** (the original plan) when credentialing is approved: onset and course prediction from records.
2. **More in-domain lesion data** (other capsule brands, many patients) to fix study-level ranking; then retry hard negatives with negatives *and* lesions from the same cameras.
3. **Study-level model**: sequence model over frame scores (temporal context, lesion runs), trained on many full studies.
4. **Patient-grouped CrohnIPI splits** if the authors release patient IDs.
5. **UC grader fine-tuning** on the GPU with LIMUC (the script structure in `gpu/` can be reused).
6. **Calibration per camera** (temperature scaling per device) to reduce the erythema / colour gap.
7. **Clinical reader study**: time saved and lesions missed with vs without CrohnSight.

---

## 16. Viva questions and answers

**Q1. Why EfficientNet?**
Best accuracy for its size on ImageNet, small enough for a 4 GB laptop GPU at 320 px and for CPU inference. B0 was the CPU-friendly start; B2 is the next size that fits.

**Q2. Why 320 px?**
It is the native CrohnIPI frame size. At 224 px small lesions lose detail. Going to native resolution plus B2 raised 7-class accuracy from 87.9% to 91.1%.

**Q3. Why rotate images during training?**
A capsule frame is a circle with no up or down. A lesion rotated is still the same lesion. This multiplies the data for free and is also why 8-view TTA is valid.

**Q4. What is EMA?**
An exponential moving average of the weights during training (decay 0.998). It smooths out noise from the last updates and usually generalises better. We evaluate the EMA weights, decided in advance.

**Q5. What is cross-fitting and why do it?**
Choosing a threshold on the same data you report on makes the result optimistic. We choose it on four folds and apply it to the fifth, for each fold. The reported 96.7% is therefore honest.

**Q6. What is quadratic weighted kappa?**
Agreement between two ordered gradings, corrected for chance, where big disagreements (no lesion vs severe) cost more than small ones (mild vs moderate). 0.94 means near-perfect agreement with the expert grade.

**Q7. Why is your accuracy higher than the published papers?**
Native resolution, a modern network and recipe (EMA, label smoothing, class weights, strong rotation augmentation), and an ensemble of 10 models. Same official folds, so the comparison is fair. Each single seed alone already gives 96.5–96.7%.

**Q8. How do you know it is not overfitting?**
Out-of-fold evaluation, fixed recipe, and above all two external datasets from other countries and cameras where it scores AUROC 0.90–0.99 for ulcers, erosions and stenosis without retraining.

**Q9. What does the heat-map show and how do you know it is right?**
Class activation map: which regions drove the decision. We checked it against expert boxes (pointing game): 69% hits on Kvasir vs 4% by chance, 93% on SEE-AI stenosis.

**Q10. What is the difference between Balanced and Screening mode?**
Two cut-offs on the same lesion score. Balanced (0.5) keeps false alarms low. Screening (0.31) was fixed on CrohnIPI to catch 99% of lesions and catches 93% on Kvasir, at the cost of more false alarms (32% of normal frames). A doctor reviewing a study would use Screening; a quick second opinion would use Balanced.

**Q11. Why did you not just lower the threshold until Kvasir looked good?**
That would be tuning on the test set. The screening cut-off was chosen on CrohnIPI only, before any external test.

**Q12. Why does the model fail at the whole-video level?**
Lesion-free videos still contain frames that look suspicious (bubbles, debris, landmarks), and a lesion video may have only a handful of lesion frames among 50,000. To learn this the model needs many lesion videos from the same camera; Kvasir has lesions in only 13 videos.

**Q13. What was the hard-negative experiment and why did it fail?**
We added normal frames from raw Kvasir videos to training. Because all those "normal" frames came from one camera and all lesions from another, the network learned to recognise the camera. Detection of real Kvasir lesions fell from 87% to 28%. We did not adopt it.

**Q14. What is data leakage and how did you show it?**
When near-identical data ends up in both training and test. On Kvasir, random frame splits gave 95.7% balanced accuracy, video-grouped splits 17.9%. Frames from one video are almost the same picture.

**Q15. Why is the app fast without a GPU?**
ONNX Runtime with an optimised graph, one view per model, batches of 16, and the heat-map comes from the same forward pass.

**Q16. How does video scanning stay fast?**
Two stages: one network scores every Nth frame; only the top 24 frames (spread out in time) go through all 10 networks.

**Q17. Why a random forest for flares and why does it not beat HBI?**
Small tabular data with missing values: forests handle that well. It ties HBI because there are only 61 flares in 27 patients; the strongest signal is the HBI itself.

**Q18. How does the UC grader work?**
Frozen ImageNet EfficientNet-B0 features and a logistic regression trained on LIMUC with patient-grouped CV; temperature scaling for calibrated probabilities.

**Q19. What would you do with more time?**
MIMIC-IV modules, more in-domain capsule data, a sequence model for whole studies, and a reader study with gastroenterologists.

**Q20. Is the data use legal?**
All datasets are public. CrohnIPI is for research use only (CC BY-NC-ND); we use it for research, do not redistribute it, and our public repository contains no dataset images.

**Q21. What is AUROC?**
The probability that a randomly chosen lesion frame gets a higher score than a randomly chosen normal frame. 0.5 is chance, 1.0 perfect. It does not depend on a cut-off.

**Q22. Why report balanced accuracy and macro-F1 too?**
With 61% normal frames, a model could score high accuracy by favouring "normal". Balanced accuracy averages recall over classes; macro-F1 averages F1 over classes. Ours: 84.0% and 0.84.

**Q23. Why 10 models and not one?**
Averaging models trained on different data splits and seeds lowers variance and gives an uncertainty signal (how many of 10 agree). One seed already meets the targets; the ensemble adds stability.

**Q24. Which classes are hardest?**
Aphthoid ulcers vs small ulcers (they look alike and differ mainly in size) and erythema (colour depends on the camera).

**Q25. What did you learn?**
Test on other hospitals, group by patient/video, fix decisions before looking, and treat a surprising gain as a bug until proven otherwise (the hard-negative "fix" taught us this).

---

## 17. Glossary

| Term | Meaning |
| --- | --- |
| AUROC | Area under the ROC curve; ranking quality from 0.5 (chance) to 1.0 |
| Sensitivity (recall) | Share of real lesions the model flags |
| Specificity | Share of normal frames the model leaves unflagged |
| Balanced accuracy | Mean of per-class recalls |
| Macro-F1 | Mean F1 over classes, each class weighted equally |
| QWK | Quadratic weighted kappa, agreement on ordered grades |
| Out-of-fold (OOF) | Prediction for a frame from the fold model that did not train on it |
| Cross-fitting | Tuning a rule on other folds and applying it to the held-out fold |
| Bootstrap CI | Confidence interval from resampling (frames, videos or patients) |
| EMA | Exponential moving average of weights |
| TTA | Test-time augmentation: predict on several transformed copies and average |
| AMP | Automatic mixed precision (float16 compute on the GPU) |
| CAM | Class activation map (heat-map) |
| Pointing game | Hit if the heat-map's hottest point falls inside the expert box |
| ONNX | Open format for trained networks; run with ONNX Runtime without PyTorch |
| Hard negatives | Difficult normal examples added to training to cut false alarms |
| Domain shift | Change in image statistics between hospitals or cameras |
| Leakage | Test data that is not truly independent of training data |
| HBI | Harvey-Bradshaw Index, a Crohn's symptom score |
| Mayo score | Endoscopic UC severity 0–3 |
| Lewis / CECDAI | Capsule-endoscopy Crohn's activity scores |

---

## 18. References and data sources

**Datasets**

* CrohnIPI: Vallée R, de Maissin A, Coutrot A, Mouchère H, Bourreille A, Normand N. *CrohnIPI: An endoscopic image database for the evaluation of automatic Crohn's disease lesions recognition algorithms.* Proc. SPIE 11317, Medical Imaging 2020. Data: https://crohnipi.ls2n.fr/ (Zenodo 14616507). CC BY-NC-ND, research use only.
* de Maissin A, Vallée R, Flamant M, et al. *Added value of a multi-expert annotation of Crohn's disease images of the small bowel for automatic detection of lesions using a convolutional recurrent attention neural network.* Endoscopy International Open, 2021. (Benchmark results we compare with.)
* Kvasir-Capsule: Smedsrud PH, Thambawita V, Hicks SA, et al. *Kvasir-Capsule, a video capsule endoscopy dataset.* Scientific Data 8, 142 (2021). OSF dv2ag. CC BY 4.0.
* SEE-AI: Yokote A, et al. *Small bowel capsule endoscopy examination and open access database with artificial intelligence: The SEE-artificial intelligence project.* DEN Open (2024). Kaggle: capsuleyokote/the-see-ai-project-dataset.
* LIMUC: Polat G, Kani HT, Ergenc I, et al. *Improving the computer-aided estimation of ulcerative colitis severity according to Mayo endoscopic score by using regression-based deep learning.* Inflammatory Bowel Diseases (published online 2022). Zenodo 5827695. CC BY 4.0.
* HyperKvasir: Borgli H, Thambawita V, Smedsrud PH, et al. *HyperKvasir, a comprehensive multi-class image and video dataset for gastrointestinal endoscopy.* Scientific Data 7, 283 (2020). OSF mh9sj. CC BY 4.0.
* HMP2 / IBDMDB: Lloyd-Price J, Arze C, Ananthakrishnan AN, et al. *Multi-omics of the gut microbial ecosystem in inflammatory bowel diseases.* Nature 569, 655–662 (2019). https://ibdmdb.org

**Methods**

* Tan M, Le Q. *EfficientNet: Rethinking model scaling for convolutional neural networks.* ICML 2019.
* Zhou B, Khosla A, Lapedriza A, Oliva A, Torralba A. *Learning deep features for discriminative localization (CAM).* CVPR 2016.
* Wightman R. *PyTorch Image Models (timm).* https://github.com/huggingface/pytorch-image-models
* Szegedy C, et al. *Rethinking the Inception architecture (label smoothing).* CVPR 2016.

**Clinical**

* Gralnek IM, et al. *Development of a capsule endoscopy scoring index for small bowel mucosal inflammatory change (Lewis score).* Aliment Pharmacol Ther 2008.
* Gal E, Geller A, Fraser G, et al. *Assessment and validation of the new Capsule Endoscopy Crohn's Disease Activity Index (CECDAI).* Dig Dis Sci 2008.
* Harvey RF, Bradshaw JM. *A simple index of Crohn's-disease activity.* Lancet 1980.
