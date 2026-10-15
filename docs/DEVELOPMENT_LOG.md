
## 2026-10-06 — Interim build for the 9 Oct evaluation (MIMIC-IV on hold)
**Why:** MIMIC-IV credentialing still pending; professor asked for a temporary but fully functional build on other data. Original onset/course plan is ON HOLD, not dropped.

**Data audit**
- Kaggle "MIMIC-IV cleaned transcripts" rejected: 232,158 rows but 209,951 exact duplicates (table join), only 7 patients / 13 admissions, 0 Crohn's, no ICD codes/labs/dates.
- Added open datasets (Hil downloaded): Kvasir-Capsule (Norway), SEE-AI (Japan), LIMUC (Turkey), HyperKvasir (Norway); plus CrohnIPI (main) and HMP2.

**Built (all in `interim_build/`)**
- Crohn's lesion model: EfficientNet-B0 fine-tuned x5 (official CrohnIPI folds), ONNX export, CAM heat-maps.
- External validation on Kvasir-Capsule + SEE-AI, pointing game, leakage experiment.
- UC Mayo grader (LIMUC -> HyperKvasir), SEE-AI findings screen (experimental), HMP2 flare forest (recalibrated).
- Web app `interim_build/app`: FastAPI + Next.js, runs with `python run.py` (CPU only). Pages: Overview, Lesion Analyzer, Study Triage, UC Grader, Flare Risk, Results, Data & Methods.
- Report doc (Claude Docs: "CrohnSight — Interim Report & Architecture") and 20-slide evaluation deck ("CrohnSight — Minor Project Evaluation") in the required 9-section format.

**Key results**
- CrohnIPI 5-fold: lesion-vs-normal AUROC 0.985 (0.981-0.989), accuracy 94.3% (published 93.7-94.6%), 7-class 86.8% (balanced 77.6%), 4-tier severity kappa 0.89.
- External, no retraining: Kvasir ulcer 0.971 / erosion 0.856 / erythema 0.787; SEE-AI stenosis 0.845 / erythema 0.834 / erosion 0.827. Pointing game 52% (chance 4%) on Kvasir.
- Leakage: random frame split 95.7% vs video split 17.9% balanced accuracy (Kvasir, 9 classes).
- UC: remission vs active AUROC 0.943 (LIMUC) -> 0.746 (HyperKvasir). Flare: RF AUROC 0.711 = HBI-only baseline.

**Safe to delete after the evaluation:** `interim_build/_packed/` (incl. `parts/`), `_features/`, `_external/`, `_to_delete/`.

**Status:** Interim build complete. Next: MIMIC-IV onset/course modules when access is approved.

## 2026-10-07/08 — Full-dataset build (overnight)
**Why:** Hil asked to use every downloaded dataset, including the videos, with no weak spots left hidden.

**Data now used**
- Kvasir-Capsule: all 47,238 labelled frames scored by the 5-model ensemble; all 43 labelled full videos scanned (1,979,285 frames, every 10th scored, top 150 per study re-scored by 5 models); every 50th frame extracted for GPU work (`interim_build/data_kvasir_video_frames`, 39,604 frames).
- HyperKvasir: 46 colonoscopy videos (15 colitis vs 31 clean bowel) through the UC grader; 1,000 polyp images with masks for a heat-map check.

**Results**
- All 47,238 Kvasir frames: inflammatory vs normal AUROC 0.907 (95% CI 0.79-0.96, video bootstrap); ulcer 0.972, erosion 0.844, erythema 0.762; top 10/20/30% of frames hold 67/80/86% of lesion frames.
- Whole videos (43 studies): lesions found inside a study (median 56% of expert lesion frames in its top 5%, 71% in top 10%), but lesion vs lesion-free study AUROC 0.55 (chance). Small-bowel window: no gain. Landmark/poor-view filter: REJECTED (removed 42% of lesions on held-out videos).
- UC grader on whole colonoscopy videos: AUROC 0.981. Polyp heat-map: 33% hit vs 16% chance, detection 13% (capsule -> colonoscopy transfer is weak).
- Severity: cross-fitted tier cut-points give QWK 0.911 (target 0.90 met without retraining).
- Improved training recipe (15 epochs, EMA, 8-view TTA): fold 1 7-class 88.1% (was 86.5%), binary@0.5 94.3% (92.0%); fold 2 86.1% (84.9%). Folds 3-5 training overnight in the cloud.

**Built**
- App: new Video Scan page (timeline, expert labels, top frames, live heat-map), Results page sections for the new tests, GPU support (onnxruntime-gpu), model switch via `crohnipi_meta.json`, Docker (`app/Dockerfile`, `docker-compose.yml`, `DOCKER.md`).
- GPU kit `interim_build/gpu/`: train_gpu.py (RTX 3050, AMP, EMA, 8-view TTA, hard-negative mode), evaluate_oof.py, export_to_app.py, eval_video_frames.py, README_GPU.md.
- Deck: 21 slides (new whole-video results slide, registered title on cover); report updated.

**Next:** GPU runs on the RTX 3050 (bigger backbone; hard-negative fix for whole-video ranking); decide whether to switch the app to the improved model.

## 2026-10-08 (early morning) — App switched to the v2 model; first hard-negative fix
**v2 Crohn's model in the app** (`app/models/crohnipi_effb0v2_fold1..5.onnx`, `crohnipi_meta.json` -> `model_prefix: crohnipi_effb0v2`). Recipe fixed before scoring all folds: 15 epochs, EMA of weights (0.998), 8-view test-time augmentation. The v1 files are kept; to go back set `model_prefix` to `crohnipi_effb0`.

**CrohnIPI, official 5-fold, out-of-fold (v1 -> v2)**
- Lesion vs normal AUROC 0.985 -> 0.989 (95% CI 0.985-0.992); cross-fitted accuracy 94.3% -> 95.5% (best published 94.6%); at 0.5: 93.9%.
- 7-class accuracy 86.8% -> 87.9% (balanced 79.5%, macro-F1 0.78). Severity: 90.0% exact, 96.1% within one tier, QWK 0.907 (argmax) / 0.919 (cross-fitted cut-points).
- Targets: kappa >= 0.90 met; binary 95.5% vs 96% target; 7-class 87.9% vs 90% target (needs GPU: bigger backbone / longer schedule).
- Averaging v1 + v2 tested: tiny gain, not adopted (choosing the mix after seeing results would be selection on test data).

**External with v2**
- Kvasir-Capsule, all 47,238 labelled frames (5 models, per-fold shards in `results/bulk/kvasir_frames_crohnipi_effb0v2_f1..5`, merged in `kvasir_frames_crohnipi_effb0v2`): inflammatory vs normal AUROC 0.915 (95% CI 0.82-0.95); ulcer 0.961, erosion 0.873, erythema 0.803; top 10/20/30% of frames hold 71/81/87% of lesion frames; top-ranked labelled frame is a lesion in 9 of 11 lesion videos (v1: 7 of 11; both leave out 2 videos whose labelled frames are almost all lesions). File: `results/kvasir_full_external_crohnipi_effb0v2.json`.
- Earlier Kvasir subset / SEE-AI: inflammatory 0.921; SEE-AI stenosis 0.863, erythema 0.846, erosion 0.825; heat-map pointing: Kvasir 52%, SEE-AI erosion 73%, stenosis 91%. File: `results/crohnipi_external_crohnipi_effb0v2.json`.
- Whole videos (v2 stage 1): median 67% of expert lesion frames in a study's top 5%; lesion vs lesion-free study AUROC 0.57.

**First hard-negative fix (head only, laptop CPU)** — `interim_build/ml_device/hard_negative_head.py` -> `results/hard_negative_head.json`
- Frozen v2 fold-1 backbone; new final layer trained on CrohnIPI folds 2-5 + 400 frames from each lesion-free video of one half of the 43 videos; each half scored by the head trained on the other half. Settings fixed before the single run (C = 0.1, sqrt class weights, seed 0).
- Study AUROC 0.57 (CI 0.37-0.77) -> 0.74 (CI 0.55-0.90); gain CI +0.03 to +0.34. Control head without negatives: 0.58. CrohnIPI fold-1 AUROC 0.982 -> 0.983 (no damage). Cost: within-study recall at top 5% 67% -> 59%.
- Next: full-network hard-negative retrain on the RTX 3050 (`gpu/README_GPU.md` step 5).

**Updated:** app results page (v2 numbers + hard-negative stat), report doc, deck (v14). Correction noted: an earlier chat message said "9 of 13" videos; the clean v1 figure is 7 of 11 (v2: 9 of 11).

## 2026-10-08 (day) — Hard-negative experiments: not adopted (camera shortcut); earlier claim corrected
- Full GPU retrain on the RTX 3050 (`gpu/train_gpu.py --folds all --kvasir-group A/B`, EfficientNet-B0, 20 epochs, ~6,000 raw-video negatives per half; `gpu/eval_video_frames.py` -> `results/hard_negative_eval.json`): study AUROC 0.51 -> 0.61 (top-1% score 0.50 -> 0.72) on every 50th frame, but held-out expert-labelled Kvasir lesion frames: AUROC 0.91 -> 0.77, sensitivity at 0.5 87% -> 28%, specificity 76% -> 99.9%.
- Cause: every negative came from Kvasir (Olympus, Norway) and every lesion from CrohnIPI (PillCam, France), so the model learned "Kvasir camera = normal".
- Re-check of the head-only fix (`ml_device/hard_negative_head_v2.py` -> `results/hard_negative_head_v2.json`): same problem — study AUROC 0.57 -> 0.74 but held-out Kvasir lesion frames AUROC 0.75 -> 0.68 (sensitivity 65% -> 2%). Domain-balanced head (adds the training half's Kvasir lesion + normal frames): study AUROC 0.49, frames 0.65 — worse.
- CORRECTION: the morning entry called the head-only result a "first fix that works". It is not; slides (v15), report and website now state the trade-off. The app stays on v2.
- Conclusion: whole-study ranking is limited by in-domain lesion data (Kvasir: 1,519 inflammatory frames from 13 videos), not by compute.
- GPU speed: ~35 images/s for B0 training, about 10x below expectation; `gpu/bench_gpu.py` added to diagnose before the B2 / 320 px runs. `train_gpu.py` now saves a resume checkpoint every epoch and retries a batch after a GPU out-of-memory error.

## 2026-10-08 (afternoon) — GPU model: EfficientNet-B2 at 320 px, 2 seeds — all targets met, now the app model
- GPU speed fix: channels-last memory layout was 7x slower on the RTX 3050 (bench_gpu.py: B0 35 -> 261 img/s; B2-320 13 -> 77 img/s); train_gpu.py now leaves it off.
- Runs: `gpu/runs_gpu/effb2_320` (seed 0) and `effb2_320_s1` (seed 1): `train_gpu.py --backbone effb2 --img 320 --epochs 40 --batch 8 --accum 4`. Rule fixed before scoring: final model = average of both seeds; switch the app only if it beats v2.
- CrohnIPI official 5-fold, out-of-fold (verified independently from the raw oof/idx files; `results/crohnipi_b2_metrics.json`): AUROC 0.992 (CI 0.989-0.995); lesion accuracy 96.7% cross-fitted (sens 95.1%, spec 97.6%), 96.3% at 0.5 (sens 97.1%); 7-class 91.1% (balanced 84.0%, macro-F1 0.84); severity 93.1% exact, 97.7% within one tier, QWK 0.942 argmax / 0.951 cross-fitted. Each seed alone: 96.5-96.7% and 91.1-91.3%.
- Targets (7-class >= 90%, binary ~96%, kappa 0.9): all met.
- App: 10 ONNX models `app/models/crohnipi_effb2s{0,1}_fold{1-5}.onnx` (parity vs PyTorch checked on real CrohnIPI frames: 8-view ONNX vs OOF max diff ~1e-3, same class 100%); `crohnipi_meta.json` -> model_prefixes [crohnipi_effb2s0, crohnipi_effb2s1], input 320; backend/capsule.py loads several seeds. ~0.9 s per frame on the laptop VM CPU. v2 kept (backup meta in `_sync/backup_20261008`).
- Website, slides (local) and report updated to B2 for CrohnIPI. External tests re-run pending: `gpu/external_gpu.py` (GPU) then `MODEL_PREFIX=crohnipi_effb2 python3 ml_device/analyze_kvasir_full.py` and `MODEL_PREFIX=crohnipi_effb2 FOLDS=1,2,3,4,5,6,7,8,9,10 python3 ml_device/external_eval.py report`, then `USE_V2=1 USE_B2=1 python3 ml_device/update_results_summary.py`.

## 2026-10-08 (evening) — External tests re-run with the B2 model (GPU): everything now from one model
- `gpu/external_gpu.py` on the RTX 3050 (~20 min): all 47,238 Kvasir frames + the 8,266-frame Kvasir subset + 4,455 SEE-AI frames, 10 models, original images resized to 320, one view.
- All 47,238 Kvasir frames (`results/kvasir_full_external_crohnipi_effb2.json`): inflammatory vs normal AUROC 0.944 (CI 0.86-0.98; v2 0.915); at 0.5: sensitivity 79.4%, specificity 97.4% (normal frames flagged 2.6%, v2 26%); ulcer 0.989, erosion 0.920, erythema 0.785; top 10/20/30% of frames hold 81/88/91% of lesion frames; top-ranked labelled frame is a lesion in 10 of 11 lesion videos (v2 9/11, v1 7/11); pylorus flagged 25%, ileocecal valve 17% (v2 64%, 45%).
- Kvasir subset / SEE-AI (`results/crohnipi_external_crohnipi_effb2.json`): Kvasir inflammatory 0.944; SEE-AI erosion 0.897, stenosis 0.916, erythema 0.897; no-finding specificity 91.6%; heat-map pointing: Kvasir 68.8% (chance 4.1%), SEE-AI erosion 77.8% (17.6%), stenosis 92.8% (54.5%).
- results_summary, website, slides (v16, incl. a new Lesion Analyzer screenshot with the B2 model) and report updated. The whole-video scan still uses the B0 v2 models (CPU) and is labelled as such.

## 2026-10-08 (night) — Screening mode, scan-your-own-video, project guide, submission report, GitHub-ready repo
**Trade-off answered (B2 at 0.5 caught 79% of Kvasir lesion frames vs v2's 88%)** — `results/screening_mode_eval.json`
- B2 is more cautious, not worse: at v2's false-alarm rate (specificity 73.8%) B2 catches 92.1% vs 88.3%; at B2's own (97.4%) 79.4% vs 65.5%.
- Screening cut-off 0.3097 = score at which CrohnIPI out-of-fold lesion frames reach 99% sensitivity, fixed on CrohnIPI before any external test (cross-fitted on CrohnIPI: sens 98.8%, spec 85.6%). Kvasir 47k: sens 93.4%, spec 68.4%. SEE-AI: erosion 88.2%, stenosis 90.0%, erythema 89.8%, no-finding spec 68.8%.
- App: Balanced / Screening switch (Lesion Analyzer shows both cut-offs; Study Triage and Video Scan toggle); `crohnipi_meta.json` -> `screening_threshold`, `operating_points`.

**Scan your own capsule video** — `app/backend/video_scan.py`, `app/frontend/components/OwnVideo.tsx`
- Upload in the browser or copy the file into `app/user_videos/` (or set CROHNSIGHT_VIDEO_DIR). Stage 1: one network on every Nth frame (~500 / 2,000 / 5,000 / all); stage 2: top 24 frames (spread in time) re-scored by all 10 B2 models; heat-map on click. OpenCV, else ffmpeg (`pip install imageio-ffmpeg`).
- Tested on this laptop: Kvasir video 0531325b64674948 (16,151 frames), 150 frames + top 24 in 38 s (Linux VM, 2 cores).

**Docs and repository**
- `interim_build/docs/PROJECT_GUIDE.md`: full guide (story, data, methods, results, negative results, 29 problems + fixes, code map, how to run, viva Q&A, glossary).
- Claude Doc "CrohnSight — Project Report (Minor Project 2026-27)": short submission report (exportable to Word/PDF).
- `interim_build` is GitHub-ready: `.gitignore` (no datasets, no intermediates, no older models, no CrohnIPI/SEE-AI sample images), `.gitattributes`, new `README.md`, `results/b2_oof/` (out-of-fold predictions of the final model: `cd gpu && python evaluate_oof.py --runs ../results/b2_oof/seed0 ../results/b2_oof/seed1` reproduces 0.992 / 96.7% / 91.1% / 0.94). 1,254 files, ~335 MB. Push from Windows (see README / chat for commands).
- `/api/samples` now skips gallery images that are not on disk (a fresh clone has only the CC BY samples).
- Fresh-clone test (tracked files only, exported to a clean folder): app starts, 10 models + UC + findings + flare load, all pages 200, frame analysis works, gallery shows the 5 Kvasir samples. The 160-frame demo study took >2.5 min on the 2-core VM with 10 B2 models, so its fixed result is now cached once per model set (`app/models/demo_study_cache.json`, loads in 0.3 s; page says "cached result"). Stale website chunks from earlier builds moved to `_to_delete/frontend_dist_stale_*`.

## 2026-10-10/11 — Disk space: ~171 GB moved to _TO_DELETE, LIMUC compressed losslessly, data backup and Docker image prepared
**Why:** E: drive had 32 GB free; the datasets took ~161 GB and interim_build ~17 GB, while the project uses only ~10 GB of data.
- Measured: Kvasir unlabelled videos 55 GB, HyperKvasir unlabelled images 30 GB, HyperKvasir videos 31 GB (46 used), Kvasir labelled videos 32 GB (scan finished), duplicate zips 5 GB, interim_build caches ~14 GB.
- Moved (not deleted) into `MP_Cohn Disease/_TO_DELETE/` with original paths: 401 items, 167.8 GB, listed in `DATA_RESTORE_moved_items.tsv`; how to restore each group (OSF / Zenodo / Kaggle links, rebuild scripts) in `DATA_RESTORE.md`. Kept: 46 HyperKvasir UC-test videos, 3 small Kvasir videos (video-scan demo), per-video scan results, trained checkpoints, `_packed/*_index.csv`, `data_kvasir_video_frames`, `_external_crohnipi_effb2`.
- App checked after the move: 10 models + UC + findings + flare load, pages and demo study respond.
- MIMIC-IV: a 52 GB raw-CSV copy found in `data/MIMIC IV` (dated Aug 2025) was removed by Hil on 11 Oct; MIMIC work waits for Hil's own PhysioNet credentialing. Never upload MIMIC data anywhere.
- Compression: JPEG/MP4/AVI datasets do not shrink (zip keeps 97.6–100%), so they stay as they are. LIMUC was uncompressed BMP: converted to lossless PNG with `interim_build/ml_device/limuc_bmp_to_png.py` (every file decoded and compared pixel by pixel), 3.43 GB -> 1.32 GB, 11,277 files; BMP originals moved to `_TO_DELETE`. pack.py / pack2.py read any extension, so nothing else changes.
- Backup "from anywhere": private Hugging Face dataset (Hil's choice). `pack_for_hf.py` packs the ~9 GB working set into 15 tar shards in `hf_upload/` (paths relative to `temorary solution/`, sha256 in MANIFEST.json); `hf_upload/get_data.py` downloads, checks and unpacks them on any machine (tested end to end with a mock). Hil runs the pack + upload (needs his Hugging Face login).
- Docker: `app/.dockerignore` (drops v1/v2 models and user videos), `docker-compose.yml` (image `hilkalathiya59/crohnsight:1.0`, `user_videos` volume), `DOCKER.md` (private Docker Hub repo steps). Image contents tested by installing requirements.txt in a clean environment and running the app (all endpoints OK, ffmpeg decoder); the image itself could not be built in the cloud sandbox (Docker Hub blocked there), Hil builds it with Docker Desktop.
- After Hil deletes `_TO_DELETE`: E: free space goes from ~86 GB to ~256 GB.
- 11 Oct: `pack_for_hf.py` done on Windows (15 shards, 7.49 GB). Added `upload_to_hf.py`: reads the Hugging Face username from the login, creates the private dataset repo `<user>/crohnsight-data`, refuses if the repo is public, uploads `hf_upload/` (resumable). Note: upgrading huggingface_hub to 2.x in the global Python breaks transformers 4.57 (needs <1.0); use `pip install "huggingface_hub>=0.34,<1.0"` (the scripts work with both).
- 11 Oct: Hil uploaded the 15 shards (7.49 GB) to the private Hugging Face dataset `crohnsight-data`. Next: verify (file list + small restore test), then Shift+Delete `_TO_DELETE` (~171 GB) and the local `hf_upload/*.tar` copies (7.5 GB); commit + push the new files to GitHub.
- 11 Oct: Upload verified: repo hilkalathiya59/crohnsight-data is private, all 15 shards present; test restore of crohnipi + hmp2 (3,501 files) matched the originals (sample of 303 files byte-identical). Safe to delete `_TO_DELETE` and `hf_upload/*.tar`.
- 11 Oct: Hil deleted `_TO_DELETE`, `restore_test` and the local shards; E: now has 264 GB free.
- 11 Oct: Licence chosen for the GitHub repo: MIT for code and docs (`interim_build/LICENSE`, (c) 2026 Hil Kalathiya and Sahaj Saliya; Sahaj to confirm); trained weights, cached results and derived data under separate non-commercial research/teaching terms with no clinical use (`interim_build/MODELS_LICENSE.md`), because CrohnIPI is CC BY-NC-ND; sample images keep CC BY 4.0. README "Licences and use" section updated.
- 11 Oct: Docker helpers in `interim_build/app/`: `docker_build_test.bat` (build `hilkalathiya59/crohnsight:1.0`, test-run on port 8765, check /api/health and the pages, log to `docker_build_log.txt`) and `docker_push.bat` (refuses to push if the Docker Hub repo is publicly visible, re-checks after the push, log to `docker_push_log.txt`); both call `docker_tools.ps1`. Docker Desktop 4.94 on the laptop is signed in as hilkalathiya59 (Personal plan, 1 private repo). Done 12 Oct (below).
- 12 Oct: PROJECT_GUIDE.md updated (project doc copy too): new 2.11 (disk-space story), 5.8 (what was deleted, why it was safe, how to restore each group, what stays, why no zipping, Hugging Face backup and restore commands), problems #30-31, and in 12.2 'Test videos after the clean-up' (videos still on the laptop, 3 ways to feed a video incl. CROHNSIGHT_VIDEO_DIR, 30-s ffmpeg test clips, lesion-rich Kvasir videos to re-download). Tested on the laptop VM: erosion clip from 0531325b64674948 (first 30 s) -> 23.7% of 300 frames flagged, labelled erosion frames 246/228 ranked 3rd/7th; normal clip from df6b47bafe5143f5 (5:00-5:30) -> 4.7%; 106 s / 90 s.
- 12 Oct: Docker done by Hil: private Docker Hub repo `hilkalathiya59/crohnsight` created; `docker_build_test.bat` -> RESULT: BUILD AND TEST OK (health: 10 capsule models + UC + findings + flare on CPU; /, /capsule/, /video/, /api/video/local all 200; image 774 MB content, 1.88 GB on disk); `docker_push.bat` -> RESULT: PUSH OK (private), digest sha256:003b9e3d7e76. GitHub: commit 2d14ffe "MIT licence, model terms, Docker build and push scripts" pushed to origin/main. Guide 12.1 now says how to run the image on another computer.

## 2026-10-13 — IEEE-format review deck (professor's changes)
Professor asked: remove old slides 12 (app screenshots), 13 (MIMIC data audit), 19 (numbers-only conclusion); make slide 11 model-centric; replace slide 8 with an in-depth model architecture; add the base paper (de Maissin et al. 2021) with its limitations and how we solve them; literature backing for every model choice (lesion CNN, UC grader, flare model); training data and train/test partitions; dataset sample sizes and attributes; which image features the model uses; plots of accuracy, AUROC and the other measures; IEEE slide format.
- New Slides artifact "CrohnSight — IEEE Review Deck" (37 slides; the 9 Oct deck is kept unchanged): sections I–VI in IEEE order, numbered citations [1]–[27] in order of first appearance (checked by script), IEEE-style TABLE I–X and Fig. 1–13 captions, 3 reference slides in IEEE format, speaker notes on every slide. Research-style conclusion replaces the old numbers-only one.
- Base paper facts taken from the article (Endosc Int Open 2021;9:E1136–E1144): 3,498 frames, 66 videos, 63 patients, 3 French centres, PillCam SB3; 3 annotation rounds; RAN = VGG-16 glimpses + GRU, RL + CE, binary, 5 × 70/10/20 splits; acc. 88.53 → 92.37 → 93.70% with better labels; ResNet-34 94.58%, VGG-16 94.40%, VGG-19 94.35%; stated limits: single device, oedema/erythema "insufficient", explainability left to future work.
- Architecture slide built from the exported ONNX graph (onnx installed in the laptop VM): stem 3×3 s2 → 7 MBConv stages (2,3,3,4,4,5,2 blocks; 16,24,48,88,120,208,352 ch) → 1×1 head 1408 → 10×10 map → GAP → FC 7; 7,677,073 parameters.
- New figures (all computed from saved result files) in interim_build/docs/figures/ by ml_device/make_deck_figures.py: ROC + PR (AUROC 0.992, AP 0.989), 7-class and 4-tier confusion matrices, per-class P/R/F1, training loss of 10 runs + per-fold accuracy (89.2–92.3%), benchmark dot plot, external ROC (Kvasir ulcer 0.989 / erosion 0.920 / erythema 0.785; SEE-AI 0.916 / 0.897 / 0.897), UC ROC + confusion, flare ROC (one repeat: RF 0.718 vs HBI 0.711) + calibration, hand-crafted feature effect sizes, CAM examples with expert boxes.
- New analysis: 9 hand-crafted colour/texture/geometry cues on CrohnIPI (results/handcrafted_features.json): logistic regression on them reaches AUROC 0.754 and 7-class 37.3% vs the CNN's 0.992 / 91.1%; stenosis frames brighter (+1.05 SD), more pale fibrin (+1.70), smaller dark lumen (−0.70); large ulcers and oedema more focal red spots (+1.27, +1.14); erythema global redness ≈ normal (+0.02).
- Note: Kvasir all-frames erosion n = 514 (506 in the erosion folder + 8 pylorus-folder frames that also carry an erosion label in metadata.csv).
