# CrohnSight app (Minor Project 2026, PDEU)

AI-based early detection and severity grading of Crohn's disease from capsule endoscopy,
with two add-ons: an ulcerative-colitis Mayo grader and a flare-risk forecaster.
Research prototype — not a medical device.

Team: Hil Kalathiya (23BIT092), Sahaj Saliya (23BIT128) · Guide: Prof. Dr. Babalu Kumar Singh

## Run the app (Windows / macOS / Linux, CPU only)

```
cd app
pip install -r requirements.txt
python run.py
```

The browser opens http://127.0.0.1:8000 . No GPU, PyTorch or internet is needed at demo time.
`python run.py --no-browser` starts the server without opening a browser; `set PORT=8080` changes the port.

## What is inside

| Folder | Contents |
| --- | --- |
| `backend/` | FastAPI service: ONNX inference, class-activation heat-maps, study triage, UC grader, flare forest |
| `frontend/` | Next.js + TypeScript + Tailwind source of the website |
| `frontend_dist/` | Pre-built website (static export) served by the backend |
| `models/` | 10 fine-tuned EfficientNet-B2 models at 320 px (5 folds × 2 seeds, ONNX) + their CAM weights; `crohnipi_meta.json` (which models load, balanced 0.5 and screening 0.31 cut-offs); shared EfficientNet-B0 backbone with the UC and findings heads; flare forest (NumPy); all metrics (`results_summary.json`) |
| `samples/` | Demo frames and a 160-frame demo study from one Kvasir-Capsule video, plus thumbnails of the 43 offline-scanned videos. The public repository holds only the CC BY images (Kvasir-Capsule, HyperKvasir, LIMUC); gallery entries whose file is missing are skipped |
| `user_videos/` | Put capsule videos here to scan them from the Video Scan page (or set `CROHNSIGHT_VIDEO_DIR`) |

Training code lives one level up: `gpu/` (PyTorch on an NVIDIA GPU: the final B2 model), `ml/` (first-phase CPU training, flare model, ONNX export)
and `ml_device/` (onnxruntime: features, heads, external tests, video scans). See `docs/PROJECT_GUIDE.md`.

## API

| Method | Path | Purpose |
| --- | --- | --- |
| GET | `/api/health` | which models are loaded |
| GET | `/api/meta` | classes, tiers, balanced and screening cut-offs, operating points |
| POST | `/api/capsule/analyze` | one capsule frame → 7-class probabilities, severity tier, heat-maps, uncertainty |
| POST | `/api/capsule/study` | many frames or a .zip → ranked review queue + study summary |
| GET | `/api/capsule/demo-study` | the bundled Kvasir-Capsule demo study |
| POST | `/api/uc/grade` | colonoscopy image → Mayo 0–3 |
| GET | `/api/flare/patients`, `/api/flare/patient/{id}` | HMP2 timelines with out-of-fold risk |
| POST | `/api/flare/predict` | what-if flare risk with feature contributions |
| GET | `/api/results` | every metric shown on the Results page |
| GET | `/api/videos`, `/api/videos/{vid}` | the 43 Kvasir-Capsule videos scanned offline |
| GET | `/api/video/local` | videos in the scan folder + available decoder |
| POST | `/api/video/upload` | stream a video into the scan folder |
| POST | `/api/video/scan` | start a two-stage scan `{name, target}` |
| GET | `/api/video/job/{id}`, `/api/video/job/{id}/frame/{n}` | scan progress/result; full analysis + heat-map of one top frame |

## Rebuild the website after editing it

```
cd frontend
npm install
npm run build
```
then replace `frontend_dist` with the new `frontend/out` folder.

## Data sources (all open, no application needed)

CrohnIPI (Zenodo 14616507, CC BY-NC-ND; see dataset Readme) · Kvasir-Capsule (OSF dv2ag, CC BY 4.0) · SEE-AI (Kaggle, see its dataset licence) ·
LIMUC (Zenodo 5827695, CC BY 4.0) · HyperKvasir (OSF mh9sj, CC BY 4.0) · HMP2 / IBDMDB.
Sample images are shown for demonstration with attribution. Video decoding: OpenCV if installed, else ffmpeg from `imageio-ffmpeg`.
