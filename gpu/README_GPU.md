# Training CrohnSight on your NVIDIA GPU (Windows)

This folder retrains the Crohn's lesion model (CrohnIPI, official 5-fold CV) on your RTX 3050 and plugs the new model into the app.
Everything stays honest: fixed recipe, no epoch picking on the test fold, and the two decision rules are cross-fitted.

## 0. One-time setup (PowerShell, in this folder)

```powershell
cd "E:\D\study\MP_Cohn Disease\interim_build\gpu"
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
pip install timm pandas pillow onnx onnxruntime
```

The first line must print `True` and `NVIDIA GeForce RTX 3050 ...`. If it prints `False`, PyTorch was installed without CUDA:
`pip install torch --index-url https://download.pytorch.org/whl/cu124` (or the CUDA version shown on pytorch.org).

The first training run downloads the ImageNet weights once from Hugging Face (internet needed only then).

## 1. Train (about 10-20 min per fold on an RTX 3050)

```powershell
# same backbone as the app, longer schedule
python train_gpu.py --backbone effb0 --epochs 30
# bigger backbone (more accurate, ~2x slower); batch 16 x 2 accumulation fits 4 GB
python train_gpu.py --backbone effb2 --epochs 30 --batch 16 --accum 2
```

Each run writes `runs_gpu/<backbone>/fold1..5.pt` plus out-of-fold predictions. You can run one fold at a time: `--folds 1`.
If you see `CUDA out of memory`, use `--batch 8 --accum 4`.

## 2. Score it (and compare with the model in the app)

```powershell
python evaluate_oof.py --runs runs_gpu/effb0
python evaluate_oof.py --runs runs_gpu/effb2 runs_gpu/effb0 --with-current   # ensemble of everything
```

The table prints each metric next to the current app model and your targets (90% 7-class, 96% binary, κ 0.90).
"Cross-fitted" means each fold's threshold or cut-points were chosen on the other four folds only.

## 3. Put the new model into the app

```powershell
python export_to_app.py --runs runs_gpu/effb2 --backbone effb2
cd ..\app ; python run.py
```

This writes `crohnipi_effb2_fold1..5.onnx` into `app/models` and switches `crohnipi_meta.json` to them.
The old EfficientNet-B0 files are kept; to go back, set `"model_prefix": "crohnipi_effb0"` in `app/models/crohnipi_meta.json`.
`http://127.0.0.1:8000/api/health` shows which model is loaded.

## 4. Run the web app on the GPU (optional)

The app is fast on CPU (under 1 s per frame). For big studies on the GPU:

```powershell
pip uninstall -y onnxruntime
pip install onnxruntime-gpu
python run.py
```

`/api/health` then reports `"device": "CUDAExecutionProvider"`. Set `CROHNSIGHT_DEVICE=cpu` to force the CPU.
onnxruntime-gpu needs the CUDA 12 / cuDNN 9 libraries; the app loads the ones installed with PyTorch automatically.

## What the recipe does

| Item | Setting |
| --- | --- |
| Data | CrohnIPI 3,484 labelled frames, official folds, full frame resized to 224 x 224 |
| Optimiser | AdamW, lr 6e-4, weight decay 0.01, 1 warm-up epoch, cosine decay |
| Loss | cross-entropy, label smoothing 0.1, square-root inverse-frequency class weights |
| Augmentation | rotation to any angle, flips, zoom 0.85-1.1, small shifts, brightness / contrast / saturation jitter |
| Weights used | exponential moving average (decay 0.998), decided in advance |
| Test-time | 8 views: 4 rotations x horizontal flip |
| Mixed precision | automatic on CUDA (fp16) |

A CPU check of this recipe on fold 1 (15 epochs) gave 88.1% 7-class accuracy and 94.3% lesion-vs-normal accuracy at 0.5,
against 86.5% and 92.0% for the current app model on the same fold.

## 5. Fix whole-video ranking with hard negatives (about 25 min on the RTX 3050)

On whole Kvasir-Capsule videos the current model finds lesions inside a study but barely tells lesion studies from
lesion-free ones (AUROC 0.57): raw video has bubbles, debris, stomach and colon frames it never saw in CrohnIPI.
**Result (8 Oct 2026): not adopted.** Study AUROC rose (0.51 -> 0.61; 0.72 on the top-1% score), but on held-out
expert-labelled Kvasir frames lesion sensitivity fell from 87% to 28% (AUROC 0.91 -> 0.77): every negative came from
Kvasir and every lesion from CrohnIPI, so the model learned the camera. The head-only version (0.57 -> 0.74) has the
same problem, and adding Kvasir lesion frames did not repair it (`ml_device/hard_negative_head_v2.py`). Kept for the record.
This step adds frames of lesion-free videos as extra "normal" training images. The 43 videos are split into two halves;
each model only gets negatives from one half and is tested on the other half, so no video tests a model that saw it.

The frames are already extracted on this laptop: `interim_build/data_kvasir_video_frames` (39,604 frames + index.csv).

```powershell
cd "E:\D\study\MP_Cohn Disease\interim_build\gpu"
python train_gpu.py --backbone effb0 --folds all --epochs 20 --kvasir-index ..\data_kvasir_video_frames\index.csv --kvasir-group A --out runs_gpu\hn_A
python train_gpu.py --backbone effb0 --folds all --epochs 20 --kvasir-index ..\data_kvasir_video_frames\index.csv --kvasir-group B --out runs_gpu\hn_B
python eval_video_frames.py --pt-A runs_gpu\hn_A\all.pt --pt-B runs_gpu\hn_B\all.pt --backbone effb0
```

`eval_video_frames.py` prints and saves `results/hard_negative_eval.json`: whole-study AUROC and lesion recall for the
new models against the original model on the same frames, plus frame-level AUROC on the expert-labelled frames of the
held-out half. `--folds all` trains on every CrohnIPI frame (no CrohnIPI test fold), so judge these models on the video results.

## 6. Speed check (if training is slow)

```powershell
python bench_gpu.py
```

Prints images per second for B0 (224 px) and B2 (320 px) with and without channels-last and mixed precision.
An RTX 3050 should train B0 at a few hundred images per second. If it is far lower, plug in the charger, set
Armoury Crate to Performance or Turbo, close other programs that use the GPU (`nvidia-smi` lists them), and use the
fastest setting from the table.

Measured on this laptop (8 Oct 2026): channels-last memory layout made training 7x slower (B0 35 vs 261 images/s;
B2 at 320 px 13 vs 77 images/s), so `train_gpu.py` now leaves it off unless you pass `--channels-last`.
