# Run CrohnSight in Docker (any computer)

The image holds the whole app: FastAPI backend, the 10 ONNX models, sample images and the built website (about 0.7 GB). It needs no GPU, no datasets and no internet at run time. Datasets are not part of the image: they live in the private Hugging Face dataset (see `DATA_RESTORE.md` / `hf_upload/README.md`).

## 1. Build and run on your laptop

Install **Docker Desktop** (Windows/macOS) or Docker Engine (Linux), then in this folder (`interim_build\app`):

```powershell
docker compose up -d --build
```

Open http://localhost:8000 and stop it with `docker compose down`. Videos you want to scan go in `app\user_videos`, which is shared with the container.

## 2. Put the image on Docker Hub as a private image

The image contains models trained on research-use-only data (CrohnIPI), so keep the repository **private**.

1. Sign in at https://hub.docker.com as `hilkalathiya59`, choose **Create repository**, name it `crohnsight`, set visibility to **Private**, and create it. Create it before the first push: a push to a repository that does not exist yet creates it with your account's default visibility, which is usually public.
2. With Docker Desktop running and signed in as `hilkalathiya59`, double-click `docker_build_test.bat` in this folder. It builds the image, test-runs it on port 8765, checks the health endpoint and the main pages, and ends with `RESULT: BUILD AND TEST OK` (the first build takes 5–10 minutes; everything is also written to `docker_build_log.txt`).
3. Double-click `docker_push.bat`. It refuses to push if `hilkalathiya59/crohnsight` is publicly visible, pushes, checks the visibility again and ends with `RESULT: PUSH OK (private)` (log: `docker_push_log.txt`).

The same by hand, in PowerShell in this folder:

```powershell
docker login -u hilkalathiya59
docker compose build
docker push hilkalathiya59/crohnsight:1.0
```

If Docker Hub says your plan has no private repository left, do not push it as public; keep the image on your laptop (step 1) instead.

## 3. Run it on any other computer

```powershell
docker login -u hilkalathiya59
mkdir user_videos
docker run -d -p 8000:8000 -v "${PWD}\user_videos:/app/user_videos" --name crohnsight hilkalathiya59/crohnsight:1.0
```

(On Linux or macOS use `-v "$PWD/user_videos:/app/user_videos"`.) Open http://localhost:8000. Update later with `docker pull hilkalathiya59/crohnsight:1.0`, then remove and re-run the container.

## 4. GPU inside Docker (optional)

Not needed: the CPU image analyses a frame in about a second. For a GPU container you need the NVIDIA Container Toolkit, a CUDA base image and `onnxruntime-gpu` instead of `onnxruntime`; then run with `--gpus all -e CROHNSIGHT_DEVICE=auto`.

## 5. Free hosting with a link (optional)

Hugging Face Spaces can run the same Dockerfile: create a **private** Docker Space, upload this folder except `frontend/`, add `app_port: 7860` to the Space's README header and `ENV PORT=7860` to the Dockerfile before `EXPOSE`.

## Licence note

The image contains sample frames from CrohnIPI (CC BY-NC-ND, research use only), Kvasir-Capsule, HyperKvasir, LIMUC (CC BY 4.0) and SEE-AI (Kaggle licence), and models trained on them. Keep any copy private or share it only with your evaluators, non-commercially and with the attribution on the Data & Methods page.
