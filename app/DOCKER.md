# Run CrohnSight in Docker (any computer) and put it online

The image holds the whole app: FastAPI backend, ONNX models, sample images and the built website. It needs no GPU and no internet at run time.

## 1. Build and run on your laptop

Install **Docker Desktop** (Windows/macOS) or Docker Engine (Linux), then in this folder (`interim_build/app`):

```powershell
docker compose up -d --build
```

Open http://localhost:8000 . Stop it with `docker compose down`.
Without compose: `docker build -t crohnsight:1.0 .` then `docker run -d -p 8000:8000 --name crohnsight crohnsight:1.0`.

The first build downloads the `python:3.11-slim` base image and the Python packages (about 250 MB). Later builds are cached.

## 2. Use it from anywhere

### Option A - Hugging Face Spaces (free, gives a public or private https link)

1. Create an account at https://huggingface.co and click **New Space**.
2. Space SDK: **Docker** (blank template). Hardware: **CPU basic (free)**. Visibility: **Private** is recommended (see the licence note below).
3. Upload the contents of this `app` folder **except `frontend/`** (web upload, or `huggingface-cli upload <user>/<space> . --repo-type space`).
4. Add this block at the very top of the Space's `README.md`:

   ```
   ---
   title: CrohnSight
   sdk: docker
   app_port: 7860
   ---
   ```

5. Add one line to the `Dockerfile` just before `EXPOSE`:  `ENV PORT=7860`
6. The Space builds by itself. Your link is `https://huggingface.co/spaces/<user>/<space>`.

### Option B - Docker Hub (anyone with Docker can pull it)

```powershell
docker login
docker tag crohnsight:1.0 <your-dockerhub-user>/crohnsight:1.0
docker push <your-dockerhub-user>/crohnsight:1.0
# on any other computer
docker run -d -p 8000:8000 <your-dockerhub-user>/crohnsight:1.0
```

## 3. GPU inside Docker (optional)

Not needed: the CPU image analyses a frame in under a second. For a GPU container you need the NVIDIA Container Toolkit,
a CUDA base image and `onnxruntime-gpu` instead of `onnxruntime`; then run with `--gpus all -e CROHNSIGHT_DEVICE=auto`.

## Licence note before making it public

The image contains sample frames from CrohnIPI (CC BY-NC-ND), Kvasir-Capsule, HyperKvasir, LIMUC (CC BY 4.0) and SEE-AI (Kaggle licence),
and models trained on them. Non-commercial academic use with attribution is fine; keep a public deployment non-commercial,
keep the attribution on the Data & Methods page, or keep the Space private and share it only with your evaluators.
