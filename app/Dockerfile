# CrohnSight - inference-only web app (FastAPI + ONNX Runtime + pre-built Next.js site). CPU by default.
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PORT=8000 CROHNSIGHT_DEVICE=cpu
# onnxruntime needs the OpenMP runtime
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY backend ./backend
COPY models ./models
COPY samples ./samples
COPY frontend_dist ./frontend_dist
COPY run.py README.md ./

# run as a non-root user
RUN useradd --create-home --uid 1000 crohnsight && chown -R crohnsight /app
USER crohnsight

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s CMD python -c "import urllib.request,os; urllib.request.urlopen(f'http://127.0.0.1:{os.environ.get(\"PORT\",\"8000\")}/api/health', timeout=4)"
CMD ["sh", "-c", "uvicorn backend.app:app --host 0.0.0.0 --port ${PORT} --workers 1"]
