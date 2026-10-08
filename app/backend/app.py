"""CrohnSight API (FastAPI). Serves the static Next.js build from ../frontend_dist and the JSON API under /api."""
import os, io, zipfile, json, time
from typing import List
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
from .core import ROOT, MODELS, read_image, thumb_b64, load_json
from .capsule import CapsuleModel, study_summary, CLASSES, NAMES, TIERS
from .heads import SharedBackbone, uc_grade, sb_findings
from .flare import FlareModel
from .video_scan import VideoJobs, list_local, local_path, upload_target

app = FastAPI(title="CrohnSight API", version="1.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
capsule = CapsuleModel(); backbone = SharedBackbone(); flare = FlareModel(); video_jobs = VideoJobs(capsule)
SAMPLES = os.path.join(ROOT, "samples")
MAX_FILES = 400
IMG_EXT = (".jpg", ".jpeg", ".png", ".bmp")

def _small(im):
    """Keep memory flat for big studies: frames are analysed at 224 px anyway."""
    if max(im.size) > 448: im = im.copy(); im.thumbnail((448, 448))
    return im

def _images_from_uploads(files: List[UploadFile]):
    out = []
    for f in files:
        data = f.file.read()
        if f.filename.lower().endswith(".zip"):
            with zipfile.ZipFile(io.BytesIO(data)) as z:
                for n in sorted(z.namelist()):
                    if n.lower().endswith(IMG_EXT) and not n.startswith("__MACOSX"):
                        out.append((os.path.basename(n), _small(read_image(z.read(n)))))
        else:
            out.append((f.filename, _small(read_image(data))))
        if len(out) > MAX_FILES: raise HTTPException(413, f"Max {MAX_FILES} frames per study")
    if not out: raise HTTPException(400, "No images found")
    return out

@app.get("/api/health")
def health():
    prov = capsule.folds[0][0].get_providers()[0] if capsule.folds else None
    return dict(ok=True, capsule_models=len(capsule.folds), capsule_model=capsule.prefix, device=prov,
                uc=("uc_mayo_head" in backbone.heads), sb_findings=("sb_findings_head" in backbone.heads), flare=flare.ok)

@app.get("/api/meta")
def meta():
    return dict(classes=[dict(code=c, name=NAMES[c]) for c in CLASSES], tiers=TIERS, threshold=capsule.threshold,
                screening_threshold=capsule.screen_threshold, operating_points=capsule.meta.get("operating_points"),
                results=load_json("results_summary.json", {}))

@app.get("/api/results")
def results():
    return load_json("results_summary.json", {})

@app.post("/api/capsule/analyze")
async def capsule_analyze(file: UploadFile = File(...)):
    if not capsule.ready: raise HTTPException(503, "Capsule model files missing")
    im = read_image(await file.read()); t = time.time(); r = capsule.analyse([im])[0]
    if "sb_findings_head" in backbone.heads: r["findings"] = sb_findings(backbone, [im])[0]
    r["ms"] = round((time.time() - t) * 1000); return r

@app.post("/api/capsule/study")
async def capsule_study(files: List[UploadFile] = File(...)):
    if not capsule.ready: raise HTTPException(503, "Capsule model files missing")
    items = _images_from_uploads(files); return _study(items)

def _study(items):
    t = time.time(); names, ims = zip(*items); res = capsule.analyse(list(ims), with_cam=False)
    findings = sb_findings(backbone, list(ims)) if "sb_findings_head" in backbone.heads else [None] * len(ims)
    frames = []
    for i, (n, im, r, f) in enumerate(zip(names, ims, res, findings)):
        frames.append(dict(idx=i, name=n, thumb=thumb_b64(im), lesion_score=r["lesion_score"], top_class=r["top_class"],
                           top_name=r["top_name"], tier=r["tier"], tier_name=r["tier_name"], tier_color=r["tier_color"],
                           probs=r["probs"], findings=f))
    s = study_summary(res, capsule.threshold, capsule.screen_threshold); s["seconds"] = round(time.time() - t, 1)
    return dict(summary=s, frames=frames)

DEMO_CACHE = os.path.join(MODELS, "demo_study_cache.json")

@app.get("/api/capsule/demo-study")
def demo_study():
    """The bundled 160-frame study is fixed input, so its result is computed once per model set and cached
    (10 B2 networks x 160 frames take a minute or more on a laptop CPU). Delete the cache file to recompute."""
    d = os.path.join(SAMPLES, "demo_study"); man = load_json_abs(os.path.join(d, "manifest.json"))
    if not man: raise HTTPException(404, "demo study not bundled")
    files = [m["file"] for m in man["frames"]]
    c = load_json_abs(DEMO_CACHE)
    if c and c.get("model") == capsule.prefix and c.get("files") == files:
        out = dict(c["result"], cached=True)
    else:
        items = [(f, _small(read_image(open(os.path.join(d, f), "rb").read()))) for f in files]
        out = _study(items)
        try:
            with open(DEMO_CACHE, "w") as fh: json.dump(dict(model=capsule.prefix, files=files, result=out), fh)
        except OSError: pass  # read-only install: just recompute next time
        out = dict(out, cached=False)
    out["source"] = man.get("source"); out["truth"] = {m["file"]: m.get("label") for m in man["frames"]}; return out

def load_json_abs(p):
    return json.load(open(p)) if os.path.exists(p) else None

@app.get("/api/samples")
def samples():
    """Sample gallery. Entries whose image is not on disk are skipped: the public repository leaves out
    research-use-only images (CrohnIPI, SEE-AI), so a fresh clone shows only the CC BY samples."""
    d = load_json_abs(os.path.join(SAMPLES, "samples.json")) or {}
    have = lambda s: os.path.exists(os.path.join(SAMPLES, s.get("url", "").replace("/samples/", "", 1)))
    return {k: [s for s in v if have(s)] if isinstance(v, list) else v for k, v in d.items()}

@app.post("/api/uc/grade")
async def uc(file: UploadFile = File(...)):
    if "uc_mayo_head" not in backbone.heads: raise HTTPException(503, "UC model missing")
    im = read_image(await file.read()); return uc_grade(backbone, [im])[0]

class FlareIn(BaseModel):
    features: dict

@app.get("/api/flare/patients")
def flare_patients():
    if not flare.ok: raise HTTPException(503, "flare model missing")
    return dict(patients=[dict(id=k, visits=len(v), max_hbi=max(x["hbi"] for x in v)) for k, v in sorted(flare.timelines.items())])

@app.get("/api/flare/patient/{pid}")
def flare_patient(pid: str):
    if pid not in flare.timelines: raise HTTPException(404, "unknown patient")
    return dict(id=pid, timeline=flare.timelines[pid])

@app.post("/api/flare/predict")
def flare_predict(body: FlareIn):
    if not flare.ok: raise HTTPException(503, "flare model missing")
    return flare.predict(body.features)

@app.get("/api/flare/schema")
def flare_schema():
    return dict(features=[dict(key=f, label=flare.nice[f]) for f in flare.features], metrics=flare.metrics) if flare.ok else {}

# ---- whole capsule videos (Kvasir-Capsule, scanned offline by ml_device/scan_kvasir_videos.py) ----
def _video_data():
    return load_json("video_timelines.json", {}), {v["video_id"]: v for v in load_json("video_study.json", {}).get("videos", [])}

@app.get("/api/videos")
def videos():
    tl, st = _video_data()
    out = []
    for v, d in tl.items():
        s = st.get(v, {}); a = s.get("all", {}); sb = s.get("sb_filtered", a)
        out.append(dict(id=v, total_frames=d["total_frames"], has_inflammatory=d["has_inflammatory"], expert_lesion_frames=len(d["expert"]),
                        peak=max(b[1] for b in d["bins"]) if d["bins"] else 0, flagged_pct=a.get("flagged_pct"), flagged_pct_sb=sb.get("flagged_pct"),
                        recall_top5=s.get("lesion_recall_top5pct"), other_findings=s.get("other_findings", [])))
    out.sort(key=lambda r: (not r["has_inflammatory"], -r["expert_lesion_frames"], r["id"]))
    return dict(videos=out, study=load_json("video_study.json", {}).get("study"))

@app.get("/api/videos/{vid}")
def video(vid: str):
    tl, st = _video_data()
    if vid not in tl: raise HTTPException(404, "unknown video")
    d = dict(tl[vid]); s = st.get(vid, {})
    tdir = os.path.join(SAMPLES, "videos", vid)
    for t in d["top"]:
        f = f"frame_{t['frame']:06d}.jpg"
        t["thumb"] = f"/samples/videos/{vid}/{f}" if os.path.exists(os.path.join(tdir, f)) else None
        t["expert_nearby"] = next((c for n, c in d["expert"] if abs(n - t["frame"]) <= 10), None)
    d["stats"] = s; d["id"] = vid
    return d

# ---------- Module 1d: scan the user's own video (upload, or a file in the videos folder) ----------
class VideoScanIn(BaseModel):
    name: str
    target: int = 2000

@app.get("/api/video/local")
def video_local():
    return list_local()

@app.post("/api/video/upload")
async def video_upload(file: UploadFile = File(...), target: int = 2000):
    if not capsule.ready: raise HTTPException(503, "Capsule model files missing")
    try: path = upload_target(file.filename)
    except ValueError as e: raise HTTPException(400, str(e))
    with open(path, "wb") as out:
        while True:
            chunk = await file.read(1 << 22)
            if not chunk: break
            out.write(chunk)
    return video_jobs.start(path, target)

@app.post("/api/video/scan")
def video_scan(req: VideoScanIn):
    if not capsule.ready: raise HTTPException(503, "Capsule model files missing")
    try: path = local_path(req.name)
    except FileNotFoundError as e: raise HTTPException(404, str(e))
    return video_jobs.start(path, max(50, min(req.target, 200000)))

@app.get("/api/video/job/{jid}")
def video_job(jid: str):
    job = video_jobs.get(jid)
    if not job: raise HTTPException(404, "unknown scan")
    return video_jobs.public(job)

@app.get("/api/video/job/{jid}/frame/{frame}")
def video_job_frame(jid: str, frame: int):
    try: return video_jobs.frame(jid, frame)
    except KeyError as e: raise HTTPException(404, str(e))

# static files: samples + built frontend
if os.path.isdir(SAMPLES): app.mount("/samples", StaticFiles(directory=SAMPLES), name="samples")
DIST = os.path.join(ROOT, "frontend_dist")
if os.path.isdir(DIST):
    app.mount("/", StaticFiles(directory=DIST, html=True), name="ui")
