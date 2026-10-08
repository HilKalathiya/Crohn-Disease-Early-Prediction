"""Module 1d - scan the user's own capsule video (upload, or a file placed in the videos folder).

Same two-stage design as the offline whole-video scan, sized for a laptop:
  stage 1  one network of the app ensemble scores every Nth frame (N chosen so about `target` frames are scored)
  stage 2  the full ensemble re-scores the most suspicious frames (spread out in time); heat-maps on request
Runs as a background job; the website polls /api/video/job/<id>.
Decoding: OpenCV if installed, otherwise ffmpeg (pip install imageio-ffmpeg bundles one; a system ffmpeg also works).
"""
import heapq, io, math, os, re, shutil, subprocess, threading, time, uuid
import numpy as np
from PIL import Image
from .core import ROOT, to_input, softmax, thumb_b64

VIDEO_DIR = os.environ.get("CROHNSIGHT_VIDEO_DIR", os.path.join(ROOT, "user_videos"))
VIDEO_EXT = (".mp4", ".avi", ".mov", ".mkv", ".mpg", ".mpeg", ".wmv", ".m4v", ".webm")
TIER_OF = np.array([0, 1, 1, 2, 2, 3, 3])
TOP_K, CANDIDATES, BATCH = 24, 240, 32


def list_local():
    os.makedirs(VIDEO_DIR, exist_ok=True)
    vids = []
    for n in sorted(os.listdir(VIDEO_DIR)):
        p = os.path.join(VIDEO_DIR, n)
        if os.path.isfile(p) and n.lower().endswith(VIDEO_EXT):
            st = os.stat(p); vids.append(dict(name=n, mb=round(st.st_size / 1e6, 1), modified=int(st.st_mtime)))
    return dict(folder=os.path.abspath(VIDEO_DIR), videos=vids, decoder=decoder_name())


def local_path(name):
    n = os.path.basename(name); p = os.path.join(VIDEO_DIR, n)
    if not (os.path.isfile(p) and n.lower().endswith(VIDEO_EXT)):
        raise FileNotFoundError(f"{n} is not a video in {VIDEO_DIR}")
    return p


def upload_target(filename):
    """a safe, non-clashing path in the videos folder for an uploaded file"""
    os.makedirs(VIDEO_DIR, exist_ok=True)
    base = re.sub(r"[^A-Za-z0-9._-]+", "_", os.path.basename(filename or "video.mp4")) or "video.mp4"
    if not base.lower().endswith(VIDEO_EXT):
        raise ValueError("Unsupported file type; use " + ", ".join(VIDEO_EXT))
    stem, ext = os.path.splitext(base); p = os.path.join(VIDEO_DIR, base); i = 1
    while os.path.exists(p):
        p = os.path.join(VIDEO_DIR, f"{stem}_{i}{ext}"); i += 1
    return p


def _cv2():
    try:
        import cv2
        return cv2
    except Exception:
        return None


def _ffmpeg():
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return shutil.which("ffmpeg")


def decoder_name():
    return "OpenCV" if _cv2() else ("ffmpeg" if _ffmpeg() else None)


def probe(path):
    """(frame count, fps, width, height)"""
    cv2 = _cv2()
    if cv2:
        cap = cv2.VideoCapture(path)
        if cap.isOpened():
            n, fps = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), float(cap.get(cv2.CAP_PROP_FPS) or 0)
            w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)); cap.release()
            if w > 0 and h > 0:
                return max(n, 0), fps, w, h
    exe = _ffmpeg()
    if not exe:
        raise RuntimeError("No video decoder found. Install one:  pip install imageio-ffmpeg")
    s = subprocess.run([exe, "-hide_banner", "-i", path], capture_output=True, text=True, errors="replace").stderr
    m = re.search(r"Duration: (\d+):(\d+):([\d.]+)", s)
    dur = int(m[1]) * 3600 + int(m[2]) * 60 + float(m[3]) if m else 0.0
    v = re.search(r"Video:.*?, (\d{2,5})x(\d{2,5})[ ,\[]", s)
    f = re.search(r", ([\d.]+) fps", s) or re.search(r", ([\d.]+) tbr", s)
    if not v:
        raise RuntimeError("Could not read this video file")
    fps = float(f[1]) if f else 0.0
    return (int(round(dur * fps)) if dur and fps else 0), fps, int(v[1]), int(v[2])


def iter_frames(path, stride, w, h):
    """yield (frame index, PIL RGB image) for every stride-th frame"""
    cv2 = _cv2()
    if cv2:
        cap = cv2.VideoCapture(path)
        if cap.isOpened():
            i = 0
            try:
                while cap.grab():
                    if i % stride == 0:
                        ok, fr = cap.retrieve()
                        if ok:
                            yield i, Image.fromarray(np.ascontiguousarray(fr[:, :, ::-1]))
                    i += 1
            finally:
                cap.release()
            return
    exe = _ffmpeg()
    cmd = [exe, "-v", "error", "-i", path, "-vf", f"select=not(mod(n\\,{stride}))", "-vsync", "0",
           "-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, bufsize=10 ** 7)
    size, k = w * h * 3, 0
    try:
        while True:
            b = p.stdout.read(size)
            if len(b) < size:
                break
            yield k * stride, Image.frombytes("RGB", (w, h), b); k += 1
    finally:
        p.stdout.close()
        if p.poll() is None:
            p.kill()
        p.wait()


def _jpeg(im, size=320, q=90):
    b = io.BytesIO(); im.convert("RGB").resize((size, size), Image.BICUBIC).save(b, "JPEG", quality=q); return b.getvalue()


class VideoJobs:
    def __init__(self, capsule):
        self.capsule, self.jobs, self.lock, self.run_lock = capsule, {}, threading.Lock(), threading.Lock()

    def start(self, path, target=2000):
        jid = uuid.uuid4().hex[:10]
        job = dict(id=jid, name=os.path.basename(path), status="queued", stage="waiting for the previous scan", done=0, todo=0,
                   started=time.time(), error=None, result=None, _images={})
        with self.lock:
            self.jobs[jid] = job
            for old in sorted(self.jobs.values(), key=lambda j: j["started"])[:-8]:  # keep the 8 most recent
                self.jobs.pop(old["id"], None)
        threading.Thread(target=self._run, args=(job, path, int(target)), daemon=True).start()
        return self.public(job)

    @staticmethod
    def public(job):
        return {k: v for k, v in job.items() if not k.startswith("_")}

    def get(self, jid):
        return self.jobs.get(jid)

    def frame(self, jid, frame):
        job = self.jobs.get(jid)
        if not job or frame not in job["_images"]:
            raise KeyError("frame not stored for this scan")
        im = Image.open(io.BytesIO(job["_images"][frame])).convert("RGB")
        r = self.capsule.analyse([im], with_cam=True)[0]
        r["image"] = "data:image/jpeg;base64," + __import__("base64").b64encode(job["_images"][frame]).decode()
        return r

    def _run(self, job, path, target):
        with self.run_lock:  # one scan at a time: they share the same CPU / GPU
            try:
                self._scan(job, path, target)
            except Exception as e:  # report, never crash the server
                job.update(status="error", error=f"{type(e).__name__}: {e}")

    def _scan(self, job, path, target):
        t0 = time.time(); cap = self.capsule
        n, fps, w, h = probe(path)
        stride = max(1, math.ceil(n / max(1, target))) if n else 10
        job.update(status="running", stage="scanning", total_frames=n, fps=fps, width=w, height=h, stride=stride,
                   todo=(n + stride - 1) // stride if n else 0)
        sess = cap.folds[0][0]  # stage 1: the first network of the app ensemble
        frames, scores, probs, cands = [], [], [], []  # cands: min-heap of (score, frame, jpeg)
        batch, idx = [], []

        def flush():
            x = to_input(batch, cap.size)
            p = softmax(sess.run(["out"], {"image": x})[0]); s = 1 - p[:, 0]
            for fi, im, si, pi in zip(idx, batch, s, p):
                frames.append(fi); scores.append(float(si)); probs.append(pi)
                if len(cands) < CANDIDATES or si > cands[0][0]:
                    item = (float(si), fi, _jpeg(im))
                    heapq.heappush(cands, item) if len(cands) < CANDIDATES else heapq.heapreplace(cands, item)
            job["done"] += len(batch); batch.clear(); idx.clear()

        for fi, im in iter_frames(path, stride, w, h):
            batch.append(im); idx.append(fi)
            if len(batch) >= BATCH:
                flush()
        if batch:
            flush()
        if not frames:
            raise RuntimeError("No frames could be decoded from this video")
        total = n if n else frames[-1] + 1
        F, S, P = np.array(frames), np.array(scores), np.stack(probs)
        # stage 2: the most suspicious frames, at least `gap` frames apart, re-scored by the full ensemble
        job.update(stage="re-scoring the top frames with all models")
        gap = max(stride * 5, total // 400); picked = []
        for s_, f_, jp in sorted(cands, reverse=True):
            if all(abs(f_ - q[1]) >= gap for q in picked):
                picked.append((s_, f_, jp))
            if len(picked) >= TOP_K:
                break
        ims = [Image.open(io.BytesIO(jp)).convert("RGB") for _, _, jp in picked]
        ens = cap.analyse(ims, with_cam=False) if ims else []
        top = []
        for (s1, f_, jp), r, im in zip(picked, ens, ims):
            job["_images"][int(f_)] = jp
            top.append(dict(frame=int(f_), time_s=round(f_ / fps, 1) if fps else None, scan_lesion=round(s1, 4),
                            lesion=round(r["lesion_score"], 4), tier=r["tier"], tier_name=r["tier_name"], top_class=r["top_class"],
                            top_name=r["top_name"], agree_count=r["agree_count"], n_models=r["n_models"], thumb=thumb_b64(im, 160)))
        top.sort(key=lambda t: -t["lesion"])
        # timeline (<= 600 bins of [first frame, max, mean]) and summary from stage 1
        nb = min(600, len(F)); edges = np.linspace(0, len(F), nb + 1).astype(int)
        bins = [[int(F[a]), round(float(S[a:b].max()), 4), round(float(S[a:b].mean()), 4)] for a, b in zip(edges[:-1], edges[1:]) if b > a]
        tierp = np.stack([P[:, TIER_OF == t].sum(1) for t in range(4)], 1); tiers = tierp.argmax(1)
        thr, sthr = cap.threshold, cap.screen_threshold
        job["result"] = dict(
            total_frames=int(total), fps=fps, duration_s=round(total / fps, 1) if fps else None, width=w, height=h, stride=stride,
            scored=int(len(F)), seconds=round(time.time() - t0, 1), stage1_model=f"{cap.prefix.split('+')[0]} fold 1", n_models=len(cap.folds),
            thresholds=dict(balanced=thr, screening=sthr),
            flagged_pct=dict(balanced=round(100 * float((S >= thr).mean()), 1), screening=round(100 * float((S >= sthr).mean()), 1)),
            tier_counts=[int((tiers == t).sum()) for t in range(4)], burden_index=round(float(100 * (tierp @ np.arange(4)).mean() / 3), 1),
            bins=bins, top=top)
        job.update(status="done", stage="done", done=job["todo"] or len(F))
